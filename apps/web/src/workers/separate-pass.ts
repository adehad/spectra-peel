import {
  Resampler,
  concatFloat32 as concat,
  SeparationStream,
  type ResamplerState,
  type SeparationModel,
  type SeparationState,
  type SpectrogramState,
} from "@spectra-peel/engine";
import { LayerBlocks } from "../audio/layer-blocks";
import { dependencies } from "../project/layers";
import type { Layer, Project } from "../project/schema";
import { loadSnapshot, saveSnapshot } from "./checkpoint";
import { LayerWriter, TileWriter, mono, type LayerWriterState } from "./layer-writer";
import type { SeparateMessage } from "./protocol";

/** A separation graph and the chunking it runs at. */
export interface PassModel {
  model: SeparationModel;
  sampleRate: number;
  segment: number;
  overlap: number;
}

/** Everything a pass holds between input blocks, so a later run can carry on exactly. */
export interface PassState {
  /** The next input block to read. */
  block: number;
  written: number;
  /** Per channel, input read but not yet matched by output. */
  pendingInput: Float32Array[];
  /** Per layer and channel, output not yet written. */
  pendingStems: Float32Array[][];
  up: ResamplerState[];
  down: ResamplerState[][];
  stream: SeparationState;
  writers: LayerWriterState[];
  restTiles: SpectrogramState | null;
}

export interface Checkpoints {
  /** Where an earlier run of this pass stopped, or null to start from the beginning. */
  resume: PassState | null;
  save: (state: PassState) => Promise<void>;
}

/** Wall time between checkpoints: often enough to lose little, rare enough to cost little. */
const CHECKPOINT_MS = 30_000;

/** A snapshot from storage, checked only as far as telling it apart from something else entirely. */
function isPassState(v: unknown): v is PassState {
  return (
    typeof v === "object" &&
    v !== null &&
    "block" in v &&
    typeof v.block === "number" &&
    "writers" in v &&
    Array.isArray(v.writers) &&
    "stream" in v
  );
}

/** Checkpoints kept under `name` in the op's job folder. */
export async function passCheckpoints(
  dir: FileSystemDirectoryHandle,
  op: string,
  name: string,
): Promise<Checkpoints> {
  const saved = await loadSnapshot(dir, op, name);
  return {
    resume: isPassState(saved) ? saved : null,
    save: (state) => saveSnapshot(dir, op, name, state),
  };
}

/**
 * Posts progress with a time estimate from the work this run has done, since a resumed job starts
 * partway through and its time so far covers only the rest.
 */
export function progressPoster(post: (m: SeparateMessage) => void, t0: number) {
  let start: number | null = null;
  return (fraction: number, stage: string) => {
    start ??= fraction;
    const ran = fraction - start;
    const elapsed = (performance.now() - t0) / 1000;
    post({
      type: "progress",
      fraction,
      stage,
      etaSeconds: ran > 0.02 ? (elapsed / ran) * (1 - fraction) : null,
    });
  };
}

const queued = (parts: Float32Array[]) => parts.reduce((n, p) => n + p.length, 0);
/** Removes and returns the first `n` samples queued in `parts`. */
const take = (parts: Float32Array[], n: number) => {
  const out = new Float32Array(n);
  let at = 0;
  while (at < n && parts.length > 0) {
    const p = parts[0] ?? new Float32Array(0);
    const k = Math.min(n - at, p.length);
    out.set(p.subarray(0, k), at);
    at += k;
    if (k === p.length) parts.shift();
    else parts[0] = p.subarray(k);
  }
  return out;
};

/**
 * Streams `input` through one model and writes one FLAC layer per source it returns, in `layers`'
 * order. With `restTiles`, the layers are every part of the input, so the pass also builds the
 * spectrogram of the input minus all of them.
 */
export async function separatePass(
  dir: FileSystemDirectoryHandle,
  project: Project,
  input: string,
  pass: PassModel,
  layers: readonly Layer[],
  restTiles: TileWriter | null,
  onProgress: (written: number) => void,
  checkpoints: Checkpoints | null = null,
): Promise<void> {
  const rate = project.source.sampleRate;
  const channels = project.source.channels;
  const total = project.source.frames;
  const resume = checkpoints?.resume ?? null;
  const blocks = await LayerBlocks.open(dir, project, dependencies(project, input));
  const writers = await Promise.all(
    layers.map((l, i) => {
      if (l.data.kind !== "flac") throw new Error("separated layers must be FLAC");
      const state = resume?.writers[i];
      return state
        ? LayerWriter.resume(dir, l.id, l.data.path, l.data.index, rate, channels, state)
        : LayerWriter.create(dir, l.id, l.data.path, l.data.index, rate, channels);
    }),
  );
  const up = [new Resampler(rate, pass.sampleRate), new Resampler(rate, pass.sampleRate)];
  const down = layers.map(() => [new Resampler(pass.sampleRate, rate), new Resampler(pass.sampleRate, rate)]);
  const stream = new SeparationStream(pass.model, {
    sources: layers.length,
    segment: pass.segment,
    overlap: pass.overlap,
  });
  const pendingInput: Float32Array[][] = Array.from({ length: channels }, () => []);
  const pendingStems: Float32Array[][][] = layers.map(() => [[], []]);
  let written = 0;
  let first = 0;
  if (resume) {
    for (const [c, r] of up.entries()) {
      const state = resume.up[c];
      if (state) r.restore(state);
    }
    for (const [s, pair] of down.entries())
      for (const [c, r] of pair.entries()) {
        const state = resume.down[s]?.[c];
        if (state) r.restore(state);
      }
    stream.restore(resume.stream);
    for (const [c, p] of resume.pendingInput.entries()) pendingInput[c]?.push(p);
    for (const [s, pair] of resume.pendingStems.entries())
      for (const [c, p] of pair.entries()) pendingStems[s]?.[c]?.push(p);
    if (resume.restTiles) restTiles?.restore(resume.restTiles);
    written = resume.written;
    first = resume.block;
  }
  let saved = performance.now();
  const snapshot = (block: number): PassState => ({
    block,
    written,
    pendingInput: pendingInput.map(concat),
    pendingStems: pendingStems.map((pair) => pair.map(concat)),
    up: up.map((r) => r.snapshot()),
    down: down.map((pair) => pair.map((r) => r.snapshot())),
    stream: stream.snapshot(),
    writers: writers.map((w) => w.snapshot()),
    restTiles: restTiles?.snapshot() ?? null,
  });

  /** Turns the model's stereo output into the project's rate and channel count and writes what lines up. */
  const accept = async (out: Float32Array[][], final: boolean) => {
    for (const [s, pair] of out.entries()) {
      for (const [c, plane] of pair.entries()) {
        const r = down[s]?.[c];
        if (!r) continue;
        pendingStems[s]?.[c]?.push(r.push(plane));
        if (final) pendingStems[s]?.[c]?.push(r.finish());
      }
    }
    const ready = Math.min(
      queued(pendingInput[0] ?? []),
      ...pendingStems.flatMap((pair) => pair.map((p) => queued(p))),
      total - written,
    );
    if (ready <= 0) return;
    const restPlanes = Array.from({ length: channels }, (_, c) => take(pendingInput[c] ?? [], ready));
    for (const [s, writer] of writers.entries()) {
      const pair = pendingStems[s] ?? [];
      const l = take(pair[0] ?? [], ready);
      const r = take(pair[1] ?? [], ready);
      const planes = channels === 2 ? [l, r] : [l.map((v, i) => (v + (r[i] ?? 0)) / 2)];
      await writer.push(planes);
      if (!restTiles) continue;
      for (const [c, p] of planes.entries()) {
        const rp = restPlanes[c];
        if (rp) for (let i = 0; i < ready; i++) rp[i] = (rp[i] ?? 0) - (p[i] ?? 0);
      }
    }
    await restTiles?.push(mono(restPlanes));
    written += ready;
    onProgress(written);
  };

  try {
    for (let k = first; k < blocks.totalBlocks; k++) {
      const planes = blocks.block(input, k);
      for (let c = 0; c < channels; c++) pendingInput[c]?.push((planes[c] ?? new Float32Array(0)).slice());
      const left = planes[0] ?? new Float32Array(blocks.blockLength(k));
      const right = planes[1] ?? left;
      const out = await stream.push(up[0]?.push(left) ?? left, up[1]?.push(right) ?? right);
      await accept(out, false);
      if (checkpoints && performance.now() - saved > CHECKPOINT_MS) {
        await checkpoints.save(snapshot(k + 1));
        saved = performance.now();
      }
      if (k % 8 === 0) await new Promise((r) => setTimeout(r, 0));
    }
    const tailL = up[0]?.finish() ?? new Float32Array(0);
    const tailR = up[1]?.finish() ?? new Float32Array(0);
    await accept(await stream.push(tailL, tailR), false);
    await accept(await stream.finish(), true);
    if (written < total) throw new Error(`separation produced ${written} of ${total} samples`);
  } catch (e) {
    for (const w of writers) w.abort();
    throw e;
  } finally {
    blocks.close();
  }
  for (const w of writers) await w.finish();
}
