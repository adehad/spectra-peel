import { DEMUCS, Resampler, SeparationStream, type SeparationModel } from "@spectra-peel/engine";
import * as ort from "onnxruntime-web/jspi";
import { z } from "zod";
import { LayerBlocks } from "../audio/layer-blocks";
import { FAST_STEMS, SPECIALISTS, type ModelId } from "../models";
import { dependencies } from "../project/layers";
import type { Layer, Project } from "../project/schema";
import { modelBytes } from "../storage/models";
import { projectDir } from "../storage/opfs";
import { loadJson, saveJson } from "./checkpoint";
import { LayerWriter, TileWriter, mono } from "./layer-writer";
import { createSession } from "./ort-session";
import type { SeparateMessage, SeparateRequest } from "./protocol";

const post = (m: SeparateMessage) => postMessage(m);

/** Runs an HT-Demucs graph and keeps the `keep` sources of its four, in that order. */
class OrtDemucs implements SeparationModel {
  private readonly session: ort.InferenceSession;
  private readonly keep: readonly number[];

  constructor(session: ort.InferenceSession, keep: readonly number[]) {
    this.session = session;
    this.keep = keep;
  }

  async run(mix: Float32Array): Promise<Float32Array> {
    const out = await this.session.run({ mix: new ort.Tensor("float32", mix, [1, 2, DEMUCS.segment]) });
    const stems = out["stems"];
    const data = await stems?.getData();
    stems?.dispose();
    if (!(data instanceof Float32Array)) throw new Error("model has no float32 'stems' output");
    if (this.keep.length === DEMUCS.sources.length) return data;
    const n = 2 * DEMUCS.segment;
    const picked = new Float32Array(this.keep.length * n);
    for (const [j, s] of this.keep.entries()) picked.set(data.subarray(s * n, (s + 1) * n), j * n);
    return picked;
  }
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

interface Pass {
  model: ModelId;
  /** Indexes into the model's sources, and into the request's stems, that this pass writes. */
  keep: number[];
}

/** The fast model for every stem not in `best`, then each chosen stem's specialist. */
function passesFor(best: readonly number[]): Pass[] {
  const fast = DEMUCS.sources.map((_, i) => i).filter((i) => !best.includes(i));
  return [
    ...(fast.length > 0 ? [{ model: FAST_STEMS, keep: fast }] : []),
    ...best.flatMap((i) => {
      const model = SPECIALISTS[i];
      return model ? [{ model, keep: [i] }] : [];
    }),
  ];
}

/**
 * Streams the input through one model and writes the kept stems. With `restTiles`, the pass writes
 * every stem, so it also builds the spectrogram of the input minus all of them.
 */
async function separatePass(
  dir: FileSystemDirectoryHandle,
  project: Project,
  input: string,
  session: ort.InferenceSession,
  keep: number[],
  stems: Layer[],
  restTiles: TileWriter | null,
  onProgress: (written: number) => void,
): Promise<void> {
  const rate = project.source.sampleRate;
  const channels = project.source.channels;
  const total = project.source.frames;
  const blocks = await LayerBlocks.open(dir, project, dependencies(project, input));
  const writers = await Promise.all(
    keep.map((s) => {
      const l = stems[s];
      if (l?.data.kind !== "flac") throw new Error("stems must be FLAC layers");
      return LayerWriter.create(dir, l.id, l.data.path, l.data.index, rate, channels);
    }),
  );
  const up = [new Resampler(rate, DEMUCS.sampleRate), new Resampler(rate, DEMUCS.sampleRate)];
  const down = keep.map(() => [
    new Resampler(DEMUCS.sampleRate, rate),
    new Resampler(DEMUCS.sampleRate, rate),
  ]);
  const stream = new SeparationStream(new OrtDemucs(session, keep), { sources: keep.length });
  const pendingInput: Float32Array[][] = Array.from({ length: channels }, () => []);
  const pendingStems: Float32Array[][][] = keep.map(() => [[], []]);
  let written = 0;

  /** Turns stereo 44.1 kHz stems into the project's rate and channel count and writes what lines up. */
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
    for (let k = 0; k < blocks.totalBlocks; k++) {
      const planes = blocks.block(input, k);
      for (let c = 0; c < channels; c++) pendingInput[c]?.push((planes[c] ?? new Float32Array(0)).slice());
      const left = planes[0] ?? new Float32Array(blocks.blockLength(k));
      const right = planes[1] ?? left;
      const out = await stream.push(up[0]?.push(left) ?? left, up[1]?.push(right) ?? right);
      await accept(out, false);
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

/** Spectrogram of the input minus the stems, read back from the stems once they are all written. */
async function writeRestTiles(dir: FileSystemDirectoryHandle, req: SeparateRequest): Promise<void> {
  const { project, stems, rest } = req;
  const withStems = { ...project, layers: [...project.layers, ...stems, rest] };
  const blocks = await LayerBlocks.open(dir, withStems, dependencies(withStems, rest.id));
  const tiles = new TileWriter(dir, rest.id, project.source.sampleRate);
  try {
    for (let k = 0; k < blocks.totalBlocks; k++) await tiles.push(mono(blocks.block(rest.id, k)));
  } finally {
    blocks.close();
  }
  await tiles.finish();
}

/** How many passes have written all their stems; a resumed run starts after them. */
const PassesDone = z.object({ done: z.number().int().nonnegative() });

async function separate(req: SeparateRequest): Promise<{ device: string; realTimeFactor: number }> {
  const { project, op, stems, rest } = req;
  const total = project.source.frames;
  const dir = await projectDir(project.id);
  const passes = passesFor(req.best);
  const t0 = performance.now();
  let device = "";
  const resumed = PassesDone.safeParse(await loadJson(dir, op.id, "passes.json"));
  const skip = resumed.success ? resumed.data.done : 0;

  for (const [p, pass] of passes.entries()) {
    if (p < skip) continue;
    const what =
      pass.keep.length > 1 ? "Separating" : `${stems[pass.keep[0] ?? 0]?.label ?? "Stem"} at best quality`;
    const label = passes.length === 1 ? what : `${what} (${p + 1} of ${passes.length})`;
    post({
      type: "progress",
      fraction: p / passes.length,
      stage: `${label}: loading model`,
      etaSeconds: null,
    });
    const created = await createSession(await modelBytes(pass.model), req.useGpu);
    device = created.device;
    const stage = `${label} on the ${device}`;
    post({ type: "progress", fraction: p / passes.length, stage, etaSeconds: null });
    const everyStem = pass.keep.length === DEMUCS.sources.length;
    const restTiles = everyStem ? new TileWriter(dir, rest.id, project.source.sampleRate) : null;
    try {
      await separatePass(dir, project, op.input, created.session, pass.keep, stems, restTiles, (written) => {
        const fraction = (p + written / total) / passes.length;
        const elapsed = (performance.now() - t0) / 1000;
        const ran = fraction - skip / passes.length;
        post({
          type: "progress",
          fraction,
          stage,
          etaSeconds: ran > 0.02 ? (elapsed / ran) * (1 - fraction) : null,
        });
      });
    } finally {
      await created.session.release();
    }
    await restTiles?.finish();
    await saveJson(dir, op.id, "passes.json", { done: p + 1 });
  }
  if (passes.length > 1) {
    post({ type: "progress", fraction: 1, stage: "Finding what the stems missed", etaSeconds: null });
    await writeRestTiles(dir, req);
  }
  return { device, realTimeFactor: (performance.now() - t0) / 1000 / (total / project.source.sampleRate) };
}

addEventListener("message", (e: MessageEvent<SeparateRequest>) => {
  separate(e.data).then(
    (r) => post({ type: "done", ...r }),
    (err: unknown) => post({ type: "error", message: err instanceof Error ? err.message : String(err) }),
  );
});
