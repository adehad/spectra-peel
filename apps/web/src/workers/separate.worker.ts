import { DEMUCS, Resampler, SeparationStream, type SeparationModel } from "@spectra-peel/engine";
import * as ort from "onnxruntime-web/jspi";
import { LayerBlocks } from "../audio/layer-blocks";
import { dependencies } from "../project/layers";
import { modelBytes } from "../storage/models";
import { projectDir } from "../storage/opfs";
import { LayerWriter, TileWriter, mono } from "./layer-writer";
import { createSession } from "./ort-session";
import type { SeparateMessage, SeparateRequest } from "./protocol";

const post = (m: SeparateMessage) => postMessage(m);

class OrtDemucs implements SeparationModel {
  private readonly session: ort.InferenceSession;

  constructor(session: ort.InferenceSession) {
    this.session = session;
  }

  async run(mix: Float32Array): Promise<Float32Array> {
    const out = await this.session.run({ mix: new ort.Tensor("float32", mix, [1, 2, DEMUCS.segment]) });
    const stems = out["stems"];
    const data = await stems?.getData();
    stems?.dispose();
    if (!(data instanceof Float32Array)) throw new Error("model has no float32 'stems' output");
    return data;
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

async function separate(req: SeparateRequest): Promise<{ device: string; realTimeFactor: number }> {
  const { project, op, stems, rest } = req;
  const rate = project.source.sampleRate;
  const channels = project.source.channels;
  const total = project.source.frames;

  post({ type: "progress", fraction: 0, stage: "Loading model", etaSeconds: null });
  const { session, device } = await createSession(await modelBytes("htdemucs-stems"), req.useGpu);
  post({
    type: "progress",
    fraction: 0,
    stage: `Separating on the ${device}`,
    etaSeconds: null,
  });

  const dir = await projectDir(project.id);
  const blocks = await LayerBlocks.open(dir, project, dependencies(project, op.input));
  const writers = await Promise.all(
    stems.map((l) => {
      if (l.data.kind !== "flac") throw new Error("stems must be FLAC layers");
      return LayerWriter.create(dir, l.id, l.data.path, l.data.index, rate, channels);
    }),
  );
  const restTiles = new TileWriter(dir, rest.id, rate);
  const up = [new Resampler(rate, DEMUCS.sampleRate), new Resampler(rate, DEMUCS.sampleRate)];
  const down = stems.map(() => [
    new Resampler(DEMUCS.sampleRate, rate),
    new Resampler(DEMUCS.sampleRate, rate),
  ]);
  const stream = new SeparationStream(new OrtDemucs(session));
  const pendingInput: Float32Array[][] = [];
  const pendingStems: Float32Array[][][] = stems.map(() => [[], []]);
  let written = 0;
  const t0 = performance.now();

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
    const dry = Array.from({ length: channels }, (_, c) => take(pendingInput[c] ?? [], ready));
    const restPlanes = dry.map((d) => d.slice());
    for (const [s, writer] of writers.entries()) {
      const pair = pendingStems[s] ?? [];
      const l = take(pair[0] ?? [], ready);
      const r = take(pair[1] ?? [], ready);
      const planes = channels === 2 ? [l, r] : [l.map((v, i) => (v + (r[i] ?? 0)) / 2)];
      await writer.push(planes);
      for (const [c, p] of planes.entries()) {
        const rp = restPlanes[c];
        if (rp) for (let i = 0; i < ready; i++) rp[i] = (rp[i] ?? 0) - (p[i] ?? 0);
      }
    }
    await restTiles.push(mono(restPlanes));
    written += ready;
    const fraction = written / total;
    const elapsed = (performance.now() - t0) / 1000;
    post({
      type: "progress",
      fraction,
      stage: `Separating on the ${device}`,
      etaSeconds: fraction > 0.02 ? (elapsed / fraction) * (1 - fraction) : null,
    });
  };

  try {
    for (let c = 0; c < channels; c++) pendingInput.push([]);
    for (let k = 0; k < blocks.totalBlocks; k++) {
      const planes = blocks.block(op.input, k);
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
    await session.release();
  }
  for (const w of writers) await w.finish();
  await restTiles.finish();
  return { device, realTimeFactor: (performance.now() - t0) / 1000 / (total / rate) };
}

addEventListener("message", (e: MessageEvent<SeparateRequest>) => {
  separate(e.data).then(
    (r) => post({ type: "done", ...r }),
    (err: unknown) => post({ type: "error", message: err instanceof Error ? err.message : String(err) }),
  );
});
