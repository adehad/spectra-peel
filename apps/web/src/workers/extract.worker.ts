import { AUDIOSEP, BAND_FILL, fillHighBand, type SeparationModel } from "@spectra-peel/engine";
import * as ort from "onnxruntime-web/jspi";
import prompts from "../extract-prompts.json";
import { LayerBlocks } from "../audio/layer-blocks";
import { dependencies } from "../project/layers";
import type { Layer } from "../project/schema";
import { modelBytes } from "../storage/models";
import { paths, projectDir, removeFile } from "../storage/opfs";
import { loadJson, saveJson } from "./checkpoint";
import { LayerWriter, TileWriter, mono } from "./layer-writer";
import { createSession } from "./ort-session";
import { extractRawLayer, type ExtractMessage, type ExtractRequest } from "./protocol";
import { passCheckpoints, progressPoster, separatePass } from "./separate-pass";

const post = (m: ExtractMessage) => postMessage(m);

/**
 * Width of the band, below the model's Nyquist, whose ownership decides who gets the band above it.
 * A third of an octave kept speech from taking pink noise's high band (-0.5 dB high-band SDR, against
 * -5.4 with an octave) while still giving it 7.4 dB over music and 12 dB over traffic, where leaving
 * the band to Rest scores 0.
 */
const FILL_OCTAVES = 1 / 3;
/** Samples filled at a time, before rounding to the fill's frame grid. */
const FILL_CHUNK = 4096 * 32;
/** Share of the progress bar the model pass takes when a fill pass follows. */
const MODEL_SHARE = 0.95;

/** Where the model pass writes the target when a fill pass will rewrite it, inside the job's folder. */
function rawLayer(req: ExtractRequest): Layer {
  const id = extractRawLayer(req.op.id);
  return {
    ...req.target,
    id,
    data: {
      kind: "flac",
      path: paths.checkpoint(req.op.id, "raw.flac"),
      index: paths.checkpoint(req.op.id, "raw.idx"),
    },
  };
}

/**
 * Rewrites the target with a share of the input's band above the model's 16 kHz, chunk by chunk.
 * Each chunk starts on the fill's frame grid and carries a window of context either side, so the
 * result is the same as filling the whole file at once. Also writes Rest's spectrogram.
 */
async function fillTarget(
  dir: FileSystemDirectoryHandle,
  req: ExtractRequest,
  raw: Layer,
  onProgress: (f: number) => void,
): Promise<void> {
  const { project, op, target, rest } = req;
  const rate = project.source.sampleRate;
  const channels = project.source.channels;
  const total = project.source.frames;
  if (target.data.kind !== "flac") throw new Error("the extracted layer must be FLAC");
  const hop = Math.round(BAND_FILL.hopSeconds * rate);
  const margin = Math.ceil(BAND_FILL.window / hop) * hop;
  const chunk = Math.ceil(FILL_CHUNK / hop) * hop;
  const withRaw = { ...project, layers: [...project.layers, raw] };
  const blocks = await LayerBlocks.open(dir, withRaw, new Set([...dependencies(withRaw, op.input), raw.id]));
  const writer = await LayerWriter.create(
    dir,
    target.id,
    target.data.path,
    target.data.index,
    rate,
    channels,
  );
  const restTiles = new TileWriter(dir, rest.id, rate);
  try {
    for (let start = 0; start < total; start += chunk) {
      const end = Math.min(total, start + chunk);
      const a = Math.max(0, start - margin);
      const b = Math.min(total, end + margin);
      const input = blocks.range(op.input, a, b);
      const model = blocks.range(raw.id, a, b);
      const planes = input.map((x, c) => {
        const y = model[c] ?? new Float32Array(x.length);
        const others = x.map((v, i) => v - (y[i] ?? 0));
        const [filled] = fillHighBand([y, others], rate, AUDIOSEP.sampleRate, FILL_OCTAVES);
        return (filled ?? y).slice(start - a, end - a);
      });
      await writer.push(planes);
      const left = planes.map((p, c) => p.map((v, i) => (input[c]?.[start - a + i] ?? 0) - v));
      await restTiles.push(mono(left));
      onProgress(end / total);
      await new Promise((r) => setTimeout(r, 0));
    }
  } catch (e) {
    writer.abort();
    throw e;
  } finally {
    blocks.close();
  }
  await writer.finish();
  await restTiles.finish();
}

/**
 * Runs AudioSep on the mid channel and returns its target as both channels: the model is mono, and
 * running each channel would double the cost. Whatever differs between the channels goes to Rest.
 */
class OrtAudioSep implements SeparationModel {
  private readonly session: ort.InferenceSession;
  private readonly condition: ort.Tensor;

  constructor(session: ort.InferenceSession, embedding: Float32Array) {
    this.session = session;
    this.condition = new ort.Tensor("float32", embedding, [1, AUDIOSEP.embedding]);
  }

  async run(mix: Float32Array): Promise<Float32Array> {
    const n = AUDIOSEP.segment;
    const mid = new Float32Array(n);
    for (let i = 0; i < n; i++) mid[i] = ((mix[i] ?? 0) + (mix[n + i] ?? 0)) / 2;
    const out = await this.session.run({
      mixture: new ort.Tensor("float32", mid, [1, 1, n]),
      condition: this.condition,
    });
    const target = out["target"];
    const data = await target?.getData();
    target?.dispose();
    if (!(data instanceof Float32Array)) throw new Error("model has no float32 'target' output");
    const both = new Float32Array(2 * n);
    both.set(data.subarray(0, n));
    both.set(data.subarray(0, n), n);
    return both;
  }
}

/** Runs AudioSep over the input and writes its target to `out`, plus Rest's tiles when given. */
async function separate(
  dir: FileSystemDirectoryHandle,
  req: ExtractRequest,
  out: Layer,
  restTiles: TileWriter | null,
  progress: (fraction: number, stage: string) => void,
  share: number,
): Promise<string> {
  const { project, op } = req;
  const total = project.source.frames;
  const embedding = (prompts as Record<string, number[] | undefined>)[req.prompt];
  if (!embedding) throw new Error(`no embedding for "${req.prompt}"`);
  const checkpoints = await passCheckpoints(dir, op.id, "pass");
  const from = ((checkpoints.resume?.written ?? 0) / total) * share;
  post({ type: "progress", fraction: from, stage: `${req.prompt}: loading model`, etaSeconds: null });
  const created = await createSession(await modelBytes("audiosep-base"), req.useGpu);
  const stage = `${req.prompt} on the ${created.device}`;
  post({ type: "progress", fraction: from, stage, etaSeconds: null });
  const model = {
    model: new OrtAudioSep(created.session, Float32Array.from(embedding)),
    sampleRate: AUDIOSEP.sampleRate,
    segment: AUDIOSEP.segment,
    overlap: AUDIOSEP.overlap,
  };
  try {
    const report = (written: number) => progress((written / total) * share, stage);
    await separatePass(dir, project, op.input, model, [out], restTiles, report, checkpoints);
  } finally {
    await created.session.release();
  }
  await restTiles?.finish();
  return created.device;
}

async function extract(req: ExtractRequest): Promise<{ device: string; realTimeFactor: number }> {
  const { project, op, target, rest } = req;
  const rate = project.source.sampleRate;
  const dir = await projectDir(project.id);
  const t0 = performance.now();
  const progress = progressPoster(post, t0);
  // Above about 30 kHz the project has a band the 32 kHz model never saw, so a fill pass follows.
  const fills = rate / 2 > (BAND_FILL.cut * AUDIOSEP.sampleRate) / 2;
  let device = "";
  if (!fills) device = await separate(dir, req, target, new TileWriter(dir, rest.id, rate), progress, 1);
  else {
    const raw = rawLayer(req);
    if ((await loadJson(dir, op.id, "separated.json")) === null) {
      device = await separate(dir, req, raw, null, progress, MODEL_SHARE);
      await saveJson(dir, op.id, "separated.json", true);
    }
    const stage = `${req.prompt}: filling above 16 kHz`;
    await fillTarget(dir, req, raw, (f) => progress(MODEL_SHARE + (1 - MODEL_SHARE) * f, stage));
    await removeFile(dir, paths.peaks(raw.id));
    await removeFile(dir, `derived/tiles/${raw.id}`);
  }
  return { device, realTimeFactor: (performance.now() - t0) / 1000 / (project.source.frames / rate) };
}

addEventListener("message", (e: MessageEvent<ExtractRequest>) => {
  extract(e.data).then(
    (r) => post({ type: "done", ...r }),
    (err: unknown) => post({ type: "error", message: err instanceof Error ? err.message : String(err) }),
  );
});
