import {
  DFN3,
  Dfn3Stream,
  FLAC_BLOCK,
  Resampler,
  crossfadeWeight,
  paddedRange,
  type Dfn3Model,
} from "@spectra-peel/engine";
import * as ort from "onnxruntime-web/jspi";
import { LayerBlocks } from "../audio/layer-blocks";
import { dependencies } from "../project/layers";
import { FADE_SECONDS } from "../project/ops";
import { modelBytes } from "../storage/models";
import { createSession } from "./ort-session";
import { projectDir } from "../storage/opfs";
import { LayerWriter, TileWriter, mono } from "./layer-writer";
import type { DenoiseMessage, DenoiseRequest } from "./protocol";

const post = (m: DenoiseMessage) => postMessage(m);

/** GRUs need well over 3 s to settle from a cold start, so give them 10 s of audio before the selection. */
const PRE_ROLL_SECONDS = 10;
const POST_ROLL_SECONDS = 0.5;

// Names reference/stream_dfn3.py exposes, in its GRUS and PADS order.
const GRU_OUTPUTS: Record<string, string> = {
  enc: "/enc/emb_gru/gru/GRU_output_0",
  erb0: "/erb_dec/emb_gru/gru/GRU_output_0",
  erb1: "/erb_dec/emb_gru/gru/GRU_1_output_0",
  df0: "/df_dec/df_gru/gru/GRU_output_0",
  df1: "/df_dec/df_gru/gru/GRU_1_output_0",
};
const PAD_INPUTS: Record<string, { tensor: string; frames: number; channels: number; width: number }> = {
  erb: { tensor: "/Pad_output_0", frames: 2, channels: 1, width: 32 },
  df: { tensor: "/Pad_1_output_0", frames: 2, channels: 2, width: 96 },
  dfp: { tensor: "/enc/df_conv0/df_conv0.4/Relu_output_0", frames: 4, channels: 64, width: 96 },
};
const HIDDEN = 256;

async function float32(t: ort.Tensor | undefined, name: string): Promise<Float32Array> {
  const data = await t?.getData();
  if (!(data instanceof Float32Array)) throw new Error(`model output ${name} missing or not float32`);
  return data;
}

/** Carries the resumable graph's GRU and convolution state between chunks. */
class OrtDfn3 implements Dfn3Model {
  private readonly session: ort.InferenceSession;
  private state = new Map<string, ort.Tensor>();

  constructor(session: ort.InferenceSession) {
    this.session = session;
    for (const name of Object.keys(GRU_OUTPUTS)) {
      this.state.set(`h_${name}`, new ort.Tensor("float32", new Float32Array(HIDDEN), [1, 1, HIDDEN]));
    }
    for (const [name, p] of Object.entries(PAD_INPUTS)) {
      const shape = [1, p.channels, p.frames, p.width];
      this.state.set(
        `hist_${name}`,
        new ort.Tensor("float32", new Float32Array(p.channels * p.frames * p.width), shape),
      );
    }
  }

  async run(featErb: Float32Array, featSpec: Float32Array, frames: number, keep: number) {
    const feeds: Record<string, ort.Tensor> = {
      feat_erb: new ort.Tensor("float32", featErb, [1, 1, frames, DFN3.erbBands]),
      feat_spec: new ort.Tensor("float32", featSpec, [1, 2, frames, DFN3.dfBins]),
      ...Object.fromEntries(this.state),
    };
    const fetches = [
      "erb_mask",
      "df_coefs",
      ...Object.values(GRU_OUTPUTS),
      ...Object.values(PAD_INPUTS).map((p) => p.tensor),
    ];
    const out = await this.session.run(feeds, fetches);
    const erbMask = await float32(out["erb_mask"], "erb_mask");
    const dfCoefs = await float32(out["df_coefs"], "df_coefs");

    const next = new Map<string, ort.Tensor>();
    for (const [name, tensor] of Object.entries(GRU_OUTPUTS)) {
      const y = await float32(out[tensor], tensor);
      next.set(
        `h_${name}`,
        new ort.Tensor("float32", y.slice((keep - 1) * HIDDEN, keep * HIDDEN), [1, 1, HIDDEN]),
      );
    }
    for (const [name, p] of Object.entries(PAD_INPUTS)) {
      const pre = await float32(out[p.tensor], p.tensor);
      const hist = new Float32Array(p.channels * p.frames * p.width);
      for (let c = 0; c < p.channels; c++) {
        const from = (c * frames + keep - p.frames) * p.width;
        hist.set(pre.subarray(from, from + p.frames * p.width), c * p.frames * p.width);
      }
      next.set(`hist_${name}`, new ort.Tensor("float32", hist, [1, p.channels, p.frames, p.width]));
    }
    for (const t of Object.values(out)) t.dispose();
    this.state = next;
    return { erbMask, dfCoefs };
  }
}

/** First-in first-out sample queue for one channel. */
class Queue {
  private parts: Float32Array[] = [];
  private head = 0;
  length = 0;

  push(x: Float32Array): void {
    if (x.length === 0) return;
    this.parts.push(x);
    this.length += x.length;
  }

  take(n: number): Float32Array {
    const out = new Float32Array(n);
    let at = 0;
    while (at < n) {
      const part = this.parts[0];
      if (!part) break;
      const k = Math.min(n - at, part.length - this.head);
      out.set(part.subarray(this.head, this.head + k), at);
      at += k;
      this.head += k;
      if (this.head === part.length) {
        this.parts.shift();
        this.head = 0;
      }
    }
    this.length -= at;
    return out;
  }
}

async function denoise(req: DenoiseRequest): Promise<number> {
  const { project, op, output, removed } = req;
  const rate = project.source.sampleRate;
  const channels = project.source.channels;
  const total = project.source.frames;
  const extent = { start: output.start, end: output.start + output.frames };
  const window = paddedRange(extent, total, PRE_ROLL_SECONDS * rate, POST_ROLL_SECONDS * rate);
  const fade = Math.round(FADE_SECONDS * rate);

  post({ type: "progress", fraction: 0, stage: "Loading model" });
  // DeepFilterNet3 is small and recurrent; the spike measured it 4x faster on wasm than WebGPU.
  const { session } = await createSession(await modelBytes("dfn3-stream"), false);

  const dir = await projectDir(project.id);
  const blocks = await LayerBlocks.open(dir, project, dependencies(project, op.input));
  if (output.data.kind !== "flac") throw new Error("denoise output must be a FLAC layer");
  const writer = await LayerWriter.create(
    dir,
    output.id,
    output.data.path,
    output.data.index,
    rate,
    channels,
  );
  const removedTiles = new TileWriter(dir, removed.id, rate);

  const up = Array.from({ length: channels }, () => new Resampler(rate, DFN3.sampleRate));
  const down = Array.from({ length: channels }, () => new Resampler(DFN3.sampleRate, rate));
  const streams = Array.from({ length: channels }, () => new Dfn3Stream(new OrtDfn3(session)));
  const inputs = Array.from({ length: channels }, () => new Queue());
  const outputs = Array.from({ length: channels }, () => new Queue());
  let emitted = window.start;
  const t0 = performance.now();

  const drain = async (final: boolean) => {
    let n = Math.min(...outputs.map((q) => q.length), ...inputs.map((q) => q.length));
    if (final) n = Math.min(...inputs.map((q) => q.length));
    if (n <= 0) return;
    const dry = inputs.map((q) => q.take(n));
    const wet = outputs.map((q) => q.take(n));
    const keepFrom = Math.max(0, extent.start - emitted);
    const keepTo = Math.min(n, extent.end - emitted);
    if (keepTo > keepFrom) {
      const mixed = dry.map((d, c) => {
        const m = new Float32Array(keepTo - keepFrom);
        const w = wet[c] ?? new Float32Array(n);
        for (let i = keepFrom; i < keepTo; i++) {
          const g = crossfadeWeight(emitted + i, op.range, fade, total);
          m[i - keepFrom] = g * (w[i] ?? 0) + (1 - g) * (d[i] ?? 0);
        }
        return m;
      });
      await writer.push(mixed);
      const residual = mixed.map((m, c) => m.map((v, i) => (dry[c]?.[keepFrom + i] ?? 0) - v));
      await removedTiles.push(mono(residual));
    }
    emitted += n;
  };

  try {
    const firstBlock = Math.floor(window.start / FLAC_BLOCK);
    const lastBlock = Math.ceil(window.end / FLAC_BLOCK);
    for (let k = firstBlock; k < lastBlock; k++) {
      const blockStart = k * FLAC_BLOCK;
      const from = Math.max(0, window.start - blockStart);
      const to = Math.min(blocks.blockLength(k), window.end - blockStart);
      const planes = blocks.block(op.input, k).map((p) => p.subarray(from, to));
      for (let c = 0; c < channels; c++) {
        const plane = planes[c] ?? new Float32Array(to - from);
        inputs[c]?.push(plane.slice());
        const s48 = up[c]?.push(plane) ?? new Float32Array(0);
        const den = (await streams[c]?.push(s48)) ?? new Float32Array(0);
        outputs[c]?.push(down[c]?.push(den) ?? new Float32Array(0));
      }
      await drain(false);
      post({
        type: "progress",
        fraction: (emitted - window.start) / (window.end - window.start),
        stage: emitted < extent.start ? "Settling" : "Denoising",
      });
    }
    for (let c = 0; c < channels; c++) {
      const tail48 = up[c]?.finish() ?? new Float32Array(0);
      const stream = streams[c];
      const resampler = down[c];
      if (!stream || !resampler) continue;
      const den = await stream.push(tail48);
      const rest = await stream.finish();
      outputs[c]?.push(resampler.push(den));
      outputs[c]?.push(resampler.push(rest));
      outputs[c]?.push(resampler.finish());
    }
    await drain(true);
  } catch (e) {
    writer.abort();
    throw e;
  } finally {
    blocks.close();
    await session.release();
  }
  await writer.finish();
  await removedTiles.finish();
  return (performance.now() - t0) / 1000 / ((window.end - window.start) / rate);
}

addEventListener("message", (e: MessageEvent<DenoiseRequest>) => {
  denoise(e.data).then(
    (realTimeFactor) => post({ type: "done", realTimeFactor }),
    (err: unknown) => post({ type: "error", message: err instanceof Error ? err.message : String(err) }),
  );
});
