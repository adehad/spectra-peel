import {
  DIARIZATION,
  Fbank,
  Resampler,
  clusterVoices,
  decodePowerset,
  frameGain,
  overlapPieces,
  separateCrosstalk,
  soloSpans,
  speakerActivity,
  windowStarts,
  type SeparatedPiece,
  type SpeakerSeparationModel,
  type Window,
} from "@spectra-peel/engine";
import * as ort from "onnxruntime-web/jspi";
import { LayerBlocks } from "../audio/layer-blocks";
import { dependencies } from "../project/layers";
import type { Project } from "../project/schema";
import { modelBytes } from "../storage/models";
import { projectDir } from "../storage/opfs";
import { LayerWriter, TileWriter, mono } from "./layer-writer";
import { createSession } from "./ort-session";
import type { SpeakersMessage, SpeakersRequest } from "./protocol";

const post = (m: SpeakersMessage) => postMessage(m);
const SR = DIARIZATION.sampleRate;
/** The TIGER export's fixed input: 6 s. */
const SEGMENT = 6 * SR;
const yieldToEvents = () => new Promise((r) => setTimeout(r, 0));

async function output(session: ort.InferenceSession, feeds: Record<string, ort.Tensor>, name: string) {
  const out = await session.run(feeds);
  const t = out[name];
  const data = await t?.getData();
  const dims = t?.dims ?? [];
  t?.dispose();
  if (!(data instanceof Float32Array)) throw new Error(`model has no float32 '${name}' output`);
  return { data, dims };
}

/** The input mixed to mono and resampled to 16 kHz, for the models. */
async function readMono16k(
  dir: FileSystemDirectoryHandle,
  project: Project,
  input: string,
  onProgress: (f: number) => void,
): Promise<Float32Array> {
  const rate = project.source.sampleRate;
  const length = Math.round((project.source.frames * SR) / rate);
  const out = new Float32Array(length);
  const resampler = new Resampler(rate, SR);
  let at = 0;
  const put = (x: Float32Array) => {
    const n = Math.min(x.length, length - at);
    out.set(x.subarray(0, n), at);
    at += n;
  };
  const blocks = await LayerBlocks.open(dir, project, dependencies(project, input));
  try {
    for (let k = 0; k < blocks.totalBlocks; k++) {
      put(resampler.push(mono(blocks.block(input, k))));
      if (k % 32 === 0) {
        onProgress(k / blocks.totalBlocks);
        await yieldToEvents();
      }
    }
    put(resampler.finish());
  } finally {
    blocks.close();
  }
  return out;
}

/** segmentation-3.0 over 10 s windows every 2 s, eight windows a run. */
async function segment(
  session: ort.InferenceSession,
  mix: Float32Array,
  onProgress: (f: number) => void,
): Promise<Window[]> {
  const { window } = DIARIZATION;
  const starts = windowStarts(mix.length);
  const windows: Window[] = [];
  for (let i = 0; i < starts.length; i += 8) {
    const group = starts.slice(i, i + 8);
    const x = new Float32Array(group.length * window);
    for (const [b, s] of group.entries()) x.set(mix.subarray(s, s + window), b * window);
    const { data, dims } = await output(
      session,
      { x: new ort.Tensor("float32", x, [group.length, 1, window]) },
      "y",
    );
    const frames = dims[1] ?? 0;
    const classes = dims[2] ?? 7;
    for (const [b, start] of group.entries()) {
      const best = new Uint8Array(frames);
      for (let f = 0; f < frames; f++) {
        const at = (b * frames + f) * classes;
        let c = 0;
        for (let q = 1; q < classes; q++)
          if ((data[at + q] ?? -Infinity) > (data[at + c] ?? -Infinity)) c = q;
        best[f] = c;
      }
      windows.push({ start, masks: decodePowerset(best), embeddings: new Map() });
    }
    onProgress(Math.min(1, (i + group.length) / starts.length));
  }
  return windows;
}

/** Embeds every local speaker that talks alone for at least a second in its window. */
async function embedWindows(
  windows: Window[],
  mix: Float32Array,
  embed: (wav: Float32Array) => Promise<Float32Array>,
  onProgress: (f: number) => void,
): Promise<Float32Array[]> {
  const all: Float32Array[] = [];
  for (const [w, win] of windows.entries()) {
    for (let j = 0; j < DIARIZATION.localSpeakers; j++) {
      const spans = soloSpans(win.masks, j)
        .map(([a, b]) => [win.start + a, Math.min(win.start + b, mix.length)] as const)
        .filter(([a, b]) => b > a);
      const total = spans.reduce((n, [a, b]) => n + b - a, 0);
      if (total < DIARIZATION.minEmbedding) continue;
      const wav = new Float32Array(total);
      let at = 0;
      for (const [a, b] of spans) {
        wav.set(mix.subarray(a, b), at);
        at += b - a;
      }
      const e = await embed(wav);
      win.embeddings.set(j, e);
      all.push(e);
    }
    onProgress((w + 1) / windows.length);
  }
  return all;
}

/**
 * Writes each person's layer at the project rate: the input wherever they talk alone, and their
 * separated crosstalk, upsampled, where they talk over someone. Also the tiles of what is left.
 */
async function writePeople(
  dir: FileSystemDirectoryHandle,
  req: SpeakersRequest,
  active: Uint8Array[],
  pieces: SeparatedPiece[],
  onProgress: (f: number) => void,
): Promise<void> {
  const { project, op, people, rest } = req;
  const rate = project.source.sampleRate;
  const channels = project.source.channels;
  const count = active[0]?.map((_, t) => active.reduce((n, a) => n + (a[t] ?? 0), 0)) ?? new Uint8Array();
  const solo = active.map((a) => a.map((v, t) => (v && count[t] === 1 ? 1 : 0)));
  const queue = pieces
    .map((p) => ({ ...p, at: Math.round((p.start * rate) / SR) }))
    .toSorted((a, b) => a.at - b.at);
  const open: { speaker: number; at: number; audio: Float32Array }[] = [];
  const writers = await Promise.all(
    people.map((l) => {
      if (l.data.kind !== "flac") throw new Error("people must be FLAC layers");
      return LayerWriter.create(dir, l.id, l.data.path, l.data.index, rate, channels);
    }),
  );
  const restTiles = new TileWriter(dir, rest.id, rate);
  const blocks = await LayerBlocks.open(dir, project, dependencies(project, op.input));
  let pos = 0;
  try {
    for (let k = 0; k < blocks.totalBlocks; k++) {
      const len = blocks.blockLength(k);
      const input = blocks.block(op.input, k);
      while ((queue[0]?.at ?? Infinity) < pos + len) {
        const p = queue.shift();
        if (!p) break;
        const up = new Resampler(SR, rate);
        const head = up.push(p.audio);
        const tail = up.finish();
        const audio = new Float32Array(head.length + tail.length);
        audio.set(head);
        audio.set(tail, head.length);
        open.push({ speaker: p.speaker, at: p.at, audio });
      }
      const left = input.map((p) => p.slice());
      const gain = new Float32Array(len);
      for (const [j, writer] of writers.entries()) {
        frameGain(solo[j] ?? new Uint8Array(), rate, pos, gain);
        const planes = input.map((p) => p.map((v, i) => v * (gain[i] ?? 0)));
        for (const piece of open) {
          if (piece.speaker !== j) continue;
          const from = Math.max(pos, piece.at);
          const to = Math.min(pos + len, piece.at + piece.audio.length);
          for (const plane of planes)
            for (let t = from; t < to; t++)
              plane[t - pos] = (plane[t - pos] ?? 0) + (piece.audio[t - piece.at] ?? 0);
        }
        await writer.push(planes);
        for (const [c, plane] of planes.entries()) {
          const l = left[c];
          if (l) for (let i = 0; i < len; i++) l[i] = (l[i] ?? 0) - (plane[i] ?? 0);
        }
      }
      await restTiles.push(mono(left));
      pos += len;
      for (let i = open.length - 1; i >= 0; i--) {
        const piece = open[i];
        if (piece && piece.at + piece.audio.length <= pos) open.splice(i, 1);
      }
      if (k % 8 === 0) {
        onProgress(k / blocks.totalBlocks);
        await yieldToEvents();
      }
    }
  } catch (e) {
    for (const w of writers) w.abort();
    throw e;
  } finally {
    blocks.close();
  }
  for (const w of writers) await w.finish();
  await restTiles.finish();
}

async function speakers(req: SpeakersRequest): Promise<{ device: string; realTimeFactor: number }> {
  const { project, op, people } = req;
  const dir = await projectDir(project.id);
  const t0 = performance.now();
  const seconds = project.source.frames / project.source.sampleRate;
  let stage = "Reading";
  const progress = (from: number, to: number) => (f: number) => {
    const fraction = from + (to - from) * f;
    const elapsed = (performance.now() - t0) / 1000;
    post({
      type: "progress",
      fraction,
      stage,
      etaSeconds: fraction > 0.05 ? (elapsed / fraction) * (1 - fraction) : null,
    });
  };

  const mix = await readMono16k(dir, project, op.input, progress(0, 0.1));

  stage = "Finding who speaks when";
  const seg = await createSession(await modelBytes("speaker-segmentation"), false);
  let windows: Window[];
  try {
    windows = await segment(seg.session, mix, progress(0.1, 0.25));
  } finally {
    await seg.session.release();
  }

  stage = "Telling voices apart";
  const emb = await createSession(await modelBytes("speaker-embedding"), false);
  const fbank = new Fbank();
  const embed = async (wav: Float32Array) => {
    const frames = Fbank.frames(wav.length);
    const feats = new ort.Tensor("float32", fbank.compute(wav), [1, frames, Fbank.bins]);
    return (await output(emb.session, { feats }, "embs")).data.slice();
  };
  let device = "CPU";
  let pieces: SeparatedPiece[] = [];
  let active: Uint8Array[];
  try {
    const embeddings = await embedWindows(windows, mix, embed, progress(0.25, 0.4));
    if (embeddings.length < people.length)
      throw new Error(`Not enough clear speech to tell ${people.length} voices apart.`);
    const voices = clusterVoices(embeddings, people.length);
    active = speakerActivity(windows, voices, mix.length);

    if (overlapPieces(active, SEGMENT).length > 0) {
      stage = "Separating crosstalk: loading model";
      progress(0.4, 0.4)(0);
      const sep = await createSession(await modelBytes("speaker-separation"), req.useGpu);
      device = sep.device;
      stage = `Separating crosstalk on the ${device}`;
      const model: SpeakerSeparationModel = {
        segment: SEGMENT,
        run: async (chunk) =>
          (await output(sep.session, { mix: new ort.Tensor("float32", chunk, [1, SEGMENT]) }, "sources"))
            .data,
      };
      try {
        const report = progress(0.4, 0.85);
        pieces = await separateCrosstalk(mix, active, model, embed, voices, (done, total) =>
          report(done / total),
        );
      } finally {
        await sep.session.release();
      }
    }
  } finally {
    await emb.session.release();
  }

  stage = "Writing each person";
  await writePeople(dir, req, active, pieces, progress(0.85, 1));
  return { device, realTimeFactor: (performance.now() - t0) / 1000 / seconds };
}

addEventListener("message", (e: MessageEvent<SpeakersRequest>) => {
  speakers(e.data).then(
    (r) => post({ type: "done", ...r }),
    (err: unknown) => post({ type: "error", message: err instanceof Error ? err.message : String(err) }),
  );
});
