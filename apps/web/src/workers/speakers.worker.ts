import {
  BAND_FILL,
  DIARIZATION,
  FLAC_BLOCK,
  Fbank,
  Resampler,
  clusterVoices,
  decodePowerset,
  fillHighBand,
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

interface Upsampled {
  speaker: number;
  /** Project-rate sample the audio starts at. */
  at: number;
  audio: Float32Array;
}

function upsample(piece: SeparatedPiece, rate: number): Upsampled {
  const up = new Resampler(SR, rate);
  const head = up.push(piece.audio);
  const tail = up.finish();
  const audio = new Float32Array(head.length + tail.length);
  audio.set(head);
  audio.set(tail, head.length);
  return { speaker: piece.speaker, at: Math.round((piece.start * rate) / SR), audio };
}

/** A person's audio from `start`: the input where they talk alone, plus their crosstalk. */
function personPlanes(
  input: readonly Float32Array[],
  start: number,
  solo: Uint8Array,
  rate: number,
  pieces: readonly Upsampled[],
  speaker: number,
): Float32Array[] {
  const length = input[0]?.length ?? 0;
  const gain = new Float32Array(length);
  frameGain(solo, rate, start, gain);
  const planes = input.map((p) => p.map((v, i) => v * (gain[i] ?? 0)));
  for (const piece of pieces) {
    if (piece.speaker !== speaker) continue;
    const from = Math.max(start, piece.at);
    const to = Math.min(start + length, piece.at + piece.audio.length);
    for (const plane of planes)
      for (let t = from; t < to; t++)
        plane[t - start] = (plane[t - start] ?? 0) + (piece.audio[t - piece.at] ?? 0);
  }
  return planes;
}

function readRange(blocks: LayerBlocks, layer: string, channels: number, start: number, end: number) {
  const out = Array.from({ length: channels }, () => new Float32Array(end - start));
  for (let k = Math.floor(start / FLAC_BLOCK); k * FLAC_BLOCK < end; k++) {
    const planes = blocks.block(layer, k);
    const at = k * FLAC_BLOCK;
    const from = Math.max(start, at);
    const to = Math.min(end, at + blocks.blockLength(k));
    for (const [c, plane] of out.entries())
      plane.set((planes[c] ?? new Float32Array()).subarray(from - at, to - at), from - start);
  }
  return out;
}

/** What the band fill adds to each person, per channel, over [start, end). */
interface HighBand {
  start: number;
  end: number;
  delta: Float32Array[][];
}

/**
 * Crosstalk comes from a 16 kHz model, so it has nothing above 8 kHz. Around a run of pieces, every
 * person and the rest get a share of the input's high band instead, faded in and out at the edges.
 */
function fillCrosstalk(
  input: readonly Float32Array[],
  start: number,
  solo: readonly Uint8Array[],
  rate: number,
  pieces: readonly Upsampled[],
): HighBand {
  const length = input[0]?.length ?? 0;
  const people = solo.map((m, j) => personPlanes(input, start, m, rate, pieces, j));
  const delta = people.map((planes) => planes.map(() => new Float32Array(length)));
  const fade = BAND_FILL.window;
  for (const [c, x] of input.entries()) {
    const layers = people.map((planes) => planes[c] ?? new Float32Array(length));
    const rest = x.map((v, i) => layers.reduce((r, l) => r - (l[i] ?? 0), v));
    const filled = fillHighBand([...layers, rest], rate, SR);
    for (const [j, layer] of layers.entries()) {
      const d = delta[j]?.[c];
      const f = filled[j];
      if (!d || !f) continue;
      for (let i = 0; i < length; i++) {
        const edge = Math.min(1, i / fade, (length - 1 - i) / fade);
        d[i] = edge * ((f[i] ?? 0) - (layer[i] ?? 0));
      }
    }
  }
  return { start, end: start + length, delta };
}

interface PieceRun {
  start: number;
  end: number;
  pieces: Upsampled[];
}

/** Pieces close enough to share one band fill, each run with the project-rate span the fill covers. */
function pieceRuns(pieces: readonly Upsampled[], total: number): PieceRun[] {
  const margin = 2 * BAND_FILL.window;
  const runs: PieceRun[] = [];
  for (const p of pieces.toSorted((a, b) => a.at - b.at)) {
    const start = Math.max(0, p.at - margin);
    const end = Math.min(total, p.at + p.audio.length + margin);
    const last = runs.at(-1);
    if (last && start <= last.end) {
      last.end = Math.max(last.end, end);
      last.pieces.push(p);
    } else runs.push({ start, end, pieces: [p] });
  }
  return runs;
}

/**
 * Writes each person's layer at the project rate: the input wherever they talk alone, and their
 * separated crosstalk, upsampled and with the high band filled, where they talk over someone. Also
 * the tiles of what is left.
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
  const runs = pieceRuns(
    pieces.map((p) => upsample(p, rate)),
    project.source.frames,
  );
  const writers = await Promise.all(
    people.map((l) => {
      if (l.data.kind !== "flac") throw new Error("people must be FLAC layers");
      return LayerWriter.create(dir, l.id, l.data.path, l.data.index, rate, channels);
    }),
  );
  const restTiles = new TileWriter(dir, rest.id, rate);
  const blocks = await LayerBlocks.open(dir, project, dependencies(project, op.input));
  const open: Upsampled[] = [];
  const bands: HighBand[] = [];
  let pos = 0;
  try {
    for (let k = 0; k < blocks.totalBlocks; k++) {
      const len = blocks.blockLength(k);
      while ((runs[0]?.start ?? Infinity) < pos + len) {
        const run = runs.shift();
        if (!run) break;
        open.push(...run.pieces);
        const x = readRange(blocks, op.input, channels, run.start, run.end);
        bands.push(fillCrosstalk(x, run.start, solo, rate, run.pieces));
      }
      const input = blocks.block(op.input, k);
      const left = input.map((p) => p.slice());
      for (const [j, writer] of writers.entries()) {
        const planes = personPlanes(input, pos, solo[j] ?? new Uint8Array(), rate, open, j);
        for (const band of bands) {
          const from = Math.max(pos, band.start);
          const to = Math.min(pos + len, band.end);
          for (const [c, plane] of planes.entries()) {
            const d = band.delta[j]?.[c];
            if (d)
              for (let t = from; t < to; t++)
                plane[t - pos] = (plane[t - pos] ?? 0) + (d[t - band.start] ?? 0);
          }
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
      for (let i = bands.length - 1; i >= 0; i--) if ((bands[i]?.end ?? 0) <= pos) bands.splice(i, 1);
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
