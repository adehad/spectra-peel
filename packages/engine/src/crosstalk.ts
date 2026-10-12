import { DIARIZATION, centred, type Voices } from "./diarize";

const SR = DIARIZATION.sampleRate;
const FRAME = DIARIZATION.frame;
/** Width of the raised-cosine edge on every speaker switch. */
const RAMP_SECONDS = 0.02;
/** TIGER was trained on speech at about this RMS, so each chunk is scaled to it, by 30 dB at most. */
const TARGET_RMS = 0.05;
const MAX_GAIN = 10 ** (30 / 20);

/**
 * A two-speaker separation graph at 16 kHz with a fixed input length. `mix` is [segment]; the result
 * is planar [2, segment].
 */
export interface SpeakerSeparationModel {
  readonly segment: number;
  run(mix: Float32Array): Promise<Float32Array>;
}

/** A speaker embedding of 16 kHz audio, unnormalised. */
export type Embed = (wav: Float32Array) => Promise<Float32Array>;

export interface OverlapPiece {
  /** Global frames [start, end) to separate. */
  start: number;
  end: number;
  /** The two voices most active over it. */
  speakers: [number, number];
}

/** Frames with at least two people talking, merged across short gaps and split to fit the model. */
export function overlapPieces(active: readonly Uint8Array[], segment: number): OverlapPiece[] {
  const frames = active[0]?.length ?? 0;
  const minLength = Math.floor((0.1 * SR) / FRAME);
  const mergeGap = Math.floor((0.3 * SR) / FRAME);
  const longest = Math.floor((segment - 2 * SR) / FRAME);
  const regions: [number, number][] = [];
  let open = -1;
  for (let t = 0; t <= frames; t++) {
    const crosstalk = t < frames && active.reduce((n, a) => n + (a[t] ?? 0), 0) >= 2;
    if (crosstalk && open < 0) open = t;
    if (!crosstalk && open >= 0) {
      const last = regions.at(-1);
      if (last && open - last[1] <= mergeGap) last[1] = t;
      else regions.push([open, t]);
      open = -1;
    }
  }
  const pieces: OverlapPiece[] = [];
  for (const [a, b] of regions) {
    if (b - a < minLength) continue;
    for (let s = a; s < b; s += longest) {
      const e = Math.min(b, s + longest);
      const talk = active.map((m, i) => {
        let n = 0;
        for (let t = s; t < e; t++) n += m[t] ?? 0;
        return [n, i] as const;
      });
      const [first, second] = talk.toSorted((p, q) => q[0] - p[0] || p[1] - q[1]);
      pieces.push({ start: s, end: e, speakers: [first?.[1] ?? 0, second?.[1] ?? 1] });
    }
  }
  return pieces;
}

const ramp = (u: number) => (u <= 0 ? 0 : u >= 1 ? 1 : u - Math.sin(2 * Math.PI * u) / (2 * Math.PI));

/**
 * A frame mask as a gain at `rate`, convolved with a 20 ms Hann so every switch is a raised-cosine
 * crossfade. Frames past either end repeat the end frame. Fills `out` from sample `start`.
 */
export function frameGain(mask: Uint8Array, rate: number, start: number, out: Float32Array): void {
  const frameSeconds = FRAME / SR;
  const at = (f: number) => mask[Math.min(Math.max(f, 0), mask.length - 1)] ?? 0;
  for (let i = 0; i < out.length; i++) {
    const t = (start + i) / rate;
    const f = Math.floor(t / frameSeconds);
    const here = at(f);
    if (at(f - 1) === here && at(f + 1) === here) {
      out[i] = here;
      continue;
    }
    let g = 0;
    for (let k = f - 1; k <= f + 1; k++) {
      if (!at(k)) continue;
      g +=
        ramp((t - k * frameSeconds) / RAMP_SECONDS + 0.5) -
        ramp((t - (k + 1) * frameSeconds) / RAMP_SECONDS + 0.5);
    }
    out[i] = g;
  }
}

const clampGain = (g: number) => Math.min(20, Math.max(0.05, g));

/** Least-squares gains so two outputs sum back to `mix`, clamped so neither is scaled wildly. */
export function projectToMixture(y: [Float32Array, Float32Array], mix: Float32Array): void {
  const [a, b] = y;
  let aa = 1e-6;
  let ab = 0;
  let bb = 1e-6;
  let am = 0;
  let bm = 0;
  for (let i = 0; i < mix.length; i++) {
    const p = a[i] ?? 0;
    const q = b[i] ?? 0;
    const m = mix[i] ?? 0;
    aa += p * p;
    ab += p * q;
    bb += q * q;
    am += p * m;
    bm += q * m;
  }
  const det = aa * bb - ab * ab;
  const ga = clampGain((bb * am - ab * bm) / det);
  const gb = clampGain((aa * bm - ab * am) / det);
  for (let i = 0; i < mix.length; i++) {
    a[i] = (a[i] ?? 0) * ga;
    b[i] = (b[i] ?? 0) * gb;
  }
}

export interface SeparatedPiece {
  speaker: number;
  /** 16 kHz sample the audio starts at. */
  start: number;
  audio: Float32Array;
}

export interface CrosstalkOptions {
  /** Pieces an earlier run already separated; separation starts at this piece of `overlapPieces`. */
  from?: number;
  /** Called with the outputs of each piece once it is separated, and awaited. */
  onPiece?: (done: number, total: number, outputs: SeparatedPiece[]) => Promise<void> | void;
}

/**
 * Separates each crosstalk piece on a model-length chunk centred on it, and gives each output to one
 * of the two voices by (a) which output carries the mix where only one of them talks inside the chunk
 * and (b) half its embedding similarity to the voice. Only the piece itself is kept, faded in and out.
 */
export async function separateCrosstalk(
  mix: Float32Array,
  active: readonly Uint8Array[],
  model: SpeakerSeparationModel,
  embed: Embed,
  voices: Voices,
  { from = 0, onPiece }: CrosstalkOptions = {},
): Promise<SeparatedPiece[]> {
  const n = model.segment;
  const pieces = overlapPieces(active, n);
  const count = new Uint8Array(active[0]?.length ?? 0).map((_, t) =>
    active.reduce((s, a) => s + (a[t] ?? 0), 0),
  );
  const out: SeparatedPiece[] = [];
  for (const [p, piece] of pieces.entries()) {
    if (p < from) continue;
    const mid = Math.floor((piece.start + piece.end) / 2) * FRAME;
    const c0 = Math.min(Math.max(mid - Math.floor(n / 2), 0), Math.max(0, mix.length - n));
    const chunk = new Float32Array(n);
    chunk.set(mix.subarray(c0, c0 + n));
    let energy = 0;
    for (const v of chunk) energy += v * v;
    const gain = Math.min(TARGET_RMS / (Math.sqrt(energy / n) + 1e-9), MAX_GAIN);
    const raw = await model.run(chunk.map((v) => v * gain));
    const y: [Float32Array, Float32Array] = [
      raw.slice(0, n).map((v) => v / gain),
      raw.slice(n, 2 * n).map((v) => v / gain),
    ];
    projectToMixture(y, chunk);

    const score = [
      [0, 0],
      [0, 0],
    ];
    const only = new Float32Array(n);
    for (const [col, sp] of piece.speakers.entries()) {
      const mask = Uint8Array.from(count, (c, t) => (c === 1 ? (active[sp]?.[t] ?? 0) : 0));
      frameGain(mask, SR, c0, only);
      let den = 1e-9;
      for (let i = 0; i < n; i++) den += (only[i] ?? 0) * (chunk[i] ?? 0) ** 2;
      for (const [j, yj] of y.entries()) {
        let num = 0;
        for (let i = 0; i < n; i++) num += (only[i] ?? 0) * (yj[i] ?? 0) * (chunk[i] ?? 0);
        const row = score[j];
        if (row) row[col] = (row[col] ?? 0) + num / den;
      }
    }
    const ra = piece.start * FRAME - c0;
    const rb = Math.min(piece.end * FRAME - c0, n);
    if (rb - ra > 0.5 * SR) {
      for (const [j, yj] of y.entries()) {
        const e = centred(await embed(yj.subarray(Math.max(0, ra - SR / 2), rb + SR / 2)), voices.mean);
        for (const [col, sp] of piece.speakers.entries()) {
          const c = voices.centroids[sp];
          let s = 0;
          for (let i = 0; i < e.length; i++) s += (e[i] ?? 0) * (c?.[i] ?? 0);
          const row = score[j];
          if (row) row[col] = (row[col] ?? 0) + 0.5 * s;
        }
      }
    }
    const at = (j: number, c: number) => score[j]?.[c] ?? 0;
    const straight = at(0, 0) + at(1, 1) >= at(0, 1) + at(1, 0);
    const owners = straight ? piece.speakers : [piece.speakers[1], piece.speakers[0]];

    const region = new Uint8Array(count.length);
    region.fill(1, piece.start, piece.end);
    const fade = new Float32Array(Math.min(n, mix.length - c0));
    frameGain(region, SR, c0, fade);
    const lo = fade.findIndex((g) => g > 0);
    const hi = fade.findLastIndex((g) => g > 0) + 1;
    const outputs = (lo < 0 ? [] : y).map((yj, j) => ({
      speaker: owners[j] ?? 0,
      start: c0 + lo,
      audio: fade.subarray(lo, hi).map((g, i) => g * (yj[lo + i] ?? 0)),
    }));
    out.push(...outputs);
    await onPiece?.(p + 1, pieces.length, outputs);
  }
  return out;
}
