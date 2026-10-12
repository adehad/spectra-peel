import { transitionWindow } from "./chunks";

export const DEMUCS = Object.freeze({
  sampleRate: 44_100,
  /** Samples per model run: HT-Demucs' 7.8 s training segment. */
  segment: Math.trunc(7.8 * 44_100),
  overlap: 0.25,
  sources: ["drums", "bass", "other", "vocals"],
});

/**
 * AudioSep's separator as reference/audiosep exports it: 32 kHz mono, one 5.11 s chunk a run, steered
 * by a 512-d CLAP text embedding. A 40% overlap fades over about 2 s, close to upstream's 1 s discarded
 * at each chunk edge, since the model is weakest near the edges.
 */
export const AUDIOSEP = Object.freeze({
  sampleRate: 32_000,
  segment: 320 * (512 - 1),
  overlap: 0.4,
  embedding: 512,
});

/**
 * A stem separation graph. `mix` is planar stereo [2, segment]; the result is planar
 * [sources, 2, segment].
 */
export interface SeparationModel {
  run(mix: Float32Array): Promise<Float32Array>;
}

interface Options {
  segment: number;
  overlap: number;
  sources: number;
}

/**
 * Runs a separation model over overlapping segments and blends them with StemSplitio's transition
 * window, streaming: output up to the start of the next segment is final and is returned as soon as it
 * is. Segments start at every multiple of the step below the input length, as in infer.py.
 */
export class SeparationStream {
  private readonly model: SeparationModel;
  private readonly segment: number;
  private readonly step: number;
  private readonly sources: number;
  private readonly window: Float32Array;
  private input: [Float32Array, Float32Array] = [new Float32Array(0), new Float32Array(0)];
  private inputStart = 0;
  private received = 0;
  private nextChunk = 0;
  private acc: Float32Array[][];
  private weights: Float32Array;
  private accStart = 0;
  private emitted = 0;

  constructor(model: SeparationModel, options: Partial<Options> = {}) {
    this.model = model;
    this.segment = options.segment ?? DEMUCS.segment;
    this.step = this.segment - Math.floor(this.segment * (options.overlap ?? DEMUCS.overlap));
    this.sources = options.sources ?? DEMUCS.sources.length;
    this.window = transitionWindow(this.segment, options.overlap ?? DEMUCS.overlap);
    this.acc = this.makeAcc(this.segment * 2);
    this.weights = new Float32Array(this.segment * 2);
  }

  /** Planar stereo in; per source, planar stereo out, for whatever became final. */
  async push(left: Float32Array, right: Float32Array): Promise<Float32Array[][]> {
    this.append(left, right);
    while (this.received >= this.nextChunk + this.segment) await this.runChunk();
    return this.emit(Math.min(this.nextChunk, this.received));
  }

  async finish(): Promise<Float32Array[][]> {
    while (this.nextChunk < this.received) await this.runChunk();
    return this.emit(this.received);
  }

  private makeAcc(length: number): Float32Array[][] {
    return Array.from({ length: this.sources }, () => [new Float32Array(length), new Float32Array(length)]);
  }

  private append(left: Float32Array, right: Float32Array): void {
    const keepFrom = this.nextChunk - this.inputStart;
    const [l, r] = this.input;
    const next: [Float32Array, Float32Array] = [
      new Float32Array(l.length - keepFrom + left.length),
      new Float32Array(r.length - keepFrom + right.length),
    ];
    next[0].set(l.subarray(keepFrom));
    next[0].set(left, l.length - keepFrom);
    next[1].set(r.subarray(keepFrom));
    next[1].set(right, r.length - keepFrom);
    this.input = next;
    this.inputStart = this.nextChunk;
    this.received += left.length;
  }

  private async runChunk(): Promise<void> {
    const start = this.nextChunk;
    const length = Math.min(this.segment, this.received - start);
    const mix = new Float32Array(2 * this.segment);
    const offset = start - this.inputStart;
    mix.set(this.input[0].subarray(offset, offset + length), 0);
    mix.set(this.input[1].subarray(offset, offset + length), this.segment);
    const stems = await this.model.run(mix);

    const at = start - this.accStart;
    this.ensureCapacity(at + length);
    for (let i = 0; i < length; i++)
      this.weights[at + i] = (this.weights[at + i] ?? 0) + (this.window[i] ?? 0);
    for (let s = 0; s < this.sources; s++) {
      for (let c = 0; c < 2; c++) {
        const acc = this.acc[s]?.[c];
        if (!acc) continue;
        const base = (s * 2 + c) * this.segment;
        for (let i = 0; i < length; i++)
          acc[at + i] = (acc[at + i] ?? 0) + (stems[base + i] ?? 0) * (this.window[i] ?? 0);
      }
    }
    this.nextChunk += this.step;
  }

  private ensureCapacity(needed: number): void {
    if (needed <= this.weights.length) return;
    const size = Math.max(needed, this.weights.length * 2);
    const weights = new Float32Array(size);
    weights.set(this.weights);
    this.weights = weights;
    this.acc = this.acc.map((pair) =>
      pair.map((a) => {
        const grown = new Float32Array(size);
        grown.set(a);
        return grown;
      }),
    );
  }

  /** Returns blended output for [emitted, end) and drops it from the accumulators. */
  private emit(end: number): Float32Array[][] {
    const n = Math.max(0, end - this.emitted);
    const from = this.emitted - this.accStart;
    const out = this.acc.map((pair) =>
      pair.map((a) => {
        const o = new Float32Array(n);
        for (let i = 0; i < n; i++) o[i] = (a[from + i] ?? 0) / Math.max(this.weights[from + i] ?? 0, 1e-8);
        return o;
      }),
    );
    this.emitted += n;
    const shift = this.emitted - this.accStart;
    if (shift > 0) {
      this.weights.copyWithin(0, shift);
      this.weights.fill(0, this.weights.length - shift);
      for (const pair of this.acc) {
        for (const a of pair) {
          a.copyWithin(0, shift);
          a.fill(0, a.length - shift);
        }
      }
      this.accStart = this.emitted;
    }
    return out;
  }
}
