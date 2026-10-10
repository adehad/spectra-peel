export interface Chunk {
  start: number;
  end: number;
}

/**
 * Chunk starts sit at every multiple of the step below `total`, as in StemSplitio's infer.py,
 * so the tail can get a short extra chunk. Matching it keeps golden parity exact at the end.
 */
export function planChunks(total: number, size: number, overlap: number): Chunk[] {
  const step = size - overlap;
  if (step <= 0) throw new RangeError(`overlap ${overlap} leaves no step for chunk size ${size}`);
  const chunks: Chunk[] = [];
  for (let start = 0; start < total; start += step) {
    chunks.push({ start, end: Math.min(start + size, total) });
  }
  return chunks;
}

/** StemSplitio's transition window: linear 0..1 fade over `overlapFraction` of each end, flat between. */
export function transitionWindow(length: number, overlapFraction = 0.25): Float32Array {
  const w = new Float32Array(length).fill(1);
  const transition = Math.floor(length * overlapFraction);
  for (let i = 0; i < transition; i++) {
    const v = transition === 1 ? 0 : i / (transition - 1);
    w[i] = v;
    w[length - 1 - i] = v;
  }
  return w;
}

/** Same shape as Demucs' apply_model weighting: ramps 1..n/2 then back down, scaled to peak at 1. */
export function triangleWeights(length: number): Float32Array {
  const w = new Float32Array(length);
  const half = Math.floor(length / 2);
  for (let i = 0; i < half; i++) w[i] = i + 1;
  for (let i = half; i < length; i++) w[i] = length - i;
  const peak = Math.max(half, length - half);
  for (let i = 0; i < length; i++) w[i] = (w[i] ?? 0) / peak;
  return w;
}

/**
 * Accumulates weighted chunks; finish() divides by the summed weight so overlaps blend without a gain bump.
 * The 1e-8 floor matches infer.py, so a sample whose only weight is 0 comes out as 0.
 */
export class OverlapAdder {
  readonly #sum: Float32Array;
  readonly #weight: Float32Array;

  constructor(length: number) {
    this.#sum = new Float32Array(length);
    this.#weight = new Float32Array(length);
  }

  add(start: number, chunk: Float32Array, weights: Float32Array = triangleWeights(chunk.length)): void {
    if (start < 0 || start + chunk.length > this.#sum.length) {
      throw new RangeError(`chunk [${start}, ${start + chunk.length}) outside [0, ${this.#sum.length})`);
    }
    for (let i = 0; i < chunk.length; i++) {
      const w = weights[i] ?? 0;
      this.#sum[start + i] = (this.#sum[start + i] ?? 0) + (chunk[i] ?? 0) * w;
      this.#weight[start + i] = (this.#weight[start + i] ?? 0) + w;
    }
  }

  finish(): Float32Array {
    const out = new Float32Array(this.#sum.length);
    for (let i = 0; i < out.length; i++) {
      const w = this.#weight[i] ?? 0;
      out[i] = (this.#sum[i] ?? 0) / Math.max(w, 1e-8);
    }
    return out;
  }
}
