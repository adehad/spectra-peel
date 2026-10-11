/** Phases above this are quantised to the nearest of this many; common rate pairs need at most 640. */
const MAX_PHASES = 4096;

function gcd(a: number, b: number): number {
  return b === 0 ? a : gcd(b, a % b);
}

/** Zeroth-order modified Bessel function of the first kind, for Kaiser windows. */
export function besselI0(x: number): number {
  let sum = 1;
  let term = 1;
  for (let k = 1; k < 50; k++) {
    term *= (x / (2 * k)) ** 2;
    sum += term;
    if (term < sum * 1e-12) break;
  }
  return sum;
}

/**
 * Streaming polyphase resampler with a Kaiser-windowed sinc. The filter is symmetric about each
 * output instant, so output sample n lines up with input time n * from / to with no added delay.
 */
export class Resampler {
  readonly from: number;
  readonly to: number;
  private readonly up: number;
  private readonly down: number;
  private readonly taps: number;
  private readonly phases: number;
  private readonly table: Float32Array;
  /** Input history; buffer[0] is absolute input sample `bufferStart`, which starts negative as zero padding. */
  private buffer: Float32Array;
  private bufferStart: number;
  private received = 0;
  private produced = 0;

  constructor(from: number, to: number, zeroCrossings = 24, beta = 8.6) {
    this.from = from;
    this.to = to;
    const g = gcd(from, to);
    this.up = to / g;
    this.down = from / g;
    const cutoff = Math.min(1, this.up / this.down) * 0.96;
    this.taps = 2 * Math.ceil(zeroCrossings / cutoff);
    this.phases = Math.min(this.up, MAX_PHASES);
    this.table = new Float32Array(this.phases * this.taps);
    this.buffer = new Float32Array(this.taps);
    this.bufferStart = -this.taps;
    const half = this.taps / 2;
    const norm = besselI0(beta);
    for (let p = 0; p < this.phases; p++) {
      for (let j = 0; j < this.taps; j++) {
        const t = j - half + 1 - p / this.phases;
        const x = cutoff * t;
        const sinc = x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
        const r = t / half;
        const window = Math.abs(r) >= 1 ? 0 : besselI0(beta * Math.sqrt(1 - r * r)) / norm;
        this.table[p * this.taps + j] = cutoff * sinc * window;
      }
    }
  }

  /** Output samples for `inputSamples` of input, rounded up. */
  outputLength(inputSamples: number): number {
    return Math.ceil((inputSamples * this.up) / this.down);
  }

  push(input: Float32Array): Float32Array {
    if (this.from === this.to) return input.slice();
    this.append(input);
    this.received += input.length;
    return this.drain(this.received);
  }

  finish(): Float32Array {
    if (this.from === this.to) return new Float32Array(0);
    this.append(new Float32Array(this.taps));
    return this.drain(Infinity);
  }

  private append(input: Float32Array): void {
    const keepFrom = Math.max(
      0,
      Math.floor((this.produced * this.down) / this.up) - 2 * this.taps - this.bufferStart,
    );
    const kept = this.buffer.subarray(keepFrom);
    const next = new Float32Array(kept.length + input.length);
    next.set(kept);
    next.set(input, kept.length);
    this.buffer = next;
    this.bufferStart += keepFrom;
  }

  private drain(available: number): Float32Array {
    const half = this.taps / 2;
    const total = this.outputLength(this.received);
    const bound = Number.isFinite(available)
      ? Math.min(total, Math.ceil(((available - half) * this.up) / this.down) + 1)
      : total;
    const out = new Float32Array(Math.max(0, bound - this.produced));
    const buffer = this.buffer;
    const table = this.table;
    const taps = this.taps;
    let count = 0;
    while (this.produced < total) {
      const pos = this.produced * this.down;
      const i = Math.floor(pos / this.up);
      const phase = Math.round(((pos % this.up) / this.up) * this.phases) % this.phases;
      const base = i + (phase === 0 && pos % this.up !== 0 ? 1 : 0);
      if (base + half >= available || count >= out.length) break;
      const row = phase * taps;
      const k0 = base - half + 1 - this.bufferStart;
      let acc = 0;
      for (let j = 0; j < taps; j++) acc += (buffer[k0 + j] ?? 0) * (table[row + j] ?? 0);
      out[count++] = acc;
      this.produced++;
    }
    return out.subarray(0, count);
  }
}
