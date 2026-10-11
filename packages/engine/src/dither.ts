const SCALE = 32767;

export class Xorshift32 {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0 || 1;
  }

  /** Uniform in [0, 1). */
  next(): number {
    let x = this.state;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.state = x >>> 0;
    return this.state / 4294967296;
  }
}

/** Float to 16-bit with triangular (TPDF) dither of +-1 LSB, which decorrelates the error from the signal. */
export function quantize16(
  input: Float32Array,
  rng: Xorshift32,
  out = new Int16Array(input.length),
): Int16Array {
  for (let i = 0; i < input.length; i++) {
    const v = Math.round((input[i] ?? 0) * SCALE + rng.next() - rng.next());
    out[i] = v > 32767 ? 32767 : v < -32768 ? -32768 : v;
  }
  return out;
}

export function toFloat(input: Int16Array | Int32Array, out = new Float32Array(input.length)): Float32Array {
  for (let i = 0; i < input.length; i++) out[i] = (input[i] ?? 0) / SCALE;
  return out;
}
