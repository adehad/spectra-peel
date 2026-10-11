/** In-place iterative radix-2 complex FFT for one fixed power-of-two size. */
export class FFT {
  readonly size: number;
  private readonly cos: Float64Array;
  private readonly sin: Float64Array;
  private readonly reversed: Uint32Array;

  constructor(size: number) {
    if (size < 2 || (size & (size - 1)) !== 0) throw new RangeError(`FFT size ${size} is not a power of two`);
    this.size = size;
    this.cos = new Float64Array(size / 2);
    this.sin = new Float64Array(size / 2);
    for (let i = 0; i < size / 2; i++) {
      this.cos[i] = Math.cos((-2 * Math.PI * i) / size);
      this.sin[i] = Math.sin((-2 * Math.PI * i) / size);
    }
    this.reversed = new Uint32Array(size);
    const bits = Math.log2(size);
    for (let i = 0; i < size; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
      this.reversed[i] = r;
    }
  }

  transform(re: Float64Array, im: Float64Array): void {
    const n = this.size;
    for (let i = 0; i < n; i++) {
      const j = this.reversed[i] ?? i;
      if (j > i) {
        const tr = re[i] ?? 0;
        re[i] = re[j] ?? 0;
        re[j] = tr;
        const ti = im[i] ?? 0;
        im[i] = im[j] ?? 0;
        im[j] = ti;
      }
    }
    for (let len = 2; len <= n; len *= 2) {
      const half = len / 2;
      const step = n / len;
      for (let start = 0; start < n; start += len) {
        for (let k = 0; k < half; k++) {
          const wr = this.cos[k * step] ?? 1;
          const wi = this.sin[k * step] ?? 0;
          const a = start + k;
          const b = a + half;
          const br = re[b] ?? 0;
          const bi = im[b] ?? 0;
          const xr = br * wr - bi * wi;
          const xi = br * wi + bi * wr;
          const ar = re[a] ?? 0;
          const ai = im[a] ?? 0;
          re[b] = ar - xr;
          im[b] = ai - xi;
          re[a] = ar + xr;
          im[a] = ai + xi;
        }
      }
    }
  }
}

/** Complex FFT for any size whose prime factors are 2, 3 and 5, by recursive mixed-radix decimation in time. */
export class MixedFFT {
  readonly size: number;
  private readonly factors: number[];
  private readonly cos: Float64Array;
  private readonly sin: Float64Array;
  private readonly scratchRe: Float64Array[];
  private readonly scratchIm: Float64Array[];

  constructor(size: number) {
    this.size = size;
    this.factors = [];
    let n = size;
    for (const p of [4, 2, 3, 5]) {
      while (n % p === 0) {
        this.factors.push(p);
        n /= p;
      }
    }
    if (n !== 1) throw new RangeError(`FFT size ${size} has a prime factor above 5`);
    this.cos = Float64Array.from({ length: size }, (_, i) => Math.cos((-2 * Math.PI * i) / size));
    this.sin = Float64Array.from({ length: size }, (_, i) => Math.sin((-2 * Math.PI * i) / size));
    this.scratchRe = this.factors.map(() => new Float64Array(size));
    this.scratchIm = this.factors.map(() => new Float64Array(size));
  }

  /** Forward transform of (inRe, inIm) into (outRe, outIm). Inputs are left untouched. */
  transform(inRe: Float64Array, inIm: Float64Array, outRe: Float64Array, outIm: Float64Array): void {
    this.recurse(inRe, inIm, 0, 1, this.size, outRe, outIm, 0, 0);
  }

  private recurse(
    xRe: Float64Array,
    xIm: Float64Array,
    offset: number,
    stride: number,
    n: number,
    yRe: Float64Array,
    yIm: Float64Array,
    yOffset: number,
    depth: number,
  ): void {
    if (n === 1) {
      yRe[yOffset] = xRe[offset] ?? 0;
      yIm[yOffset] = xIm[offset] ?? 0;
      return;
    }
    const r = this.factors[depth] ?? n;
    const m = n / r;
    const tRe = this.scratchRe[depth] ?? new Float64Array(n);
    const tIm = this.scratchIm[depth] ?? new Float64Array(n);
    for (let q = 0; q < r; q++) {
      this.recurse(xRe, xIm, offset + q * stride, stride * r, m, tRe, tIm, yOffset + q * m, depth + 1);
    }
    const twiddleStep = this.size / n;
    for (let k = 0; k < m; k++) {
      for (let p = 0; p < r; p++) {
        const out = k + p * m;
        let accRe = 0;
        let accIm = 0;
        for (let q = 0; q < r; q++) {
          const e = ((q * out) % n) * twiddleStep;
          const wr = this.cos[e] ?? 1;
          const wi = this.sin[e] ?? 0;
          const ar = tRe[yOffset + q * m + k] ?? 0;
          const ai = tIm[yOffset + q * m + k] ?? 0;
          accRe += ar * wr - ai * wi;
          accIm += ar * wi + ai * wr;
        }
        yRe[yOffset + out] = accRe;
        yIm[yOffset + out] = accIm;
      }
    }
  }
}
