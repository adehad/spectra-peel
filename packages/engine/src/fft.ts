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
