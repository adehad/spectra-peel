/** Samples per min/max pair at the finest level. Each coarser level halves the resolution. */
export const PEAKS_BASE = 256;

/** Min/max pairs over all channels, interleaved as [min0, max0, min1, max1, ...] in -1..1. */
export interface PeaksPyramid {
  base: number;
  totalSamples: number;
  levels: Float32Array[];
}

export interface PeaksState {
  pairs: Float32Array;
  min: number;
  max: number;
  fill: number;
  total: number;
}

export class PeaksBuilder {
  private readonly pairs: number[] = [];
  private min = Infinity;
  private max = -Infinity;
  private fill = 0;
  private total = 0;

  snapshot(): PeaksState {
    const { min, max, fill, total } = this;
    return { pairs: Float32Array.from(this.pairs), min, max, fill, total };
  }

  restore(state: PeaksState): void {
    this.pairs.length = 0;
    for (const v of state.pairs) this.pairs.push(v);
    this.min = state.min;
    this.max = state.max;
    this.fill = state.fill;
    this.total = state.total;
  }

  push(channels: readonly Float32Array[]): void {
    const length = channels[0]?.length ?? 0;
    for (let i = 0; i < length; i++) {
      for (const c of channels) {
        const v = c[i] ?? 0;
        if (v < this.min) this.min = v;
        if (v > this.max) this.max = v;
      }
      if (++this.fill === PEAKS_BASE) this.flush();
    }
    this.total += length;
  }

  finish(): PeaksPyramid {
    if (this.fill > 0) this.flush();
    const levels = [Float32Array.from(this.pairs)];
    for (let prev = levels[0]; prev && prev.length > 2;) {
      const next = new Float32Array(Math.ceil(prev.length / 4) * 2);
      for (let i = 0; i < next.length; i += 2) {
        const a = 2 * i;
        next[i] = Math.min(prev[a] ?? Infinity, prev[a + 2] ?? Infinity);
        next[i + 1] = Math.max(prev[a + 1] ?? -Infinity, prev[a + 3] ?? -Infinity);
      }
      levels.push(next);
      prev = next;
    }
    return { base: PEAKS_BASE, totalSamples: this.total, levels };
  }

  private flush(): void {
    this.pairs.push(this.min, this.max);
    this.min = Infinity;
    this.max = -Infinity;
    this.fill = 0;
  }
}

/**
 * Min/max for each of `columns` equal slices of [start, end) samples, from the coarsest level that still
 * has at least one pair per column. Returns interleaved pairs like the pyramid levels.
 */
export function peaksForRange(p: PeaksPyramid, start: number, end: number, columns: number): Float32Array {
  const out = new Float32Array(columns * 2);
  const perColumn = (end - start) / columns;
  let level = 0;
  while (level + 1 < p.levels.length && p.base * 2 ** (level + 1) <= perColumn) level++;
  const data = p.levels[level];
  const span = p.base * 2 ** level;
  if (!data) return out;
  const pairs = data.length / 2;
  for (let c = 0; c < columns; c++) {
    const from = Math.max(0, Math.floor((start + c * perColumn) / span));
    const to = Math.min(pairs, Math.max(from + 1, Math.ceil((start + (c + 1) * perColumn) / span)));
    let min = Infinity;
    let max = -Infinity;
    for (let i = from; i < to; i++) {
      min = Math.min(min, data[2 * i] ?? Infinity);
      max = Math.max(max, data[2 * i + 1] ?? -Infinity);
    }
    out[2 * c] = min === Infinity ? 0 : min;
    out[2 * c + 1] = max === -Infinity ? 0 : max;
  }
  return out;
}

export function serializePeaks(p: PeaksPyramid): ArrayBuffer {
  const header = 3 + p.levels.length;
  const floats = p.levels.reduce((n, l) => n + l.length, 0);
  const buf = new ArrayBuffer(header * 8 + floats * 4);
  const head = new Float64Array(buf, 0, header);
  head.set([p.base, p.totalSamples, p.levels.length, ...p.levels.map((l) => l.length)]);
  let at = header * 8;
  for (const l of p.levels) {
    new Float32Array(buf, at, l.length).set(l);
    at += l.length * 4;
  }
  return buf;
}

export function deserializePeaks(buf: ArrayBuffer): PeaksPyramid {
  const [base = PEAKS_BASE, totalSamples = 0, count = 0] = new Float64Array(buf, 0, 3);
  const lengths = new Float64Array(buf, 24, count);
  const levels: Float32Array[] = [];
  let at = (3 + count) * 8;
  for (const length of lengths) {
    levels.push(new Float32Array(buf.slice(at, at + length * 4)));
    at += length * 4;
  }
  return { base, totalSamples, levels };
}
