import { FFT } from "./fft";

export interface SpectrogramConfig {
  fftSize: number;
  hop: number;
  bands: number;
  minHz: number;
  /** dB below full scale that maps to 0; 0 dBFS maps to 255. */
  floorDb: number;
  tileColumns: number;
}

export const SPECTROGRAM: SpectrogramConfig = {
  fftSize: 2048,
  hop: 512,
  bands: 256,
  minHz: 20,
  floorDb: 100,
  tileColumns: 1024,
};

/** Column-major tile: column c occupies bytes [c * bands, (c + 1) * bands), band 0 is the lowest. */
export interface SpectrogramTile {
  index: number;
  columns: number;
  data: Uint8Array;
}

/** First and one-past-last FFT bin of each log-spaced band; narrow low bands repeat a bin. */
export function bandEdges(sampleRate: number, c: SpectrogramConfig): Uint16Array {
  const nyquist = sampleRate / 2;
  const binHz = sampleRate / c.fftSize;
  const edges = new Uint16Array(c.bands * 2);
  for (let b = 0; b < c.bands; b++) {
    const lo = c.minHz * (nyquist / c.minHz) ** (b / c.bands);
    const hi = c.minHz * (nyquist / c.minHz) ** ((b + 1) / c.bands);
    const first = Math.min(c.fftSize / 2 - 1, Math.floor(lo / binHz));
    edges[2 * b] = first;
    edges[2 * b + 1] = Math.min(c.fftSize / 2, Math.max(first + 1, Math.ceil(hi / binHz)));
  }
  return edges;
}

/** Builds magnitude columns centred on multiples of the hop from a mono stream, emitted as fixed-size tiles. */
export class SpectrogramBuilder {
  readonly config: SpectrogramConfig;
  private readonly fft: FFT;
  private readonly window: Float64Array;
  private readonly edges: Uint16Array;
  private readonly re: Float64Array;
  private readonly im: Float64Array;
  private buffer: Float32Array;
  private buffered: number;
  private consumed = 0;
  private total = 0;
  private columns = 0;
  private tile: Uint8Array;
  private tileFill = 0;
  private tileIndex = 0;

  constructor(sampleRate: number, config: SpectrogramConfig = SPECTROGRAM) {
    this.config = config;
    const n = config.fftSize;
    this.fft = new FFT(n);
    this.window = Float64Array.from({ length: n }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n));
    this.edges = bandEdges(sampleRate, config);
    this.re = new Float64Array(n);
    this.im = new Float64Array(n);
    this.buffer = new Float32Array(n * 4);
    this.buffered = n / 2;
    this.tile = new Uint8Array(config.tileColumns * config.bands);
  }

  push(mono: Float32Array): SpectrogramTile[] {
    this.append(mono);
    this.total += mono.length;
    return this.drain(false);
  }

  finish(): SpectrogramTile[] {
    this.append(new Float32Array(this.config.fftSize));
    const out = this.drain(true);
    if (this.tileFill > 0) out.push(this.emitTile());
    return out;
  }

  private append(samples: Float32Array): void {
    if (this.buffered + samples.length > this.buffer.length) {
      const keep = this.buffer.subarray(this.consumed, this.buffered);
      const next = new Float32Array(Math.max(this.buffer.length, (keep.length + samples.length) * 2));
      next.set(keep);
      this.buffer = next;
      this.buffered = keep.length;
      this.consumed = 0;
    }
    this.buffer.set(samples, this.buffered);
    this.buffered += samples.length;
  }

  private drain(final: boolean): SpectrogramTile[] {
    const { fftSize, hop, tileColumns } = this.config;
    const out: SpectrogramTile[] = [];
    const limit = final ? Math.ceil(this.total / hop) : Infinity;
    while (this.buffered - this.consumed >= fftSize && this.columns < limit) {
      this.column(this.buffer.subarray(this.consumed, this.consumed + fftSize));
      this.consumed += hop;
      this.columns++;
      if (this.tileFill === tileColumns) out.push(this.emitTile());
    }
    return out;
  }

  private column(frame: Float32Array): void {
    const { fftSize, bands, floorDb } = this.config;
    for (let i = 0; i < fftSize; i++) {
      this.re[i] = (frame[i] ?? 0) * (this.window[i] ?? 0);
      this.im[i] = 0;
    }
    this.fft.transform(this.re, this.im);
    const fullScale = (fftSize / 4) ** 2;
    const base = this.tileFill * bands;
    for (let b = 0; b < bands; b++) {
      let peak = 0;
      const end = this.edges[2 * b + 1] ?? 0;
      for (let k = this.edges[2 * b] ?? 0; k < end; k++) {
        const p = (this.re[k] ?? 0) ** 2 + (this.im[k] ?? 0) ** 2;
        if (p > peak) peak = p;
      }
      const db = peak > 0 ? 10 * Math.log10(peak / fullScale) : -Infinity;
      this.tile[base + b] = Math.max(0, Math.min(255, Math.round(((db + floorDb) / floorDb) * 255)));
    }
    this.tileFill++;
  }

  private emitTile(): SpectrogramTile {
    const tile = {
      index: this.tileIndex++,
      columns: this.tileFill,
      data: this.tile.slice(0, this.tileFill * this.config.bands),
    };
    this.tile.fill(0);
    this.tileFill = 0;
    return tile;
  }
}
