import { concatFloat32 } from "./buffers";
import { MixedFFT } from "./fft";

/**
 * DeepFilterNet3's DSP around its neural graph, following libDF v0.5.6. reference/dfn3.py is the
 * NumPy twin, checked against the official package; the fixtures in fixtures/dfn3 come from it.
 */
export const DFN3 = Object.freeze({
  sampleRate: 48_000,
  fftSize: 960,
  hop: 480,
  bins: 481,
  erbBands: 32,
  dfBins: 96,
  dfOrder: 5,
  dfLookahead: 2,
  /** Frames the graph looks ahead; a chunk must be fed this many extra feature frames. */
  convLookahead: 2,
  alpha: 0.99,
});

/**
 * The neural graph. `featErb` is [frames, 32]; `featSpec` is [2, frames, 96], real plane then imaginary.
 * Returns `erbMask` [frames, 32] and `dfCoefs` [5, frames, 96, 2]. The implementation carries recurrent
 * state between calls and must advance it by exactly `keep` frames: the frames after that are lookahead
 * that the next call feeds again.
 */
export interface Dfn3Model {
  run(
    featErb: Float32Array,
    featSpec: Float32Array,
    frames: number,
    keep: number,
  ): Promise<{ erbMask: Float32Array; dfCoefs: Float32Array }>;
}

export function vorbisWindow(): Float32Array {
  const n = DFN3.fftSize;
  return Float32Array.from({ length: n }, (_, i) =>
    Math.sin(0.5 * Math.PI * Math.sin((Math.PI * (i + 0.5)) / n) ** 2),
  );
}

const freqToErb = (f: number) => 9.265 * Math.log1p(f / (24.7 * 9.265));
const erbToFreq = (e: number) => 24.7 * 9.265 * (Math.exp(e / 9.265) - 1);

export function erbWidths(): number[] {
  const lo = freqToErb(0);
  const step = (freqToErb(DFN3.sampleRate / 2) - lo) / DFN3.erbBands;
  const widths: number[] = [];
  let prev = 0;
  let over = 0;
  for (let i = 1; i <= DFN3.erbBands; i++) {
    const fb = Math.round(erbToFreq(lo + i * step) / (DFN3.sampleRate / DFN3.fftSize));
    let n = fb - prev - over;
    if (n < 2) {
      over = 2 - n;
      n = 2;
    } else over = 0;
    widths.push(n);
    prev = fb;
  }
  const last = widths.length - 1;
  widths[last] = (widths[last] ?? 0) + 1;
  widths[last] = (widths[last] ?? 0) - (widths.reduce((a, b) => a + b, 0) - DFN3.bins);
  return widths;
}

interface Frame {
  /** 481 complex bins, interleaved re/im, scaled as libDF does (forward FFT divided by 960). */
  spec: Float32Array;
  erb: Float32Array;
  /** 96 unit-normalised bins, interleaved re/im. */
  cplx: Float32Array;
}

/** Turns 480-sample hops into spectra and normalised features, carrying libDF's running state. */
export class Dfn3Analysis {
  private readonly fft = new MixedFFT(DFN3.fftSize);
  private readonly window = vorbisWindow();
  private readonly widths = erbWidths();
  private readonly memory = new Float32Array(DFN3.hop);
  private readonly mean: Float32Array;
  private readonly unit: Float32Array;
  private readonly re = new Float64Array(DFN3.fftSize);
  private readonly im = new Float64Array(DFN3.fftSize);
  private readonly outRe = new Float64Array(DFN3.fftSize);
  private readonly outIm = new Float64Array(DFN3.fftSize);

  constructor() {
    this.mean = Float32Array.from({ length: DFN3.erbBands }, (_, i) => -60 + (-30 * i) / (DFN3.erbBands - 1));
    this.unit = Float32Array.from(
      { length: DFN3.dfBins },
      (_, i) => 0.001 - (0.0009 * i) / (DFN3.dfBins - 1),
    );
  }

  frame(hop: Float32Array): Frame {
    const { fftSize, hop: h, bins, alpha } = DFN3;
    for (let i = 0; i < h; i++) {
      this.re[i] = (this.memory[i] ?? 0) * (this.window[i] ?? 0);
      this.re[h + i] = (hop[i] ?? 0) * (this.window[h + i] ?? 0);
    }
    this.im.fill(0);
    this.memory.set(hop);
    this.fft.transform(this.re, this.im, this.outRe, this.outIm);
    const spec = new Float32Array(bins * 2);
    for (let k = 0; k < bins; k++) {
      spec[2 * k] = (this.outRe[k] ?? 0) / fftSize;
      spec[2 * k + 1] = (this.outIm[k] ?? 0) / fftSize;
    }

    const erb = new Float32Array(DFN3.erbBands);
    let bin = 0;
    for (const [b, width] of this.widths.entries()) {
      let power = 0;
      for (let j = 0; j < width; j++, bin++)
        power += (spec[2 * bin] ?? 0) ** 2 + (spec[2 * bin + 1] ?? 0) ** 2;
      const db = 10 * Math.log10(power / width + 1e-10);
      const mean = Math.fround(db * (1 - alpha) + (this.mean[b] ?? 0) * alpha);
      this.mean[b] = mean;
      erb[b] = (db - mean) / 40;
    }

    const cplx = new Float32Array(DFN3.dfBins * 2);
    for (let k = 0; k < DFN3.dfBins; k++) {
      const re = spec[2 * k] ?? 0;
      const im = spec[2 * k + 1] ?? 0;
      const unit = Math.fround(Math.hypot(re, im) * (1 - alpha) + (this.unit[k] ?? 0) * alpha);
      this.unit[k] = unit;
      const scale = 1 / Math.sqrt(unit);
      cplx[2 * k] = re * scale;
      cplx[2 * k + 1] = im * scale;
    }
    return { spec, erb, cplx };
  }
}

/** Overlap-add synthesis matching libDF: unnormalised inverse FFT, window, then add the previous half. */
export class Dfn3Synthesis {
  private readonly fft = new MixedFFT(DFN3.fftSize);
  private readonly window = vorbisWindow();
  private readonly memory = new Float32Array(DFN3.hop);
  private readonly re = new Float64Array(DFN3.fftSize);
  private readonly im = new Float64Array(DFN3.fftSize);
  private readonly outRe = new Float64Array(DFN3.fftSize);
  private readonly outIm = new Float64Array(DFN3.fftSize);

  frame(spec: Float32Array): Float32Array {
    const n: number = DFN3.fftSize;
    const { hop: h, bins } = DFN3;
    for (let k = 0; k < bins; k++) {
      this.re[k] = spec[2 * k] ?? 0;
      this.im[k] = -(spec[2 * k + 1] ?? 0);
    }
    for (let k = bins; k < n; k++) {
      this.re[k] = this.re[n - k] ?? 0;
      this.im[k] = -(this.im[n - k] ?? 0);
    }
    this.im[0] = 0;
    this.im[n / 2] = 0;
    this.fft.transform(this.re, this.im, this.outRe, this.outIm);
    const out = new Float32Array(h);
    for (let i = 0; i < h; i++) {
      out[i] = (this.outRe[i] ?? 0) * (this.window[i] ?? 0) + (this.memory[i] ?? 0);
      this.memory[i] = (this.outRe[h + i] ?? 0) * (this.window[h + i] ?? 0);
    }
    return out;
  }
}

/**
 * ERB gain on every bin, then the order-5 deep filter over frames t-2..t+2 of the noisy spectrum on the
 * first 96 bins. `noisy(t)` returns a frame's noisy spectrum, or undefined outside the signal.
 */
export function applyDfn3(
  noisy: (frame: number) => Float32Array | undefined,
  frame: number,
  mask: Float32Array,
  coefs: (order: number, bin: number) => [number, number],
  widths: readonly number[],
): Float32Array {
  const spec = noisy(frame) ?? new Float32Array(DFN3.bins * 2);
  const out = new Float32Array(DFN3.bins * 2);
  let bin = 0;
  for (const [b, width] of widths.entries()) {
    const g = mask[b] ?? 0;
    for (let j = 0; j < width; j++, bin++) {
      out[2 * bin] = (spec[2 * bin] ?? 0) * g;
      out[2 * bin + 1] = (spec[2 * bin + 1] ?? 0) * g;
    }
  }
  const before = DFN3.dfOrder - 1 - DFN3.dfLookahead;
  for (let k = 0; k < DFN3.dfBins; k++) {
    let re = 0;
    let im = 0;
    for (let n = 0; n < DFN3.dfOrder; n++) {
      const x = noisy(frame - before + n);
      if (!x) continue;
      const [cr, ci] = coefs(n, k);
      const xr = x[2 * k] ?? 0;
      const xi = x[2 * k + 1] ?? 0;
      re += xr * cr - xi * ci;
      im += xr * ci + xi * cr;
    }
    out[2 * k] = re;
    out[2 * k + 1] = im;
  }
  return out;
}

/**
 * Denoises a mono 48 kHz stream in chunks. Output is delay-compensated and exactly as long as the input,
 * as in DeepFilterNet's offline `enhance`: the input gets one FFT frame of zeros appended and the first
 * hop of synthesis is dropped.
 */
export class Dfn3Stream {
  private readonly model: Dfn3Model;
  private readonly chunk: number;
  private readonly analysis = new Dfn3Analysis();
  private readonly synthesis = new Dfn3Synthesis();
  private readonly widths = erbWidths();
  private readonly frames = new Map<number, Frame>();
  private hop = new Float32Array(DFN3.hop);
  private hopFill = 0;
  private analysed = 0;
  private enhanced = 0;
  private received = 0;
  private emitted = 0;
  private skip = DFN3.hop;

  constructor(model: Dfn3Model, chunkFrames = 500) {
    this.model = model;
    this.chunk = chunkFrames;
  }

  async push(samples: Float32Array): Promise<Float32Array> {
    this.received += samples.length;
    this.analyse(samples);
    const parts: Float32Array[] = [];
    while (this.analysed - this.enhanced >= this.chunk + DFN3.convLookahead) {
      parts.push(await this.runChunk(this.enhanced + this.chunk, false));
    }
    return concatFloat32(parts);
  }

  async finish(): Promise<Float32Array> {
    this.analyse(new Float32Array(DFN3.fftSize));
    const parts: Float32Array[] = [];
    while (this.enhanced < this.analysed) {
      parts.push(await this.runChunk(Math.min(this.analysed, this.enhanced + this.chunk), true));
    }
    return concatFloat32(parts);
  }

  private analyse(samples: Float32Array): void {
    let at = 0;
    while (at < samples.length) {
      const n = Math.min(DFN3.hop - this.hopFill, samples.length - at);
      this.hop.set(samples.subarray(at, at + n), this.hopFill);
      this.hopFill += n;
      at += n;
      if (this.hopFill === DFN3.hop) {
        this.frames.set(this.analysed++, this.analysis.frame(this.hop));
        this.hop = new Float32Array(DFN3.hop);
        this.hopFill = 0;
      }
    }
  }

  private async runChunk(end: number, final: boolean): Promise<Float32Array> {
    const start = this.enhanced;
    const fedEnd = final ? Math.min(this.analysed, end + DFN3.convLookahead) : end + DFN3.convLookahead;
    const fed = fedEnd - start;
    const featErb = new Float32Array(fed * DFN3.erbBands);
    const featSpec = new Float32Array(2 * fed * DFN3.dfBins);
    for (let t = 0; t < fed; t++) {
      const f = this.frames.get(start + t);
      if (!f) throw new Error(`frame ${start + t} missing`);
      featErb.set(f.erb, t * DFN3.erbBands);
      for (let k = 0; k < DFN3.dfBins; k++) {
        featSpec[t * DFN3.dfBins + k] = f.cplx[2 * k] ?? 0;
        featSpec[(fed + t) * DFN3.dfBins + k] = f.cplx[2 * k + 1] ?? 0;
      }
    }
    const keep = end - start;
    const { erbMask, dfCoefs } = await this.model.run(featErb, featSpec, fed, keep);

    const out = new Float32Array(keep * DFN3.hop);
    const noisy = (t: number) => (t >= 0 && t < this.analysed ? this.frames.get(t) : undefined)?.spec;
    for (let t = 0; t < keep; t++) {
      const mask = erbMask.subarray(t * DFN3.erbBands, (t + 1) * DFN3.erbBands);
      const coefs = (n: number, k: number): [number, number] => {
        const i = ((n * fed + t) * DFN3.dfBins + k) * 2;
        return [dfCoefs[i] ?? 0, dfCoefs[i + 1] ?? 0];
      };
      out.set(this.synthesis.frame(applyDfn3(noisy, start + t, mask, coefs, this.widths)), t * DFN3.hop);
    }
    this.enhanced = end;
    for (const t of this.frames.keys()) if (t < end - DFN3.dfOrder) this.frames.delete(t);
    return this.trim(out);
  }

  /** Drops the first hop of synthesis delay and anything past the input length. */
  private trim(out: Float32Array): Float32Array {
    const drop = Math.min(this.skip, out.length);
    this.skip -= drop;
    const kept = out.subarray(drop, drop + Math.max(0, this.received - this.emitted));
    this.emitted += kept.length;
    return kept;
  }
}
