import { besselI0 } from "./resample";

/**
 * Integrated loudness and true peak per ITU-R BS.1770-4, as EBU R128 and the streaming services use it:
 * K-weighting, 400 ms blocks every 100 ms, an absolute gate at -70 LUFS and a relative gate 10 LU below
 * the ungated mean. Left, right and mono channels all weigh 1, so mono reads about 3 LU below the same
 * audio duplicated to stereo.
 */

/** RBJ biquad with a0 normalised to 1. */
class Biquad {
  private x1 = 0;
  private x2 = 0;
  private y1 = 0;
  private y2 = 0;
  readonly b: [number, number, number];
  readonly a: [number, number];

  constructor(b: [number, number, number], a: [number, number, number]) {
    const a0 = a[0];
    this.b = [b[0] / a0, b[1] / a0, b[2] / a0];
    this.a = [a[1] / a0, a[2] / a0];
  }

  run(x: number): number {
    const y =
      this.b[0] * x + this.b[1] * this.x1 + this.b[2] * this.x2 - this.a[0] * this.y1 - this.a[1] * this.y2;
    this.x2 = this.x1;
    this.x1 = x;
    this.y2 = this.y1;
    this.y1 = y;
    return y;
  }
}

/**
 * The two K-weighting stages for any sample rate, using libebur128's parameterisation. At 48 kHz it
 * reproduces the coefficients tabulated in BS.1770, including stage 2's unnormalised [1, -2, 1] numerator
 * that the -0.691 dB constant compensates for.
 */
export function kWeighting(sampleRate: number): [Biquad, Biquad] {
  let k = Math.tan((Math.PI * 1681.974450955533) / sampleRate);
  let q = 0.7071752369554196;
  const vh = 10 ** (3.999843853973347 / 20);
  const vb = vh ** 0.4996667741545416;
  let a0 = 1 + k / q + k * k;
  const shelf = new Biquad(
    [(vh + (vb * k) / q + k * k) / a0, (2 * (k * k - vh)) / a0, (vh - (vb * k) / q + k * k) / a0],
    [1, (2 * (k * k - 1)) / a0, (1 - k / q + k * k) / a0],
  );
  k = Math.tan((Math.PI * 38.13547087602444) / sampleRate);
  q = 0.5003270373238773;
  a0 = 1 + k / q + k * k;
  const highPass = new Biquad([1, -2, 1], [1, (2 * (k * k - 1)) / a0, (1 - k / q + k * k) / a0]);
  return [shelf, highPass];
}

const lufs = (z: number) => -0.691 + 10 * Math.log10(z);
const mean = (zs: number[]) => zs.reduce((a, b) => a + b, 0) / zs.length;

const OVERSAMPLE = 4;
const TAPS_PER_PHASE = 12;

/** 4x polyphase interpolator taps: a Kaiser-windowed sinc, 48 taps, cut off at the original Nyquist. */
function interpolatorTaps(): Float32Array[] {
  const n = OVERSAMPLE * TAPS_PER_PHASE;
  const beta = 8;
  const taps = Array.from({ length: n }, (_, i) => {
    const t = (i - (n - 1) / 2) / OVERSAMPLE;
    const sinc = t === 0 ? 1 : Math.sin(Math.PI * t) / (Math.PI * t);
    const r = (2 * i) / (n - 1) - 1;
    return sinc * (besselI0(beta * Math.sqrt(1 - r * r)) / besselI0(beta));
  });
  return Array.from({ length: OVERSAMPLE }, (_, p) =>
    Float32Array.from({ length: TAPS_PER_PHASE }, (_unused, k) => taps[k * OVERSAMPLE + p] ?? 0),
  );
}

export interface LoudnessResult {
  /** Gated integrated loudness in LUFS; -Infinity for silence. */
  integrated: number;
  /** Highest 4x-oversampled sample in dBTP. */
  truePeak: number;
}

export class LoudnessMeter {
  private readonly filters: [Biquad, Biquad][];
  private readonly step: number;
  private readonly phases = interpolatorTaps();
  private readonly history: Float32Array[];
  private readonly historyAt: Int32Array;
  /** Energy of each 100 ms step, summed over channels. */
  private readonly steps: number[] = [];
  private stepEnergy = 0;
  private stepFill = 0;
  private peak = 0;

  constructor(sampleRate: number, channels: number) {
    this.filters = Array.from({ length: channels }, () => kWeighting(sampleRate));
    this.step = Math.round(sampleRate / 10);
    // Each sample is written twice, TAPS_PER_PHASE apart, so the last taps are always contiguous.
    this.history = Array.from({ length: channels }, () => new Float32Array(2 * TAPS_PER_PHASE));
    this.historyAt = new Int32Array(channels);
  }

  push(planes: readonly Float32Array[]): void {
    const n = planes[0]?.length ?? 0;
    for (let i = 0; i < n; i++) {
      let energy = 0;
      for (const [c, [shelf, highPass]] of this.filters.entries()) {
        const x = planes[c]?.[i] ?? 0;
        const y = highPass.run(shelf.run(x));
        energy += y * y;
        this.truePeakSample(c, x);
      }
      this.stepEnergy += energy;
      if (++this.stepFill === this.step) {
        this.steps.push(this.stepEnergy);
        this.stepEnergy = 0;
        this.stepFill = 0;
      }
    }
  }

  result(): LoudnessResult {
    const blocks: number[] = [];
    for (let i = 3; i < this.steps.length; i++) {
      const sum =
        (this.steps[i] ?? 0) + (this.steps[i - 1] ?? 0) + (this.steps[i - 2] ?? 0) + (this.steps[i - 3] ?? 0);
      blocks.push(sum / (4 * this.step));
    }
    const absolute = blocks.filter((z) => lufs(z) > -70);
    let integrated = -Infinity;
    if (absolute.length > 0) {
      const relative = lufs(mean(absolute)) - 10;
      const gated = absolute.filter((z) => lufs(z) > relative);
      if (gated.length > 0) integrated = lufs(mean(gated));
    }
    return { integrated, truePeak: this.peak > 0 ? 20 * Math.log10(this.peak) : -Infinity };
  }

  private truePeakSample(channel: number, x: number): void {
    const h = this.history[channel];
    if (!h) return;
    const p = ((this.historyAt[channel] ?? 0) + 1) % TAPS_PER_PHASE;
    this.historyAt[channel] = p;
    h[p] = x;
    h[p + TAPS_PER_PHASE] = x;
    const newest = p + TAPS_PER_PHASE;
    for (const taps of this.phases) {
      let y = 0;
      for (let k = 0; k < TAPS_PER_PHASE; k++) y += (h[newest - k] ?? 0) * (taps[k] ?? 0);
      const a = Math.abs(y);
      if (a > this.peak) this.peak = a;
    }
    const a = Math.abs(x);
    if (a > this.peak) this.peak = a;
  }
}

/** Gain in dB that brings `measured` to `target` LUFS without true peak going above `ceiling` dBTP. */
export function normalizationGain(measured: LoudnessResult, target: number, ceiling: number): number {
  if (!Number.isFinite(measured.integrated)) return 0;
  const toTarget = target - measured.integrated;
  return Number.isFinite(measured.truePeak) ? Math.min(toTarget, ceiling - measured.truePeak) : toTarget;
}
