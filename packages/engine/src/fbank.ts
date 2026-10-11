import { FFT } from "./fft";

const SR = 16_000;
const LENGTH = 400;
const SHIFT = 160;
const SIZE = 512;
const BINS = 80;
const EPSILON = 1.1920928955078125e-7;

const mel = (f: number) => 1127 * Math.log(1 + f / 700);

/** Kaldi's triangular mel filters from 20 Hz to Nyquist, over the first SIZE / 2 FFT bins. */
function melBanks(): Float64Array[] {
  const lo = mel(20);
  const delta = (mel(SR / 2) - lo) / (BINS + 1);
  return Array.from({ length: BINS }, (_, b) => {
    const left = lo + b * delta;
    const centre = left + delta;
    const right = centre + delta;
    const bank = new Float64Array(SIZE / 2);
    for (let i = 0; i < SIZE / 2; i++) {
      const m = mel((i * SR) / SIZE);
      bank[i] = Math.max(0, Math.min((m - left) / (centre - left), (right - m) / (right - centre)));
    }
    return bank;
  });
}

/**
 * Log mel filterbank features as WeSpeaker computes them with torchaudio's Kaldi fbank: 16 kHz, 25 ms
 * Hamming frames every 10 ms, DC removed, 0.97 pre-emphasis, 80 bins, no dither, then the mean over
 * frames subtracted. Returns [frames, 80] row-major.
 */
export class Fbank {
  static readonly bins = BINS;
  private readonly fft = new FFT(SIZE);
  private readonly banks = melBanks();
  private readonly window = Float64Array.from(
    { length: LENGTH },
    (_, n) => 0.54 - 0.46 * Math.cos((2 * Math.PI * n) / (LENGTH - 1)),
  );
  private readonly re = new Float64Array(SIZE);
  private readonly im = new Float64Array(SIZE);

  static frames(samples: number): number {
    return samples < LENGTH ? 0 : 1 + Math.floor((samples - LENGTH) / SHIFT);
  }

  compute(wav: Float32Array): Float32Array {
    const frames = Fbank.frames(wav.length);
    const out = new Float32Array(frames * BINS);
    const mean = new Float64Array(BINS);
    const { re, im, fft, window, banks } = this;
    for (let f = 0; f < frames; f++) {
      const at = f * SHIFT;
      let dc = 0;
      for (let n = 0; n < LENGTH; n++) dc += wav[at + n] ?? 0;
      dc /= LENGTH;
      let previous = ((wav[at] ?? 0) - dc) * 32768;
      for (let n = 0; n < LENGTH; n++) {
        const x = ((wav[at + n] ?? 0) - dc) * 32768;
        re[n] = (x - 0.97 * previous) * (window[n] ?? 0);
        previous = x;
      }
      re.fill(0, LENGTH);
      im.fill(0);
      fft.transform(re, im);
      for (let i = 0; i < SIZE / 2; i++) re[i] = (re[i] ?? 0) ** 2 + (im[i] ?? 0) ** 2;
      for (const [b, bank] of banks.entries()) {
        let e = 0;
        for (let i = 0; i < SIZE / 2; i++) e += (bank[i] ?? 0) * (re[i] ?? 0);
        const v = Math.log(Math.max(e, EPSILON));
        out[f * BINS + b] = v;
        mean[b] = (mean[b] ?? 0) + v;
      }
    }
    for (let b = 0; b < BINS; b++) {
      const m = (mean[b] ?? 0) / Math.max(frames, 1);
      for (let f = 0; f < frames; f++) out[f * BINS + b] = (out[f * BINS + b] ?? 0) - m;
    }
    return out;
  }
}
