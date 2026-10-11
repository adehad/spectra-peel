import { describe, expect, it } from "vitest";
import { FFT, MixedFFT } from "./fft";
import { SPECTROGRAM, SpectrogramBuilder, bandEdges, type SpectrogramTile } from "./spectrogram";

describe("FFT", () => {
  it("matches a direct DFT", () => {
    const n = 64;
    const x = Float64Array.from({ length: n }, (_, i) => Math.sin(i * 0.3) + (i % 5) * 0.1);
    const re = Float64Array.from(x);
    const im = new Float64Array(n);
    new FFT(n).transform(re, im);
    for (let k = 0; k < n; k++) {
      let dr = 0;
      let di = 0;
      for (let t = 0; t < n; t++) {
        dr += (x[t] ?? 0) * Math.cos((-2 * Math.PI * k * t) / n);
        di += (x[t] ?? 0) * Math.sin((-2 * Math.PI * k * t) / n);
      }
      expect(re[k]).toBeCloseTo(dr, 9);
      expect(im[k]).toBeCloseTo(di, 9);
    }
  });

  it("rejects sizes that are not powers of two", () => {
    expect(() => new FFT(1000)).toThrow(RangeError);
  });
});

describe("MixedFFT", () => {
  it.each([960, 12, 30, 45])("matches a direct DFT at size %i", (n) => {
    const re = Float64Array.from({ length: n }, (_, i) => Math.sin(i * 0.37) + (i % 7) * 0.1);
    const im = Float64Array.from({ length: n }, (_, i) => Math.cos(i * 0.11) * 0.5);
    const outRe = new Float64Array(n);
    const outIm = new Float64Array(n);
    new MixedFFT(n).transform(re, im, outRe, outIm);
    let worst = 0;
    for (let k = 0; k < n; k++) {
      let dr = 0;
      let di = 0;
      for (let t = 0; t < n; t++) {
        const a = (-2 * Math.PI * k * t) / n;
        dr += (re[t] ?? 0) * Math.cos(a) - (im[t] ?? 0) * Math.sin(a);
        di += (re[t] ?? 0) * Math.sin(a) + (im[t] ?? 0) * Math.cos(a);
      }
      worst = Math.max(worst, Math.abs((outRe[k] ?? 0) - dr), Math.abs((outIm[k] ?? 0) - di));
    }
    expect(worst).toBeLessThan(1e-9);
  });

  it("rejects sizes with a prime factor above 5", () => {
    expect(() => new MixedFFT(14)).toThrow(RangeError);
  });
});

function run(signal: Float32Array, rate = 44_100, pushSize = 3000): SpectrogramTile[] {
  const b = new SpectrogramBuilder(rate);
  const tiles: SpectrogramTile[] = [];
  for (let at = 0; at < signal.length; at += pushSize)
    tiles.push(...b.push(signal.subarray(at, at + pushSize)));
  tiles.push(...b.finish());
  return tiles;
}

describe("SpectrogramBuilder", () => {
  it("emits one column per hop, split into tiles", () => {
    const total = SPECTROGRAM.hop * 2500 + 7;
    const tiles = run(new Float32Array(total));
    expect(tiles.map((t) => t.columns)).toEqual([1024, 1024, 453]);
    expect(tiles.map((t) => t.index)).toEqual([0, 1, 2]);
    expect(tiles.every((t) => t.data.every((v) => v === 0))).toBe(true);
  });

  it("puts a full-scale 1 kHz sine at about 0 dB in the 1 kHz band and far lower elsewhere", () => {
    const rate = 44_100;
    const sine = Float32Array.from({ length: rate }, (_, i) => Math.sin((2 * Math.PI * 1000 * i) / rate));
    const tile = run(sine, rate)[0];
    if (!tile) throw new Error("no tile");
    const column = tile.data.subarray(40 * SPECTROGRAM.bands, 41 * SPECTROGRAM.bands);
    const edges = bandEdges(rate, SPECTROGRAM);
    const bin = Math.round(1000 / (rate / SPECTROGRAM.fftSize));
    const band = Array.from({ length: SPECTROGRAM.bands }, (_, b) => b).find(
      (b) => (edges[2 * b] ?? 0) <= bin && bin < (edges[2 * b + 1] ?? 0),
    );
    expect(column[band ?? -1]).toBeGreaterThanOrEqual(250);
    expect(column[SPECTROGRAM.bands - 1]).toBeLessThan(60);
  });
});

describe("bandEdges", () => {
  it("covers increasing, non-empty bin ranges up to Nyquist", () => {
    const e = bandEdges(48_000, SPECTROGRAM);
    for (let b = 0; b < SPECTROGRAM.bands; b++) {
      expect(e[2 * b + 1]).toBeGreaterThan(e[2 * b] ?? 0);
      expect(e[2 * b]).toBeGreaterThanOrEqual(e[2 * b - 2] ?? 0);
    }
    expect(e[2 * SPECTROGRAM.bands - 1]).toBe(SPECTROGRAM.fftSize / 2);
  });
});
