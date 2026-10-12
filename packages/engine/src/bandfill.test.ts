import { describe, expect, it } from "vitest";
import { BAND_FILL, fillHighBand } from "./bandfill";
import { Xorshift32 } from "./dither";
import { Resampler } from "./resample";

const RATE = 48_000;
const N = RATE;

function noise(seed: number, gain: number): Float32Array {
  const rng = new Xorshift32(seed);
  return Float32Array.from({ length: N }, () => gain * (rng.next() - 0.5) * 2);
}

/** Through 16 kHz and back, as model output reaches a layer. */
function viaModelRate(x: Float32Array): Float32Array {
  const down = new Resampler(RATE, 16_000);
  const low = Float32Array.from([...down.push(x), ...down.finish()]);
  const up = new Resampler(16_000, RATE);
  const back = Float32Array.from([...up.push(low), ...up.finish()]);
  return back.subarray(0, N);
}

/** High-band energy: the fourth difference weights a bin by (2 sin(pi f / rate))^8. */
function highEnergy(x: Float32Array): number {
  let d = x;
  for (let r = 0; r < 4; r++) d = d.subarray(1).map((v, i) => v - (d[i] ?? 0));
  return d.reduce((s, v) => s + v * v, 0);
}

const sum = (layers: Float32Array[]) =>
  Float32Array.from({ length: N }, (_, i) => layers.reduce((s, l) => s + (l[i] ?? 0), 0));
const inner = (start: number, end: number) => (i: number) => i >= start && i < end;

describe("fillHighBand", () => {
  it("keeps the layers summing to the mixture", () => {
    const a = viaModelRate(noise(1, 0.5));
    const mix = noise(2, 0.6).map((v, i) => v + (a[i] ?? 0));
    const rest = mix.map((v, i) => v - (a[i] ?? 0));
    const filled = fillHighBand([a, rest], RATE, 16_000);
    const back = sum(filled);
    const keep = inner(BAND_FILL.window, N - BAND_FILL.window);
    let worst = 0;
    for (let i = 0; i < N; i++)
      if (keep(i)) worst = Math.max(worst, Math.abs((back[i] ?? 0) - (mix[i] ?? 0)));
    expect(worst).toBeLessThan(1e-4);
  });

  it("leaves a layer that is the whole mixture as it was", () => {
    const mix = noise(3, 0.5);
    const [whole] = fillHighBand([mix, new Float32Array(N)], RATE, 16_000);
    const keep = inner(BAND_FILL.window, N - BAND_FILL.window);
    let err = 0;
    let sig = 0;
    for (let i = 0; i < N; i++) {
      if (!keep(i)) continue;
      err += ((whole?.[i] ?? 0) - (mix[i] ?? 0)) ** 2;
      sig += (mix[i] ?? 0) ** 2;
    }
    expect(10 * Math.log10(sig / err)).toBeGreaterThan(50);
  });

  it("gives a model-rate layer the high band in proportion to its share below the cut", () => {
    const a = noise(4, 0.8);
    const b = noise(5, 0.2);
    const mix = a.map((v, i) => v + (b[i] ?? 0));
    const lowA = viaModelRate(a);
    const rest = mix.map((v, i) => v - (lowA[i] ?? 0));
    const [filled] = fillHighBand([lowA, rest], RATE, 16_000);
    const before = highEnergy(lowA);
    const after = highEnergy(filled ?? new Float32Array());
    const full = highEnergy(mix);
    expect(before / full).toBeLessThan(0.05);
    // a holds 0.64 / 0.68 of the power, so about 0.94 of the mixture's high band, at 0.89 of its power.
    expect(after / full).toBeGreaterThan(0.8);
    expect(after / full).toBeLessThan(0.95);
  });

  it("gives less of the high band to a layer that fades before the cut when the reference is narrower", () => {
    const down = new Resampler(RATE, 10_000);
    const up = new Resampler(10_000, RATE);
    const low = Float32Array.from([...down.push(noise(3, 0.5)), ...down.finish()]);
    const dull = Float32Array.from([...up.push(low), ...up.finish()]).subarray(0, N);
    const mix = noise(4, 0.1).map((v, i) => v + (dull[i] ?? 0));
    const rest = mix.map((v, i) => v - (dull[i] ?? 0));
    const [octave] = fillHighBand([dull, rest], RATE, 16_000);
    const [third] = fillHighBand([dull, rest], RATE, 16_000, 1 / 3);
    const extra = (f: Float32Array | undefined) => highEnergy(f ?? dull) - highEnergy(dull);
    expect(extra(third)).toBeLessThan(0.2 * extra(octave));
  });
});
