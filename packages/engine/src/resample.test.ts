import { describe, expect, it } from "vitest";
import { concatFloat32 } from "./buffers";
import { Resampler } from "./resample";

function sine(rate: number, hz: number, seconds: number): Float32Array {
  return Float32Array.from({ length: Math.round(rate * seconds) }, (_, i) =>
    Math.sin((2 * Math.PI * hz * i) / rate),
  );
}

function run(r: Resampler, input: Float32Array, block: number): Float32Array {
  const parts: Float32Array[] = [];
  for (let at = 0; at < input.length; at += block) parts.push(r.push(input.subarray(at, at + block)));
  parts.push(r.finish());
  return concatFloat32(parts);
}

function errorDb(actual: Float32Array, expected: Float32Array, from: number, to: number): number {
  let err = 0;
  let sig = 0;
  for (let i = from; i < to; i++) {
    err += ((actual[i] ?? 0) - (expected[i] ?? 0)) ** 2;
    sig += (expected[i] ?? 0) ** 2;
  }
  return 10 * Math.log10(err / sig);
}

describe("Resampler", () => {
  it.each([
    [44_100, 48_000],
    [48_000, 44_100],
    [32_000, 48_000],
    [96_000, 48_000],
  ])("turns a %i Hz sine into the same sine at %i Hz", (from, to) => {
    const out = run(new Resampler(from, to), sine(from, 1000, 0.5), 4096);
    const expected = sine(to, 1000, 0.5);
    expect(out.length).toBe(Math.ceil((sine(from, 1000, 0.5).length * to) / from));
    expect(errorDb(out, expected, 200, out.length - 200)).toBeLessThan(-80);
  });

  it("gives the same output whatever the block size", () => {
    const input = sine(44_100, 3000, 0.3).map((v, i) => v * Math.sin(i / 300));
    const a = run(new Resampler(44_100, 48_000), input, 100_000);
    const b = run(new Resampler(44_100, 48_000), input, 777);
    expect(b).toEqual(a);
  });

  it("round-trips 44.1 to 48 kHz and back with small error away from the edges", () => {
    const input = sine(44_100, 440, 0.5).map((v, i) => v * 0.5 + 0.3 * Math.sin(i / 3.1));
    const back = run(new Resampler(48_000, 44_100), run(new Resampler(44_100, 48_000), input, 5000), 5000);
    expect(back.length).toBe(input.length);
    expect(errorDb(back, input, 200, input.length - 200)).toBeLessThan(-70);
  });

  it("removes content above the new Nyquist when downsampling", () => {
    const out = run(new Resampler(48_000, 16_000), sine(48_000, 12_000, 0.3), 4096);
    const rms = Math.sqrt(out.subarray(200, -200).reduce((n, v) => n + v * v, 0) / (out.length - 400));
    expect(20 * Math.log10(rms)).toBeLessThan(-60);
  });

  it("passes audio through untouched at equal rates", () => {
    const input = sine(48_000, 100, 0.01);
    expect(run(new Resampler(48_000, 48_000), input, 64)).toEqual(input);
  });
});
