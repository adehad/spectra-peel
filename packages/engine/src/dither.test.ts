import { describe, expect, it } from "vitest";
import { Xorshift32, quantize16, toFloat } from "./dither";

describe("quantize16", () => {
  it("keeps every sample within one LSB of the scaled input", () => {
    const input = Float32Array.from({ length: 10_000 }, (_, i) => 0.8 * Math.sin(i / 7));
    const out = quantize16(input, new Xorshift32(1));
    for (let i = 0; i < input.length; i++) {
      expect(Math.abs((out[i] ?? 0) - (input[i] ?? 0) * 32767)).toBeLessThanOrEqual(1.5);
    }
  });

  it("clips out-of-range input instead of wrapping", () => {
    const out = quantize16(Float32Array.of(2, -2), new Xorshift32(1));
    expect(Array.from(out)).toEqual([32767, -32768]);
  });

  it("dithers silence to at most one LSB of noise with zero mean", () => {
    const out = quantize16(new Float32Array(100_000), new Xorshift32(7));
    let sum = 0;
    for (const v of out) {
      expect(Math.abs(v)).toBeLessThanOrEqual(1);
      sum += v;
    }
    expect(Math.abs(sum / out.length)).toBeLessThan(0.01);
  });

  it("is deterministic for a given seed", () => {
    const input = Float32Array.from({ length: 64 }, (_, i) => i / 100);
    expect(quantize16(input, new Xorshift32(3))).toEqual(quantize16(input, new Xorshift32(3)));
  });
});

describe("toFloat", () => {
  it("inverts the scale used by quantize16", () => {
    expect(Array.from(toFloat(Int16Array.of(32767, 0, -32767)))).toEqual([1, 0, -1]);
  });
});
