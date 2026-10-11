import { describe, expect, it } from "vitest";
import { PEAKS_BASE, PeaksBuilder, deserializePeaks, peaksForRange, serializePeaks } from "./peaks";

function ramp(length: number): Float32Array {
  return Float32Array.from({ length }, (_, i) => (i % 1000) / 1000 - 0.5);
}

describe("PeaksBuilder", () => {
  it("takes min and max across channels per base window, across push boundaries", () => {
    const b = new PeaksBuilder();
    const left = new Float32Array(PEAKS_BASE * 2);
    const right = new Float32Array(PEAKS_BASE * 2);
    left[10] = 0.9;
    right[PEAKS_BASE + 5] = -0.7;
    b.push([left.subarray(0, 100), right.subarray(0, 100)]);
    b.push([left.subarray(100), right.subarray(100)]);
    const p = b.finish();
    expect(p.totalSamples).toBe(PEAKS_BASE * 2);
    expect(Array.from(p.levels[0] ?? [])).toEqual([0, Math.fround(0.9), Math.fround(-0.7), 0]);
  });

  it("builds levels that halve until one pair is left", () => {
    const b = new PeaksBuilder();
    b.push([ramp(PEAKS_BASE * 10 + 3)]);
    const p = b.finish();
    expect(p.levels.map((l) => l.length / 2)).toEqual([11, 6, 3, 2, 1]);
    const top = p.levels.at(-1) ?? [];
    expect(top[0]).toBeCloseTo(-0.5);
    expect(top[1]).toBeCloseTo(0.499);
  });
});

describe("peaksForRange", () => {
  it("matches a brute-force min/max per column", () => {
    const signal = Float32Array.from({ length: 200_000 }, (_, i) => Math.sin(i / 50) * (i / 200_000));
    const b = new PeaksBuilder();
    b.push([signal]);
    const p = b.finish();
    const out = peaksForRange(p, 0, signal.length, 100);
    const span = signal.length / 100;
    for (let c = 0; c < 100; c++) {
      const slice = signal.subarray(c * span, (c + 1) * span);
      expect(out[2 * c + 1]).toBeGreaterThanOrEqual(Math.max(...slice) - 1e-6);
      const around = signal.subarray(Math.max(0, c * span - 2048), (c + 1) * span + 2048);
      expect(out[2 * c + 1]).toBeLessThanOrEqual(Math.max(...around) + 1e-6);
    }
  });
});

describe("serializePeaks", () => {
  it("round-trips", () => {
    const b = new PeaksBuilder();
    b.push([ramp(50_000)]);
    const p = b.finish();
    expect(deserializePeaks(serializePeaks(p))).toEqual(p);
  });
});
