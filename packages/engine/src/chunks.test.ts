import { describe, expect, it } from "vitest";
import { OverlapAdder, planChunks, transitionWindow, triangleWeights } from "./chunks";

function noise(length: number, seed = 1): Float32Array {
  const out = new Float32Array(length);
  let s = seed;
  for (let i = 0; i < length; i++) {
    s = (s * 16807) % 2147483647;
    out[i] = s / 2147483647 - 0.5;
  }
  return out;
}

function maxAbsError(a: Float32Array, b: Float32Array, from = 0): number {
  let max = 0;
  for (let i = from; i < a.length; i++) max = Math.max(max, Math.abs((a[i] ?? 0) - (b[i] ?? 0)));
  return max;
}

describe("planChunks", () => {
  it("covers every frame once the overlaps are accounted for", () => {
    const chunks = planChunks(10_000, 1_000, 250);
    const hits = new Uint8Array(10_000);
    for (const c of chunks) for (let i = c.start; i < c.end; i++) hits[i] = 1;
    expect(hits.every((h) => h === 1)).toBe(true);
    expect(chunks.at(-1)?.end).toBe(10_000);
  });

  it("starts a chunk at every step below the total, like infer.py", () => {
    expect(planChunks(2_500, 1_000, 250)).toEqual([
      { start: 0, end: 1000 },
      { start: 750, end: 1750 },
      { start: 1500, end: 2500 },
      { start: 2250, end: 2500 },
    ]);
  });

  it("returns one short chunk when the input is shorter than a chunk", () => {
    expect(planChunks(300, 1_000, 250)).toEqual([{ start: 0, end: 300 }]);
  });

  it("returns nothing for empty input", () => {
    expect(planChunks(0, 1_000, 250)).toEqual([]);
  });

  it("rejects an overlap that leaves no step", () => {
    expect(() => planChunks(100, 10, 10)).toThrow(RangeError);
  });
});

describe("triangleWeights", () => {
  it("peaks at 1 in the middle and stays positive at the edges", () => {
    const w = triangleWeights(8);
    expect(Math.max(...w)).toBe(1);
    expect(Math.min(...w)).toBeGreaterThan(0);
    expect(Array.from(w)).toEqual(Array.from(w).toReversed());
  });
});

describe("transitionWindow", () => {
  it("fades 0 to 1 over a quarter at each end and is flat in the middle", () => {
    const w = transitionWindow(16);
    expect(Array.from(w.subarray(0, 4))).toEqual([0, 1 / 3, 2 / 3, 1].map(Math.fround));
    expect(Array.from(w.subarray(4, 12))).toEqual(Array.from({ length: 8 }, () => 1));
    expect(Array.from(w)).toEqual(Array.from(w).toReversed());
  });
});

describe("OverlapAdder", () => {
  it("reconstructs the input exactly when each chunk passes through unchanged", () => {
    const input = noise(48_000);
    const adder = new OverlapAdder(input.length);
    for (const c of planChunks(input.length, 4_096, 1_024)) {
      adder.add(c.start, input.subarray(c.start, c.end));
    }
    expect(maxAbsError(adder.finish(), input)).toBeLessThan(1e-6);
  });

  it("reconstructs the input with the transition window except the zero-weight first sample", () => {
    const input = noise(20_000, 3);
    const adder = new OverlapAdder(input.length);
    for (const c of planChunks(input.length, 4_096, 1_024)) {
      adder.add(
        c.start,
        input.subarray(c.start, c.end),
        transitionWindow(4_096).subarray(0, c.end - c.start),
      );
    }
    const out = adder.finish();
    expect(out[0]).toBe(0);
    expect(maxAbsError(out, input, 1)).toBeLessThan(1e-6);
  });

  it("rejects a chunk that runs past the end", () => {
    const adder = new OverlapAdder(10);
    expect(() => adder.add(8, new Float32Array(4))).toThrow(RangeError);
  });
});
