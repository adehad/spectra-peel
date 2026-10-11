import { describe, expect, it } from "vitest";
import { FLAC_BLOCK } from "./flac";
import { crossfadeWeight, layerExtent, paddedRange } from "./ranges";

describe("layerExtent", () => {
  it("widens by the fade and snaps outwards to FLAC frames", () => {
    expect(layerExtent({ start: 10_000, end: 20_000 }, 100_000, 600)).toEqual({
      start: 2 * FLAC_BLOCK,
      end: 6 * FLAC_BLOCK,
    });
  });

  it("stays inside the file", () => {
    expect(layerExtent({ start: 100, end: 99_000 }, 99_500, 600)).toEqual({ start: 0, end: 99_500 });
  });
});

describe("paddedRange", () => {
  it("adds pre- and post-roll within the file", () => {
    expect(paddedRange({ start: 50_000, end: 60_000 }, 70_000, 48_000, 24_000)).toEqual({
      start: 2000,
      end: 70_000,
    });
  });
});

describe("crossfadeWeight", () => {
  const sel = { start: 1000, end: 2000 };
  it("is 1 inside, 0 well outside, and ramps monotonically across each fade", () => {
    expect(crossfadeWeight(1500, sel, 100, 10_000)).toBe(1);
    expect(crossfadeWeight(800, sel, 100, 10_000)).toBe(0);
    expect(crossfadeWeight(2200, sel, 100, 10_000)).toBe(0);
    let prev = 0;
    for (let s = 900; s <= 1000; s++) {
      const w = crossfadeWeight(s, sel, 100, 10_000);
      expect(w).toBeGreaterThanOrEqual(prev);
      expect(w).toBeGreaterThan(0);
      prev = w;
    }
    prev = 1;
    for (let s = 1999; s <= 2100; s++) {
      const w = crossfadeWeight(s, sel, 100, 10_000);
      expect(w).toBeLessThanOrEqual(prev);
      prev = w;
    }
  });

  it("does not fade at the file edges", () => {
    expect(crossfadeWeight(0, { start: 0, end: 500 }, 100, 1000)).toBe(1);
    expect(crossfadeWeight(999, { start: 500, end: 1000 }, 100, 1000)).toBe(1);
  });
});
