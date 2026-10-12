import { SPECTROGRAM } from "@spectra-peel/engine";
import { describe, expect, it } from "vitest";
import { COLORMAP } from "./colormap";
import { shadeColumn, type Tint } from "./paint";
import { oklchToRgb } from "./palette";

const STEP = 255 / SPECTROGRAM.floorDb;
const red = (px: number) => px & 0xff;
const green = (px: number) => (px >> 8) & 0xff;
const blue = (px: number) => (px >> 16) & 0xff;

function shade(base: number, levels: number[], tints: Tint[], focus = false): number {
  const out = new Uint32Array(1);
  shadeColumn(
    Uint8Array.of(base),
    levels.map((v) => Uint8Array.of(v)),
    tints,
    out,
    focus,
  );
  return out[0] ?? 0;
}

const tint = (gainDb: number, component = true, rgb: Tint["rgb"] = [255, 0, 0]): Tint => ({
  layer: "x",
  rgb,
  gainDb,
  component,
  coloured: true,
});

describe("oklchToRgb", () => {
  it("matches CSS for white, black and a saturated red", () => {
    expect(oklchToRgb(1, 0, 0)).toEqual([255, 255, 255]);
    expect(oklchToRgb(0, 0, 0)).toEqual([0, 0, 0]);
    expect(oklchToRgb(0.628, 0.2577, 29.23)).toEqual([255, 0, 0]);
  });
});

describe("shadeColumn", () => {
  it("draws the base alone when nothing is tinted", () => {
    expect(shade(180, [], [])).toBe(COLORMAP[180]);
  });

  it("fades a stem's tint as its gain goes down", () => {
    const reds = [0, -10, -20, -40].map((db) => red(shade(120, [200], [tint(db, false)])));
    for (let i = 1; i < reds.length; i++) expect(reds[i]).toBeLessThan(reds[i - 1] ?? 0);
  });

  it("hides a muted stem's tint", () => {
    expect(shade(180, [200], [tint(-Infinity, false)])).toBe(COLORMAP[180]);
  });

  it("darkens the base by the power a component loses", () => {
    // A stem 3 dB below the base holds half its power, so muting it takes the base 3 dB down.
    const stem = 180 - Math.round(10 * Math.log10(2) * STEP);
    const left = 1 - 10 ** ((stem - 180) / STEP / 10);
    const expected = Math.round(180 + 10 * Math.log10(left) * STEP);
    expect(shade(180, [stem], [tint(-Infinity)])).toBe(COLORMAP[expected]);
    expect(expected).toBeLessThan(175);
    expect(shade(180, [stem], [{ ...tint(-12), coloured: false }])).toBe(
      COLORMAP[Math.round(180 + 10 * Math.log10(1 - (1 - left) * (1 - 10 ** -1.2)) * STEP)],
    );
  });

  it("colours a pixel by whichever stem holds most of its energy", () => {
    const px = shade(200, [200, 120], [tint(0, true, [255, 0, 0]), tint(0, true, [0, 255, 0])]);
    expect(red(px)).toBeGreaterThan(green(px));
  });

  it("greys the base and heat-maps the focused layer by its share of the energy", () => {
    const grey = shade(200, [0], [tint(0, true)], true);
    expect(red(grey)).toBe(green(grey));
    expect(green(grey)).toBe(blue(grey));
    expect(shade(200, [200], [tint(0, true)], true)).toBe(COLORMAP[200]);
    const half = shade(200, [150], [tint(0, true)], true);
    expect(red(half)).not.toBe(blue(half));
    expect(red(half)).toBeLessThan(red(COLORMAP[150] ?? 0) + red(grey));
  });
});
