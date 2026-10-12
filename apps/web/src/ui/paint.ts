import { SPECTROGRAM } from "@spectra-peel/engine";
import { COLORMAP } from "./colormap";
import type { Rgb } from "./palette";

/** A layer whose energy is drawn in its own colour over the base spectrogram. */
export interface Tint {
  layer: string;
  rgb: Rgb;
  /** Gain the energy is drawn at, in dB; -Infinity hides it. */
  gainDb: number;
  /**
   * The layer is one part of the base, as a stem is of the mix it came from, so its gain also scales
   * the base: turning it down darkens the base where its energy was.
   */
  component: boolean;
  /** False for a component that should only scale the base, without a colour of its own. */
  coloured: boolean;
}

/** Tile bytes per dB. */
const STEP = 255 / SPECTROGRAM.floorDb;
/** Relative power of a tile byte, after gains have moved it by up to ±512, indexed by byte + 512. */
const POWER = Float64Array.from({ length: 1025 }, (_, i) => 10 ** ((i - 512) / STEP / 10));
const power = (level: number) => POWER[Math.max(0, Math.min(1024, Math.round(level) + 512))] ?? 0;

/** Alpha grows faster than linearly with level so faint energy stays out of the way. */
const alphaOf = (level: number) => (level <= 0 ? 0 : Math.min(0.85, (level / 255) ** 2 * 1.2));

/** Grey ramp for the base under a focused layer, kept below white so the layer's own colours stand out. */
const GREY = Uint32Array.from({ length: 256 }, (_, i) => {
  const v = Math.round(i * 0.7);
  return (255 << 24) | (v << 16) | (v << 8) | v;
});

const mix = (a: number, b: number, t: number) =>
  (255 << 24) |
  (Math.round(((a >> 16) & 0xff) * (1 - t) + ((b >> 16) & 0xff) * t) << 16) |
  (Math.round(((a >> 8) & 0xff) * (1 - t) + ((b >> 8) & 0xff) * t) << 8) |
  Math.round((a & 0xff) * (1 - t) + (b & 0xff) * t);

/**
 * Colours for one spectrogram column, one per band. `levels[i]` holds tint i's band bytes, or null
 * where its tiles have not loaded. Components rescale the base by how far their gains move the sum of
 * their powers, treating stems as uncorrelated, so the picture follows what the mix sounds like.
 * With `focus`, the base is grey and coloured layers are drawn in the heat colormap instead of their
 * own colour, stronger where they hold more of the band's energy.
 */
export function shadeColumn(
  base: Uint8Array,
  levels: readonly (Uint8Array | null)[],
  tints: readonly Tint[],
  out: Uint32Array,
  focus = false,
): void {
  const shift = tints.map((t) => t.gainDb * STEP);
  // Shifting the level alone barely fades a loud stem at -10 dB, so cuts also thin the colour.
  const fade = tints.map((t) => Math.min(1, 10 ** (t.gainDb / 40)));
  const linear = tints.map((t) => (t.gainDb === -Infinity ? 0 : 10 ** (t.gainDb / 10)));
  const ceiling = Math.max(1, ...tints.map((t, i) => (t.component ? (linear[i] ?? 0) : 0)));
  for (let b = 0; b < out.length; b++) {
    let level = base[b] ?? 0;
    let change = 0;
    let weight = 0;
    let r = 0;
    let g = 0;
    let bl = 0;
    let alpha = 0;
    let heat = 0;
    let share = 0;
    for (const [i, tint] of tints.entries()) {
      const v = levels[i]?.[b] ?? 0;
      if (v === 0) continue;
      if (tint.component) change += ((linear[i] ?? 0) - 1) * power(v);
      if (!tint.coloured) continue;
      const shown = v + (shift[i] ?? 0);
      if (focus) {
        const s = power(shown) * (fade[i] ?? 1);
        if (s > share) [share, heat] = [s, shown];
        continue;
      }
      const a = alphaOf(shown) * (fade[i] ?? 1);
      if (a === 0) continue;
      const w = power(shown);
      weight += w;
      r += w * tint.rgb[0];
      g += w * tint.rgb[1];
      bl += w * tint.rgb[2];
      if (a > alpha) alpha = a;
    }
    if (change !== 0) {
      const ratio = Math.min(ceiling, Math.max(1e-10, 1 + change / power(level)));
      level = Math.max(0, Math.min(255, level + 10 * Math.log10(ratio) * STEP));
    }
    if (focus) {
      const grey = GREY[Math.round(level)] ?? 0;
      const t = share > 0 ? Math.sqrt(Math.min(1, share / power(level))) : 0;
      out[b] = t > 0 ? mix(grey, COLORMAP[Math.max(0, Math.min(255, Math.round(heat)))] ?? 0, t) : grey;
      continue;
    }
    let px = COLORMAP[Math.round(level)] ?? 0;
    if (alpha > 0) {
      px =
        (255 << 24) |
        (Math.round(((px >> 16) & 0xff) * (1 - alpha) + (bl / weight) * alpha) << 16) |
        (Math.round(((px >> 8) & 0xff) * (1 - alpha) + (g / weight) * alpha) << 8) |
        Math.round((px & 0xff) * (1 - alpha) + (r / weight) * alpha);
    }
    out[b] = px;
  }
}
