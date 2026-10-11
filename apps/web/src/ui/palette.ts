export type Rgb = readonly [number, number, number];

/** Linear light to an 8-bit sRGB channel. */
const encode = (v: number) => {
  const x = Math.min(1, Math.max(0, v));
  return Math.round(255 * (x <= 0.0031308 ? 12.92 * x : 1.055 * x ** (1 / 2.4) - 0.055));
};

/** OKLCH (lightness 0-1, chroma, hue in degrees) to 8-bit sRGB, clipped to the gamut. */
export function oklchToRgb(l: number, c: number, h: number): Rgb {
  const a = c * Math.cos((h * Math.PI) / 180);
  const b = c * Math.sin((h * Math.PI) / 180);
  const lc = (l + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const mc = (l - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const sc = (l - 0.0894841775 * a - 1.291485548 * b) ** 3;
  const linear = [
    4.0767416621 * lc - 3.3077115913 * mc + 0.2309699292 * sc,
    -1.2684380046 * lc + 2.6097574011 * mc - 0.3413193965 * sc,
    -0.0041960863 * lc - 0.7034186147 * mc + 1.707614701 * sc,
  ] as const;
  return [encode(linear[0]), encode(linear[1]), encode(linear[2])];
}

export interface LayerColor {
  css: string;
  rgb: Rgb;
}

const OKLCH: Record<string, readonly [number, number, number]> = {
  Source: [0.78, 0.1, 230],
  Denoised: [0.8, 0.14, 150],
  Removed: [0.75, 0.15, 40],
  Drums: [0.72, 0.17, 25],
  Bass: [0.7, 0.15, 290],
  Other: [0.82, 0.12, 95],
  Vocals: [0.8, 0.13, 180],
  Rest: [0.6, 0.03, 270],
};

/** The colour a layer is drawn in everywhere: its legend dot, its meter and its spectrogram tint. */
export function layerColor(label: string): LayerColor {
  const lch = OKLCH[label];
  if (!lch) return { css: "white", rgb: [255, 255, 255] };
  return { css: `oklch(${lch.join(" ")})`, rgb: oklchToRgb(...lch) };
}
