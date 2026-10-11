/** Magma-like ramp sampled at eight stops, expanded to a 256-entry RGBA table packed little-endian. */
const STOPS = [
  [0, 0, 4],
  [40, 11, 84],
  [101, 21, 110],
  [159, 42, 99],
  [212, 72, 66],
  [245, 125, 21],
  [250, 193, 39],
  [252, 253, 191],
] as const;

export const COLORMAP = (() => {
  const lut = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (STOPS.length - 1);
    const a = STOPS[Math.floor(t)] ?? STOPS[0];
    const b = STOPS[Math.min(STOPS.length - 1, Math.floor(t) + 1)] ?? a;
    const f = t - Math.floor(t);
    const [r, g, bl] = [0, 1, 2].map((k) => Math.round((a[k] ?? 0) + ((b[k] ?? 0) - (a[k] ?? 0)) * f));
    lut[i] = (255 << 24) | ((bl ?? 0) << 16) | ((g ?? 0) << 8) | (r ?? 0);
  }
  return lut;
})();
