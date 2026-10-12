import { FFT } from "./fft";

export const BAND_FILL = Object.freeze({
  /** Above this fraction of the model's Nyquist the resampler has rolled the model output off. */
  cut: 0.94,
  /** STFT length; a fill changes samples up to this far either side of a changed layer. */
  window: 2048,
  hopSeconds: 0.01,
  /** Least share any layer gets, so silence in every layer still splits the band. */
  floor: 1e-3,
});

/**
 * Fills the band a lower-rate model could not produce. Below 0.94 x `modelRate` / 2 every layer is
 * left as it is. Above it, each layer becomes a share of the mixture (the sum of all layers), frame
 * by frame: the layer's share of power in the `octaves` below the model's Nyquist, weighted by the
 * mixture's power there. Shares sum to one, so the layers still sum to the mixture. `layers` must
 * include the residual, or the high band of anything the model left out goes to the separated
 * layers. A narrower reference band follows sources that fall off faster than the background: over
 * pink noise, a third of an octave kept AudioSep's speech layer from taking the noise's high band.
 */
export function fillHighBand(
  layers: readonly Float32Array[],
  rate: number,
  modelRate: number,
  octaves = 1,
): Float32Array[] {
  const { window: W, hopSeconds, floor, cut } = BAND_FILL;
  const n = layers[0]?.length ?? 0;
  const k = layers.length;
  const hop = Math.round(hopSeconds * rate);
  const half = W / 2;
  const bins = half + 1;
  const binHz = rate / W;
  const cutBin = Math.min(bins, Math.ceil((cut * modelRate) / 2 / binHz));
  const topBin = Math.ceil(modelRate / 2 / 2 ** octaves / binHz);
  const win = Float64Array.from({ length: W }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / W));
  const fft = new FFT(W);
  const frames = Math.floor(n / hop) + 1;
  const out = layers.map(() => new Float64Array(n + W));
  const norm = new Float64Array(n + W);
  const re = layers.map(() => new Float64Array(W));
  const im = layers.map(() => new Float64Array(W));
  const xr = new Float64Array(bins);
  const xi = new Float64Array(bins);

  for (let t = 0; t < frames; t++) {
    const at = t * hop - half;
    for (let l = 0; l < k; l++) {
      const src = layers[l] ?? new Float32Array();
      const r = re[l] ?? new Float64Array(W);
      const m = im[l] ?? new Float64Array(W);
      for (let i = 0; i < W; i++) {
        const s = at + i;
        r[i] = s >= 0 && s < n ? (src[s] ?? 0) * (win[i] ?? 0) : 0;
      }
      m.fill(0);
      fft.transform(r, m);
    }
    xr.fill(0);
    xi.fill(0);
    for (let l = 0; l < k; l++)
      for (let f = 0; f < bins; f++) {
        xr[f] = (xr[f] ?? 0) + (re[l]?.[f] ?? 0);
        xi[f] = (xi[f] ?? 0) + (im[l]?.[f] ?? 0);
      }
    const share = new Float64Array(k);
    let weight = 0;
    for (let f = topBin; f < cutBin; f++) {
      const px = (xr[f] ?? 0) ** 2 + (xi[f] ?? 0) ** 2;
      let total = 0;
      for (let l = 0; l < k; l++) total += (re[l]?.[f] ?? 0) ** 2 + (im[l]?.[f] ?? 0) ** 2;
      for (let l = 0; l < k; l++) {
        const p = (re[l]?.[f] ?? 0) ** 2 + (im[l]?.[f] ?? 0) ** 2;
        share[l] = (share[l] ?? 0) + (floor + (1 - k * floor) * (p / (total + 1e-20))) * px;
      }
      weight += px;
    }
    for (let l = 0; l < k; l++) {
      const part = weight > 0 ? (share[l] ?? 0) / weight : 1 / k;
      const r = re[l] ?? new Float64Array(W);
      const m = im[l] ?? new Float64Array(W);
      for (let f = cutBin; f < bins; f++) {
        r[f] = part * (xr[f] ?? 0);
        m[f] = part * (xi[f] ?? 0);
      }
      // Inverse by conjugation; the upper half mirrors the lower so the result is real.
      for (let f = 1; f < half; f++) {
        r[W - f] = r[f] ?? 0;
        m[W - f] = -(m[f] ?? 0);
      }
      m[0] = 0;
      m[half] = 0;
      for (let i = 0; i < W; i++) m[i] = -(m[i] ?? 0);
      fft.transform(r, m);
      const o = out[l] ?? new Float64Array();
      for (let i = 0; i < W; i++) {
        const s = at + i;
        if (s < 0 || s >= n) continue;
        o[s] = (o[s] ?? 0) + ((r[i] ?? 0) / W) * (win[i] ?? 0);
      }
    }
    for (let i = 0; i < W; i++) {
      const s = at + i;
      if (s >= 0 && s < n) norm[s] = (norm[s] ?? 0) + (win[i] ?? 0) ** 2;
    }
  }
  return out.map((o) =>
    Float32Array.from({ length: n }, (_, i) => (o[i] ?? 0) / Math.max(norm[i] ?? 0, 1e-8)),
  );
}
