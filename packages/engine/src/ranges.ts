import { FLAC_BLOCK } from "./flac";

export interface SampleRange {
  start: number;
  end: number;
}

/** The selection widened by the crossfade on each side, then out to whole FLAC frames, within the file. */
export function layerExtent(selection: SampleRange, total: number, fade: number): SampleRange {
  const start = Math.max(0, Math.floor((selection.start - fade) / FLAC_BLOCK) * FLAC_BLOCK);
  const end = Math.min(total, Math.ceil((selection.end + fade) / FLAC_BLOCK) * FLAC_BLOCK);
  return { start, end };
}

/**
 * The window a model runs over: the extent plus pre-roll, so recurrent state has settled by the time the
 * selection starts, and post-roll for lookahead.
 */
export function paddedRange(
  extent: SampleRange,
  total: number,
  preRoll: number,
  postRoll: number,
): SampleRange {
  return { start: Math.max(0, extent.start - preRoll), end: Math.min(total, extent.end + postRoll) };
}

/**
 * Weight of the processed signal at `sample`: 1 inside the selection, 0 beyond the fades, a raised-cosine
 * ramp across `fade` samples outside each edge. A selection edge at the file boundary gets no fade.
 */
export function crossfadeWeight(sample: number, selection: SampleRange, fade: number, total: number): number {
  if (sample >= selection.start && sample < selection.end) return 1;
  let distance: number;
  if (sample < selection.start) {
    if (selection.start === 0) return 0;
    distance = selection.start - sample;
  } else {
    if (selection.end >= total) return 0;
    distance = sample - selection.end + 1;
  }
  if (fade <= 0 || distance > fade) return 0;
  return 0.5 + 0.5 * Math.cos((Math.PI * distance) / (fade + 1));
}
