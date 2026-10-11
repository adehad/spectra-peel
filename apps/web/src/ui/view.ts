import { create } from "zustand";

/** Longest span the zoomed spectrogram shows; the overview covers the whole file. */
export const MAX_SPAN = 300;
export const MIN_SPAN = 0.5;

interface ViewState {
  duration: number;
  /** Left edge of the zoomed view, seconds. */
  start: number;
  span: number;
  follow: boolean;
  reset: (duration: number) => void;
  setView: (start: number, span?: number) => void;
  zoom: (factor: number, anchor: number) => void;
  setFollow: (follow: boolean) => void;
}

function clamp(duration: number, start: number, span: number) {
  const s = Math.min(Math.max(span, MIN_SPAN), MAX_SPAN, Math.max(duration, MIN_SPAN));
  return { span: s, start: Math.min(Math.max(0, start), Math.max(0, duration - s)) };
}

export const useView = create<ViewState>((set) => ({
  duration: 0,
  start: 0,
  span: 30,
  follow: true,
  reset: (duration) => set({ duration, follow: true, ...clamp(duration, 0, 30) }),
  setView: (start, span) => set((v) => clamp(v.duration, start, span ?? v.span)),
  zoom: (factor, anchor) =>
    set((v) => {
      const span = v.span * factor;
      const fraction = (anchor - v.start) / v.span;
      return clamp(v.duration, anchor - fraction * span, span);
    }),
  setFollow: (follow) => set({ follow }),
}));
