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
  /** Seconds; null means the whole file. */
  selection: { start: number; end: number } | null;
  reset: (duration: number) => void;
  setSelection: (selection: { start: number; end: number } | null) => void;
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
  selection: null,
  reset: (duration) => set({ duration, follow: true, selection: null, ...clamp(duration, 0, 30) }),
  setSelection: (selection) =>
    set((v) => {
      if (!selection) return { selection: null };
      const start = Math.max(0, Math.min(selection.start, selection.end));
      const end = Math.min(v.duration, Math.max(selection.start, selection.end));
      return { selection: end - start > 0.05 ? { start, end } : null };
    }),
  setView: (start, span) => set((v) => clamp(v.duration, start, span ?? v.span)),
  zoom: (factor, anchor) =>
    set((v) => {
      const span = v.span * factor;
      const fraction = (anchor - v.start) / v.span;
      return clamp(v.duration, anchor - fraction * span, span);
    }),
  setFollow: (follow) => set({ follow }),
}));
