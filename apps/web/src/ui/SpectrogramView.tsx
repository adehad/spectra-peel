import { SPECTROGRAM } from "@spectra-peel/engine";
import { useRef } from "react";
import type { Player } from "../audio/player";
import { formatHz, formatTime } from "./format";
import { shadeColumn, type Tint } from "./paint";
import type { ProjectTiles } from "./tiles";
import { useCanvas, useFrame } from "./useCanvas";
import { useView } from "./view";

interface Props {
  tiles: ProjectTiles;
  /** Layer whose spectrogram fills the view. */
  base: string;
  /** Layers drawn in their own colours over the base, such as what a denoise removed or each stem. */
  tints: readonly Tint[];
  player: Player;
  /** Single-finger drags select a time range instead of panning, for touch screens. */
  selectMode: boolean;
}

const FREQ_TICKS = [50, 100, 200, 500, 1000, 2000, 5000, 10_000, 20_000];

export function SpectrogramView({ tiles, base, tints, player, selectMode }: Props) {
  const [ref, size] = useCanvas();
  const [overlayRef, overlaySize] = useCanvas();
  const { start, span, duration, setView, zoom, setSelection } = useView();
  const rate = player.sampleRate;
  const nyquist = rate / 2;
  const drawn = useRef("");

  useFrame(() => {
    const ctx = ref.current?.getContext("2d");
    const { width, height } = size;
    if (!ctx || width === 0 || height === 0) return;
    const view = useView.getState();
    const key = `${view.start} ${view.span} ${width} ${height} ${tiles.version} ${base} ${JSON.stringify(tints)}`;
    if (key === drawn.current) return;
    drawn.current = key;
    const image = ctx.createImageData(width, height);
    const pixels = new Uint32Array(image.data.buffer);
    const perSecond = rate / SPECTROGRAM.hop;
    const c0 = view.start * perSecond;
    const perPixel = (view.span * perSecond) / width;
    const rowBand = Uint16Array.from({ length: height }, (_, y) =>
      Math.min(SPECTROGRAM.bands - 1, Math.floor((1 - (y + 0.5) / height) * SPECTROGRAM.bands)),
    );
    const max = new Uint8Array(SPECTROGRAM.bands);
    const levels = tints.map(() => new Uint8Array(SPECTROGRAM.bands));
    const shaded = new Uint32Array(SPECTROGRAM.bands);
    const gather = (
      column: (c: number) => Uint8Array | undefined,
      out: Uint8Array,
      from: number,
      to: number,
    ) => {
      out.fill(0);
      for (let c = from; c < to; c++) {
        const col = column(c);
        if (!col) continue;
        for (let b = 0; b < SPECTROGRAM.bands; b++) if ((col[b] ?? 0) > (out[b] ?? 0)) out[b] = col[b] ?? 0;
      }
    };
    for (let x = 0; x < width; x++) {
      const from = Math.floor(c0 + x * perPixel);
      const to = Math.max(from + 1, Math.floor(c0 + (x + 1) * perPixel));
      gather((c) => tiles.column(base, c), max, from, to);
      for (const [i, t] of tints.entries()) {
        const out = levels[i];
        if (out) gather((c) => tiles.ownColumn(t.layer, c), out, from, to);
      }
      shadeColumn(max, levels, tints, shaded);
      for (let y = 0; y < height; y++) pixels[y * width + x] = shaded[rowBand[y] ?? 0] ?? 0;
    }
    ctx.putImageData(image, 0, 0);
  });

  useFrame(() => {
    const ctx = overlayRef.current?.getContext("2d");
    const { width, height, dpr } = overlaySize;
    if (!ctx || width === 0) return;
    const t = player.position() / rate;
    const v = useView.getState();
    if (v.follow && player.isPlaying && (t < v.start || t > v.start + v.span * 0.9))
      v.setView(t - v.span * 0.1);
    const x = ((t - v.start) / v.span) * width;
    ctx.clearRect(0, 0, width, height);
    ctx.font = `${11 * dpr}px ui-monospace, monospace`;
    ctx.fillStyle = "rgba(255,255,255,0.7)";
    const minHz = SPECTROGRAM.minHz;
    for (const hz of FREQ_TICKS) {
      if (hz >= nyquist) continue;
      const y = (1 - Math.log(hz / minHz) / Math.log(nyquist / minHz)) * height;
      ctx.fillRect(0, y, 6 * dpr, 1);
      ctx.fillText(formatHz(hz), 8 * dpr, y + 4 * dpr);
    }
    const step = niceStep(v.span / 8);
    for (let s = Math.ceil(v.start / step) * step; s < v.start + v.span; s += step) {
      const tx = ((s - v.start) / v.span) * width;
      ctx.fillRect(tx, height - 6 * dpr, 1, 6 * dpr);
      ctx.fillText(formatTime(s, step < 1), tx + 3 * dpr, height - 8 * dpr);
    }
    if (v.selection) {
      const sx = ((v.selection.start - v.start) / v.span) * width;
      const ex = ((v.selection.end - v.start) / v.span) * width;
      ctx.fillStyle = "rgba(255, 190, 90, 0.16)";
      ctx.fillRect(sx, 0, ex - sx, height);
      ctx.fillStyle = "rgba(255, 190, 90, 0.9)";
      ctx.fillRect(sx, 0, dpr, height);
      ctx.fillRect(ex - dpr, 0, dpr, height);
    }
    ctx.fillStyle = "oklch(0.78 0.15 65)";
    ctx.fillRect(Math.round(x), 0, Math.max(1, dpr), height);
  });

  const gesture = useRef<{
    x: number;
    start: number;
    moved: boolean;
    pointers: Map<number, number>;
    pinch?: number;
    selectFrom?: number;
  }>({
    x: 0,
    start: 0,
    moved: false,
    pointers: new Map(),
  });
  const timeAt = (clientX: number, el: HTMLElement) => {
    const r = el.getBoundingClientRect();
    return start + ((clientX - r.left) / r.width) * span;
  };

  return (
    <div className="relative h-full min-h-0 w-full touch-none select-none" data-testid="spectrogram">
      <canvas ref={ref} className="absolute inset-0 h-full w-full" />
      <canvas
        ref={overlayRef}
        className="absolute inset-0 h-full w-full cursor-crosshair"
        onWheel={(e) => {
          useView.getState().setFollow(false);
          if (e.ctrlKey || Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
            zoom(Math.exp(e.deltaY * 0.002), timeAt(e.clientX, e.currentTarget));
          } else setView(start + (e.deltaX / e.currentTarget.clientWidth) * span);
        }}
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          const g = gesture.current;
          g.pointers.set(e.pointerId, e.clientX);
          if (g.pointers.size === 1) {
            Object.assign(g, { x: e.clientX, start, moved: false });
            if (e.shiftKey || selectMode) g.selectFrom = timeAt(e.clientX, e.currentTarget);
            else delete g.selectFrom;
          } else {
            g.pinch = pinchWidth(g.pointers);
            delete g.selectFrom;
          }
        }}
        onPointerMove={(e) => {
          const g = gesture.current;
          if (!g.pointers.has(e.pointerId)) return;
          g.pointers.set(e.pointerId, e.clientX);
          const width = e.currentTarget.clientWidth;
          if (g.pointers.size >= 2 && g.pinch) {
            const now = pinchWidth(g.pointers);
            const center = [...g.pointers.values()].reduce((a, b) => a + b, 0) / g.pointers.size;
            zoom(g.pinch / Math.max(now, 1), timeAt(center, e.currentTarget));
            g.pinch = now;
            g.moved = true;
          } else if (g.selectFrom !== undefined) {
            if (Math.abs(e.clientX - g.x) > 4) g.moved = true;
            if (g.moved) setSelection({ start: g.selectFrom, end: timeAt(e.clientX, e.currentTarget) });
          } else if (Math.abs(e.clientX - g.x) > 4 || g.moved) {
            g.moved = true;
            useView.getState().setFollow(false);
            setView(g.start - ((e.clientX - g.x) / width) * span);
          }
        }}
        onPointerUp={(e) => {
          const g = gesture.current;
          g.pointers.delete(e.pointerId);
          if (g.pointers.size === 0 && !g.moved) {
            if (g.selectFrom === undefined) setSelection(null);
            player.seek(timeAt(e.clientX, e.currentTarget) * rate);
            useView.getState().setFollow(true);
          }
          if (g.pointers.size < 2) delete g.pinch;
        }}
        onPointerCancel={(e) => gesture.current.pointers.delete(e.pointerId)}
      />
      {duration === 0 ? null : (
        <span className="pointer-events-none absolute top-1 right-2 font-mono text-xs text-white/70">
          {formatTime(start, span < 10)} – {formatTime(start + span, span < 10)}
        </span>
      )}
    </div>
  );
}

function pinchWidth(pointers: Map<number, number>): number {
  const xs = [...pointers.values()];
  return Math.max(...xs) - Math.min(...xs);
}

function niceStep(raw: number): number {
  const steps = [0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
  return steps.find((s) => s >= raw) ?? 600;
}
