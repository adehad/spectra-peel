import { SPECTROGRAM } from "@spectra-peel/engine";
import { useMemo, useRef } from "react";
import type { Player } from "../audio/player";
import { COLORMAP } from "./colormap";
import { formatHz, formatTime } from "./format";
import { TileCache } from "./tiles";
import { useCanvas, useFrame } from "./useCanvas";
import { useView } from "./view";

interface Props {
  dir: FileSystemDirectoryHandle;
  layer: string;
  player: Player;
}

const FREQ_TICKS = [50, 100, 200, 500, 1000, 2000, 5000, 10_000, 20_000];

export function SpectrogramView({ dir, layer, player }: Props) {
  const [ref, size] = useCanvas();
  const [overlayRef, overlaySize] = useCanvas();
  const tiles = useMemo(() => new TileCache(dir, layer), [dir, layer]);
  const { start, span, duration, setView, zoom } = useView();
  const rate = player.sampleRate;
  const nyquist = rate / 2;
  const drawn = useRef("");

  useFrame(() => {
    const ctx = ref.current?.getContext("2d");
    const { width, height } = size;
    if (!ctx || width === 0 || height === 0) return;
    const view = useView.getState();
    const key = `${view.start} ${view.span} ${width} ${height} ${tiles.version}`;
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
    for (let x = 0; x < width; x++) {
      const from = Math.floor(c0 + x * perPixel);
      const to = Math.max(from + 1, Math.floor(c0 + (x + 1) * perPixel));
      max.fill(0);
      for (let c = from; c < to; c++) {
        const col = tiles.column(c);
        if (!col) continue;
        for (let b = 0; b < SPECTROGRAM.bands; b++) if ((col[b] ?? 0) > (max[b] ?? 0)) max[b] = col[b] ?? 0;
      }
      for (let y = 0; y < height; y++) pixels[y * width + x] = COLORMAP[max[rowBand[y] ?? 0] ?? 0] ?? 0;
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
    ctx.fillStyle = "oklch(0.78 0.15 65)";
    ctx.fillRect(Math.round(x), 0, Math.max(1, dpr), height);
  });

  const gesture = useRef<{
    x: number;
    start: number;
    moved: boolean;
    pointers: Map<number, number>;
    pinch?: number;
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
          if (g.pointers.size === 1) Object.assign(g, { x: e.clientX, start, moved: false });
          else g.pinch = pinchWidth(g.pointers);
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
