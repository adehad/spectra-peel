import { peaksForRange, type PeaksPyramid } from "@spectra-peel/engine";
import { useEffect } from "react";
import type { Player } from "../audio/player";
import { useCanvas, useFrame } from "./useCanvas";
import { useView } from "./view";

interface Props {
  peaks: PeaksPyramid;
  player: Player;
}

function moveTo(clientX: number, el: HTMLElement) {
  const r = el.getBoundingClientRect();
  const v = useView.getState();
  v.setFollow(false);
  v.setView(((clientX - r.left) / r.width) * v.duration - v.span / 2);
}

/** Whole-file waveform with the zoomed view's extent; dragging moves the view. */
export function Overview({ peaks, player }: Props) {
  const [waveRef, waveSize] = useCanvas();
  const [overlayRef, overlaySize] = useCanvas();
  const duration = useView((v) => v.duration);

  useEffect(() => {
    const ctx = waveRef.current?.getContext("2d");
    const { width, height } = waveSize;
    if (!ctx || width === 0) return;
    const cols = peaksForRange(peaks, 0, peaks.totalSamples, width);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "oklch(0.78 0.1 230)";
    const mid = height / 2;
    for (let x = 0; x < width; x++) {
      const lo = cols[2 * x] ?? 0;
      const hi = cols[2 * x + 1] ?? 0;
      ctx.fillRect(x, mid - hi * mid, 1, Math.max(1, (hi - lo) * mid));
    }
  }, [peaks, waveSize, waveRef]);

  useFrame(() => {
    const ctx = overlayRef.current?.getContext("2d");
    const { width, height, dpr } = overlaySize;
    if (!ctx || width === 0 || duration === 0) return;
    const v = useView.getState();
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "rgba(255,255,255,0.12)";
    ctx.strokeStyle = "rgba(255,255,255,0.6)";
    ctx.lineWidth = dpr;
    const x0 = (v.start / duration) * width;
    const w = Math.max(2 * dpr, (v.span / duration) * width);
    ctx.fillRect(x0, 0, w, height);
    ctx.strokeRect(x0 + 0.5, 0.5, w - 1, height - 1);
    ctx.fillStyle = "oklch(0.78 0.15 65)";
    ctx.fillRect((player.position() / player.sampleRate / duration) * width, 0, dpr, height);
  });

  return (
    <div className="relative h-14 w-full shrink-0 touch-none select-none border-b border-edge bg-panel">
      <canvas ref={waveRef} className="absolute inset-0 h-full w-full" />
      <canvas
        ref={overlayRef}
        aria-label="Whole-file overview"
        className="absolute inset-0 h-full w-full cursor-grab"
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          moveTo(e.clientX, e.currentTarget);
        }}
        onPointerMove={(e) => {
          if (e.buttons) moveTo(e.clientX, e.currentTarget);
        }}
      />
    </div>
  );
}
