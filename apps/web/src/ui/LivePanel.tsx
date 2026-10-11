import { useRef, useState } from "react";
import { COLORMAP } from "./colormap";
import { useCanvas, useFrame } from "./useCanvas";

export interface LiveLayer {
  id: string;
  color: string;
  analyser: AnalyserNode;
}

interface Props {
  layers: LiveLayer[];
  sampleRate: number;
}

const MIN_HZ = 20;
const MIN_DB = -100;
const MAX_DB = -10;

/** What is playing right now: one spectrum curve per layer, or a scrolling waterfall of the first layer. */
export function LivePanel({ layers, sampleRate }: Props) {
  const [mode, setMode] = useState<"curves" | "waterfall">("curves");
  const [ref, size] = useCanvas();
  const bins = useRef(new Float32Array(0));
  const nyquist = sampleRate / 2;

  useFrame(() => {
    const ctx = ref.current?.getContext("2d");
    const { width, height, dpr } = size;
    if (!ctx || width === 0) return;
    const xForBin = (k: number, n: number) =>
      (Math.log(Math.max(MIN_HZ, (k * nyquist) / n) / MIN_HZ) / Math.log(nyquist / MIN_HZ)) * width;
    const level = (db: number) => Math.max(0, Math.min(1, (db - MIN_DB) / (MAX_DB - MIN_DB)));

    if (mode === "curves") {
      ctx.clearRect(0, 0, width, height);
      for (const layer of layers) {
        const n = layer.analyser.frequencyBinCount;
        if (bins.current.length !== n) bins.current = new Float32Array(n);
        layer.analyser.getFloatFrequencyData(bins.current);
        ctx.beginPath();
        ctx.strokeStyle = layer.color;
        ctx.lineWidth = 1.5 * dpr;
        for (let k = 1; k < n; k++) {
          const y = (1 - level(bins.current[k] ?? MIN_DB)) * height;
          if (k === 1) ctx.moveTo(xForBin(k, n), y);
          else ctx.lineTo(xForBin(k, n), y);
        }
        ctx.stroke();
      }
      return;
    }

    const first = layers[0];
    if (!first) return;
    const n = first.analyser.frequencyBinCount;
    if (bins.current.length !== n) bins.current = new Float32Array(n);
    first.analyser.getFloatFrequencyData(bins.current);
    ctx.drawImage(ctx.canvas, 0, 0, width, height - dpr, 0, dpr, width, height - dpr);
    const row = ctx.createImageData(width, 1);
    const px = new Uint32Array(row.data.buffer);
    let k = 1;
    for (let x = 0; x < width; x++) {
      while (k < n - 1 && xForBin(k + 1, n) <= x) k++;
      px[x] = COLORMAP[Math.round(level(bins.current[k] ?? MIN_DB) * 255)] ?? 0;
    }
    for (let y = 0; y < dpr; y++) ctx.putImageData(row, 0, y);
  });

  return (
    <section className="flex h-full min-h-0 flex-col" aria-label="Live spectrum">
      <div className="flex items-center gap-1 px-2 py-1 text-xs">
        <span className="mr-auto text-dim uppercase tracking-wider">
          <span className="hidden md:inline">Live</span>
        </span>
        {(["curves", "waterfall"] as const).map((m) => (
          <button
            key={m}
            type="button"
            aria-pressed={mode === m}
            onClick={() => {
              setMode(m);
              ref.current?.getContext("2d")?.clearRect(0, 0, size.width, size.height);
            }}
            className="rounded px-2 py-0.5 capitalize text-dim aria-pressed:bg-edge aria-pressed:text-zinc-100"
          >
            {m}
          </button>
        ))}
      </div>
      <div className="relative min-h-0 flex-1 bg-ink">
        <canvas ref={ref} className="absolute inset-0 h-full w-full" />
      </div>
    </section>
  );
}
