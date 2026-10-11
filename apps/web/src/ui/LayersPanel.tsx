import type { Layer } from "../project/schema";

interface Props {
  layers: Layer[];
  colors: Record<string, string>;
  gains: Record<string, number>;
  muted: string[];
  onGain: (id: string, db: number) => void;
  onMute: (id: string, muted: boolean) => void;
}

export function LayersPanel({ layers, colors, gains, muted, onGain, onMute }: Props) {
  return (
    <section className="flex h-full min-h-0 flex-col" aria-label="Layers">
      <div className="hidden px-2 py-1 text-xs text-dim uppercase tracking-wider md:block">Layers</div>
      <ul className="min-h-0 flex-1 overflow-y-auto">
        {layers.map((layer) => {
          const isMuted = muted.includes(layer.id);
          const db = gains[layer.id] ?? 0;
          return (
            <li key={layer.id} className="flex items-center gap-2 px-2 py-1.5 text-sm">
              <svg viewBox="0 0 10 10" className="size-2.5 shrink-0" aria-hidden="true">
                <circle cx="5" cy="5" r="5" fill={colors[layer.id] ?? "currentColor"} />
              </svg>
              <span className="min-w-0 flex-1 truncate">{layer.label}</span>
              <input
                type="range"
                min={-24}
                max={6}
                step={0.5}
                value={db}
                aria-label={`${layer.label} gain`}
                onChange={(e) => onGain(layer.id, Number(e.target.value))}
                className="w-24 accent-accent"
              />
              <span className="w-12 text-right font-mono text-xs text-dim tabular-nums">
                {db > 0 ? "+" : ""}
                {db.toFixed(1)}
              </span>
              <button
                type="button"
                aria-pressed={isMuted}
                onClick={() => onMute(layer.id, !isMuted)}
                className="rounded px-1.5 py-0.5 text-xs text-dim aria-pressed:bg-red-500/30 aria-pressed:text-red-200"
              >
                M
              </button>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
