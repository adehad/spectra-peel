import { useMemo, useRef } from "react";
import { SOURCE_LAYER, type Layer, type Mix, type Project } from "../project/schema";
import { formatTime } from "./format";

interface Props {
  project: Project;
  colors: ReadonlyMap<string, string>;
  onMix: (update: (mix: Mix) => Mix) => void;
  /** A gain slider moved or was grabbed; `holding` stays true until it is let go. */
  onAdjust: (layer: string, holding: boolean) => void;
  onUndo: () => void;
  busy: boolean;
}

interface RowProps {
  layer: Layer;
  project: Project;
  color: string;
  onMix: Props["onMix"];
  onAdjust: Props["onAdjust"];
}

function LayerRow({ layer, project, color, onMix, onAdjust }: RowProps) {
  const { mix } = project;
  const muted = mix.muted.includes(layer.id);
  const solo = mix.solo === layer.id;
  const db = mix.gains[layer.id] ?? 0;
  const held = useRef(false);
  const slider = useMemo(() => ({ accentColor: color }), [color]);
  return (
    <li
      className="flex items-center gap-2 py-1 pr-2 pl-4 text-sm"
      data-testid={`layer-${layer.label.toLowerCase()}`}
    >
      <svg viewBox="0 0 10 10" className="size-2.5 shrink-0" aria-hidden="true">
        <circle cx="5" cy="5" r="5" fill={color} />
      </svg>
      <span className={`min-w-0 flex-1 truncate ${muted && !solo ? "text-dim" : ""}`}>{layer.label}</span>
      <input
        type="range"
        min={-40}
        max={6}
        step={0.5}
        value={db}
        aria-label={`${layer.label} gain`}
        onChange={(e) => {
          const value = Number(e.target.value);
          onMix((m) => ({ ...m, gains: { ...m.gains, [layer.id]: value } }));
          onAdjust(layer.id, held.current);
        }}
        onPointerDown={() => {
          held.current = true;
          onAdjust(layer.id, true);
        }}
        onPointerUp={() => {
          held.current = false;
          onAdjust(layer.id, false);
        }}
        onPointerCancel={() => {
          held.current = false;
          onAdjust(layer.id, false);
        }}
        style={slider}
        className="w-20 sm:w-24"
      />
      <span className="w-10 text-right font-mono text-xs text-dim tabular-nums">
        {db > 0 ? "+" : ""}
        {db.toFixed(0)}
      </span>
      <button
        type="button"
        aria-pressed={muted}
        aria-label={`Mute ${layer.label}`}
        onClick={() =>
          onMix((m) => ({
            ...m,
            muted: muted ? m.muted.filter((id) => id !== layer.id) : [...m.muted, layer.id],
          }))
        }
        className="rounded px-1.5 py-0.5 text-xs text-dim aria-pressed:bg-red-500/30 aria-pressed:text-red-200"
      >
        M
      </button>
      <button
        type="button"
        aria-pressed={solo}
        aria-label={`Solo ${layer.label}`}
        onClick={() => onMix((m) => ({ ...m, solo: solo ? null : layer.id }))}
        className="rounded px-1.5 py-0.5 text-xs text-dim aria-pressed:bg-amber-400/30 aria-pressed:text-amber-100"
      >
        S
      </button>
    </li>
  );
}

/** Layers grouped under the op that made them, newest last. */
export function LayersPanel({ project, colors, onMix, onAdjust, onUndo, busy }: Props) {
  const rate = project.source.sampleRate;
  const source = project.layers.find((l) => l.id === SOURCE_LAYER);
  const last = project.ops.at(-1);
  const color = (id: string) => colors.get(id) ?? "currentColor";
  return (
    <section className="flex h-full min-h-0 flex-col" aria-label="Layers">
      <div className="flex items-center gap-2 px-2 py-1 text-xs">
        <span className="mr-auto hidden text-dim uppercase tracking-wider md:inline">Layers</span>
        <button
          type="button"
          aria-pressed={project.mix.ab === "input"}
          onClick={() => onMix((m) => ({ ...m, ab: m.ab === "input" ? "mix" : "input" }))}
          className="ml-auto rounded px-2 py-0.5 text-dim aria-pressed:bg-accent aria-pressed:text-ink"
          data-testid="ab"
        >
          Hear input
        </button>
      </div>
      <ul className="min-h-0 flex-1 overflow-y-auto">
        {source ? (
          <LayerRow
            layer={source}
            project={project}
            color={color(source.id)}
            onMix={onMix}
            onAdjust={onAdjust}
          />
        ) : null}
        {project.ops.map((op) => (
          <li key={op.id}>
            <div className="flex items-center gap-2 px-2 pt-1.5 text-xs text-dim">
              <span className="flex-1 truncate">
                {{ denoise: "Denoise", separate: "Stems", speakers: "People" }[op.kind]} of{" "}
                {project.layers.find((l) => l.id === op.input)?.label ?? "a removed layer"} ·{" "}
                {formatTime(op.range.start / rate)}–{formatTime(op.range.end / rate)}
              </span>
              {op === last ? (
                <button
                  type="button"
                  disabled={busy}
                  onClick={onUndo}
                  className="px-1 hover:text-zinc-100 disabled:opacity-40"
                >
                  Undo
                </button>
              ) : null}
            </div>
            <ul>
              {project.layers
                .filter((l) => l.producedBy === op.id)
                .map((l) => (
                  <LayerRow
                    key={l.id}
                    layer={l}
                    project={project}
                    color={color(l.id)}
                    onMix={onMix}
                    onAdjust={onAdjust}
                  />
                ))}
            </ul>
          </li>
        ))}
      </ul>
    </section>
  );
}
