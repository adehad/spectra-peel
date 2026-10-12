import { Headphones, Volume2, VolumeX } from "lucide-react";
import { useMemo, useRef } from "react";
import { SOURCE_LAYER, type Layer, type Mix, type Project } from "../project/schema";
import { formatTime } from "./format";
import { opTitle } from "./labels";
import { useFrame } from "./useCanvas";

interface Props {
  project: Project;
  colors: ReadonlyMap<string, string>;
  onMix: (update: (mix: Mix) => Mix) => void;
  /** A gain slider moved or was grabbed; `holding` stays true until it is let go. */
  onAdjust: (layer: string, holding: boolean) => void;
  onUndo: () => void;
  busy: boolean;
  /** Each layer's signal before its gain, for the level meter under its row. */
  analysers: ReadonlyMap<string, AnalyserNode>;
}

interface RowProps {
  layer: Layer;
  project: Project;
  color: string;
  onMix: Props["onMix"];
  onAdjust: Props["onAdjust"];
  analyser: AnalyserNode | undefined;
}

/** -60 dBFS to full scale across the row. */
const METER_FLOOR_DB = -60;

/** The layer's level at its slider gain while it plays, as a thin bar along the bottom of its row. */
function LevelMeter({
  analyser,
  gainDb,
  color,
  heard,
}: {
  analyser: AnalyserNode | undefined;
  gainDb: number;
  color: string;
  heard: boolean;
}) {
  const bar = useRef<HTMLDivElement>(null);
  const samples = useRef(new Float32Array(0));
  const style = useMemo(() => ({ backgroundColor: color, opacity: heard ? 0.9 : 0.3 }), [color, heard]);
  useFrame(() => {
    const el = bar.current;
    if (!el || !analyser) return;
    if (samples.current.length !== analyser.fftSize) samples.current = new Float32Array(analyser.fftSize);
    analyser.getFloatTimeDomainData(samples.current);
    let sum = 0;
    for (const v of samples.current) sum += v * v;
    const db = 10 * Math.log10(sum / samples.current.length + 1e-12) + gainDb;
    el.style.width = `${Math.max(0, Math.min(100, (1 - db / METER_FLOOR_DB) * 100))}%`;
  });
  return (
    <div className="pointer-events-none absolute inset-x-4 bottom-0 h-0.5" aria-hidden="true">
      <div ref={bar} style={style} className="h-full w-0 rounded-full" />
    </div>
  );
}

function LayerRow({ layer, project, color, onMix, onAdjust, analyser }: RowProps) {
  const { mix } = project;
  const muted = mix.muted.includes(layer.id);
  const solo = mix.solo === layer.id;
  const comparing = mix.ab === "input";
  const soloed = mix.solo === null ? null : project.layers.find((l) => l.id === mix.solo);
  // Solo and Compare with source decide what plays over each layer's own mute, so the row shows
  // what is heard and locks the controls they override.
  let heard = soloed ? solo : !muted;
  let silencedBy: string | null = soloed && !solo ? `Muted while ${soloed.label} plays solo` : null;
  if (comparing) {
    heard = layer.id === SOURCE_LAYER;
    silencedBy = "Comparing with the source";
  }
  const db = mix.gains[layer.id] ?? 0;
  const held = useRef(false);
  const slider = useMemo(() => ({ accentColor: color }), [color]);
  return (
    <li
      className="relative flex items-center gap-2 py-1 pr-2 pl-4 text-sm pointer-coarse:py-0"
      data-testid={`layer-${layer.label.toLowerCase()}`}
    >
      <svg viewBox="0 0 10 10" className="size-2.5 shrink-0" aria-hidden="true">
        <circle cx="5" cy="5" r="5" fill={color} />
      </svg>
      <span className={`min-w-0 flex-1 truncate ${heard ? "" : "text-dim"}`}>{layer.label}</span>
      <input
        type="range"
        min={-40}
        max={6}
        step={0.5}
        value={db}
        disabled={comparing}
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
        className="w-20 disabled:opacity-40 sm:w-24"
      />
      <span className="w-14 text-right font-mono text-xs whitespace-nowrap text-dim tabular-nums">
        {db > 0 ? "+" : ""}
        {db.toFixed(0)} dB
      </span>
      <button
        type="button"
        aria-pressed={muted}
        aria-label={`Mute ${layer.label}`}
        title={silencedBy ?? (muted ? "Unmute" : "Mute")}
        disabled={silencedBy !== null}
        onClick={() =>
          onMix((m) => ({
            ...m,
            muted: muted ? m.muted.filter((id) => id !== layer.id) : [...m.muted, layer.id],
          }))
        }
        className="grid place-items-center rounded p-1 text-dim disabled:opacity-40 aria-pressed:bg-red-500/30 aria-pressed:text-red-200"
      >
        {heard ? <Volume2 className="size-4" aria-hidden /> : <VolumeX className="size-4" aria-hidden />}
      </button>
      <button
        type="button"
        aria-pressed={solo}
        aria-label={`Solo ${layer.label}`}
        title={
          comparing ? "Comparing with the source" : solo ? "Hear every layer again" : "Hear only this layer"
        }
        disabled={comparing}
        onClick={() => onMix((m) => ({ ...m, solo: solo ? null : layer.id }))}
        className="grid place-items-center rounded p-1 text-dim disabled:opacity-40 aria-pressed:bg-amber-400/30 aria-pressed:text-amber-100"
      >
        <Headphones className="size-4" aria-hidden />
      </button>
      <LevelMeter analyser={analyser} gainDb={db} color={color} heard={heard} />
    </li>
  );
}

/** Layers grouped under the op that made them, newest last. */
export function LayersPanel({ project, colors, onMix, onAdjust, onUndo, busy, analysers }: Props) {
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
          className="ml-auto rounded border border-edge px-2 py-0.5 text-dim aria-pressed:border-transparent aria-pressed:bg-accent aria-pressed:text-ink"
          data-testid="ab"
        >
          Compare with source
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
            analyser={analysers.get(source.id)}
          />
        ) : null}
        {project.ops.map((op) => (
          <li key={op.id}>
            <div className="flex items-center gap-2 px-2 pt-1.5 text-xs text-dim">
              <span className="flex-1 truncate">
                {opTitle(project, op)} · {formatTime(op.range.start / rate)}–{formatTime(op.range.end / rate)}
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
                    analyser={analysers.get(l.id)}
                  />
                ))}
            </ul>
          </li>
        ))}
      </ul>
    </section>
  );
}
