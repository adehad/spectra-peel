import { deserializePeaks, type PeaksPyramid } from "@spectra-peel/engine";
import { useEffect, useMemo, useRef, useState } from "react";
import { Player } from "../audio/player";
import { SOURCE_LAYER, type Project } from "../project/schema";
import { paths, projectDir, readBytes } from "../storage/opfs";
import { formatTime } from "./format";
import { LayersPanel } from "./LayersPanel";
import { LivePanel } from "./LivePanel";
import { Overview } from "./Overview";
import { SpectrogramView } from "./SpectrogramView";
import { useFrame } from "./useCanvas";
import { MAX_SPAN, MIN_SPAN, useView } from "./view";

interface Loaded {
  dir: FileSystemDirectoryHandle;
  peaks: PeaksPyramid;
  player: Player;
}

const COLORS: Record<string, string> = { [SOURCE_LAYER]: "oklch(0.78 0.1 230)" };

export function Editor({ project, onClose }: { project: Project; onClose: () => void }) {
  const [loaded, setLoaded] = useState<Loaded>();
  const [error, setError] = useState<string>();

  useEffect(() => {
    let player: Player | undefined;
    const state = { cancelled: false };
    (async () => {
      const dir = await projectDir(project.id);
      const peaks = deserializePeaks(await readBytes(dir, paths.peaks(SOURCE_LAYER)));
      if (state.cancelled) return;
      player = new Player(project);
      useView.getState().reset(project.source.frames / project.source.sampleRate);
      setLoaded({ dir, peaks, player });
    })().catch((e: unknown) => setError(String(e)));
    return () => {
      state.cancelled = true;
      void player?.close();
    };
  }, [project]);

  if (error) return <p className="p-4 text-red-300">Could not open project: {error}</p>;
  if (!loaded) return <p className="p-4 text-dim">Opening…</p>;
  return <Workspace project={project} loaded={loaded} onClose={onClose} />;
}

function Workspace({ project, loaded, onClose }: { project: Project; loaded: Loaded; onClose: () => void }) {
  const { player, peaks, dir } = loaded;
  const [playing, setPlaying] = useState(false);
  const [now, setNow] = useState(0);
  const [tab, setTab] = useState<"live" | "layers">("live");
  const [gains, setGains] = useState<Record<string, number>>({});
  const [muted, setMuted] = useState<string[]>([]);
  const { follow, setFollow, span, zoom } = useView();
  const duration = project.source.frames / project.source.sampleRate;

  useEffect(() => player.setOnEnded(() => setPlaying(false)), [player]);
  useEffect(() => {
    const isMuted = muted.includes(SOURCE_LAYER);
    player.setGain(isMuted ? 0 : 10 ** ((gains[SOURCE_LAYER] ?? 0) / 20));
  }, [player, gains, muted]);

  const lastTick = useRef(0);
  useFrame(() => {
    if (performance.now() - lastTick.current < 100) return;
    lastTick.current = performance.now();
    setNow(player.position() / player.sampleRate);
  });

  const toggle = async () => {
    if (player.isPlaying) {
      player.pause();
      setPlaying(false);
    } else {
      await player.play();
      setFollow(true);
      setPlaying(true);
    }
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== "Space" || e.target instanceof HTMLInputElement) return;
      e.preventDefault();
      void toggle();
    };
    addEventListener("keydown", onKey);
    return () => removeEventListener("keydown", onKey);
  });

  const liveLayers = useMemo(
    () => [{ id: SOURCE_LAYER, color: COLORS[SOURCE_LAYER] ?? "white", analyser: player.analyser }],
    [player],
  );
  const live = <LivePanel layers={liveLayers} sampleRate={player.sampleRate} />;
  const layers = (
    <LayersPanel
      layers={project.layers}
      colors={COLORS}
      gains={gains}
      muted={muted}
      onGain={(id, db) => setGains((g) => ({ ...g, [id]: db }))}
      onMute={(id, m) => setMuted((list) => (m ? [...list, id] : list.filter((x) => x !== id)))}
    />
  );

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center gap-3 border-b border-edge px-3 py-2">
        <button type="button" onClick={onClose} className="text-sm text-dim hover:text-zinc-100">
          ← Projects
        </button>
        <h1 className="min-w-0 flex-1 truncate font-medium" data-testid="project-name">
          {project.name}
        </h1>
        <span className="text-xs text-dim tabular-nums">
          {(project.source.sampleRate / 1000).toFixed(1)} kHz ·{" "}
          {project.source.channels === 1 ? "mono" : "stereo"}
        </span>
      </header>

      <Overview peaks={peaks} player={player} />
      <div className="min-h-0 flex-1">
        <SpectrogramView dir={dir} layer={SOURCE_LAYER} player={player} />
      </div>

      <div className="h-44 shrink-0 border-t border-edge bg-panel md:grid md:h-48 md:grid-cols-[2fr_1fr] md:grid-rows-1 md:divide-x md:divide-edge">
        <div className="flex border-b border-edge text-xs md:hidden" role="tablist">
          {(["live", "layers"] as const).map((t) => (
            <button
              key={t}
              type="button"
              role="tab"
              aria-selected={tab === t}
              onClick={() => setTab(t)}
              className="flex-1 py-1.5 capitalize text-dim aria-selected:text-zinc-100 aria-selected:shadow-[inset_0_-2px_0_var(--color-accent)]"
            >
              {t}
            </button>
          ))}
        </div>
        <div className={`h-[calc(100%-2rem)] md:block md:h-full ${tab === "live" ? "" : "hidden"}`}>
          {live}
        </div>
        <div className={`h-[calc(100%-2rem)] md:block md:h-full ${tab === "layers" ? "" : "hidden"}`}>
          {layers}
        </div>
      </div>

      <footer className="flex items-center gap-2 border-t border-edge px-3 py-2 pb-[max(0.5rem,env(safe-area-inset-bottom))]">
        <button
          type="button"
          onClick={() => void toggle()}
          aria-label={playing ? "Pause" : "Play"}
          data-testid="play"
          className="grid size-10 place-items-center rounded-full bg-accent text-ink"
        >
          {playing ? "❚❚" : "▶"}
        </button>
        <span className="font-mono text-sm tabular-nums" data-testid="time">
          {formatTime(now, true)} / {formatTime(duration)}
        </span>
        <span className="flex-1" />
        <button
          type="button"
          aria-pressed={follow}
          onClick={() => setFollow(!follow)}
          className="rounded px-2 py-1 text-xs text-dim aria-pressed:bg-edge aria-pressed:text-zinc-100"
        >
          Follow
        </button>
        <button
          type="button"
          aria-label="Zoom out"
          disabled={span >= Math.min(MAX_SPAN, duration)}
          onClick={() => zoom(2, now)}
          className="size-8 rounded bg-edge disabled:opacity-40"
        >
          −
        </button>
        <button
          type="button"
          aria-label="Zoom in"
          disabled={span <= MIN_SPAN}
          onClick={() => zoom(0.5, now)}
          className="size-8 rounded bg-edge disabled:opacity-40"
        >
          +
        </button>
      </footer>
    </div>
  );
}
