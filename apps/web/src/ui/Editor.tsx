import { deserializePeaks, type PeaksPyramid } from "@spectra-peel/engine";
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { Player } from "../audio/player";
import { useJobs } from "../jobs";
import { applyDenoise, currentLayer, planDenoise, undoLast, type PlannedDenoise } from "../project/ops";
import { SOURCE_LAYER, type Mix, type Project } from "../project/schema";
import { paths, projectDir, readBytes, removeFile, writeProject } from "../storage/opfs";
import type { DenoiseMessage, DenoiseRequest } from "../workers/protocol";
import { ExportPanel } from "./ExportPanel";
import { formatTime } from "./format";
import { LayersPanel } from "./LayersPanel";
import { LivePanel } from "./LivePanel";
import { Overview } from "./Overview";
import { SpectrogramView } from "./SpectrogramView";
import { ProjectTiles } from "./tiles";
import { useFrame } from "./useCanvas";
import { MAX_SPAN, MIN_SPAN, useView } from "./view";

const PALETTE: Record<string, string> = {
  Source: "oklch(0.78 0.1 230)",
  Denoised: "oklch(0.8 0.14 150)",
  Removed: "oklch(0.75 0.15 40)",
};

interface Job {
  plan: PlannedDenoise;
  worker: Worker;
  fraction: number;
  stage: string;
}

async function deleteFiles(dir: FileSystemDirectoryHandle, files: string[]): Promise<void> {
  for (const f of files) await removeFile(dir, f);
}

function planFiles(plan: PlannedDenoise): string[] {
  const { output, removed } = plan;
  const flac = output.data.kind === "flac" ? [output.data.path, output.data.index] : [];
  return [...flac, paths.peaks(output.id), `derived/tiles/${output.id}`, `derived/tiles/${removed.id}`];
}

export function Editor({ project: initial, onClose }: { project: Project; onClose: () => void }) {
  const [project, setProject] = useState(initial);
  const [dir, setDir] = useState<FileSystemDirectoryHandle>();
  const [peaks, setPeaks] = useState<PeaksPyramid>();
  const [error, setError] = useState<string>();
  const projectRef = useRef(project);
  useLayoutEffect(() => {
    projectRef.current = project;
  });

  useEffect(() => {
    (async () => {
      const d = await projectDir(initial.id);
      setPeaks(deserializePeaks(await readBytes(d, paths.peaks(SOURCE_LAYER))));
      setDir(d);
      useView.getState().reset(initial.source.frames / initial.source.sampleRate);
    })().catch((e: unknown) => setError(String(e)));
  }, [initial]);

  const [handoff] = useState<Handoff>(() => ({ closing: Promise.resolve(), position: 0 }));
  if (error) return <p className="p-4 text-red-300">Could not open project: {error}</p>;
  if (!dir || !peaks) return <p className="p-4 text-dim">Opening…</p>;
  // A new layer set needs a new Player, so the session remounts on it.
  return (
    <Session
      key={project.layers.map((l) => l.id).join(",")}
      handoff={handoff}
      projectRef={projectRef}
      project={project}
      setProject={setProject}
      dir={dir}
      peaks={peaks}
      onClose={onClose}
    />
  );
}

/** Passes the play position on to the next Player, and makes it wait until the old one closed its handles. */
interface Handoff {
  closing: Promise<void>;
  position: number;
}

function Session({
  handoff,
  projectRef,
  ...props
}: Omit<WorkspaceProps, "player"> & { handoff: Handoff; projectRef: { current: Project } }) {
  const [player, setPlayer] = useState<Player>();
  useEffect(() => {
    let p: Player | undefined;
    let cancelled = false;
    void handoff.closing.then(() => {
      if (!cancelled) {
        p = new Player(projectRef.current, handoff.position);
        setPlayer(p);
      }
      return undefined;
    });
    return () => {
      cancelled = true;
      if (p) {
        handoff.position = p.position();
        handoff.closing = p.close();
      }
    };
  }, [handoff, projectRef]);
  if (!player) return <p className="p-4 text-dim">Opening…</p>;
  return <Workspace {...props} player={player} />;
}

interface WorkspaceProps {
  project: Project;
  setProject: (p: Project) => void;
  dir: FileSystemDirectoryHandle;
  peaks: PeaksPyramid;
  player: Player;
  onClose: () => void;
}

function Workspace({ project, setProject, dir, peaks, player, onClose }: WorkspaceProps) {
  const [playing, setPlaying] = useState(player.isPlaying);
  const [now, setNow] = useState(0);
  const [tab, setTab] = useState<"live" | "layers">("live");
  const [job, setJob] = useState<Job>();
  const [notice, setNotice] = useState<string>();
  const [selectMode, setSelectMode] = useState(false);
  const [exporting, setExporting] = useState(false);
  const lastOp = project.ops.at(-1);
  const [base, setBase] = useState(lastOp?.outputs[0] ?? SOURCE_LAYER);
  const [showRemoved, setShowRemoved] = useState(true);
  const { follow, setFollow, span, zoom, selection } = useView();
  const rate = project.source.sampleRate;
  const duration = project.source.frames / rate;
  // Workspace remounts when layers change, so tiles built once per mount stay valid.
  const [tiles] = useState(() => new ProjectTiles(dir, project));
  const colors = useMemo(
    () => new Map(project.layers.map((l) => [l.id, PALETTE[l.label] ?? "white"])),
    [project],
  );
  const removedLayer = project.layers.findLast((l) => l.data.kind === "residual");

  useEffect(() => player.setOnEnded(() => setPlaying(false)), [player]);

  const lastTick = useRef(0);
  useFrame(() => {
    if (performance.now() - lastTick.current < 100) return;
    lastTick.current = performance.now();
    setNow(player.position() / rate);
  });

  const saveTimer = useRef<ReturnType<typeof setTimeout>>(undefined);
  const onMix = useCallback(
    (update: (mix: Mix) => Mix) => {
      const next = { ...project, mix: update(project.mix) };
      setProject(next);
      player.setMix(next);
      clearTimeout(saveTimer.current);
      saveTimer.current = setTimeout(() => {
        writeProject(dir, next).catch((e: unknown) => setNotice(String(e)));
      }, 300);
    },
    [project, setProject, player, dir],
  );

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

  const startDenoise = () => {
    const sel = useView.getState().selection;
    const range = sel
      ? { start: Math.round(sel.start * rate), end: Math.round(sel.end * rate) }
      : { start: 0, end: project.source.frames };
    const plan = planDenoise(project, currentLayer(project), range);
    const worker = new Worker(new URL("../workers/denoise.worker.ts", import.meta.url), { type: "module" });
    setNotice(undefined);
    setJob({ plan, worker, fraction: 0, stage: "Starting" });
    useJobs.getState().start(plan.op.id);
    worker.addEventListener("message", (e: MessageEvent<DenoiseMessage>) => {
      const m = e.data;
      if (m.type === "progress") {
        setJob((j) => (j ? { ...j, fraction: m.fraction, stage: m.stage } : j));
        return;
      }
      worker.terminate();
      useJobs.getState().end(plan.op.id);
      if (m.type === "error") {
        setJob(undefined);
        setNotice(`Denoise failed: ${m.message}`);
        void deleteFiles(dir, planFiles(plan));
        return;
      }
      const next = applyDenoise(project, plan);
      void (async () => {
        try {
          await writeProject(dir, next);
          useView.getState().setSelection(null);
          setProject(next);
        } catch (err) {
          setNotice(String(err));
        }
        setJob(undefined);
      })();
    });
    const req: DenoiseRequest = {
      project,
      op: plan.op,
      output: plan.output,
      removed: plan.removed,
    };
    worker.postMessage(req);
  };

  const cancelJob = () => {
    if (!job) return;
    job.worker.terminate();
    useJobs.getState().end(job.plan.op.id);
    setJob(undefined);
    // Terminating releases the worker's handles shortly after, not synchronously.
    setTimeout(() => void deleteFiles(dir, planFiles(job.plan)).catch(() => undefined), 500);
  };

  const undo = () => {
    const { project: next, files } = undoLast(project);
    void (async () => {
      try {
        await writeProject(dir, next);
        setProject(next);
        // The player keeps reading these files until it is replaced.
        setTimeout(() => void deleteFiles(dir, files).catch((e: unknown) => setNotice(String(e))), 1500);
      } catch (err) {
        setNotice(String(err));
      }
    })();
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
      if (e.code === "Space") {
        e.preventDefault();
        void toggle();
      } else if (e.code === "Escape") useView.getState().setSelection(null);
    };
    addEventListener("keydown", onKey);
    return () => removeEventListener("keydown", onKey);
  });

  const liveLayers = useMemo(
    () =>
      project.layers.flatMap((l) => {
        const analyser = player.analysers.get(l.id);
        return analyser ? [{ id: l.id, label: l.label, color: colors.get(l.id) ?? "white", analyser }] : [];
      }),
    [project, player, colors],
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
        <span className="hidden text-xs text-dim tabular-nums sm:inline">
          {(rate / 1000).toFixed(1)} kHz · {project.source.channels === 1 ? "mono" : "stereo"}
        </span>
      </header>

      <Overview peaks={peaks} player={player} />
      <div className="flex items-center gap-2 border-b border-edge px-3 py-1 text-xs">
        <label className="flex items-center gap-1 text-dim">
          View
          <select
            value={base}
            onChange={(e) => setBase(e.target.value)}
            className="rounded bg-edge px-1 py-0.5 text-zinc-100"
            data-testid="view-layer"
          >
            {project.layers.map((l) => (
              <option key={l.id} value={l.id}>
                {l.label}
                {l.producedBy ? ` (${project.ops.findIndex((o) => o.id === l.producedBy) + 1})` : ""}
              </option>
            ))}
          </select>
        </label>
        {removedLayer ? (
          <button
            type="button"
            aria-pressed={showRemoved}
            onClick={() => setShowRemoved(!showRemoved)}
            className="rounded px-2 py-0.5 text-dim aria-pressed:bg-cyan-400/25 aria-pressed:text-cyan-100"
          >
            Removed energy
          </button>
        ) : null}
        <span className="ml-auto font-mono text-dim tabular-nums">
          {selection ? `${formatTime(selection.start, true)} – ${formatTime(selection.end, true)}` : ""}
        </span>
      </div>
      <div className="relative min-h-0 flex-1">
        <SpectrogramView
          tiles={tiles}
          base={base}
          overlay={showRemoved && removedLayer ? removedLayer.id : null}
          player={player}
          selectMode={selectMode}
        />
        {job ? (
          <div
            className="absolute inset-x-0 top-0 flex items-center gap-2 bg-ink/80 px-3 py-1.5 text-xs"
            data-testid="job"
          >
            <span>
              {job.stage}… {Math.round(job.fraction * 100)}%
            </span>
            <progress value={job.fraction} className="h-1 flex-1 accent-accent" />
            <button type="button" onClick={cancelJob} className="text-dim hover:text-zinc-100">
              Cancel
            </button>
          </div>
        ) : null}
        {notice ? (
          <p
            role="alert"
            className="absolute inset-x-3 top-2 rounded bg-red-500/20 px-3 py-1.5 text-xs text-red-100"
          >
            {notice}
          </p>
        ) : null}
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
          <LivePanel layers={liveLayers} sampleRate={rate} />
        </div>
        <div className={`h-[calc(100%-2rem)] md:block md:h-full ${tab === "layers" ? "" : "hidden"}`}>
          <LayersPanel
            project={project}
            colors={colors}
            onMix={onMix}
            onUndo={undo}
            busy={job !== undefined}
          />
        </div>
      </div>

      <footer className="flex flex-wrap items-center gap-2 border-t border-edge px-3 py-2 pb-[max(0.5rem,env(safe-area-inset-bottom))]">
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
          aria-pressed={selectMode}
          onClick={() => setSelectMode(!selectMode)}
          className="rounded px-2 py-1 text-xs text-dim aria-pressed:bg-edge aria-pressed:text-zinc-100"
          title="Drag on the spectrogram to select a range (or hold Shift)"
        >
          Select
        </button>
        <button
          type="button"
          disabled={job !== undefined}
          onClick={startDenoise}
          className="rounded bg-edge px-3 py-1 text-sm disabled:opacity-40"
          data-testid="denoise"
        >
          Denoise {selection ? "selection" : "all"}
        </button>
        <button
          type="button"
          onClick={() => setExporting(true)}
          className="rounded bg-edge px-3 py-1 text-sm"
          data-testid="export"
        >
          Export
        </button>
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
      {exporting ? (
        <ExportPanel project={project} dir={dir} selection={selection} onClose={() => setExporting(false)} />
      ) : null}
    </div>
  );
}
