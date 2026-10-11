import { deserializePeaks, type PeaksPyramid } from "@spectra-peel/engine";
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { Player } from "../audio/player";
import { useJobs } from "../jobs";
import { applyDenoise, applySeparate, planDenoise, planSeparate, undoLast } from "../project/ops";
import { SOURCE_LAYER, layerGains, type Layer, type Mix, type Op, type Project } from "../project/schema";
import { paths, projectDir, readBytes, removeFile, writeProject } from "../storage/opfs";
import { modelStatus } from "../storage/models";
import type { DenoiseRequest, SeparateRequest } from "../workers/protocol";
import { ModelsPanel } from "./ModelsPanel";
import { StemsConfirm } from "./StemsConfirm";
import { ExportPanel } from "./ExportPanel";
import { formatTime } from "./format";
import { LayersPanel } from "./LayersPanel";
import { LivePanel } from "./LivePanel";
import { Overview } from "./Overview";
import type { Tint } from "./paint";
import { layerColor } from "./palette";
import { SpectrogramView } from "./SpectrogramView";
import { ProjectTiles } from "./tiles";
import { useFrame } from "./useCanvas";
import { MAX_SPAN, MIN_SPAN, useView } from "./view";

/** How long a slider's highlight stays on the spectrogram after it is let go. */
const HIGHLIGHT_MS = 1200;

/** The op whose layers explain the viewed layer: the newest that took it as input or made it. */
function viewedOp(project: Project, base: string): Op | undefined {
  return project.ops.findLast((o) => o.input === base || o.outputs.includes(base));
}

/**
 * Layers drawn over `base`. When `base` was split by an op, its parts always scale it by the gain they
 * play at, so muting a stem takes its energy out of the picture. A slider being moved colours just its
 * layer. Otherwise, with colours on, a denoise colours what it removed and a separation each stem.
 * Layers that are not parts of the base are coloured at their slider gain, muted or not.
 */
function tintsFor(project: Project, base: string, focus: string | null, colours: boolean): Tint[] {
  const op = viewedOp(project, base);
  const parts = op?.input === base ? op.outputs : [];
  let coloured: string[] = [];
  if (focus !== null) coloured = [focus];
  else if (colours && op)
    coloured = op.outputs.filter(
      (id) =>
        id !== base &&
        (op.kind === "separate" || project.layers.find((l) => l.id === id)?.data.kind === "residual"),
    );
  const playing = layerGains({ ...project, mix: { ...project.mix, ab: "mix" } });
  return [...new Set([...parts, ...coloured])].flatMap((id) => {
    const layer = project.layers.find((l) => l.id === id);
    if (!layer) return [];
    const component = parts.includes(id);
    return [
      {
        layer: id,
        rgb: layerColor(layer.label).rgb,
        gainDb: component ? 20 * Math.log10(playing.get(id) ?? 0) : (project.mix.gains[id] ?? 0),
        component,
        coloured: coloured.includes(id),
      },
    ];
  });
}

interface Job {
  id: string;
  files: string[];
  worker: Worker;
  fraction: number;
  stage: string;
  etaSeconds: number | null;
  wakeLock: WakeLockSentinel | null;
}

type JobMessage =
  | { type: "progress"; fraction: number; stage: string; etaSeconds?: number | null }
  | { type: "done" }
  | { type: "error"; message: string };

async function deleteFiles(dir: FileSystemDirectoryHandle, files: string[]): Promise<void> {
  for (const f of files) await removeFile(dir, f);
}

/** Every file an op's layers may have written, for cleaning up after a failed or cancelled job. */
function layerFiles(layers: readonly Layer[]): string[] {
  return layers.flatMap((l) => [
    ...(l.data.kind === "flac" ? [l.data.path, l.data.index] : []),
    paths.peaks(l.id),
    `derived/tiles/${l.id}`,
  ]);
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
  const [stemsAsk, setStemsAsk] = useState(false);
  const [modelsOpen, setModelsOpen] = useState(false);
  const lastOp = project.ops.at(-1);
  // Stems read best over the mix they came from; a denoise over its result, tinted by what it removed.
  const [base, setBase] = useState(
    (lastOp?.kind === "separate" ? lastOp.input : lastOp?.outputs[0]) ?? SOURCE_LAYER,
  );
  const [colours, setColours] = useState(true);
  const [focus, setFocus] = useState<string | null>(null);
  const focusTimer = useRef<ReturnType<typeof setTimeout>>(undefined);
  const onAdjust = useCallback((layer: string, holding: boolean) => {
    clearTimeout(focusTimer.current);
    setFocus(layer);
    if (!holding) focusTimer.current = setTimeout(() => setFocus(null), HIGHLIGHT_MS);
  }, []);
  const { follow, setFollow, span, zoom, selection } = useView();
  const rate = project.source.sampleRate;
  const duration = project.source.frames / rate;
  // Workspace remounts when layers change, so tiles built once per mount stay valid.
  const [tiles] = useState(() => new ProjectTiles(dir, project));
  const colors = useMemo(
    () => new Map(project.layers.map((l) => [l.id, layerColor(l.label).css])),
    [project],
  );
  const op = viewedOp(project, base);
  const tints = useMemo(() => tintsFor(project, base, focus, colours), [project, base, focus, colours]);

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

  /** Runs an op in a worker, keeps the screen awake, and adds its layers when it finishes. */
  const runJob = (
    id: string,
    layers: Layer[],
    worker: Worker,
    request: DenoiseRequest | SeparateRequest,
    finish: (p: Project) => Project,
  ) => {
    setNotice(undefined);
    const files = layerFiles(layers);
    setJob({ id, files, worker, fraction: 0, stage: "Starting", etaSeconds: null, wakeLock: null });
    useJobs.getState().start(id);
    ("wakeLock" in navigator
      ? navigator.wakeLock.request("screen")
      : Promise.reject(new Error("no wake lock"))
    ).then(
      (wakeLock) => {
        setJob((j) => (j?.id === id ? { ...j, wakeLock } : j));
        return undefined;
      },
      () => undefined,
    );
    worker.addEventListener("message", (e: MessageEvent<JobMessage>) => {
      const m = e.data;
      if (m.type === "progress") {
        setJob((j) =>
          j ? { ...j, fraction: m.fraction, stage: m.stage, etaSeconds: m.etaSeconds ?? null } : j,
        );
        return;
      }
      worker.terminate();
      useJobs.getState().end(id);
      setJob((j) => {
        void j?.wakeLock?.release();
        return j;
      });
      if (m.type === "error") {
        setJob(undefined);
        setNotice(`Failed: ${m.message}`);
        void deleteFiles(dir, files);
        return;
      }
      const next = finish(project);
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
    worker.postMessage(request);
  };

  const startDenoise = () => {
    const sel = useView.getState().selection;
    const range = sel
      ? { start: Math.round(sel.start * rate), end: Math.round(sel.end * rate) }
      : { start: 0, end: project.source.frames };
    const plan = planDenoise(project, base, range);
    const worker = new Worker(new URL("../workers/denoise.worker.ts", import.meta.url), { type: "module" });
    const req: DenoiseRequest = { project, op: plan.op, output: plan.output, removed: plan.removed };
    runJob(plan.op.id, [plan.output, plan.removed], worker, req, (p) => applyDenoise(p, plan));
  };

  const askStems = () => {
    modelStatus("htdemucs-stems").then(
      (s) => {
        if (s.ready) setStemsAsk(true);
        else setModelsOpen(true);
        return undefined;
      },
      (e: unknown) => setNotice(String(e)),
    );
  };

  const startStems = (useGpu: boolean) => {
    setStemsAsk(false);
    const plan = planSeparate(project, base);
    const worker = new Worker(new URL("../workers/separate.worker.ts", import.meta.url), { type: "module" });
    const req: SeparateRequest = { project, op: plan.op, stems: plan.stems, rest: plan.rest, useGpu };
    runJob(plan.op.id, [...plan.stems, plan.rest], worker, req, (p) => applySeparate(p, plan));
  };

  const cancelJob = () => {
    if (!job) return;
    job.worker.terminate();
    void job.wakeLock?.release();
    useJobs.getState().end(job.id);
    setJob(undefined);
    // Terminating releases the worker's handles shortly after, not synchronously.
    setTimeout(() => void deleteFiles(dir, job.files).catch(() => undefined), 500);
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
        {op ? (
          <button
            type="button"
            aria-pressed={colours}
            onClick={() => setColours(!colours)}
            className="rounded px-2 py-0.5 text-dim aria-pressed:bg-edge aria-pressed:text-zinc-100"
            data-testid="colours"
          >
            {op.kind === "separate" ? "Stem colours" : "Removed energy"}
          </button>
        ) : null}
        <span className="ml-auto font-mono text-dim tabular-nums">
          {selection ? `${formatTime(selection.start, true)} – ${formatTime(selection.end, true)}` : ""}
        </span>
      </div>
      <div className="relative min-h-0 flex-1">
        <SpectrogramView tiles={tiles} base={base} tints={tints} player={player} selectMode={selectMode} />
        {job ? (
          <div
            className="absolute inset-x-0 top-0 flex items-center gap-2 bg-ink/80 px-3 py-1.5 text-xs"
            data-testid="job"
          >
            <span>
              {job.stage}… {Math.round(job.fraction * 100)}%
              {job.etaSeconds === null ? "" : ` · about ${formatTime(job.etaSeconds)} left`}
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
            onAdjust={onAdjust}
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
          disabled={job !== undefined}
          onClick={askStems}
          className="rounded bg-edge px-3 py-1 text-sm disabled:opacity-40"
          data-testid="stems"
        >
          Stems
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
      {stemsAsk ? (
        <StemsConfirm
          project={project}
          input={project.layers.find((l) => l.id === base)?.label ?? "Source"}
          onRun={startStems}
          onClose={() => setStemsAsk(false)}
        />
      ) : null}
      {modelsOpen ? <ModelsPanel onClose={() => setModelsOpen(false)} /> : null}
      {exporting ? (
        <ExportPanel project={project} dir={dir} selection={selection} onClose={() => setExporting(false)} />
      ) : null}
    </div>
  );
}
