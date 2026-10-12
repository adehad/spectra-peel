import { useCallback, useEffect, useState } from "react";
import { useJobs } from "../jobs";
import type { Project } from "../project/schema";
import { deleteProject, listProjects, type ProjectListing } from "../storage/opfs";
import type { ImportMessage, ImportRequest } from "../workers/protocol";
import { formatBytes, formatTime } from "./format";
import { ModelsPanel } from "./ModelsPanel";

interface Props {
  onOpen: (project: Project) => void;
}

export function ProjectList({ onOpen }: Props) {
  const [listing, setListing] = useState<ProjectListing>();
  const [progress, setProgress] = useState<{ id: string; name: string; fraction: number }>();
  const [error, setError] = useState<string>();
  const [confirm, setConfirm] = useState<string>();
  const [storage, setStorage] = useState<StorageEstimate>();
  const [models, setModels] = useState(false);

  const refresh = useCallback(() => {
    Promise.all([listProjects(), navigator.storage.estimate()]).then(
      ([list, estimate]) => {
        setListing(list);
        setStorage(estimate);
        return undefined;
      },
      (e: unknown) => setError(String(e)),
    );
  }, []);
  useEffect(refresh, [refresh]);

  const startImport = (file: File) => {
    setError(undefined);
    void navigator.storage.persist();
    const name = file.name.replace(/\.[^.]+$/, "");
    const id = crypto.randomUUID();
    const worker = new Worker(new URL("../workers/import.worker.ts", import.meta.url), { type: "module" });
    setProgress({ id, name, fraction: 0 });
    useJobs.getState().start(id);
    worker.addEventListener("message", (e: MessageEvent<ImportMessage>) => {
      const m = e.data;
      if (m.type === "progress") setProgress({ id, name, fraction: m.fraction });
      else {
        worker.terminate();
        useJobs.getState().end(id);
        setProgress(undefined);
        if (m.type === "done") onOpen(m.project);
        else {
          setError(m.message);
          refresh();
        }
      }
    });
    const req: ImportRequest = { file, projectId: id, name };
    worker.postMessage(req);
  };

  const busy = progress !== undefined;
  const stale = listing?.incomplete.filter((s) => s.id !== progress?.id) ?? [];
  useEffect(() => {
    const over = (e: DragEvent) => e.preventDefault();
    const drop = (e: DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer?.files[0];
      if (file && !busy) startImport(file);
    };
    addEventListener("dragover", over);
    addEventListener("drop", drop);
    return () => {
      removeEventListener("dragover", over);
      removeEventListener("drop", drop);
    };
  });

  return (
    <main className="mx-auto flex h-full max-w-2xl flex-col gap-4 px-4 py-6">
      <header className="flex items-baseline gap-3">
        <h1 className="text-xl font-semibold">Spectra Peel</h1>
        <span className="hidden text-sm text-dim sm:inline">On-device audio cleanup and separation</span>
        <button
          type="button"
          onClick={() => setModels(true)}
          className="ml-auto rounded bg-edge px-2 py-1 text-sm"
          data-testid="models"
        >
          Models
        </button>
      </header>
      {models ? <ModelsPanel onClose={() => setModels(false)} /> : null}

      <label
        className={`flex cursor-pointer flex-col items-center gap-1 rounded-lg border border-dashed border-edge bg-panel px-4 py-8 text-center ${progress ? "pointer-events-none opacity-60" : "hover:border-accent"}`}
      >
        <span className="font-medium">Import audio or video</span>
        <span className="text-sm text-dim">Drop a file here or choose one. It stays on this device.</span>
        <input
          type="file"
          accept="audio/*,video/*,.flac,.wav,.mp3,.m4a,.ogg,.opus,.webm,.mkv"
          className="sr-only"
          data-testid="import"
          onChange={(e) => {
            const file = e.target.files?.[0];
            e.target.value = "";
            if (file) startImport(file);
          }}
        />
      </label>

      {progress ? (
        <div className="flex flex-col gap-1" data-testid="import-progress">
          <span className="text-sm">Importing {progress.name}…</span>
          <progress value={progress.fraction} className="h-1.5 w-full accent-accent" />
        </div>
      ) : null}
      {error ? (
        <p role="alert" className="rounded bg-red-500/15 px-3 py-2 text-sm text-red-200">
          {error}
        </p>
      ) : null}

      <section className="flex flex-col gap-1" aria-label="Projects">
        <div className="flex items-baseline justify-between text-xs text-dim uppercase tracking-wider">
          <span>Projects</span>
          {storage?.usage === undefined ? null : (
            <span className="normal-case tracking-normal tabular-nums">
              {formatBytes(storage.usage)} used of {formatBytes(storage.quota ?? 0)}
            </span>
          )}
        </div>
        {listing?.projects.length === 0 ? <p className="py-4 text-sm text-dim">No projects yet.</p> : null}
        {stale.length > 0 ? (
          <p className="flex items-center gap-2 py-1 text-xs text-dim">
            <span className="flex-1">
              {stale.length} interrupted import{stale.length > 1 ? "s" : ""} using{" "}
              {formatBytes(stale.reduce((n, s) => n + s.bytes, 0))}
            </span>
            <button
              type="button"
              className="px-2 py-1 hover:text-red-200"
              onClick={() => {
                Promise.all(stale.map((s) => deleteProject(s.id))).then(refresh, (e: unknown) =>
                  setError(String(e)),
                );
              }}
            >
              Clear
            </button>
          </p>
        ) : null}
        <ul className="divide-y divide-edge">
          {listing?.projects.map(({ project, bytes }) => (
            <li key={project.id} className="flex items-center gap-3 py-1" data-testid="project-row">
              <button
                type="button"
                onClick={() => onOpen(project)}
                className="-mx-2 flex min-w-0 flex-1 items-center gap-2 rounded px-2 py-1 text-left hover:bg-edge/60 active:bg-edge"
              >
                <span className="min-w-0 flex-1">
                  <span className="block truncate">{project.name}</span>
                  <span className="block text-xs text-dim tabular-nums">
                    {formatTime(project.source.frames / project.source.sampleRate)} ·{" "}
                    {(project.source.sampleRate / 1000).toFixed(1)} kHz · {formatBytes(bytes)}
                  </span>
                </span>
                <span className="text-dim" aria-hidden="true">
                  ›
                </span>
              </button>
              {confirm === project.id ? (
                <>
                  <button
                    type="button"
                    className="rounded bg-red-500/30 px-2 py-1 text-xs text-red-100"
                    onClick={() => {
                      setConfirm(undefined);
                      deleteProject(project.id).then(refresh, (e: unknown) => setError(String(e)));
                    }}
                  >
                    Delete
                  </button>
                  <button
                    type="button"
                    className="px-2 py-1 text-xs text-dim"
                    onClick={() => setConfirm(undefined)}
                  >
                    Keep
                  </button>
                </>
              ) : (
                <button
                  type="button"
                  className="px-2 py-1 text-xs text-dim hover:text-red-200"
                  onClick={() => setConfirm(project.id)}
                >
                  Delete…
                </button>
              )}
            </li>
          ))}
        </ul>
      </section>
    </main>
  );
}
