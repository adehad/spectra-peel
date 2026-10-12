import { useCallback, useEffect, useState } from "react";
import { MODEL_IDS, MODELS, SPECIALISTS, type ModelId } from "../models";
import {
  addModelFromFile,
  deleteModel,
  downloadModel,
  modelStatus,
  type ModelStatus,
} from "../storage/models";
import { formatBytes } from "./format";

/** The models a task needs, listed alone, with a way on once they are all on this device. */
export interface ModelsNeed {
  ids: readonly ModelId[];
  title: string;
  about: string;
  /** Continue works with none of them added, as for optional upgrades. */
  optional?: boolean;
  onContinue: () => void;
}

type Work = (progress: (f: number) => void) => Promise<void>;

interface RowProps {
  status: ModelStatus;
  busy: { id: ModelId; fraction: number } | undefined;
  run: (id: ModelId, work: Work) => void;
}

const BUTTON = "inline-flex items-center justify-center rounded px-2 py-1 pointer-coarse:flex-1";

function ModelRow({ status: s, busy, run }: RowProps) {
  const spec = MODELS[s.id];
  const ready = spec.bundled || s.ready;
  let state = "Not on this device";
  if (spec.bundled) state = "Built in";
  else if (s.ready) state = "Ready";
  else if (!spec.url) state = "Not hosted";
  return (
    <li className="flex flex-col gap-1 py-2" data-testid={`model-${s.id}`}>
      <div className="flex items-baseline gap-2">
        <span className="font-medium">{spec.label}</span>
        <span className="ml-auto text-xs whitespace-nowrap text-dim tabular-nums">
          {spec.bytes > 0 ? formatBytes(spec.bytes) : ""}
        </span>
      </div>
      <p className="text-xs text-dim">
        {spec.purpose} · {spec.license}
      </p>
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className={ready ? "text-emerald-300" : "text-dim"}>{state}</span>
        <span className="flex-1 pointer-coarse:basis-full" />
        {!ready && spec.url ? (
          <button
            type="button"
            disabled={busy !== undefined}
            onClick={() => run(s.id, (p) => downloadModel(s.id, p))}
            className={`${BUTTON} bg-accent text-ink disabled:opacity-40`}
          >
            Download
          </button>
        ) : null}
        {ready ? null : (
          <label className={`${BUTTON} cursor-pointer bg-edge`}>
            Add from file…
            <input
              type="file"
              accept=".onnx"
              className="sr-only"
              disabled={busy !== undefined}
              onChange={(e) => {
                const file = e.target.files?.[0];
                e.target.value = "";
                if (file) run(s.id, (p) => addModelFromFile(s.id, file, p));
              }}
            />
          </label>
        )}
        {s.ready && !spec.bundled ? (
          <button
            type="button"
            disabled={busy !== undefined}
            onClick={() => run(s.id, () => deleteModel(s.id))}
            className="rounded px-2 py-1 text-dim hover:text-red-200"
          >
            Delete
          </button>
        ) : null}
      </div>
      {busy?.id === s.id ? <progress value={busy.fraction} className="h-1 w-full accent-accent" /> : null}
    </li>
  );
}

/** Models with their size and state; downloads and file imports are checked against a pinned SHA-256. */
export function ModelsPanel({ onClose, need }: { onClose: () => void; need?: ModelsNeed }) {
  const ids = need?.ids ?? MODEL_IDS;
  const [status, setStatus] = useState<ModelStatus[]>([]);
  const [busy, setBusy] = useState<{ id: ModelId; fraction: number }>();
  const [error, setError] = useState<string>();
  const [persisted, setPersisted] = useState<boolean>();

  const refresh = useCallback(() => {
    Promise.all(ids.map((id) => modelStatus(id))).then(
      (s) => {
        setStatus(s);
        return undefined;
      },
      (e: unknown) => setError(String(e)),
    );
    navigator.storage.persisted().then(
      (p) => {
        setPersisted(p);
        return undefined;
      },
      () => undefined,
    );
  }, [ids]);
  useEffect(refresh, [refresh]);
  const missing = status.filter((s) => !s.ready && !MODELS[s.id].bundled).map((s) => MODELS[s.id].label);
  const canContinue = status.length === ids.length && (missing.length === 0 || need?.optional === true);

  const run = (id: ModelId, work: Work) => {
    setError(undefined);
    setBusy({ id, fraction: 0 });
    void navigator.storage.persist();
    work((fraction) => setBusy({ id, fraction }))
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => {
        setBusy(undefined);
        refresh();
      });
  };

  // The full list folds the four best-quality stem models into one group; a job's list shows them flat.
  const grouped = need ? [] : status.filter((s) => SPECIALISTS.includes(s.id));
  const flat = status.filter((s) => !grouped.includes(s));
  const row = (s: ModelStatus) => <ModelRow key={s.id} status={s} busy={busy} run={run} />;

  return (
    <div
      className="fixed inset-0 z-10 grid place-items-end bg-black/50 sm:place-items-center"
      role="presentation"
    >
      <dialog
        open
        aria-label={need?.title ?? "Models"}
        className="static m-0 flex max-h-[90dvh] w-full max-w-lg flex-col rounded-t-xl border border-edge bg-panel p-4 pb-[max(1rem,env(safe-area-inset-bottom))] text-zinc-100 sm:rounded-xl"
      >
        <h2 className="mb-3 font-medium">{need?.title ?? "Models"}</h2>
        {need ? <p className="mb-3 text-sm text-dim">{need.about}</p> : null}
        <div className="mb-3 min-h-0 overflow-y-auto">
          <ul className="divide-y divide-edge">{flat.map(row)}</ul>
          {grouped.length > 0 ? (
            <details className="border-t border-edge" open={grouped.some((s) => s.ready)}>
              <summary className="cursor-pointer py-2 font-medium">
                Best-quality stems (optional)
                <span className="ml-2 text-xs font-normal text-dim">
                  {grouped.filter((s) => s.ready).length} of {grouped.length} on this device
                </span>
              </summary>
              <ul className="divide-y divide-edge pl-3">{grouped.map(row)}</ul>
            </details>
          ) : null}
        </div>
        {error ? (
          <p role="alert" className="mb-3 text-sm text-red-200">
            {error}
          </p>
        ) : null}
        {need ? null : (
          <p className="mb-3 text-xs text-dim">
            Models marked Not hosted are added from an .onnx file.{" "}
            {persisted
              ? "This browser will keep downloaded models and projects."
              : "The browser may clear models and projects when storage runs low."}
          </p>
        )}
        <div className="flex items-center justify-end gap-2 text-sm">
          {need && !canContinue && missing.length > 0 ? (
            <span className="mr-auto text-xs text-dim">Add {missing.join(", ")} to continue.</span>
          ) : null}
          <button type="button" onClick={onClose} className="px-3 py-1.5">
            {need ? "Cancel" : "Close"}
          </button>
          {need ? (
            <button
              type="button"
              disabled={!canContinue || busy !== undefined}
              onClick={need.onContinue}
              className="rounded bg-accent px-3 py-1.5 text-ink disabled:opacity-40"
              data-testid="models-continue"
            >
              Continue
            </button>
          ) : null}
        </div>
      </dialog>
    </div>
  );
}
