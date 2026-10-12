import { useCallback, useEffect, useState } from "react";
import { MODEL_IDS, MODELS, type ModelId } from "../models";
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
  const allReady = status.length === ids.length && status.every((s) => s.ready);

  const run = (id: ModelId, work: (progress: (f: number) => void) => Promise<void>) => {
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
        <ul className="mb-3 min-h-0 divide-y divide-edge overflow-y-auto">
          {status.map((s) => {
            const spec = MODELS[s.id];
            const working = busy?.id === s.id;
            return (
              <li key={s.id} className="flex flex-col gap-1.5 py-2" data-testid={`model-${s.id}`}>
                <div className="flex flex-wrap items-baseline gap-x-2">
                  <span className="font-medium">{spec.label}</span>
                  <span className="order-last basis-full text-xs text-dim sm:order-none sm:basis-auto">
                    {spec.purpose}
                  </span>
                  <span className="ml-auto text-xs whitespace-nowrap text-dim tabular-nums">
                    {spec.bytes > 0 ? formatBytes(spec.bytes) : ""}
                  </span>
                </div>
                <div className="flex flex-wrap items-center gap-2 text-xs">
                  <span className={s.ready ? "text-emerald-300" : "text-dim"}>
                    {spec.bundled
                      ? "Built in"
                      : s.ready
                        ? "Ready"
                        : spec.url
                          ? "Not on this device"
                          : "Not hosted: add its .onnx file"}
                  </span>
                  <span className="text-dim">· {spec.license}</span>
                  <span className="flex-1" />
                  {!spec.bundled && !s.ready && spec.url ? (
                    <button
                      type="button"
                      disabled={busy !== undefined}
                      onClick={() => run(s.id, (p) => downloadModel(s.id, p))}
                      className="rounded bg-accent px-2 py-1 text-ink disabled:opacity-40"
                    >
                      Download
                    </button>
                  ) : null}
                  {!spec.bundled && !s.ready ? (
                    <label className="inline-flex cursor-pointer items-center rounded bg-edge px-2 py-1">
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
                  ) : null}
                  {!spec.bundled && s.ready ? (
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
                {working ? <progress value={busy.fraction} className="h-1 w-full accent-accent" /> : null}
              </li>
            );
          })}
        </ul>
        {error ? (
          <p role="alert" className="mb-3 text-sm text-red-200">
            {error}
          </p>
        ) : null}
        <div className="flex items-center gap-2 text-xs text-dim">
          <span className="flex-1">
            {persisted
              ? "This browser will keep downloaded models and projects."
              : "The browser may clear models and projects when storage runs low."}
          </span>
          <button type="button" onClick={onClose} className="px-3 py-1.5 text-sm">
            {need ? "Cancel" : "Close"}
          </button>
          {need ? (
            <button
              type="button"
              disabled={!(allReady || need.optional) || busy !== undefined}
              onClick={need.onContinue}
              className="rounded bg-accent px-3 py-1.5 text-sm text-ink disabled:opacity-40"
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
