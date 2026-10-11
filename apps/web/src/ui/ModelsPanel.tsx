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

/** Models with their size and state; downloads and file imports are checked against a pinned SHA-256. */
export function ModelsPanel({ onClose }: { onClose: () => void }) {
  const [status, setStatus] = useState<ModelStatus[]>([]);
  const [busy, setBusy] = useState<{ id: ModelId; fraction: number }>();
  const [error, setError] = useState<string>();
  const [persisted, setPersisted] = useState<boolean>();

  const refresh = useCallback(() => {
    Promise.all(MODEL_IDS.map((id) => modelStatus(id))).then(
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
  }, []);
  useEffect(refresh, [refresh]);

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
        aria-label="Models"
        className="static m-0 w-full max-w-lg rounded-t-xl border border-edge bg-panel p-4 pb-[max(1rem,env(safe-area-inset-bottom))] text-zinc-100 sm:rounded-xl"
      >
        <h2 className="mb-3 font-medium">Models</h2>
        <ul className="mb-3 divide-y divide-edge">
          {status.map((s) => {
            const spec = MODELS[s.id];
            const working = busy?.id === s.id;
            return (
              <li key={s.id} className="flex flex-col gap-1.5 py-2" data-testid={`model-${s.id}`}>
                <div className="flex items-baseline gap-2">
                  <span className="font-medium">{spec.label}</span>
                  <span className="text-xs text-dim">{spec.purpose}</span>
                  <span className="ml-auto text-xs text-dim tabular-nums">
                    {spec.bytes > 0 ? formatBytes(spec.bytes) : ""}
                  </span>
                </div>
                <div className="flex flex-wrap items-center gap-2 text-xs">
                  <span className={s.ready ? "text-emerald-300" : "text-dim"}>
                    {spec.bundled ? "Built in" : s.ready ? "Ready" : "Not on this device"}
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
                    <label className="cursor-pointer rounded bg-edge px-2 py-1">
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
            Close
          </button>
        </div>
      </dialog>
    </div>
  );
}
