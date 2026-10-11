import { useEffect, useState } from "react";
import { registerSW } from "virtual:pwa-register";
import { useJobs } from "../jobs";

/** Offers a new version once it is downloaded; reloading is disabled while a job would be lost. */
export function UpdatePrompt() {
  const [update, setUpdate] = useState<(reload: boolean) => Promise<void>>();
  const [offline, setOffline] = useState(false);
  const busy = useJobs((s) => s.running.size > 0);

  useEffect(() => {
    if (!("serviceWorker" in navigator)) return;
    const updateSW = registerSW({
      onNeedRefresh: () => setUpdate(() => updateSW),
      onOfflineReady: () => setOffline(true),
    });
  }, []);

  if (!update && !offline) return null;
  return (
    <div className="fixed top-12 left-1/2 z-20 flex -translate-x-1/2 items-center gap-3 rounded-lg border border-edge bg-panel px-3 py-2 text-sm shadow-lg">
      {update ? (
        <>
          <span>
            {busy ? "Update ready. It installs once the current job finishes." : "A new version is ready."}
          </span>
          <button
            type="button"
            disabled={busy}
            onClick={() => void update(true)}
            className="rounded bg-accent px-2 py-1 text-ink disabled:opacity-40"
          >
            Reload
          </button>
        </>
      ) : (
        <>
          <span>Ready to work offline.</span>
          <button type="button" onClick={() => setOffline(false)} className="text-dim">
            OK
          </button>
        </>
      )}
    </div>
  );
}
