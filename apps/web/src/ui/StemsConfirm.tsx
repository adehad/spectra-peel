import { SPECTROGRAM } from "@spectra-peel/engine";
import { useEffect, useState } from "react";
import type { Project } from "../project/schema";
import { formatBytes, formatTime } from "./format";

/** Real-time factors measured on a desktop RTX-class GPU and 8 CPU threads; phones are likely slower. */
const RTF = { gpu: 0.8, cpu: 1.6 };

interface Props {
  project: Project;
  input: string;
  onRun: () => void;
  onClose: () => void;
}

/** What separating will cost before it starts: disk for four FLAC stems and a rough time. */
export function StemsConfirm({ project, input, onRun, onClose }: Props) {
  const [free, setFree] = useState<number>();
  const seconds = project.source.frames / project.source.sampleRate;
  // FLAC at about 60% of 16-bit PCM for music, plus a spectrogram per stem and for the rest layer.
  const bytes =
    4 * project.source.frames * project.source.channels * 2 * 0.6 +
    5 * (project.source.frames / SPECTROGRAM.hop) * SPECTROGRAM.bands;
  const gpu = "gpu" in navigator;

  useEffect(() => {
    navigator.storage.estimate().then(
      ({ quota = Infinity, usage = 0 }) => {
        setFree(quota - usage);
        return undefined;
      },
      () => undefined,
    );
  }, []);

  const tooBig = free !== undefined && bytes > free;
  return (
    <div
      className="fixed inset-0 z-10 grid place-items-end bg-black/50 sm:place-items-center"
      role="presentation"
    >
      <dialog
        open
        aria-label="Separate stems"
        className="static m-0 w-full max-w-md rounded-t-xl border border-edge bg-panel p-4 pb-[max(1rem,env(safe-area-inset-bottom))] text-zinc-100 sm:rounded-xl"
      >
        <h2 className="mb-2 font-medium">Separate {input} into stems</h2>
        <p className="mb-3 text-sm text-dim">
          HT-Demucs splits the whole file into drums, bass, other and vocals. A fifth layer, Rest, holds
          whatever the four stems miss.
        </p>
        <dl className="mb-3 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm tabular-nums">
          <dt className="text-dim">Storage</dt>
          <dd>
            about {formatBytes(bytes)}
            {free === undefined ? "" : ` of ${formatBytes(free)} free`}
          </dd>
          <dt className="text-dim">Time</dt>
          <dd>
            roughly {formatTime(seconds * (gpu ? RTF.gpu : RTF.cpu))} on a desktop {gpu ? "GPU" : "CPU"}; a
            phone takes longer
          </dd>
        </dl>
        {tooBig ? (
          <p role="alert" className="mb-3 text-sm text-red-200">
            Not enough free storage. Delete a project or a model first.
          </p>
        ) : null}
        <p className="mb-3 text-xs text-dim">Keep this tab open; the screen stays on while it runs.</p>
        <div className="flex justify-end gap-2">
          <button type="button" onClick={onClose} className="px-3 py-1.5 text-sm text-dim">
            Cancel
          </button>
          <button
            type="button"
            disabled={tooBig}
            onClick={onRun}
            className="rounded bg-accent px-3 py-1.5 text-sm text-ink disabled:opacity-40"
            data-testid="stems-run"
          >
            Separate
          </button>
        </div>
      </dialog>
    </div>
  );
}
