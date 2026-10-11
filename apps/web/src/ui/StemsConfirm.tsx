import { SPECTROGRAM } from "@spectra-peel/engine";
import { useEffect, useState } from "react";
import { describeAdapter, isIntegrated, requestGpu } from "../audio/gpu";
import type { Project } from "../project/schema";
import { formatBytes, formatTime } from "./format";

/** Real-time factors measured on a desktop RTX-class GPU and 8 CPU threads; phones are likely slower. */
const RTF = { gpu: 0.8, cpu: 1.6 };

interface Props {
  project: Project;
  input: string;
  onRun: (useGpu: boolean) => void;
  onClose: () => void;
}

interface Gpu {
  label: string;
  integrated: boolean;
}

/** What separating will cost before it starts: disk for four FLAC stems, where it runs and a rough time. */
export function StemsConfirm({ project, input, onRun, onClose }: Props) {
  const [free, setFree] = useState<number>();
  const [gpu, setGpu] = useState<Gpu | null>();
  const [useGpu, setUseGpu] = useState(false);
  const threads = Math.max(1, Math.min(navigator.hardwareConcurrency, 8));
  const seconds = project.source.frames / project.source.sampleRate;
  // FLAC at about 60% of 16-bit PCM for music, plus a spectrogram per stem and for the rest layer.
  const bytes =
    4 * project.source.frames * project.source.channels * 2 * 0.6 +
    5 * (project.source.frames / SPECTROGRAM.hop) * SPECTROGRAM.bands;

  useEffect(() => {
    requestGpu().then(
      (adapter) => {
        const found = adapter ? { label: describeAdapter(adapter), integrated: isIntegrated(adapter) } : null;
        setGpu(found);
        setUseGpu(found !== null && !found.integrated);
        return undefined;
      },
      () => setGpu(null),
    );
    navigator.storage.estimate().then(
      ({ quota = Infinity, usage = 0 }) => {
        setFree(quota - usage);
        return undefined;
      },
      () => undefined,
    );
  }, []);

  const tooBig = free !== undefined && bytes > free;
  let time: string;
  if (useGpu && gpu?.integrated)
    time = `roughly ${formatTime(seconds * RTF.cpu)}; Intel graphics ran about as fast as the CPU`;
  else {
    const where = useGpu ? "GPU" : "CPU";
    time = `roughly ${formatTime(seconds * (useGpu ? RTF.gpu : RTF.cpu))} on a desktop ${where}; a phone takes longer`;
  }

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
          <dt className="text-dim">Run on</dt>
          <dd className="flex flex-col gap-1">
            <label className={`flex items-center gap-2 ${gpu ? "" : "opacity-40"}`}>
              <input
                type="radio"
                checked={useGpu}
                disabled={!gpu}
                onChange={() => setUseGpu(true)}
                className="accent-accent"
              />
              GPU{gpu ? ` · ${gpu.label}` : gpu === null ? " · none available" : ""}
            </label>
            <label className="flex items-center gap-2">
              <input
                type="radio"
                checked={!useGpu}
                onChange={() => setUseGpu(false)}
                className="accent-accent"
              />
              CPU · {threads} threads
            </label>
          </dd>
          <dt className="text-dim">Time</dt>
          <dd>{time}</dd>
        </dl>
        {gpu?.integrated ? (
          <p className="mb-3 text-xs text-dim">
            The browser was given the integrated GPU. On Windows, set the browser to High performance under
            Settings, System, Display, Graphics to use a discrete GPU, then restart it.
          </p>
        ) : null}
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
            disabled={tooBig || gpu === undefined}
            onClick={() => onRun(useGpu)}
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
