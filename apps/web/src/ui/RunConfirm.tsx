import { useEffect, useState, type ReactNode } from "react";
import { describeAdapter, isIntegrated, requestGpu } from "../audio/gpu";
import { formatBytes, formatTime } from "./format";

interface Gpu {
  label: string;
  integrated: boolean;
}

interface Props {
  /** The dialog's accessible name; the title also names the input. */
  name: string;
  title: string;
  /** What the job does, above the details. */
  about: ReactNode;
  /** Rows before Storage, as <dt>/<dd> pairs. */
  children?: ReactNode;
  /** Disk the job will write. */
  bytes: number;
  /** Seconds of audio the job processes. */
  seconds: number;
  /** Measured real-time factors on a desktop GPU and on 8 CPU threads, and on a phone's GPU if known. */
  rtf: { gpu: number; cpu: number; phone?: number };
  runnable: boolean;
  /** The job always runs on the CPU, so there is no device to choose. */
  cpuOnly?: boolean;
  /** Runs the job on the chosen device, with the time estimate shown here. */
  onRun: (useGpu: boolean, estimateSeconds: number) => void;
  onClose: () => void;
  runLabel: string;
  runTestId: string;
}

const ON_PHONE = /Android|iPhone|iPad/.test(navigator.userAgent);
const ON_WINDOWS = navigator.userAgent.includes("Windows");

/** What a long job will cost before it starts: disk, where it runs and a rough time. */
export function RunConfirm(props: Props) {
  const { title, about, children, bytes, seconds, rtf, runnable, onRun, onClose } = props;
  const [free, setFree] = useState<number>();
  const [gpu, setGpu] = useState<Gpu | null>();
  const [useGpu, setUseGpu] = useState(false);
  const threads = Math.max(1, Math.min(navigator.hardwareConcurrency, 8));

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
  const onGpu = useGpu && !props.cpuOnly;
  const phone = ON_PHONE && onGpu ? rtf.phone : undefined;
  const estimate = seconds * (phone ?? (onGpu && !gpu?.integrated ? rtf.gpu : rtf.cpu));
  let time: string;
  if (phone !== undefined) time = `roughly ${formatTime(estimate)}, as measured on a Pixel 10a`;
  else if (ON_PHONE) time = `longer than the ${formatTime(estimate)} a desktop takes`;
  else if (onGpu && gpu?.integrated)
    time = `roughly ${formatTime(estimate)}; Intel graphics ran about as fast as the CPU`;
  else time = `roughly ${formatTime(estimate)} on a desktop ${onGpu ? "GPU" : "CPU"}`;

  return (
    <div
      className="fixed inset-0 z-10 grid place-items-end bg-black/50 sm:place-items-center"
      role="presentation"
    >
      <dialog
        open
        aria-label={props.name}
        className="static m-0 max-h-[90dvh] w-full max-w-md overflow-y-auto rounded-t-xl border border-edge bg-panel p-4 pb-[max(1rem,env(safe-area-inset-bottom))] text-zinc-100 sm:rounded-xl"
      >
        <h2 className="mb-2 font-medium">{title}</h2>
        <p className="mb-3 text-sm text-dim">{about}</p>
        <dl className="mb-3 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm tabular-nums">
          {children}
          <dt className="text-dim">Storage</dt>
          <dd>
            about {formatBytes(bytes)}
            {free === undefined ? "" : ` of ${formatBytes(free)} free`}
          </dd>
          <dt className={props.cpuOnly ? "hidden" : "text-dim"}>Run on</dt>
          <dd className={props.cpuOnly ? "hidden" : "flex flex-col gap-1"}>
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
        {gpu?.integrated && onGpu && ON_WINDOWS ? (
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
            disabled={tooBig || gpu === undefined || !runnable}
            onClick={() => onRun(useGpu, estimate)}
            className="rounded bg-accent px-3 py-1.5 text-sm text-ink disabled:opacity-40"
            data-testid={props.runTestId}
          >
            {props.runLabel}
          </button>
        </div>
      </dialog>
    </div>
  );
}
