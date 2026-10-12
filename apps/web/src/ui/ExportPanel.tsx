import { useEffect, useState } from "react";
import { useJobs } from "../jobs";
import type { Project } from "../project/schema";
import { fileAt } from "../storage/opfs";
import type { ExportFormat, ExportMessage, ExportRequest, LoudnessReport } from "../workers/protocol";
import { formatBytes, formatTime } from "./format";

const LABELS: Record<ExportFormat, string> = {
  wav: "WAV · 24-bit",
  flac: "FLAC · lossless 16-bit",
  m4a: "M4A · AAC",
  opus: "Opus · Ogg",
};

/** Integrated loudness targets in LUFS. Mono reads 3 LU below the same audio in stereo under BS.1770. */
const TARGETS = [
  { lufs: -14, label: "-14 LUFS · music streaming" },
  { lufs: -16, label: "-16 LUFS · stereo speech, podcasts" },
  { lufs: -19, label: "-19 LUFS · mono speech, podcasts" },
  { lufs: -23, label: "-23 LUFS · EBU R128 broadcast" },
];
const CEILING_DBTP = -1;

function describeLoudness(l: LoudnessReport, target: number): string {
  if (!Number.isFinite(l.measured)) return "Silent, so the level was left alone.";
  const sign = l.gainDb >= 0 ? "+" : "";
  const short = target - (l.measured + l.gainDb);
  const capped =
    short > 0.05
      ? ` Stopped ${short.toFixed(1)} LU short of ${target} to keep peaks at ${CEILING_DBTP} dBTP.`
      : "";
  return `Measured ${l.measured.toFixed(1)} LUFS, peak ${l.truePeak.toFixed(1)} dBTP; applied ${sign}${l.gainDb.toFixed(1)} dB.${capped}`;
}

const LOSSY = new Set(["aac", "opus", "mp3", "vorbis"]);

/**
 * The source's own bitrate when it was lossy, since re-encoding above it only grows the file; otherwise a
 * transparent rate for the format and channel count.
 */
function defaultKbps(project: Project, format: ExportFormat, supported: readonly number[]): number {
  const { codec, bitrate, channels } = project.source;
  let want: number;
  if (codec !== undefined && bitrate !== undefined && LOSSY.has(codec)) want = bitrate / 1000;
  else if (format === "opus") want = channels === 1 ? 64 : 128;
  else want = channels === 1 ? 96 : 192;
  return supported.find((k) => k >= want * 0.95) ?? supported.at(-1) ?? want;
}

interface Props {
  project: Project;
  dir: FileSystemDirectoryHandle;
  /** Seconds, or null for the whole file. */
  selection: { start: number; end: number } | null;
  onClose: () => void;
}

function newWorker(): Worker {
  return new Worker(new URL("../workers/export.worker.ts", import.meta.url), { type: "module" });
}

interface SaveFilePickerWindow {
  showSaveFilePicker?: (options: { suggestedName: string }) => Promise<FileSystemFileHandle>;
}

/** Streams the export out of OPFS into a file the user picks, so it never sits in memory. */
async function saveAs(file: File, name: string): Promise<boolean> {
  const picker = (window as Window & SaveFilePickerWindow).showSaveFilePicker;
  if (!picker) return false;
  try {
    const handle = await picker({ suggestedName: name });
    await file.stream().pipeTo(await handle.createWritable());
    return true;
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") return false;
    throw e;
  }
}

function download(file: File, name: string): void {
  const url = URL.createObjectURL(file);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}

function canShare(file: File): boolean {
  return "canShare" in navigator && navigator.canShare({ files: [file] });
}

/** Exports what you hear: the current mix of audible layers at their gains. */
export function ExportPanel({ project, dir, selection, onClose }: Props) {
  const [formats, setFormats] = useState<ExportFormat[]>(["wav", "flac"]);
  const [supportedKbps, setSupportedKbps] = useState<Partial<Record<ExportFormat, number[]>>>({});
  const [format, setFormat] = useState<ExportFormat>("flac");
  const [scope, setScope] = useState<"all" | "selection">(selection ? "selection" : "all");
  const [kbps, setKbps] = useState<number>();
  const [progress, setProgress] = useState<number>();
  const [result, setResult] = useState<{ file: File; label: string; loudness: string | null }>();
  const [target, setTarget] = useState<number | null>(null);
  const [error, setError] = useState<string>();
  const rate = project.source.sampleRate;

  useEffect(() => {
    const worker = newWorker();
    worker.addEventListener("message", (e: MessageEvent<ExportMessage>) => {
      if (e.data.type === "formats") {
        setFormats(e.data.formats);
        setSupportedKbps(e.data.kbps);
      }
      worker.terminate();
    });
    worker.postMessage({ type: "probe", project });
    return () => worker.terminate();
  }, [project]);

  const kbpsOptions = supportedKbps[format] ?? [];
  const chosenKbps =
    kbps !== undefined && kbpsOptions.includes(kbps) ? kbps : defaultKbps(project, format, kbpsOptions);
  const range =
    scope === "selection" && selection
      ? { start: Math.round(selection.start * rate), end: Math.round(selection.end * rate) }
      : { start: 0, end: project.source.frames };
  const suffix = project.ops.length > 0 ? "-peeled" : "";
  const fileName = `${project.name}${suffix}.${format}`;

  const start = () => {
    setError(undefined);
    setResult(undefined);
    setProgress(0);
    const job = `export-${Date.now()}`;
    useJobs.getState().start(job);
    const worker = newWorker();
    worker.addEventListener("message", (e: MessageEvent<ExportMessage>) => {
      const m = e.data;
      if (m.type === "progress") {
        setProgress(m.fraction);
        return;
      }
      worker.terminate();
      useJobs.getState().end(job);
      setProgress(undefined);
      if (m.type === "error") setError(m.message);
      else if (m.type === "done") {
        fileAt(dir, m.path)
          .then((h) => h.getFile())
          .then((f) => {
            setResult({
              file: new File([f], fileName, { type: f.type }),
              label: `${fileName} · ${formatBytes(m.bytes)}`,
              loudness: m.loudness && target !== null ? describeLoudness(m.loudness, target) : null,
            });
            return undefined;
          })
          .catch((err: unknown) => setError(String(err)));
      }
    });
    const req: ExportRequest = {
      project,
      format,
      range,
      fileName,
      bitrate: chosenKbps * 1000,
      normalize: target === null ? null : { target, ceiling: CEILING_DBTP },
    };
    worker.postMessage(req);
  };

  return (
    <div
      className="fixed inset-0 z-10 grid place-items-end bg-black/50 sm:place-items-center"
      role="presentation"
    >
      <dialog
        open
        aria-label="Export"
        className="static m-0 max-h-[90dvh] w-full max-w-md overflow-y-auto rounded-t-xl text-zinc-100 border border-edge bg-panel p-4 pb-[max(1rem,env(safe-area-inset-bottom))] sm:rounded-xl"
      >
        <h2 className="mb-3 font-medium">Export the mix</h2>
        <fieldset className="mb-3 flex flex-col gap-1.5 text-sm">
          {(["wav", "flac", "m4a", "opus"] as const).map((f) => (
            <label key={f} className={`flex items-center gap-2 ${formats.includes(f) ? "" : "opacity-40"}`}>
              <input
                type="radio"
                name="format"
                value={f}
                checked={format === f}
                disabled={!formats.includes(f)}
                onChange={() => setFormat(f)}
                className="accent-accent"
              />
              {LABELS[f]}
              {formats.includes(f) ? null : (
                <span className="text-xs text-dim">not supported in this browser</span>
              )}
            </label>
          ))}
        </fieldset>
        {format === "m4a" || format === "opus" ? (
          <label className="mb-3 flex flex-wrap items-center gap-2 text-sm">
            Bitrate
            <select
              value={chosenKbps}
              onChange={(e) => setKbps(Number(e.target.value))}
              className="rounded bg-edge px-1 py-0.5"
              data-testid="export-bitrate"
            >
              {kbpsOptions.map((k) => (
                <option key={k} value={k}>
                  {k} kbps
                </option>
              ))}
            </select>
            {project.source.bitrate !== undefined && project.source.codec !== undefined ? (
              <span className="text-xs text-dim">
                Source: {project.source.codec.toUpperCase()} {Math.round(project.source.bitrate / 1000)} kbps.
                Going higher only makes the file bigger.
              </span>
            ) : null}
          </label>
        ) : null}
        <label className="mb-3 flex flex-wrap items-center gap-2 text-sm">
          Loudness
          <select
            value={target ?? "off"}
            onChange={(e) => setTarget(e.target.value === "off" ? null : Number(e.target.value))}
            className="rounded bg-edge px-1 py-0.5"
            data-testid="export-loudness"
          >
            <option value="off">Keep the mix's level</option>
            {TARGETS.map((t) => (
              <option key={t.lufs} value={t.lufs}>
                {t.label}
              </option>
            ))}
          </select>
          {target === null ? null : (
            <span className="text-xs text-dim">
              BS.1770 integrated loudness, one gain for the whole export, peaks kept at or below{" "}
              {CEILING_DBTP} dBTP.
            </span>
          )}
        </label>
        <fieldset className="mb-3 flex gap-4 text-sm">
          <label className="flex items-center gap-2">
            <input
              type="radio"
              checked={scope === "all"}
              onChange={() => setScope("all")}
              className="accent-accent"
            />
            Whole file
          </label>
          <label className={`flex items-center gap-2 ${selection ? "" : "opacity-40"}`}>
            <input
              type="radio"
              checked={scope === "selection"}
              disabled={!selection}
              onChange={() => setScope("selection")}
              className="accent-accent"
            />
            Selection
            {selection ? (
              <span className="font-mono text-xs text-dim">
                {formatTime(selection.start)}–{formatTime(selection.end)}
              </span>
            ) : null}
          </label>
        </fieldset>
        <p className="mb-3 text-xs text-dim">
          Exports the layers you can hear at their current gains.{" "}
          {project.mix.ab === "input" ? "Hear input is on, so this is the untouched source." : ""}
        </p>
        {progress === undefined ? null : (
          <progress value={progress} className="mb-3 h-1.5 w-full accent-accent" />
        )}
        {result ? (
          <div className="mb-3 flex flex-wrap items-center gap-2 text-sm" data-testid="export-result">
            <span className="mr-auto">
              {result.label}
              {result.loudness ? <span className="block text-xs text-dim">{result.loudness}</span> : null}
            </span>
            {"showSaveFilePicker" in window ? (
              <button
                type="button"
                className="rounded bg-edge px-2 py-1"
                onClick={() => {
                  saveAs(result.file, result.file.name).catch((err: unknown) => setError(String(err)));
                }}
              >
                Save as…
              </button>
            ) : null}
            <button
              type="button"
              className="rounded bg-edge px-2 py-1"
              onClick={() => download(result.file, result.file.name)}
              data-testid="export-download"
            >
              Download
            </button>
            {canShare(result.file) ? (
              <button
                type="button"
                className="rounded bg-edge px-2 py-1"
                onClick={() => {
                  navigator.share({ files: [result.file], title: result.file.name }).catch((err: unknown) => {
                    if (!(err instanceof DOMException && err.name === "AbortError")) setError(String(err));
                  });
                }}
              >
                Share
              </button>
            ) : null}
          </div>
        ) : null}
        {error ? (
          <p role="alert" className="mb-3 text-sm text-red-200">
            {error}
          </p>
        ) : null}
        <div className="flex justify-end gap-2">
          <button type="button" onClick={onClose} className="px-3 py-1.5 text-sm text-dim">
            Close
          </button>
          <button
            type="button"
            disabled={progress !== undefined}
            onClick={start}
            className="rounded bg-accent px-3 py-1.5 text-sm text-ink disabled:opacity-40"
            data-testid="export-start"
          >
            Export {format.toUpperCase()}
          </button>
        </div>
      </dialog>
    </div>
  );
}
