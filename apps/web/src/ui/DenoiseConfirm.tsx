import { SPECTROGRAM, type SampleRange } from "@spectra-peel/engine";
import type { Project } from "../project/schema";
import { RunConfirm } from "./RunConfirm";

/** Measured in the app: 0.14x real time for both channels of a 44.1 kHz file, on 4 wasm threads. */
const RTF = { gpu: 0.14, cpu: 0.14 };

interface Props {
  project: Project;
  input: string;
  range: SampleRange;
  onRun: (estimateSeconds: number) => void;
  onClose: () => void;
}

export function DenoiseConfirm({ project, input, range, onRun, onClose }: Props) {
  const frames = range.end - range.start;
  const whole = frames === project.source.frames;
  const bytes =
    frames * project.source.channels * 2 * 0.5 + 2 * (frames / SPECTROGRAM.hop) * SPECTROGRAM.bands;
  return (
    <RunConfirm
      name="Denoise"
      title={whole ? `Denoise ${input}` : `Denoise the selection of ${input}`}
      about="Takes background noise out of speech with DeepFilterNet3. What it removes goes to a Removed layer, so you can bring any of it back."
      bytes={bytes}
      seconds={frames / project.source.sampleRate}
      rtf={RTF}
      cpuOnly
      runnable
      onRun={(_, estimate) => onRun(estimate)}
      onClose={onClose}
      runLabel="Denoise"
      runTestId="denoise-run"
    />
  );
}
