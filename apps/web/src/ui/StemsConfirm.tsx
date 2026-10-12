import { SPECTROGRAM } from "@spectra-peel/engine";
import { useState } from "react";
import { stemModels } from "../models";
import { STEMS } from "../project/ops";
import type { Project } from "../project/schema";
import { RunConfirm } from "./RunConfirm";

/** Real-time factors measured on a desktop RTX-class GPU and 8 CPU threads; phones are likely slower. */
const RTF = { gpu: 0.8, cpu: 1.6 };

interface Props {
  project: Project;
  input: string;
  /** Whether the fast model is stored, and each stem's specialist, in stem order. */
  ready: StemsReady;
  onRun: (best: number[], useGpu: boolean, estimateSeconds: number) => void;
  onOpenModels: () => void;
  onClose: () => void;
}

export interface StemsReady {
  fast: boolean;
  specialists: boolean[];
}

export function StemsConfirm({ project, input, ready, onRun, onOpenModels, onClose }: Props) {
  // Without the fast model every stem must come from its specialist.
  const [best, setBest] = useState<number[]>(() => (ready.fast ? [] : STEMS.map((_, i) => i)));
  const passes = stemModels(best).length;
  const runnable = (ready.fast || best.length === STEMS.length) && best.every((i) => ready.specialists[i]);
  // FLAC at about 60% of 16-bit PCM for music, plus a spectrogram per stem and for the rest layer.
  const bytes =
    4 * project.source.frames * project.source.channels * 2 * 0.6 +
    5 * (project.source.frames / SPECTROGRAM.hop) * SPECTROGRAM.bands;

  return (
    <RunConfirm
      name="Separate stems"
      title={`Separate ${input} into stems`}
      about="HT-Demucs splits the whole file into drums, bass, other and vocals. A fifth layer, Rest, holds whatever the four stems miss."
      bytes={bytes}
      seconds={(project.source.frames / project.source.sampleRate) * passes}
      rtf={RTF}
      runnable={runnable}
      onRun={(useGpu, estimate) => onRun(best, useGpu, estimate)}
      onClose={onClose}
      runLabel="Separate"
      runTestId="stems-run"
    >
      <dt className="text-dim">Best quality</dt>
      <dd className="flex flex-col gap-1">
        <span className="flex flex-wrap gap-x-3 gap-y-1">
          {STEMS.map((label, i) => (
            <label
              key={label}
              className={`flex items-center gap-1.5 ${ready.specialists[i] ? "" : "opacity-40"}`}
            >
              <input
                type="checkbox"
                checked={best.includes(i)}
                disabled={!ready.specialists[i] || (!ready.fast && best.includes(i))}
                onChange={(e) =>
                  setBest(
                    e.target.checked ? [...best, i].toSorted((a, b) => a - b) : best.filter((s) => s !== i),
                  )
                }
                className="accent-accent"
                data-testid={`best-${label.toLowerCase()}`}
              />
              {label}
            </label>
          ))}
        </span>
        <p className="text-xs text-dim">
          Each ticked stem runs its own fine-tuned model, one more pass over the file.
          {ready.specialists.every(Boolean) ? null : (
            <>
              {" "}
              <button type="button" onClick={onOpenModels} className="text-accent underline">
                Add the models
              </button>{" "}
              to tick more.
            </>
          )}
        </p>
      </dd>
    </RunConfirm>
  );
}
