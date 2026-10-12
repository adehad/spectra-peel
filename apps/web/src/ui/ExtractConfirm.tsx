import { SPECTROGRAM } from "@spectra-peel/engine";
import { useState } from "react";
import { EXTRACT_PROMPTS, type ExtractPrompt } from "../models";
import type { Project } from "../project/schema";
import { RunConfirm } from "./RunConfirm";

/**
 * Measured: about 0.53x real time on an Intel iGPU through WebGPU and 1.2x on 8 wasm
 * threads, counting the chunk overlap; 0.8x on a Pixel 10a's GPU in the app. One mono pass whatever
 * the channel count.
 */
const RTF = { gpu: 0.5, cpu: 1.2, phone: 0.8 };

interface Props {
  project: Project;
  input: string;
  onRun: (prompt: ExtractPrompt, useGpu: boolean, estimateSeconds: number) => void;
  onClose: () => void;
}

export function ExtractConfirm({ project, input, onRun, onClose }: Props) {
  const [prompt, setPrompt] = useState<ExtractPrompt>("Speech");
  const bytes =
    project.source.frames * project.source.channels * 2 * 0.5 +
    2 * (project.source.frames / SPECTROGRAM.hop) * SPECTROGRAM.bands;

  return (
    <RunConfirm
      name="Extract a sound"
      title={`Extract a sound from ${input}`}
      about="Pulls one kind of sound out of the whole file into its own layer, and everything else into Rest."
      bytes={bytes}
      seconds={project.source.frames / project.source.sampleRate}
      rtf={RTF}
      runnable
      onRun={(useGpu, estimate) => onRun(prompt, useGpu, estimate)}
      onClose={onClose}
      runLabel="Extract"
      runTestId="extract-run"
    >
      <dt className="text-dim">Sound</dt>
      <dd className="flex flex-wrap gap-x-3">
        {EXTRACT_PROMPTS.map((p) => (
          <label key={p} className="flex items-center gap-1.5">
            <input
              type="radio"
              checked={prompt === p}
              onChange={() => setPrompt(p)}
              className="accent-accent"
              data-testid={`prompt-${p.toLowerCase().replaceAll(" ", "-")}`}
            />
            {p}
          </label>
        ))}
        <p className="basis-full text-xs text-dim">
          Speech is the most reliable. The model hears up to 16 kHz, so anything higher stays in Rest.
        </p>
      </dd>
    </RunConfirm>
  );
}
