import { SPECTROGRAM } from "@spectra-peel/engine";
import { useState } from "react";
import { EXTRACT_PROMPTS, type ExtractPrompt } from "../models";
import type { Project } from "../project/schema";
import { RunConfirm } from "./RunConfirm";

/**
 * Measured: about 0.53x real time on an Intel iGPU through WebGPU and 1.2x on 8 wasm
 * threads, counting the chunk overlap. One mono pass whatever the channel count.
 */
const RTF = { gpu: 0.5, cpu: 1.2 };

interface Props {
  project: Project;
  input: string;
  onRun: (prompt: ExtractPrompt, useGpu: boolean) => void;
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
      about="Pulls one kind of sound out of the whole file into a layer of its own; Rest holds everything else. The model hears up to 16 kHz, so anything higher stays in Rest. Speech is the most reliable; the others can leave some of the sound behind."
      bytes={bytes}
      seconds={project.source.frames / project.source.sampleRate}
      rtf={RTF}
      runnable
      onRun={(useGpu) => onRun(prompt, useGpu)}
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
      </dd>
    </RunConfirm>
  );
}
