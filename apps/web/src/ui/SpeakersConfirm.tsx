import { SPECTROGRAM } from "@spectra-peel/engine";
import { useState } from "react";
import type { Project } from "../project/schema";
import { RunConfirm } from "./RunConfirm";

/**
 * Measured whole-pipeline real-time factors: the separator only runs on crosstalk, a
 * third to a half of the audio in conversation, at 0.23x on an RTX-class GPU and 2.6x on 8 CPU threads.
 */
const RTF = { gpu: 0.2, cpu: 1.3 };

interface Props {
  project: Project;
  input: string;
  onRun: (people: number, useGpu: boolean) => void;
  onClose: () => void;
}

export function SpeakersConfirm({ project, input, onRun, onClose }: Props) {
  const [people, setPeople] = useState(2);
  const bytes =
    people * project.source.frames * project.source.channels * 2 * 0.5 +
    (people + 1) * (project.source.frames / SPECTROGRAM.hop) * SPECTROGRAM.bands;

  return (
    <RunConfirm
      name="Separate people"
      title={`Separate the people in ${input}`}
      about="Gives each person a layer of their own. Where one person talks alone their layer is the original audio; where people talk over each other it is separated, at phone-call bandwidth, and some of the other voice remains. Rest holds pauses and background."
      bytes={bytes}
      seconds={project.source.frames / project.source.sampleRate}
      rtf={RTF}
      runnable
      onRun={(useGpu) => onRun(people, useGpu)}
      onClose={onClose}
      runLabel="Separate"
      runTestId="speakers-run"
    >
      <dt className="text-dim">People</dt>
      <dd className="flex gap-3">
        {[2, 3].map((n) => (
          <label key={n} className="flex items-center gap-1.5">
            <input
              type="radio"
              checked={people === n}
              onChange={() => setPeople(n)}
              className="accent-accent"
              data-testid={`people-${n}`}
            />
            {n}
          </label>
        ))}
      </dd>
    </RunConfirm>
  );
}
