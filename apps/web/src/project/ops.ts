import { ENGINE_VERSION, layerExtent } from "@spectra-peel/engine";
import { MODELS, SPEAKER_MODELS, stemModels, type ModelId } from "../models";
import { paths } from "../storage/opfs";
import type { Layer, Op, Project } from "./schema";

/** Crossfade at each selection edge between the input and the processed signal. */
export const FADE_SECONDS = 0.015;

const shortId = () => crypto.randomUUID().slice(0, 8);

export interface PlannedDenoise {
  op: Op;
  output: Layer;
  removed: Layer;
}

/** The op and layers a denoise of `input` over `range` (source samples) will produce. */
export function planDenoise(
  project: Project,
  input: string,
  range: { start: number; end: number },
): PlannedDenoise {
  const total = project.source.frames;
  const extent = layerExtent(range, total, Math.round(FADE_SECONDS * project.source.sampleRate));
  const id = shortId();
  const output: Layer = {
    id: `${id}-denoised`,
    label: "Denoised",
    producedBy: id,
    start: extent.start,
    frames: extent.end - extent.start,
    data: {
      kind: "flac",
      path: paths.layerFlac(`${id}-denoised`),
      index: paths.layerIndex(`${id}-denoised`),
      fallback: input,
    },
  };
  const removed: Layer = {
    id: `${id}-removed`,
    label: "Removed",
    producedBy: id,
    start: extent.start,
    frames: extent.end - extent.start,
    data: { kind: "residual", minuend: input, subtrahends: [output.id] },
  };
  const model = MODELS["dfn3-stream"];
  const op: Op = {
    id,
    kind: "denoise",
    input,
    range,
    model: { id: "dfn3-stream", sha256: model.sha256 },
    engineVersion: ENGINE_VERSION,
    outputs: [output.id, removed.id],
    createdAt: new Date().toISOString(),
  };
  return { op, output, removed };
}

/** Adds a finished op's layers. Its input goes quiet, as do `muted` outputs, so the mix is the result. */
export function applyOp(project: Project, op: Op, layers: Layer[], muted: string[]): Project {
  const quiet = new Set([...project.mix.muted, op.input, ...muted]);
  return {
    ...project,
    ops: [...project.ops, op],
    layers: [...project.layers, ...layers],
    mix: { ...project.mix, muted: [...quiet], solo: null },
  };
}

export function applyDenoise(project: Project, { op, output, removed }: PlannedDenoise): Project {
  return applyOp(project, op, [output, removed], [removed.id]);
}

export const STEMS = ["Drums", "Bass", "Other", "Vocals"] as const;

interface WholeFileSplit {
  op: Op;
  parts: Layer[];
  /** The input minus every part: what the models left out. */
  rest: Layer;
}

/** An op that splits `input` over the whole file into FLAC parts, plus a Rest layer for what they miss. */
function splitWholeFile(
  project: Project,
  input: string,
  kind: Op["kind"],
  models: readonly ModelId[],
  labels: readonly string[],
): WholeFileSplit {
  const id = shortId();
  const frames = project.source.frames;
  const parts: Layer[] = labels.map((label) => {
    const layer = `${id}-${label.toLowerCase().replaceAll(" ", "")}`;
    return {
      id: layer,
      label,
      producedBy: id,
      start: 0,
      frames,
      data: { kind: "flac", path: paths.layerFlac(layer), index: paths.layerIndex(layer) },
    };
  });
  const rest: Layer = {
    id: `${id}-rest`,
    label: "Rest",
    producedBy: id,
    start: 0,
    frames,
    data: { kind: "residual", minuend: input, subtrahends: parts.map((l) => l.id) },
  };
  const op: Op = {
    id,
    kind,
    input,
    range: { start: 0, end: frames },
    model: models.map((m) => ({ id: m, sha256: MODELS[m].sha256 })),
    engineVersion: ENGINE_VERSION,
    outputs: [...parts.map((l) => l.id), rest.id],
    createdAt: new Date().toISOString(),
  };
  return { op, parts, rest };
}

export interface PlannedSeparate {
  op: Op;
  stems: Layer[];
  rest: Layer;
}

/**
 * Four stems of `input` over the whole file, in HT-Demucs' source order. `best` lists the stems to take
 * from their fine-tuned specialists; the rest come from the fast model.
 */
export function planSeparate(project: Project, input: string, best: readonly number[]): PlannedSeparate {
  const { op, parts, rest } = splitWholeFile(project, input, "separate", stemModels(best), STEMS);
  return { op, stems: parts, rest };
}

export function applySeparate(project: Project, { op, stems, rest }: PlannedSeparate): Project {
  return applyOp(project, op, [...stems, rest], [rest.id]);
}

export interface PlannedSpeakers {
  op: Op;
  people: Layer[];
  rest: Layer;
}

/** One layer per person in `input` over the whole file, in the order their voices are first heard. */
export function planSpeakers(project: Project, input: string, count: number): PlannedSpeakers {
  const labels = Array.from({ length: count }, (_, i) => `Person ${i + 1}`);
  const { op, parts, rest } = splitWholeFile(project, input, "speakers", SPEAKER_MODELS, labels);
  return { op, people: parts, rest };
}

export function applySpeakers(project: Project, { op, people, rest }: PlannedSpeakers): Project {
  return applyOp(project, op, [...people, rest], [rest.id]);
}

export interface PlannedExtract {
  op: Op;
  target: Layer;
  rest: Layer;
}

/** `prompt` pulled out of `input` over the whole file, and Rest for everything else. */
export function planExtract(project: Project, input: string, prompt: string): PlannedExtract {
  const { op, parts, rest } = splitWholeFile(project, input, "extract", ["audiosep-base"], [prompt]);
  const [target] = parts;
  if (!target) throw new Error("extract plans one layer");
  return { op, target, rest };
}

export function applyExtract(project: Project, { op, target, rest }: PlannedExtract): Project {
  return applyOp(project, op, [target, rest], [rest.id]);
}

/** Removes the last op and its layers; returns the project and the files to delete. */
export function undoLast(project: Project): { project: Project; files: string[] } {
  const op = project.ops.at(-1);
  if (!op) return { project, files: [] };
  const gone = new Set(op.outputs);
  const files: string[] = [];
  for (const layer of project.layers) {
    if (!gone.has(layer.id)) continue;
    if (layer.data.kind === "flac") files.push(layer.data.path, layer.data.index);
    files.push(paths.peaks(layer.id), `derived/tiles/${layer.id}`);
  }
  const stillUsed = project.ops.slice(0, -1).some((o) => o.input === op.input);
  const muted = project.mix.muted.filter((id) => !gone.has(id) && (stillUsed || id !== op.input));
  const gains = Object.fromEntries(Object.entries(project.mix.gains).filter(([id]) => !gone.has(id)));
  return {
    project: {
      ...project,
      ops: project.ops.slice(0, -1),
      layers: project.layers.filter((l) => !gone.has(l.id)),
      mix: {
        ...project.mix,
        muted,
        gains,
        solo: project.mix.solo !== null && gone.has(project.mix.solo) ? null : project.mix.solo,
      },
    },
    files,
  };
}
