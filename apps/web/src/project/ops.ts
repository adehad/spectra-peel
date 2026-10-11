import { ENGINE_VERSION, layerExtent } from "@spectra-peel/engine";
import { MODELS } from "../models";
import { paths } from "../storage/opfs";
import { SOURCE_LAYER, type Layer, type Op, type Project } from "./schema";

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
    data: { kind: "residual", minuend: input, subtrahend: output.id },
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

/** Adds a finished op. The input goes quiet and the removed part starts muted, so the mix is the result. */
export function applyDenoise(project: Project, { op, output, removed }: PlannedDenoise): Project {
  const muted = new Set(project.mix.muted);
  muted.add(op.input);
  muted.add(removed.id);
  return {
    ...project,
    ops: [...project.ops, op],
    layers: [...project.layers, output, removed],
    mix: { ...project.mix, muted: [...muted], solo: null },
  };
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

/** The layer a new op should process: the newest FLAC layer, so ops chain. */
export function currentLayer(project: Project): string {
  return project.layers.findLast((l) => l.data.kind === "flac")?.id ?? SOURCE_LAYER;
}
