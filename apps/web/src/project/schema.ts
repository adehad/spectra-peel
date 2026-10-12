import { z } from "zod";

const Samples = z.number().int().nonnegative();

/**
 * A layer differs from its fallback only over [start, start + frames), which is aligned to FLAC frames.
 * Outside that extent a FLAC layer plays its fallback layer and a residual layer is silent.
 */
export const LayerSchema = z.object({
  id: z.string(),
  label: z.string(),
  producedBy: z.string().optional(),
  start: Samples,
  frames: Samples,
  data: z.discriminatedUnion("kind", [
    z.object({
      kind: z.literal("flac"),
      path: z.string(),
      index: z.string(),
      fallback: z.string().optional(),
    }),
    z.object({ kind: z.literal("residual"), minuend: z.string(), subtrahends: z.array(z.string()) }),
  ]),
});

const ModelRef = z.object({ id: z.string(), sha256: z.string() });

export const OpSchema = z.object({
  id: z.string(),
  kind: z.enum(["denoise", "separate", "speakers", "extract"]),
  input: z.string(),
  /** The selection the user asked for, in source samples, before padding and crossfades. */
  range: z.object({ start: Samples, end: Samples }),
  /** The model it ran, or the models it ran in turn. */
  model: ModelRef.or(z.array(ModelRef)),
  engineVersion: z.string(),
  outputs: z.array(z.string()),
  createdAt: z.string(),
});

export const MixSchema = z.object({
  /** dB per layer; missing means 0 dB. */
  gains: z.record(z.string(), z.number()),
  muted: z.array(z.string()),
  solo: z.string().nullable(),
  /** "input" plays only the source, for comparing against the mix. */
  ab: z.enum(["mix", "input"]),
});

export const ProjectSchema = z.object({
  version: z.literal(1),
  id: z.string(),
  name: z.string(),
  createdAt: z.string(),
  engineVersion: z.string(),
  source: z.object({
    fileName: z.string(),
    sampleRate: z.number().int().positive(),
    channels: z.union([z.literal(1), z.literal(2)]),
    frames: Samples,
    /** The imported file's codec and average bitrate, so lossy exports can match them. */
    codec: z.string().optional(),
    bitrate: z.number().positive().optional(),
  }),
  layers: z.array(LayerSchema),
  ops: z.array(OpSchema),
  mix: MixSchema,
});

export type Layer = z.infer<typeof LayerSchema>;
export type Op = z.infer<typeof OpSchema>;
export type Mix = z.infer<typeof MixSchema>;
export type Project = z.infer<typeof ProjectSchema>;
export const SOURCE_LAYER = "source";

/** Linear gain each layer plays at under `mix`. */
export function layerGains(project: Project): Map<string, number> {
  const { mix } = project;
  const out = new Map<string, number>();
  for (const layer of project.layers) {
    let on = !mix.muted.includes(layer.id);
    if (mix.solo !== null) on = layer.id === mix.solo;
    if (mix.ab === "input") on = layer.id === SOURCE_LAYER;
    out.set(layer.id, on ? 10 ** ((mix.gains[layer.id] ?? 0) / 20) : 0);
  }
  return out;
}
