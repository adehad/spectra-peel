import { z } from "zod";

export const LayerSchema = z.object({
  id: z.string(),
  label: z.string(),
  data: z.object({ kind: z.literal("flac"), path: z.string(), index: z.string() }),
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
    frames: z.number().int().nonnegative(),
  }),
  layers: z.array(LayerSchema),
  mix: z.object({ gains: z.record(z.string(), z.number()), muted: z.array(z.string()) }),
});

export type Layer = z.infer<typeof LayerSchema>;
export type Project = z.infer<typeof ProjectSchema>;
export const SOURCE_LAYER = "source";
