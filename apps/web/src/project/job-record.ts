import { z } from "zod";
import { LayerSchema, OpSchema, ProjectSchema, type Layer } from "./schema";

/** A running op as sent to its worker, saved so a reload can offer to resume it. */
export const JobRecordSchema = z.discriminatedUnion("kind", [
  z.object({
    kind: z.literal("denoise"),
    request: z.object({ project: ProjectSchema, op: OpSchema, output: LayerSchema, removed: LayerSchema }),
  }),
  z.object({
    kind: z.literal("separate"),
    request: z.object({
      project: ProjectSchema,
      op: OpSchema,
      stems: z.array(LayerSchema),
      rest: LayerSchema,
      best: z.array(z.number().int()),
      useGpu: z.boolean(),
    }),
  }),
  z.object({
    kind: z.literal("speakers"),
    request: z.object({
      project: ProjectSchema,
      op: OpSchema,
      people: z.array(LayerSchema),
      rest: LayerSchema,
      useGpu: z.boolean(),
    }),
  }),
]);

export type JobRecord = z.infer<typeof JobRecordSchema>;

export const JOB_TITLES: Record<JobRecord["kind"], string> = {
  denoise: "Denoising",
  separate: "Separating stems",
  speakers: "Separating people",
};

/** The layers the job writes, in the order its op lists them. */
export function jobLayers(record: JobRecord): Layer[] {
  if (record.kind === "denoise") return [record.request.output, record.request.removed];
  if (record.kind === "separate") return [...record.request.stems, record.request.rest];
  return [...record.request.people, record.request.rest];
}
