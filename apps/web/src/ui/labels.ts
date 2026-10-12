import type { Op, Project } from "../project/schema";

export const OP_NAMES: Record<Op["kind"], string> = {
  denoise: "Denoise",
  separate: "Stems",
  speakers: "People",
  extract: "Extract",
};

/** "Denoise of Source": what an op did and to which layer. */
export function opTitle(project: Project, op: Op): string {
  return `${OP_NAMES[op.kind]} of ${project.layers.find((l) => l.id === op.input)?.label ?? "a removed layer"}`;
}
