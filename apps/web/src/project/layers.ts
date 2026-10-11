import type { Project } from "./schema";

/** Layers the job must read: the input and everything it falls back to or subtracts. */
export function dependencies(project: Project, id: string, out = new Set<string>()): Set<string> {
  const layer = project.layers.find((l) => l.id === id);
  if (!layer || out.has(id)) return out;
  out.add(id);
  if (layer.data.kind === "flac" && layer.data.fallback !== undefined)
    dependencies(project, layer.data.fallback, out);
  if (layer.data.kind === "residual") {
    dependencies(project, layer.data.minuend, out);
    for (const s of layer.data.subtrahends) dependencies(project, s, out);
  }
  return out;
}
