import { ProjectSchema, type Project } from "../project/schema";

/**
 * OPFS layout and the main-thread operations on it. Workers own the files they write through sync access
 * handles, which are exclusive, so the main thread only reads finished files. See docs/opfs-ownership.md.
 */
export const paths = {
  project: "project.json",
  sourceFlac: "source.flac",
  sourceIndex: "source.idx",
  layerFlac: (layer: string) => `layers/${layer}.flac`,
  layerIndex: (layer: string) => `layers/${layer}.idx`,
  peaks: (layer: string) => `derived/peaks/${layer}.bin`,
  tile: (layer: string, n: number) => `derived/tiles/${layer}/${n}.bin`,
};

export async function projectsDir(): Promise<FileSystemDirectoryHandle> {
  const root = await navigator.storage.getDirectory();
  return root.getDirectoryHandle("projects", { create: true });
}

export async function projectDir(id: string, create = false): Promise<FileSystemDirectoryHandle> {
  return (await projectsDir()).getDirectoryHandle(id, { create });
}

/** Resolves a slash-separated path under `dir`, creating directories when asked. */
export async function fileAt(
  dir: FileSystemDirectoryHandle,
  path: string,
  create = false,
): Promise<FileSystemFileHandle> {
  const parts = path.split("/");
  const name = parts.pop() ?? path;
  let d = dir;
  for (const p of parts) d = await d.getDirectoryHandle(p, { create });
  return d.getFileHandle(name, { create });
}

export async function readBytes(dir: FileSystemDirectoryHandle, path: string): Promise<ArrayBuffer> {
  return (await (await fileAt(dir, path)).getFile()).arrayBuffer();
}

/** Main thread only: project.json is never held by a sync handle, so a writable stream is safe. */
export async function writeProject(dir: FileSystemDirectoryHandle, project: Project): Promise<void> {
  const writable = await (await fileAt(dir, paths.project, true)).createWritable();
  await writable.write(JSON.stringify(ProjectSchema.parse(project), null, 2));
  await writable.close();
}

/** Deletes a file if it exists, creating no directories on the way. */
export async function removeFile(dir: FileSystemDirectoryHandle, path: string): Promise<void> {
  const parts = path.split("/");
  const name = parts.pop() ?? path;
  try {
    let d = dir;
    for (const p of parts) d = await d.getDirectoryHandle(p);
    await d.removeEntry(name, { recursive: true });
  } catch (e) {
    if (!(e instanceof DOMException && e.name === "NotFoundError")) throw e;
  }
}

export async function readProject(dir: FileSystemDirectoryHandle): Promise<Project> {
  const text = await (await (await fileAt(dir, paths.project)).getFile()).text();
  return ProjectSchema.parse(JSON.parse(text));
}

async function sizeOf(dir: FileSystemDirectoryHandle): Promise<number> {
  let total = 0;
  for await (const entry of dir.values()) {
    total += entry.kind === "file" ? (await entry.getFile()).size : await sizeOf(entry);
  }
  return total;
}

export interface ProjectSummary {
  project: Project;
  bytes: number;
}

export interface ProjectListing {
  projects: ProjectSummary[];
  /** Folders without a valid project.json: imports that are running or were interrupted. */
  incomplete: { id: string; bytes: number }[];
}

export async function listProjects(): Promise<ProjectListing> {
  const listing: ProjectListing = { projects: [], incomplete: [] };
  for await (const entry of (await projectsDir()).values()) {
    if (entry.kind !== "directory") continue;
    const bytes = await sizeOf(entry);
    try {
      listing.projects.push({ project: await readProject(entry), bytes });
    } catch {
      listing.incomplete.push({ id: entry.name, bytes });
    }
  }
  listing.projects.sort((a, b) => b.project.createdAt.localeCompare(a.project.createdAt));
  return listing;
}

export async function deleteProject(id: string): Promise<void> {
  await (await projectsDir()).removeEntry(id, { recursive: true });
}
