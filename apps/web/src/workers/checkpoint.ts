import { paths, readBytes } from "../storage/opfs";
import { writeWhole } from "./layer-writer";

/** Files a worker keeps in its job's folder so a resumed run can skip work it already did. */
export async function saveCheckpoint(
  dir: FileSystemDirectoryHandle,
  op: string,
  name: string,
  data: AllowSharedBufferSource,
): Promise<void> {
  await writeWhole(dir, paths.checkpoint(op, name), data);
}

export async function loadCheckpoint(
  dir: FileSystemDirectoryHandle,
  op: string,
  name: string,
): Promise<ArrayBuffer | null> {
  try {
    return await readBytes(dir, paths.checkpoint(op, name));
  } catch (e) {
    if (e instanceof DOMException && e.name === "NotFoundError") return null;
    throw e;
  }
}

export async function saveJson(
  dir: FileSystemDirectoryHandle,
  op: string,
  name: string,
  value: unknown,
): Promise<void> {
  await saveCheckpoint(dir, op, name, new TextEncoder().encode(JSON.stringify(value)));
}

export async function loadJson(dir: FileSystemDirectoryHandle, op: string, name: string): Promise<unknown> {
  const bytes = await loadCheckpoint(dir, op, name);
  return bytes === null ? null : JSON.parse(new TextDecoder().decode(bytes));
}
