import { Sha256 } from "@spectra-peel/engine";
import { MODELS, type ModelId } from "../models";

async function modelsDir(): Promise<FileSystemDirectoryHandle> {
  return (await navigator.storage.getDirectory()).getDirectoryHandle("models", { create: true });
}

const fileName = (sha256: string) => `${sha256}.onnx`;

export interface ModelStatus {
  id: ModelId;
  ready: boolean;
  bytes: number;
}

export async function modelStatus(id: ModelId): Promise<ModelStatus> {
  const spec = MODELS[id];
  if (spec.bundled) return { id, ready: true, bytes: spec.bytes };
  if (!spec.sha256) return { id, ready: false, bytes: 0 };
  try {
    const file = await (await (await modelsDir()).getFileHandle(fileName(spec.sha256))).getFile();
    return { id, ready: file.size === spec.bytes, bytes: file.size };
  } catch {
    return { id, ready: false, bytes: 0 };
  }
}

/**
 * Streams a model into OPFS while hashing it, and keeps it only if the hash matches. `size` is the
 * expected length, for progress.
 */
async function store(
  id: ModelId,
  stream: ReadableStream<Uint8Array<ArrayBuffer>>,
  onProgress: (fraction: number) => void,
  signal?: AbortSignal,
): Promise<void> {
  const spec = MODELS[id];
  const dir = await modelsDir();
  const name = fileName(spec.sha256);
  const handle = await dir.getFileHandle(`${name}.part`, { create: true });
  const writable = await handle.createWritable();
  const hash = new Sha256();
  let received = 0;
  const reader = stream.getReader();
  try {
    for (;;) {
      if (signal?.aborted) throw new DOMException("Download cancelled", "AbortError");
      const { done, value } = await reader.read();
      if (done) break;
      hash.update(value);
      await writable.write(value);
      received += value.length;
      onProgress(spec.bytes > 0 ? received / spec.bytes : 0);
    }
    await writable.close();
  } catch (e) {
    await writable.abort().catch(() => undefined);
    await dir.removeEntry(`${name}.part`).catch(() => undefined);
    throw e;
  }
  const digest = hash.hex();
  if (digest !== spec.sha256) {
    await dir.removeEntry(`${name}.part`);
    throw new Error(`This file is not ${spec.label}: its SHA-256 is ${digest}, expected ${spec.sha256}.`);
  }
  await handle.move(name);
}

export async function downloadModel(id: ModelId, onProgress: (f: number) => void, signal?: AbortSignal) {
  const { url } = MODELS[id];
  if (!url) throw new Error("This model has no download location yet. Add it from a file.");
  const res = await fetch(url, signal ? { signal } : {});
  if (!res.ok || !res.body) throw new Error(`Download failed: ${res.status} ${res.statusText}`);
  await store(id, res.body, onProgress, signal);
}

export async function addModelFromFile(id: ModelId, file: File, onProgress: (f: number) => void) {
  await store(id, file.stream(), onProgress);
}

export async function deleteModel(id: ModelId): Promise<void> {
  const spec = MODELS[id];
  if (spec.bundled) return;
  await (await modelsDir()).removeEntry(fileName(spec.sha256)).catch(() => undefined);
}

/**
 * The model's bytes, for creating an inference session in a worker. Bundled models are checked here;
 * stored ones were checked when they were stored.
 */
export async function modelBytes(id: ModelId): Promise<Uint8Array> {
  const spec = MODELS[id];
  if (spec.bundled && spec.url) {
    const res = await fetch(spec.url);
    if (!res.ok) throw new Error(`model download failed: ${res.status} ${res.statusText}`);
    const bytes = new Uint8Array(await res.arrayBuffer());
    const digest = new Sha256().update(bytes).hex();
    if (digest !== spec.sha256)
      throw new Error(`model checksum mismatch: expected ${spec.sha256}, got ${digest}`);
    return bytes;
  }
  const file = await (await (await modelsDir()).getFileHandle(fileName(spec.sha256))).getFile();
  return new Uint8Array(await file.arrayBuffer());
}
