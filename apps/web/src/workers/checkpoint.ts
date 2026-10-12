import { z } from "zod";
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

const align = (n: number) => Math.ceil(n / 8) * 8;

const Header = z.object({
  json: z.string(),
  arrays: z.array(
    z.tuple([
      z.enum(["Float32Array", "Float64Array", "Int16Array", "Int32Array", "Uint8Array", "Uint32Array"]),
      z.number(),
    ]),
  ),
});

const TYPED = { Float32Array, Float64Array, Int16Array, Int32Array, Uint8Array, Uint32Array };

/**
 * Plain data with typed arrays inside, as [header length][JSON header][array bytes, each 8-byte
 * aligned]. JSON alone would turn typed arrays into slow objects and infinities into null.
 */
export function pack(value: unknown): Uint8Array<ArrayBuffer> {
  const arrays: ArrayBufferView[] = [];
  const json = JSON.stringify(value, (_key, v: unknown) => {
    if (ArrayBuffer.isView(v)) {
      arrays.push(v);
      return { $typed: v.constructor.name, $i: arrays.length - 1 };
    }
    if (typeof v === "number" && !Number.isFinite(v)) return { $number: String(v) };
    return v;
  });
  const header = new TextEncoder().encode(
    JSON.stringify({ json, arrays: arrays.map((a) => [a.constructor.name, a.byteLength]) }),
  );
  let size = align(4 + header.length);
  for (const a of arrays) size += align(a.byteLength);
  const out = new Uint8Array(size);
  new DataView(out.buffer).setUint32(0, header.length, true);
  out.set(header, 4);
  let at = align(4 + header.length);
  for (const a of arrays) {
    out.set(new Uint8Array(a.buffer, a.byteOffset, a.byteLength), at);
    at += align(a.byteLength);
  }
  return out;
}

export function unpack(bytes: ArrayBuffer): unknown {
  const length = new DataView(bytes).getUint32(0, true);
  const header = Header.parse(JSON.parse(new TextDecoder().decode(new Uint8Array(bytes, 4, length))));
  let at = align(4 + length);
  const arrays = header.arrays.map(([name, byteLength]) => {
    const Ctor = TYPED[name];
    const a = new Ctor(bytes.slice(at, at + byteLength));
    at += align(byteLength);
    return a;
  });
  return JSON.parse(header.json, (_key, v: unknown) => {
    if (v && typeof v === "object" && "$i" in v && typeof v.$i === "number") return arrays[v.$i];
    if (v && typeof v === "object" && "$number" in v) return Number(v.$number);
    return v;
  });
}

/**
 * Saves a large snapshot into one of two slots, then points at it. A page killed mid-write leaves
 * the pointer on the older, complete slot.
 */
export async function saveSnapshot(
  dir: FileSystemDirectoryHandle,
  op: string,
  name: string,
  value: unknown,
): Promise<void> {
  const current = await loadJson(dir, op, `${name}.json`);
  const slot = current && typeof current === "object" && "slot" in current && current.slot === 0 ? 1 : 0;
  await saveCheckpoint(dir, op, `${name}.${slot}.bin`, pack(value));
  await saveJson(dir, op, `${name}.json`, { slot });
}

export async function loadSnapshot(
  dir: FileSystemDirectoryHandle,
  op: string,
  name: string,
): Promise<unknown> {
  const pointer = await loadJson(dir, op, `${name}.json`);
  if (!pointer || typeof pointer !== "object" || !("slot" in pointer)) return null;
  const bytes = await loadCheckpoint(dir, op, `${name}.${String(pointer.slot)}.bin`);
  return bytes === null ? null : unpack(bytes);
}
