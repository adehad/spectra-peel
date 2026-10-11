import { FLAC_BLOCK, decodeFrame, toFloat } from "@spectra-peel/engine";
import type { Layer, Project } from "../project/schema";
import { fileAt } from "../storage/opfs";

/** One FLAC layer file opened with a sync handle, so workers only. */
class FlacFile {
  private readonly handle: FileSystemSyncAccessHandle;
  private readonly index: Float64Array;

  constructor(handle: FileSystemSyncAccessHandle, index: Float64Array) {
    this.handle = handle;
    this.index = index;
  }

  get blocks(): number {
    return this.index.length - 1;
  }

  decode(block: number, channels: number): Float32Array[] {
    const start = this.index[block] ?? 0;
    const end = this.index[block + 1] ?? start;
    const bytes = new Uint8Array(end - start);
    this.handle.read(bytes, { at: start });
    const frame = decodeFrame(bytes, 0);
    return Array.from({ length: channels }, (_, c) =>
      toFloat(frame.channels[c] ?? frame.channels[0] ?? new Int32Array(frame.blockSize)),
    );
  }

  close(): void {
    this.handle.close();
  }
}

/**
 * Every layer's audio, one 4096-sample block at a time, resolving extents: inside its extent a FLAC layer
 * plays its own file and a residual layer is minuend minus subtrahend; outside, a FLAC layer plays its
 * fallback and a residual layer is silent. Decoded blocks are cached for the current block index only.
 */
export class LayerBlocks {
  readonly project: Project;
  private readonly files: Map<string, FlacFile>;
  private readonly layers: Map<string, Layer>;
  private cacheBlock = -1;
  private readonly cache = new Map<string, Float32Array[]>();

  private constructor(project: Project, files: Map<string, FlacFile>) {
    this.project = project;
    this.files = files;
    this.layers = new Map(project.layers.map((l) => [l.id, l]));
  }

  static async open(
    dir: FileSystemDirectoryHandle,
    project: Project,
    only?: Set<string>,
  ): Promise<LayerBlocks> {
    const files = new Map<string, FlacFile>();
    try {
      for (const layer of project.layers) {
        if (layer.data.kind !== "flac" || (only && !only.has(layer.id))) continue;
        const index = new Float64Array(
          await (await (await fileAt(dir, layer.data.index)).getFile()).arrayBuffer(),
        );
        const handle = await (
          await fileAt(dir, layer.data.path)
        ).createSyncAccessHandle({ mode: "read-only" });
        files.set(layer.id, new FlacFile(handle, index));
      }
    } catch (e) {
      for (const f of files.values()) f.close();
      throw e;
    }
    return new LayerBlocks(project, files);
  }

  get totalBlocks(): number {
    return Math.ceil(this.project.source.frames / FLAC_BLOCK);
  }

  blockLength(block: number): number {
    return Math.max(0, Math.min(FLAC_BLOCK, this.project.source.frames - block * FLAC_BLOCK));
  }

  block(layerId: string, block: number): Float32Array[] {
    if (block !== this.cacheBlock) {
      this.cache.clear();
      this.cacheBlock = block;
    }
    const cached = this.cache.get(layerId);
    if (cached) return cached;
    const out = this.resolve(layerId, block);
    this.cache.set(layerId, out);
    return out;
  }

  close(): void {
    for (const f of this.files.values()) f.close();
  }

  private resolve(layerId: string, block: number): Float32Array[] {
    const channels = this.project.source.channels;
    const length = this.blockLength(block);
    const silence = () => Array.from({ length: channels }, () => new Float32Array(length));
    const layer = this.layers.get(layerId);
    if (!layer) return silence();
    const sample = block * FLAC_BLOCK;
    const inside = sample >= layer.start && sample < layer.start + layer.frames;
    if (layer.data.kind === "residual") {
      if (!inside) return silence();
      const a = this.block(layer.data.minuend, block);
      const b = this.block(layer.data.subtrahend, block);
      return a.map((plane, c) => plane.map((v, i) => v - (b[c]?.[i] ?? 0)));
    }
    if (!inside)
      return layer.data.fallback === undefined ? silence() : this.block(layer.data.fallback, block);
    const file = this.files.get(layerId);
    if (!file) return silence();
    return file.decode((sample - layer.start) / FLAC_BLOCK, channels).map((p) => p.subarray(0, length));
  }
}
