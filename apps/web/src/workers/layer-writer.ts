import {
  FLAC_HEADER_BYTES,
  FlacWriter,
  PeaksBuilder,
  SpectrogramBuilder,
  Xorshift32,
  quantize16,
  serializePeaks,
  type FlacWriterState,
  type PeaksState,
  type SpectrogramState,
  type SpectrogramTile,
} from "@spectra-peel/engine";
import { fileAt, paths } from "../storage/opfs";

/**
 * Opens a sync access handle, waiting up to about 5 s for another one on the same file to close.
 * A page reloaded mid-job resumes while its old worker's handles are still being released.
 */
async function openForWrite(file: FileSystemFileHandle): Promise<FileSystemSyncAccessHandle> {
  for (let wait = 100; ; wait *= 2) {
    try {
      return await file.createSyncAccessHandle();
    } catch (e) {
      if (!(e instanceof DOMException && e.name === "NoModificationAllowedError") || wait > 3200) throw e;
      await new Promise((r) => setTimeout(r, wait));
    }
  }
}

export async function writeWhole(
  dir: FileSystemDirectoryHandle,
  path: string,
  data: AllowSharedBufferSource,
): Promise<void> {
  const handle = await openForWrite(await fileAt(dir, path, true));
  handle.truncate(0);
  handle.write(data, { at: 0 });
  handle.close();
}

/** Writes a layer's spectrogram tiles as they fill. */
export class TileWriter {
  private readonly builder: SpectrogramBuilder;
  private readonly dir: FileSystemDirectoryHandle;
  private readonly layer: string;

  constructor(dir: FileSystemDirectoryHandle, layer: string, sampleRate: number) {
    this.dir = dir;
    this.layer = layer;
    this.builder = new SpectrogramBuilder(sampleRate);
  }

  /** Mono samples; awaiting each tile write keeps them from queueing up behind OPFS. */
  async push(samples: Float32Array): Promise<void> {
    await this.save(this.builder.push(samples));
  }

  async finish(): Promise<void> {
    await this.save(this.builder.finish());
  }

  snapshot(): SpectrogramState {
    return this.builder.snapshot();
  }

  restore(state: SpectrogramState): void {
    this.builder.restore(state);
  }

  private async save(tiles: SpectrogramTile[]): Promise<void> {
    for (const t of tiles) await writeWhole(this.dir, paths.tile(this.layer, t.index), t.data);
  }
}

export function mono(planes: readonly Float32Array[]): Float32Array {
  const [left, right] = planes;
  if (!left) return new Float32Array(0);
  if (!right) return left;
  return left.map((v, i) => (v + (right[i] ?? 0)) / 2);
}

export interface LayerWriterState {
  /** Bytes of the FLAC file written so far; anything past it is from after the snapshot. */
  at: number;
  flac: FlacWriterState;
  peaks: PeaksState;
  tiles: SpectrogramState;
  rng: number;
}

/** Streams a FLAC layer to OPFS together with its frame index, peaks pyramid and spectrogram tiles. */
export class LayerWriter {
  private readonly dir: FileSystemDirectoryHandle;
  private readonly layer: string;
  private readonly indexPath: string;
  private readonly flac: FileSystemSyncAccessHandle;
  private readonly writer: FlacWriter;
  private readonly peaks = new PeaksBuilder();
  private readonly tiles: TileWriter;
  private readonly rng = new Xorshift32(0x5eed);
  private at = FLAC_HEADER_BYTES;

  private constructor(
    dir: FileSystemDirectoryHandle,
    layer: string,
    indexPath: string,
    flac: FileSystemSyncAccessHandle,
    sampleRate: number,
    channels: 1 | 2,
  ) {
    this.dir = dir;
    this.layer = layer;
    this.indexPath = indexPath;
    this.flac = flac;
    this.writer = new FlacWriter({ sampleRate, channels });
    this.tiles = new TileWriter(dir, layer, sampleRate);
  }

  static async create(
    dir: FileSystemDirectoryHandle,
    layer: string,
    flacPath: string,
    indexPath: string,
    sampleRate: number,
    channels: 1 | 2,
  ): Promise<LayerWriter> {
    const flac = await openForWrite(await fileAt(dir, flacPath, true));
    flac.truncate(0);
    flac.write(new Uint8Array(FLAC_HEADER_BYTES), { at: 0 });
    return new LayerWriter(dir, layer, indexPath, flac, sampleRate, channels);
  }

  /** Opens a layer an earlier run was writing and puts it back where `state` was taken. */
  static async resume(
    dir: FileSystemDirectoryHandle,
    layer: string,
    flacPath: string,
    indexPath: string,
    sampleRate: number,
    channels: 1 | 2,
    state: LayerWriterState,
  ): Promise<LayerWriter> {
    const flac = await openForWrite(await fileAt(dir, flacPath));
    flac.truncate(state.at);
    const writer = new LayerWriter(dir, layer, indexPath, flac, sampleRate, channels);
    writer.at = state.at;
    writer.writer.restore(state.flac);
    writer.peaks.restore(state.peaks);
    writer.tiles.restore(state.tiles);
    writer.rng.restore(state.rng);
    return writer;
  }

  /** Flushes the FLAC written so far and returns where the layer is, for `resume`. */
  snapshot(): LayerWriterState {
    this.flac.flush();
    return {
      at: this.at,
      flac: this.writer.snapshot(),
      peaks: this.peaks.snapshot(),
      tiles: this.tiles.snapshot(),
      rng: this.rng.snapshot(),
    };
  }

  get frames(): number {
    return this.writer.totalSamples;
  }

  async push(planes: readonly Float32Array[]): Promise<void> {
    for (const frame of this.writer.push(planes.map((p) => quantize16(p, this.rng)))) this.write(frame);
    this.peaks.push(planes);
    await this.tiles.push(mono(planes));
  }

  async finish(): Promise<void> {
    try {
      for (const frame of this.writer.finish()) this.write(frame);
      this.flac.write(this.writer.header(), { at: 0 });
      this.flac.flush();
    } finally {
      this.flac.close();
    }
    await this.tiles.finish();
    await writeWhole(this.dir, this.indexPath, Float64Array.from([...this.writer.index, this.at]));
    await writeWhole(this.dir, paths.peaks(this.layer), serializePeaks(this.peaks.finish()));
  }

  /** Closes the FLAC handle without finishing, for a cancelled job. */
  abort(): void {
    this.flac.close();
  }

  private write(frame: Uint8Array): void {
    this.flac.write(frame, { at: this.at });
    this.at += frame.length;
  }
}
