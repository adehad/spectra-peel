import {
  FLAC_HEADER_BYTES,
  FlacWriter,
  PeaksBuilder,
  SpectrogramBuilder,
  Xorshift32,
  quantize16,
  serializePeaks,
  type SpectrogramTile,
} from "@spectra-peel/engine";
import { fileAt, paths } from "../storage/opfs";

export async function writeWhole(
  dir: FileSystemDirectoryHandle,
  path: string,
  data: AllowSharedBufferSource,
): Promise<void> {
  const handle = await (await fileAt(dir, path, true)).createSyncAccessHandle();
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
    const flac = await (await fileAt(dir, flacPath, true)).createSyncAccessHandle();
    flac.truncate(0);
    flac.write(new Uint8Array(FLAC_HEADER_BYTES), { at: 0 });
    return new LayerWriter(dir, layer, indexPath, flac, sampleRate, channels);
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
