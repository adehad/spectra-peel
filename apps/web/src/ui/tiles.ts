import { SPECTROGRAM } from "@spectra-peel/engine";
import { paths, readBytes } from "../storage/opfs";

/** Loads spectrogram tiles from OPFS on demand and keeps them; an hour of one layer is about 80 MB. */
export class TileCache {
  /** Bumps whenever a tile arrives, so a renderer can tell its last frame is stale. */
  version = 0;
  private readonly tiles = new Map<number, Uint8Array>();
  private readonly loading = new Set<number>();
  private readonly dir: FileSystemDirectoryHandle;
  private readonly layer: string;
  private readonly onLoad: () => void;

  constructor(dir: FileSystemDirectoryHandle, layer: string, onLoad: () => void = () => undefined) {
    this.dir = dir;
    this.layer = layer;
    this.onLoad = onLoad;
  }

  /** Band values for `column`, or undefined while its tile is loading. */
  column(column: number): Uint8Array | undefined {
    const n = Math.floor(column / SPECTROGRAM.tileColumns);
    const tile = this.tiles.get(n);
    if (!tile) {
      this.load(n);
      return undefined;
    }
    const at = (column - n * SPECTROGRAM.tileColumns) * SPECTROGRAM.bands;
    return at + SPECTROGRAM.bands <= tile.length ? tile.subarray(at, at + SPECTROGRAM.bands) : undefined;
  }

  private load(n: number): void {
    if (this.loading.has(n)) return;
    this.loading.add(n);
    readBytes(this.dir, paths.tile(this.layer, n)).then(
      (buf) => {
        this.tiles.set(n, new Uint8Array(buf));
        this.version++;
        this.onLoad();
        return undefined;
      },
      () => this.tiles.set(n, new Uint8Array(0)),
    );
  }
}
