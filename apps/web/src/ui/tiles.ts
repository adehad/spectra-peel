import { SPECTROGRAM } from "@spectra-peel/engine";
import type { Layer, Project } from "../project/schema";
import { paths, readBytes } from "../storage/opfs";

/** Loads one layer's spectrogram tiles from OPFS on demand and keeps them; an hour is about 80 MB. */
class TileCache {
  private readonly tiles = new Map<number, Uint8Array>();
  private readonly loading = new Set<number>();
  private readonly dir: FileSystemDirectoryHandle;
  private readonly layer: string;
  private readonly onLoad: () => void;

  constructor(dir: FileSystemDirectoryHandle, layer: string, onLoad: () => void) {
    this.dir = dir;
    this.layer = layer;
    this.onLoad = onLoad;
  }

  /** Band values for a layer-relative column, or undefined while its tile loads. */
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
        this.onLoad();
        return undefined;
      },
      () => this.tiles.set(n, new Uint8Array(0)),
    );
  }
}

/**
 * Spectrogram columns for any layer of a project, resolving extents like playback does: outside its
 * extent a FLAC layer shows its fallback and a residual layer shows nothing.
 */
export class ProjectTiles {
  /** Bumps whenever a tile arrives, so a renderer can tell its last frame is stale. */
  version = 0;
  private readonly caches = new Map<string, TileCache>();
  private readonly layers: Map<string, Layer>;

  constructor(dir: FileSystemDirectoryHandle, project: Project) {
    this.layers = new Map(project.layers.map((l) => [l.id, l]));
    for (const layer of project.layers) {
      this.caches.set(
        layer.id,
        new TileCache(dir, layer.id, () => {
          this.version++;
        }),
      );
    }
  }

  /** `column` counts spectrogram hops from the start of the file. */
  column(layerId: string, column: number): Uint8Array | undefined {
    const layer = this.layers.get(layerId);
    if (!layer) return undefined;
    const first = layer.start / SPECTROGRAM.hop;
    const inside = column >= first && column < (layer.start + layer.frames) / SPECTROGRAM.hop;
    if (inside) return this.caches.get(layerId)?.column(column - first);
    if (layer.data.kind === "flac" && layer.data.fallback !== undefined)
      return this.column(layer.data.fallback, column);
    return undefined;
  }
}
