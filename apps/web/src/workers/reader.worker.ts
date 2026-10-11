import { AudioRing, FLAC_BLOCK } from "@spectra-peel/engine";
import { LayerBlocks } from "../audio/layer-blocks";
import { projectDir } from "../storage/opfs";
import type { ReaderMessage, ReaderRequest } from "./protocol";

const post = (m: ReaderMessage) => postMessage(m);
const FILL_INTERVAL_MS = 10;

/** Keeps the ring full of every layer's channels, in project layer order. */
class Reader {
  private readonly blocks: LayerBlocks;
  private readonly ring: AudioRing;
  private nextBlock = 0;
  private pending: Float32Array[] = [];
  private pendingAt = 0;

  constructor(blocks: LayerBlocks, ring: AudioRing) {
    this.blocks = blocks;
    this.ring = ring;
  }

  seek(sample: number): void {
    this.ring.seek(sample);
    this.nextBlock = Math.floor(sample / FLAC_BLOCK);
    this.pending = [];
    this.pendingAt = 0;
    if (this.decodeNext()) {
      this.pendingAt = Math.min(sample - (this.nextBlock - 1) * FLAC_BLOCK, this.pendingLength);
    }
  }

  fill(): void {
    for (;;) {
      if (this.pendingAt >= this.pendingLength && !this.decodeNext()) return;
      const rest = this.pending.map((p) => p.subarray(this.pendingAt));
      const written = this.ring.write(rest);
      this.pendingAt += written;
      if (written < (rest[0]?.length ?? 0)) return;
    }
  }

  close(): void {
    this.blocks.close();
  }

  private get pendingLength(): number {
    return this.pending[0]?.length ?? 0;
  }

  private decodeNext(): boolean {
    if (this.nextBlock >= this.blocks.totalBlocks) return false;
    const k = this.nextBlock++;
    this.pending = this.blocks.project.layers.flatMap((layer) => this.blocks.block(layer.id, k));
    this.pendingAt = 0;
    return true;
  }
}

let reader: Reader | undefined;
let timer: ReturnType<typeof setInterval> | undefined;

async function open(req: Extract<ReaderRequest, { type: "open" }>): Promise<void> {
  const blocks = await LayerBlocks.open(await projectDir(req.project.id), req.project);
  reader = new Reader(blocks, new AudioRing(req.ring, req.channels, req.capacity));
  reader.seek(req.position);
  timer = setInterval(() => reader?.fill(), FILL_INTERVAL_MS);
  post({ type: "ready" });
}

addEventListener("message", (e: MessageEvent<ReaderRequest>) => {
  const req = e.data;
  if (req.type === "open") {
    open(req).catch((err: unknown) => post({ type: "error", message: String(err) }));
  } else if (req.type === "seek") {
    reader?.seek(req.frame);
    reader?.fill();
  } else {
    clearInterval(timer);
    reader?.close();
    reader = undefined;
    post({ type: "closed" });
  }
});
