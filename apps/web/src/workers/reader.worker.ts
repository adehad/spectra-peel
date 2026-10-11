import { AudioRing, FLAC_BLOCK, decodeFrame, toFloat } from "@spectra-peel/engine";
import { fileAt, paths, projectDir } from "../storage/opfs";
import type { ReaderMessage, ReaderRequest } from "./protocol";

const post = (m: ReaderMessage) => postMessage(m);
const FILL_INTERVAL_MS = 10;

class Reader {
  private readonly flac: FileSystemSyncAccessHandle;
  private readonly index: Float64Array;
  private readonly ring: AudioRing;
  private readonly channels: number;
  /** FLAC frame to decode next. */
  private nextFrame = 0;
  private pending: Float32Array[] = [];
  private pendingAt = 0;

  constructor(flac: FileSystemSyncAccessHandle, index: Float64Array, ring: AudioRing, channels: number) {
    this.flac = flac;
    this.index = index;
    this.ring = ring;
    this.channels = channels;
  }

  get frameCount(): number {
    return this.index.length - 1;
  }

  seek(sample: number): void {
    this.ring.seek(sample);
    this.nextFrame = Math.floor(sample / FLAC_BLOCK);
    this.pending = [];
    this.pendingAt = 0;
    if (this.decodeNext())
      this.pendingAt = Math.min(sample - (this.nextFrame - 1) * FLAC_BLOCK, this.pendingLength);
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
    this.flac.close();
  }

  private get pendingLength(): number {
    return this.pending[0]?.length ?? 0;
  }

  private decodeNext(): boolean {
    if (this.nextFrame >= this.frameCount) return false;
    const start = this.index[this.nextFrame] ?? 0;
    const end = this.index[this.nextFrame + 1] ?? start;
    const bytes = new Uint8Array(end - start);
    this.flac.read(bytes, { at: start });
    const frame = decodeFrame(bytes, 0);
    this.pending = Array.from({ length: this.channels }, (_, c) =>
      toFloat(frame.channels[c] ?? frame.channels[0] ?? new Int32Array(frame.blockSize)),
    );
    this.pendingAt = 0;
    this.nextFrame++;
    return true;
  }
}

let reader: Reader | undefined;
let timer: ReturnType<typeof setInterval> | undefined;

async function open(req: Extract<ReaderRequest, { type: "open" }>): Promise<void> {
  const dir = await projectDir(req.projectId);
  const index = new Float64Array(
    await (await (await fileAt(dir, paths.sourceIndex)).getFile()).arrayBuffer(),
  );
  const flac = await (await fileAt(dir, paths.sourceFlac)).createSyncAccessHandle();
  reader = new Reader(flac, index, new AudioRing(req.ring, req.channels, req.capacity), req.channels);
  reader.seek(0);
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
