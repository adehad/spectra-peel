const WRITE = 0;
const READ = 1;
const EPOCH = 2;
const EPOCH_START = 3;
const FLAGS_INT32 = 4;
/** Float64 slots after the Int32 header: the source frame at which the current epoch starts, and the play head. */
const START_FRAME = 0;
const PLAYED_FRAME = 1;
const HEADER_BYTES = FLAGS_INT32 * 4 + 2 * 8;

/**
 * Single-producer, single-consumer ring of planar float audio over a SharedArrayBuffer, for a reader worker
 * feeding an AudioWorklet. Indices count frames and wrap at 2^32, which a power-of-two capacity divides, so
 * `(write - read) | 0` is the fill level. A seek is
 * an epoch: the producer marks where the new audio begins, and the consumer skips to it on its next read, so
 * neither side ever has to wait on the other.
 */
export class AudioRing {
  readonly channels: number;
  readonly capacity: number;
  private readonly header: Int32Array;
  private readonly positions: Float64Array;
  private readonly data: Float32Array[];
  private seenEpoch = 0;

  static bytes(channels: number, capacity: number): number {
    return HEADER_BYTES + channels * capacity * 4;
  }

  constructor(buffer: SharedArrayBuffer, channels: number, capacity: number) {
    if (capacity < 1 || (capacity & (capacity - 1)) !== 0) {
      throw new RangeError(`ring capacity ${capacity} is not a power of two`);
    }
    this.channels = channels;
    this.capacity = capacity;
    this.header = new Int32Array(buffer, 0, FLAGS_INT32);
    this.positions = new Float64Array(buffer, FLAGS_INT32 * 4, 2);
    this.data = Array.from(
      { length: channels },
      (_, c) => new Float32Array(buffer, HEADER_BYTES + c * capacity * 4, capacity),
    );
    this.seenEpoch = Atomics.load(this.header, EPOCH);
  }

  /** Frames the consumer can read in the current epoch. */
  available(): number {
    const write = Atomics.load(this.header, WRITE);
    const from = this.epochChanged()
      ? Atomics.load(this.header, EPOCH_START)
      : Atomics.load(this.header, READ);
    return (write - from) | 0;
  }

  /** Frames the producer can write without overwriting unread audio. */
  space(): number {
    return this.capacity - ((Atomics.load(this.header, WRITE) - Atomics.load(this.header, READ)) | 0);
  }

  /** Producer: discard everything queued and start a new epoch whose first frame is `startFrame`. */
  seek(startFrame: number): void {
    this.positions[START_FRAME] = startFrame;
    Atomics.store(this.header, EPOCH_START, Atomics.load(this.header, WRITE));
    Atomics.add(this.header, EPOCH, 1);
  }

  /** Producer: copies up to `frames` frames, returns how many fitted. */
  write(channels: readonly Float32Array[], frames = channels[0]?.length ?? 0): number {
    const n = Math.min(frames, this.space());
    const write = Atomics.load(this.header, WRITE);
    for (let c = 0; c < this.channels; c++) {
      const src = channels[c];
      const dst = this.data[c];
      if (!src || !dst) continue;
      const at = write & (this.capacity - 1);
      const first = Math.min(n, this.capacity - at);
      dst.set(src.subarray(0, first), at);
      if (n > first) dst.set(src.subarray(first, n), 0);
    }
    Atomics.store(this.header, WRITE, (write + n) | 0);
    return n;
  }

  /**
   * Consumer: fills `out` from the current epoch and zero-fills any shortfall. Returns frames read; the source
   * frame of the first one is `playedFrame()` before the call.
   */
  read(out: readonly Float32Array[], frames: number): number {
    if (this.epochChanged()) {
      this.seenEpoch = Atomics.load(this.header, EPOCH);
      const start = Atomics.load(this.header, EPOCH_START);
      this.positions[PLAYED_FRAME] = this.positions[START_FRAME] ?? 0;
      Atomics.store(this.header, READ, start);
    }
    const read = Atomics.load(this.header, READ);
    const n = Math.min(frames, (Atomics.load(this.header, WRITE) - read) | 0);
    for (let c = 0; c < this.channels; c++) {
      const src = this.data[c];
      const dst = out[c];
      if (!src || !dst) continue;
      const at = read & (this.capacity - 1);
      const first = Math.min(n, this.capacity - at);
      dst.set(src.subarray(at, at + first));
      if (n > first) dst.set(src.subarray(0, n - first), first);
      dst.fill(0, n, frames);
    }
    Atomics.store(this.header, READ, (read + n) | 0);
    this.positions[PLAYED_FRAME] = (this.positions[PLAYED_FRAME] ?? 0) + n;
    return n;
  }

  /** Source frame the consumer will play next. Readable from any thread. */
  playedFrame(): number {
    return this.positions[PLAYED_FRAME] ?? 0;
  }

  private epochChanged(): boolean {
    return Atomics.load(this.header, EPOCH) !== this.seenEpoch;
  }
}
