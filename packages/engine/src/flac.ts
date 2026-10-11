import { BitReader, BitWriter, crc16, crc8 } from "./bits";

/**
 * A FLAC subset for layer storage: 16-bit, one or two channels, fixed 4096-sample blocks so frame n
 * starts at sample n * 4096, and fixed predictors with partitioned Rice residuals. Output is a valid
 * FLAC stream. The decoder also reads LPC subframes so it can open files from other encoders.
 */
export const FLAC_BLOCK = 4096;
export const FLAC_HEADER_BYTES = 42;
const BPS = 16;
const MAX_PARTITION_ORDER = 8;
const MAX_RICE = 14;

export interface FlacFormat {
  sampleRate: number;
  channels: 1 | 2;
}

export interface StreamInfo extends FlacFormat {
  totalSamples: number;
}

export interface DecodedFrame {
  channels: Int32Array[];
  blockSize: number;
  /** Encoded length of the frame in bytes. */
  bytes: number;
}

export function streamInfo(format: FlacFormat, totalSamples: number): Uint8Array<ArrayBuffer> {
  const w = new BitWriter(FLAC_HEADER_BYTES);
  for (const c of "fLaC") w.byte(c.charCodeAt(0));
  w.bits(1, 1);
  w.bits(0, 7);
  w.bits(34, 24);
  w.bits(FLAC_BLOCK, 16);
  w.bits(FLAC_BLOCK, 16);
  w.bits(0, 24);
  w.bits(0, 24);
  w.bits(format.sampleRate, 20);
  w.bits(format.channels - 1, 3);
  w.bits(BPS - 1, 5);
  w.long(totalSamples, 36);
  for (let i = 0; i < 16; i++) w.byte(0);
  return w.bytes;
}

export function readStreamInfo(bytes: Uint8Array): StreamInfo {
  if (String.fromCharCode(...bytes.subarray(0, 4)) !== "fLaC") throw new Error("not a FLAC stream");
  const r = new BitReader(bytes, 8);
  r.bits(16);
  r.bits(16);
  r.bits(24);
  r.bits(24);
  const sampleRate = r.bits(20);
  const channels = r.bits(3) + 1;
  if (channels !== 1 && channels !== 2) throw new Error(`unsupported channel count ${channels}`);
  r.bits(5);
  return { sampleRate, channels, totalSamples: r.long(36) };
}

/** Collects PCM, emits encoded frames, and records each frame's byte offset in the finished file. */
export class FlacWriter {
  readonly index: number[] = [];
  private readonly block: Int32Array[];
  private fill = 0;
  private frames = 0;
  private samples = 0;
  private offset = FLAC_HEADER_BYTES;
  readonly format: FlacFormat;

  constructor(format: FlacFormat) {
    this.format = format;
    this.block = Array.from({ length: format.channels }, () => new Int32Array(FLAC_BLOCK));
  }

  get totalSamples(): number {
    return this.samples;
  }

  /** The 42-byte stream header; write it last, over the start of the file, once the length is known. */
  header(): Uint8Array<ArrayBuffer> {
    return streamInfo(this.format, this.samples);
  }

  push(channels: readonly Int16Array[]): Uint8Array<ArrayBuffer>[] {
    const length = channels[0]?.length ?? 0;
    const out: Uint8Array<ArrayBuffer>[] = [];
    let at = 0;
    while (at < length) {
      const n = Math.min(FLAC_BLOCK - this.fill, length - at);
      for (let c = 0; c < this.format.channels; c++) {
        const src = channels[c] ?? channels[0];
        if (src) this.block[c]?.set(src.subarray(at, at + n), this.fill);
      }
      this.fill += n;
      at += n;
      if (this.fill === FLAC_BLOCK) out.push(this.emit());
    }
    return out;
  }

  finish(): Uint8Array<ArrayBuffer>[] {
    return this.fill > 0 ? [this.emit()] : [];
  }

  private emit(): Uint8Array<ArrayBuffer> {
    const frame = encodeFrame(
      this.frames,
      this.block.map((b) => b.subarray(0, this.fill)),
    );
    this.index.push(this.offset);
    this.offset += frame.length;
    this.samples += this.fill;
    this.frames++;
    this.fill = 0;
    return frame;
  }
}

const Assignment = { Independent: 1, LeftSide: 8, SideRight: 9, MidSide: 10 } as const;

export function encodeFrame(frameNumber: number, channels: readonly Int32Array[]): Uint8Array<ArrayBuffer> {
  const blockSize = channels[0]?.length ?? 0;
  const w = new BitWriter(32 + channels.length * (4 + Math.ceil(((BPS + 1) * blockSize) / 8)));

  let subframes: { signal: Int32Array; bps: number }[];
  let assignment: number;
  const [left, right] = channels;
  if (channels.length === 2 && left && right) {
    const mid = new Int32Array(blockSize);
    const side = new Int32Array(blockSize);
    for (let i = 0; i < blockSize; i++) {
      const l = left[i] ?? 0;
      const r = right[i] ?? 0;
      mid[i] = (l + r) >> 1;
      side[i] = l - r;
    }
    const cost = (s: Int32Array) => bestFixedOrder(s).sum;
    const [cl, cr, cm, cs] = [cost(left), cost(right), cost(mid), cost(side)];
    const options: [number, number, { signal: Int32Array; bps: number }[]][] = [
      [
        cl + cr,
        Assignment.Independent,
        [
          { signal: left, bps: BPS },
          { signal: right, bps: BPS },
        ],
      ],
      [
        cl + cs,
        Assignment.LeftSide,
        [
          { signal: left, bps: BPS },
          { signal: side, bps: BPS + 1 },
        ],
      ],
      [
        cs + cr,
        Assignment.SideRight,
        [
          { signal: side, bps: BPS + 1 },
          { signal: right, bps: BPS },
        ],
      ],
      [
        cm + cs,
        Assignment.MidSide,
        [
          { signal: mid, bps: BPS },
          { signal: side, bps: BPS + 1 },
        ],
      ],
    ];
    const best = options.reduce((a, b) => (b[0] < a[0] ? b : a));
    assignment = best[1];
    subframes = best[2];
  } else {
    assignment = channels.length - 1;
    subframes = channels.map((signal) => ({ signal, bps: BPS }));
  }

  w.bits(0x3ffe, 14);
  w.bits(0, 1);
  w.bits(0, 1);
  w.bits(blockSize === FLAC_BLOCK ? 12 : 7, 4);
  w.bits(0, 4);
  w.bits(assignment, 4);
  w.bits(4, 3);
  w.bits(0, 1);
  writeUtf8(w, frameNumber);
  if (blockSize !== FLAC_BLOCK) w.bits(blockSize - 1, 16);
  w.byte(crc8(w.bytes, 0, w.byteLength));

  for (const s of subframes) writeSubframe(w, s.signal, s.bps);
  w.alignToByte();
  const crc = crc16(w.bytes, 0, w.byteLength);
  w.bits(crc, 16);
  return w.bytes.slice(0, w.byteLength);
}

function writeUtf8(w: BitWriter, n: number): void {
  if (n < 0x80) {
    w.byte(n);
    return;
  }
  let bytes = 2;
  while (n >= 2 ** (5 * bytes + 1)) bytes++;
  w.byte(((0xff00 >> bytes) & 0xff) | Math.floor(n / 2 ** (6 * (bytes - 1))));
  for (let i = bytes - 2; i >= 0; i--) w.byte(0x80 | (Math.floor(n / 2 ** (6 * i)) & 0x3f));
}

/** Prediction of the fixed polynomial predictor of the given order. */
function predict(signal: Int32Array, order: number, i: number): number {
  const x = (k: number) => signal[i - k] ?? 0;
  switch (order) {
    case 0:
      return 0;
    case 1:
      return x(1);
    case 2:
      return 2 * x(1) - x(2);
    case 3:
      return 3 * x(1) - 3 * x(2) + x(3);
    default:
      return 4 * x(1) - 6 * x(2) + 4 * x(3) - x(4);
  }
}

function residual(signal: Int32Array, order: number, i: number): number {
  return (signal[i] ?? 0) - predict(signal, order, i);
}

function bestFixedOrder(signal: Int32Array): { order: number; sum: number } {
  let s0 = 0,
    s1 = 0,
    s2 = 0,
    s3 = 0,
    s4 = 0;
  let x1 = 0,
    x2 = 0,
    x3 = 0,
    x4 = 0;
  for (let i = 0; i < signal.length; i++) {
    const x0 = signal[i] ?? 0;
    const e1 = x0 - x1;
    const e2 = e1 - (x1 - x2);
    const e3 = e2 - (x1 - 2 * x2 + x3);
    const e4 = e3 - (x1 - 3 * x2 + 3 * x3 - x4);
    s0 += Math.abs(x0);
    if (i >= 1) s1 += Math.abs(e1);
    if (i >= 2) s2 += Math.abs(e2);
    if (i >= 3) s3 += Math.abs(e3);
    if (i >= 4) s4 += Math.abs(e4);
    x4 = x3;
    x3 = x2;
    x2 = x1;
    x1 = x0;
  }
  const sums = [s0, s1, s2, s3, s4].slice(0, Math.min(4, signal.length - 1) + 1);
  let order = 0;
  for (const [o, sum] of sums.entries()) if (sum < (sums[order] ?? Infinity)) order = o;
  return { order, sum: sums[order] ?? 0 };
}

function riceParam(sum: number, n: number): number {
  if (n === 0 || sum <= n) return 0;
  return Math.min(MAX_RICE, Math.floor(Math.log2(sum / n)));
}

function writeSubframe(w: BitWriter, signal: Int32Array, bps: number): void {
  const n = signal.length;
  const first = signal[0] ?? 0;
  if (signal.every((v) => v === first)) {
    w.byte(0);
    w.bits(first, bps);
    return;
  }

  const { order } = bestFixedOrder(signal);
  const u = new Uint32Array(n);
  for (let i = order; i < n; i++) {
    const r = residual(signal, order, i);
    u[i] = r >= 0 ? r * 2 : -r * 2 - 1;
  }

  let maxP = 0;
  while (maxP < MAX_PARTITION_ORDER && n % 2 ** (maxP + 1) === 0 && n >> (maxP + 1) > order) maxP++;
  let best = { p: 0, bits: Infinity, ks: [] as number[] };
  for (let p = 0; p <= maxP; p++) {
    const size = n >> p;
    const ks: number[] = [];
    let bits = 0;
    for (let part = 0; part < 2 ** p; part++) {
      const start = part === 0 ? order : part * size;
      const end = (part + 1) * size;
      let sum = 0;
      for (let i = start; i < end; i++) sum += u[i] ?? 0;
      const k = riceParam(sum, end - start);
      let partBits = 4 + (end - start) * (k + 1);
      for (let i = start; i < end; i++) partBits += (u[i] ?? 0) >>> k;
      ks.push(k);
      bits += partBits;
    }
    if (bits < best.bits) best = { p, bits, ks };
  }

  if (order * bps + 6 + best.bits >= n * bps) {
    w.byte(1 << 1);
    for (const v of signal) w.bits(v, bps);
    return;
  }

  w.byte((0b001000 | order) << 1);
  for (let i = 0; i < order; i++) w.bits(signal[i] ?? 0, bps);
  w.bits(0, 2);
  w.bits(best.p, 4);
  const size = n >> best.p;
  for (const [part, k] of best.ks.entries()) {
    w.bits(k, 4);
    const end = (part + 1) * size;
    for (let i = part === 0 ? order : part * size; i < end; i++) {
      const v = u[i] ?? 0;
      w.unary(v >>> k);
      if (k > 0) w.bits(v, k);
    }
  }
}

export function decodeFrame(bytes: Uint8Array, offset: number): DecodedFrame {
  const r = new BitReader(bytes, offset);
  if (r.bits(14) !== 0x3ffe) throw new Error(`no FLAC frame sync at byte ${offset}`);
  r.bits(2);
  const bsCode = r.bits(4);
  const srCode = r.bits(4);
  const assignment = r.bits(4);
  const ssCode = r.bits(3);
  r.bits(1);
  const lead = r.bits(8);
  let extra = 0;
  while (lead & (0x80 >> extra)) extra++;
  for (let i = 1; i < extra; i++) r.bits(8);

  let blockSize: number;
  if (bsCode === 1) blockSize = 192;
  else if (bsCode <= 5) blockSize = 576 * 2 ** (bsCode - 2);
  else if (bsCode === 6) blockSize = r.bits(8) + 1;
  else if (bsCode === 7) blockSize = r.bits(16) + 1;
  else blockSize = 256 * 2 ** (bsCode - 8);
  if (srCode === 12) r.bits(8);
  else if (srCode === 13 || srCode === 14) r.bits(16);
  const bps = [BPS, 8, 12, 0, 16, 20, 24, 32][ssCode] ?? BPS;
  r.bits(8);

  const count = assignment < 8 ? assignment + 1 : 2;
  const channels: Int32Array[] = [];
  for (let c = 0; c < count; c++) {
    const side =
      (assignment === Assignment.LeftSide && c === 1) ||
      (assignment === Assignment.SideRight && c === 0) ||
      (assignment === Assignment.MidSide && c === 1);
    channels.push(readSubframe(r, blockSize, bps + (side ? 1 : 0)));
  }
  r.alignToByte();
  r.bits(16);

  const [a, b] = channels;
  if (a && b && assignment >= 8) {
    for (let i = 0; i < blockSize; i++) {
      const x = a[i] ?? 0;
      const y = b[i] ?? 0;
      if (assignment === Assignment.LeftSide) b[i] = x - y;
      else if (assignment === Assignment.SideRight) a[i] = x + y;
      else {
        const mid = x * 2 + (y & 1);
        a[i] = (mid + y) >> 1;
        b[i] = (mid - y) >> 1;
      }
    }
  }
  return { channels, blockSize, bytes: r.bytePos - offset };
}

function readSubframe(r: BitReader, n: number, bps: number): Int32Array {
  r.bits(1);
  const type = r.bits(6);
  let wasted = 0;
  if (r.bits(1)) {
    wasted = r.unary() + 1;
    bps -= wasted;
  }
  const out = new Int32Array(n);
  if (type === 0) out.fill(r.signed(bps));
  else if (type === 1) for (let i = 0; i < n; i++) out[i] = r.signed(bps);
  else if (type >= 8 && type <= 12) {
    const order = type - 8;
    for (let i = 0; i < order; i++) out[i] = r.signed(bps);
    readResidual(r, out, order);
    for (let i = order; i < n; i++) out[i] = (out[i] ?? 0) + predict(out, order, i);
  } else if (type >= 32) {
    const order = type - 31;
    for (let i = 0; i < order; i++) out[i] = r.signed(bps);
    const precision = r.bits(4) + 1;
    const shift = r.signed(5);
    const coefs = Array.from({ length: order }, () => r.signed(precision));
    readResidual(r, out, order);
    for (let i = order; i < n; i++) {
      let sum = 0;
      for (let j = 0; j < order; j++) sum += (coefs[j] ?? 0) * (out[i - j - 1] ?? 0);
      out[i] = (out[i] ?? 0) + Math.floor(sum / 2 ** shift);
    }
  } else throw new Error(`reserved FLAC subframe type ${type}`);
  if (wasted > 0) for (let i = 0; i < n; i++) out[i] = (out[i] ?? 0) * 2 ** wasted;
  return out;
}

function readResidual(r: BitReader, out: Int32Array, order: number): void {
  const paramBits = r.bits(2) === 0 ? 4 : 5;
  const escape = (1 << paramBits) - 1;
  const p = r.bits(4);
  const size = out.length >> p;
  let i = order;
  for (let part = 0; part < 2 ** p; part++) {
    const end = (part + 1) * size;
    const k = r.bits(paramBits);
    if (k === escape) {
      const width = r.bits(5);
      for (; i < end; i++) out[i] = width === 0 ? 0 : r.signed(width);
    } else for (; i < end; i++) out[i] = r.rice(k);
  }
}
