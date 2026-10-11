import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { hasFfmpeg } from "./testing/ffmpeg";
import { crc16 } from "./bits";
import { FLAC_BLOCK, FlacWriter, decodeFrame, readStreamInfo, type FlacFormat } from "./flac";

function signal(length: number, seed: number, noise = 200): Int16Array {
  const out = new Int16Array(length);
  let s = seed;
  for (let i = 0; i < length; i++) {
    s = (s * 16807) % 2147483647;
    out[i] = Math.round(8000 * Math.sin(i / (20 + seed)) + noise * (s / 2147483647 - 0.5));
  }
  return out;
}

function encode(format: FlacFormat, channels: Int16Array[], pushSize = 10_000) {
  const writer = new FlacWriter(format);
  const parts: Uint8Array[] = [];
  const total = channels[0]?.length ?? 0;
  for (let at = 0; at < total; at += pushSize) {
    parts.push(...writer.push(channels.map((c) => c.subarray(at, at + pushSize))));
  }
  parts.push(...writer.finish());
  const header = writer.header();
  const body = new Uint8Array(parts.reduce((n, p) => n + p.length, header.length));
  body.set(header);
  let at = header.length;
  for (const p of parts) {
    body.set(p, at);
    at += p.length;
  }
  return { bytes: body, index: writer.index };
}

function decodeAll(bytes: Uint8Array, index: readonly number[], channels: number, total: number) {
  const out = Array.from({ length: channels }, () => new Int32Array(total));
  for (const [n, offset] of index.entries()) {
    const frame = decodeFrame(bytes, offset);
    for (let c = 0; c < channels; c++) out[c]?.set(frame.channels[c] ?? [], n * FLAC_BLOCK);
  }
  return out;
}

const STEREO: FlacFormat = { sampleRate: 44_100, channels: 2 };

describe("FlacWriter and decodeFrame", () => {
  it("round-trips correlated stereo exactly, short last block included", () => {
    const left = signal(3 * FLAC_BLOCK + 1234, 1);
    const right = left.map((v, i) => Math.round(0.7 * v + ((i * 31) % 50)));
    const { bytes, index } = encode(STEREO, [left, right]);
    expect(index).toHaveLength(4);
    const [l, r] = decodeAll(bytes, index, 2, left.length);
    expect(l).toEqual(Int32Array.from(left));
    expect(r).toEqual(Int32Array.from(right));
  });

  it("round-trips mono, silence, full-scale noise and extremes", () => {
    const n = 2 * FLAC_BLOCK + 17;
    const cases = [
      new Int16Array(n),
      Int16Array.from({ length: n }, (_, i) => ((i * 2654435761) >>> 16) - 32768),
      Int16Array.from({ length: n }, (_, i) => (i % 2 ? 32767 : -32768)),
      signal(n, 5, 30_000),
    ];
    for (const samples of cases) {
      const { bytes, index } = encode({ sampleRate: 48_000, channels: 1 }, [samples], 4096);
      expect(decodeAll(bytes, index, 1, n)[0]).toEqual(Int32Array.from(samples));
    }
  });

  it("decodes any frame on its own from the index", () => {
    const left = signal(5 * FLAC_BLOCK, 2);
    const { bytes, index } = encode(STEREO, [left, left]);
    const frame = decodeFrame(bytes, index[3] ?? -1);
    expect(frame.blockSize).toBe(FLAC_BLOCK);
    expect(frame.channels[0]).toEqual(Int32Array.from(left.subarray(3 * FLAC_BLOCK, 4 * FLAC_BLOCK)));
    expect((index[3] ?? 0) + frame.bytes).toBe(index[4]);
  });

  it("writes frames whose CRC-16 checks out", () => {
    const { bytes, index } = encode(STEREO, [signal(FLAC_BLOCK, 3), signal(FLAC_BLOCK, 4)]);
    const start = index[0] ?? 0;
    expect(crc16(bytes, start, bytes.length)).toBe(0);
  });

  it("compresses a quiet tonal signal well below raw size", () => {
    const left = signal(20 * FLAC_BLOCK, 1, 50);
    const { bytes } = encode(STEREO, [left, left]);
    expect(bytes.length / (left.length * 4)).toBeLessThan(0.35);
  });

  it("records format and total length in STREAMINFO", () => {
    const { bytes } = encode(STEREO, [signal(10_000, 1), signal(10_000, 2)]);
    expect(readStreamInfo(bytes)).toEqual({ sampleRate: 44_100, channels: 2, totalSamples: 10_000 });
  });

  it.skipIf(!hasFfmpeg)("produces files ffmpeg decodes to the same samples", () => {
    const left = signal(4 * FLAC_BLOCK + 99, 7, 3000);
    const right = signal(4 * FLAC_BLOCK + 99, 8, 3000);
    const { bytes } = encode(STEREO, [left, right]);
    const dir = mkdtempSync(join(tmpdir(), "spectra-flac-"));
    writeFileSync(join(dir, "a.flac"), bytes);
    execFileSync("ffmpeg", ["-v", "error", "-i", join(dir, "a.flac"), "-f", "s16le", join(dir, "a.raw")]);
    const raw = readFileSync(join(dir, "a.raw"));
    const pcm = new Int16Array(raw.buffer, raw.byteOffset, raw.byteLength / 2);
    expect(pcm.length).toBe(left.length * 2);
    for (let i = 0; i < left.length; i++) {
      if (pcm[2 * i] !== left[i] || pcm[2 * i + 1] !== right[i]) throw new Error(`mismatch at ${i}`);
    }
  });
});
