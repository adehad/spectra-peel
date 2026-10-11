import {
  FLAC_BLOCK,
  FLAC_HEADER_BYTES,
  FlacWriter,
  LoudnessMeter,
  Resampler,
  Xorshift32,
  normalizationGain,
  quantize16,
} from "@spectra-peel/engine";
import {
  AudioSample,
  AudioSampleSource,
  Mp4OutputFormat,
  OggOutputFormat,
  Output,
  Quality,
  StreamTarget,
  canEncodeAudio,
  type StreamTargetChunk,
} from "mediabunny";
import { LayerBlocks } from "../audio/layer-blocks";
import { layerGains, type Project } from "../project/schema";
import { fileAt, projectDir } from "../storage/opfs";
import type { ExportFormat, ExportMessage, ExportRequest, LoudnessReport } from "./protocol";

const post = (m: ExportMessage) => postMessage(m);
const OPUS_RATE = 48_000;

interface Sink {
  write(planes: readonly Float32Array[]): Promise<void>;
  finish(): Promise<void>;
}

function wavHeader(frames: number, channels: number, rate: number): Uint8Array<ArrayBuffer> {
  const bytes = frames * channels * 3;
  const h = new DataView(new ArrayBuffer(44));
  const text = (at: number, s: string) => {
    for (let i = 0; i < s.length; i++) h.setUint8(at + i, s.charCodeAt(i));
  };
  text(0, "RIFF");
  h.setUint32(4, 36 + bytes, true);
  text(8, "WAVEfmt ");
  h.setUint32(16, 16, true);
  h.setUint16(20, 1, true);
  h.setUint16(22, channels, true);
  h.setUint32(24, rate, true);
  h.setUint32(28, rate * channels * 3, true);
  h.setUint16(32, channels * 3, true);
  h.setUint16(34, 24, true);
  text(36, "data");
  h.setUint32(40, bytes, true);
  return new Uint8Array(h.buffer);
}

/** 24-bit PCM: the mix is float, and 24 bits keep gain changes below the 16-bit layers' own noise floor. */
function wavSink(w: FileSystemWritableFileStream, frames: number, channels: number, rate: number): Sink {
  let at = 0;
  return {
    async write(planes) {
      if (at === 0) {
        await w.write({ type: "write", position: 0, data: wavHeader(frames, channels, rate) });
        at = 44;
      }
      const n = planes[0]?.length ?? 0;
      const out = new Uint8Array(n * channels * 3);
      let o = 0;
      for (let i = 0; i < n; i++) {
        for (let c = 0; c < channels; c++) {
          const v = Math.max(-8_388_608, Math.min(8_388_607, Math.round((planes[c]?.[i] ?? 0) * 8_388_607)));
          out[o++] = v & 0xff;
          out[o++] = (v >> 8) & 0xff;
          out[o++] = (v >> 16) & 0xff;
        }
      }
      await w.write({ type: "write", position: at, data: out });
      at += out.length;
    },
    async finish() {
      await w.close();
    },
  };
}

function flacSink(w: FileSystemWritableFileStream, channels: 1 | 2, rate: number): Sink {
  const writer = new FlacWriter({ sampleRate: rate, channels });
  const rng = new Xorshift32(7);
  let at = FLAC_HEADER_BYTES;
  const put = async (frames: Uint8Array<ArrayBuffer>[]) => {
    for (const f of frames) {
      await w.write({ type: "write", position: at, data: f });
      at += f.length;
    }
  };
  return {
    async write(planes) {
      await put(writer.push(planes.map((p) => quantize16(p, rng))));
    },
    async finish() {
      await put(writer.finish());
      await w.write({ type: "write", position: 0, data: writer.header() });
      await w.close();
    },
  };
}

/** AAC in MP4 or Opus in Ogg through WebCodecs. Opus only runs at 48 kHz, so it resamples first. */
async function encodedSink(
  w: FileSystemWritableFileStream,
  format: "m4a" | "opus",
  channels: number,
  rate: number,
  bitrate: number,
): Promise<Sink> {
  const codec = format === "m4a" ? "aac" : "opus";
  const outRate = format === "opus" ? OPUS_RATE : rate;
  const writable = new WritableStream<StreamTargetChunk>({
    write: (chunk) => w.write({ type: "write", position: chunk.position, data: chunk.data }),
  });
  const output = new Output({
    format: format === "m4a" ? new Mp4OutputFormat() : new OggOutputFormat(),
    target: new StreamTarget(writable),
  });
  const source = new AudioSampleSource({ codec, quality: new Quality({ bitrate }) });
  output.addAudioTrack(source);
  await output.start();
  const resamplers =
    outRate === rate ? null : Array.from({ length: channels }, () => new Resampler(rate, outRate));
  let written = 0;
  const add = async (planes: readonly Float32Array[]) => {
    const n = planes[0]?.length ?? 0;
    if (n === 0) return;
    const data = new Float32Array(n * channels);
    for (let c = 0; c < channels; c++) data.set(planes[c]?.subarray(0, n) ?? new Float32Array(n), c * n);
    const sample = new AudioSample({
      data,
      format: "f32-planar",
      numberOfChannels: channels,
      sampleRate: outRate,
      timestamp: written / outRate,
    });
    written += n;
    await source.add(sample);
    sample.close();
  };
  return {
    async write(planes) {
      await add(resamplers ? planes.map((p, c) => resamplers[c]?.push(p) ?? p) : planes);
    },
    async finish() {
      if (resamplers) await add(resamplers.map((r) => r.finish()));
      await output.finalize();
      await w.close();
    },
  };
}

async function exportMix(
  req: ExportRequest,
): Promise<{ path: string; bytes: number; loudness: LoudnessReport | null }> {
  const { project, format, range } = req;
  const rate = project.source.sampleRate;
  const channels = project.source.channels;
  const dir = await projectDir(project.id);
  const path = `exports/${req.fileName}`;
  const handle = await fileAt(dir, path, true);
  const w = await handle.createWritable();
  await w.truncate(0);
  const frames = range.end - range.start;
  const sink =
    format === "wav"
      ? wavSink(w, frames, channels, rate)
      : format === "flac"
        ? flacSink(w, channels, rate)
        : await encodedSink(w, format, channels, rate, req.bitrate);

  const gains = [...layerGains(project)].filter(([, g]) => g > 0);
  const blocks = await LayerBlocks.open(dir, project);
  const first = Math.floor(range.start / FLAC_BLOCK);
  const last = Math.ceil(range.end / FLAC_BLOCK);
  const mixBlock = (k: number, gain: number) => {
    const base = k * FLAC_BLOCK;
    const from = Math.max(0, range.start - base);
    const to = Math.min(blocks.blockLength(k), range.end - base);
    const mix = Array.from({ length: channels }, () => new Float32Array(to - from));
    for (const [id, g] of gains) {
      const planes = blocks.block(id, k);
      for (let c = 0; c < channels; c++) {
        const src = planes[c];
        const dst = mix[c];
        if (!src || !dst) continue;
        for (let i = 0; i < dst.length; i++) dst[i] = (dst[i] ?? 0) + g * gain * (src[from + i] ?? 0);
      }
    }
    return mix;
  };
  const passes = req.normalize ? 2 : 1;
  const progress = (pass: number, k: number) => {
    if (k % 32 === 0)
      post({ type: "progress", fraction: (pass + (k - first) / Math.max(1, last - first)) / passes });
  };
  let loudness: LoudnessReport | null = null;
  try {
    let gain = 1;
    if (req.normalize) {
      const meter = new LoudnessMeter(rate, channels);
      for (let k = first; k < last; k++) {
        meter.push(mixBlock(k, 1));
        progress(0, k);
      }
      const measured = meter.result();
      const gainDb = normalizationGain(measured, req.normalize.target, req.normalize.ceiling);
      loudness = { measured: measured.integrated, truePeak: measured.truePeak, gainDb };
      gain = 10 ** (gainDb / 20);
    }
    for (let k = first; k < last; k++) {
      await sink.write(mixBlock(k, gain));
      progress(passes - 1, k);
    }
    await sink.finish();
  } catch (e) {
    await w.abort().catch(() => undefined);
    throw e;
  } finally {
    blocks.close();
  }
  return { path, bytes: (await handle.getFile()).size, loudness };
}

const CANDIDATE_KBPS = [32, 48, 64, 96, 128, 160, 192, 256];

/**
 * Which lossy formats and bitrates this browser can encode. Platform encoders differ: Windows' AAC encoder
 * only takes a few fixed bitrates, none below 96 kbps.
 */
async function probe(project: Project) {
  const channels = project.source.channels;
  const supported = async (codec: "aac" | "opus", sampleRate: number) => {
    const ok = await Promise.all(
      CANDIDATE_KBPS.map((k) =>
        canEncodeAudio(codec, {
          numberOfChannels: channels,
          sampleRate,
          quality: new Quality({ bitrate: k * 1000 }),
        }),
      ),
    );
    return CANDIDATE_KBPS.filter((_, i) => ok[i]);
  };
  const kbps = {
    m4a: await supported("aac", project.source.sampleRate),
    opus: await supported("opus", OPUS_RATE),
  };
  const formats: ExportFormat[] = ["wav", "flac"];
  if (kbps.m4a.length > 0) formats.push("m4a");
  if (kbps.opus.length > 0) formats.push("opus");
  return { formats, kbps };
}

addEventListener("message", (e: MessageEvent<{ type: "probe"; project: Project } | ExportRequest>) => {
  const req = e.data;
  if ("type" in req) {
    probe(req.project).then(
      (r) => post({ type: "formats", ...r }),
      (err: unknown) => post({ type: "error", message: String(err) }),
    );
    return;
  }
  exportMix(req).then(
    (r) => post({ type: "done", ...r }),
    (err: unknown) => post({ type: "error", message: err instanceof Error ? err.message : String(err) }),
  );
});
