import {
  ENGINE_VERSION,
  FLAC_HEADER_BYTES,
  FlacWriter,
  PeaksBuilder,
  SPECTROGRAM,
  SpectrogramBuilder,
  Xorshift32,
  quantize16,
  serializePeaks,
  type SpectrogramTile,
} from "@spectra-peel/engine";
import { ALL_FORMATS, AudioSampleSink, BlobSource, Input } from "mediabunny";
import { SOURCE_LAYER, type Project } from "../project/schema";
import { fileAt, paths, projectDir, projectsDir } from "../storage/opfs";
import type { ImportMessage, ImportRequest } from "./protocol";

const post = (m: ImportMessage) => postMessage(m);

async function writeWhole(dir: FileSystemDirectoryHandle, path: string, data: AllowSharedBufferSource) {
  const handle = await (await fileAt(dir, path, true)).createSyncAccessHandle();
  handle.truncate(0);
  handle.write(data, { at: 0 });
  handle.close();
}

const gb = (n: number) => (n / 1e9).toFixed(2);

/** FLAC at its worst case for 16-bit audio plus the spectrogram tiles; peaks are negligible. */
async function checkSpace(frames: number, channels: number): Promise<void> {
  const needed = frames * channels * 2 + (frames / SPECTROGRAM.hop) * SPECTROGRAM.bands;
  const { quota = Infinity, usage = 0 } = await navigator.storage.estimate();
  if (needed > quota - usage) {
    throw new Error(
      `Importing needs up to ${gb(needed)} GB but only ${gb(quota - usage)} GB of storage is free.`,
    );
  }
}

async function importFile({ file, projectId, name }: ImportRequest): Promise<Project> {
  const input = new Input({ source: new BlobSource(file), formats: ALL_FORMATS });
  const track = await input.getPrimaryAudioTrack();
  if (!track) throw new Error("No audio track found in this file.");
  if (!(await track.canDecode()))
    throw new Error(`This browser cannot decode ${track.codec ?? "this codec"}.`);
  const sampleRate = await track.getSampleRate();
  const channels = Math.min(2, await track.getNumberOfChannels()) === 1 ? 1 : 2;
  const expected = Math.max(1, (await track.computeDuration()) * sampleRate);
  await checkSpace(expected, channels);

  await (await projectsDir()).removeEntry(projectId, { recursive: true }).catch(() => undefined);
  const dir = await projectDir(projectId, true);
  const flac = await (await fileAt(dir, paths.sourceFlac, true)).createSyncAccessHandle();
  flac.truncate(0);
  flac.write(new Uint8Array(FLAC_HEADER_BYTES), { at: 0 });
  let flacAt = FLAC_HEADER_BYTES;

  const writer = new FlacWriter({ sampleRate, channels });
  const peaks = new PeaksBuilder();
  const spectrogram = new SpectrogramBuilder(sampleRate);
  const rng = new Xorshift32(0x5eed);
  const saveTiles = async (list: SpectrogramTile[]) => {
    for (const t of list) await writeWhole(dir, paths.tile(SOURCE_LAYER, t.index), t.data);
  };

  let frames = 0;
  let lastReport = 0;
  try {
    for await (const sample of new AudioSampleSink(track).samples()) {
      const skip =
        sample.timestamp < 0
          ? Math.min(sample.numberOfFrames, Math.round(-sample.timestamp * sampleRate))
          : 0;
      const n = sample.numberOfFrames - skip;
      const planes: Float32Array[] = [];
      for (let c = 0; c < channels; c++) {
        const plane = new Float32Array(n);
        if (n > 0) {
          const source = Math.min(c, sample.numberOfChannels - 1);
          sample.copyTo(plane, {
            planeIndex: source,
            format: "f32-planar",
            frameOffset: skip,
            frameCount: n,
          });
        }
        planes.push(plane);
      }
      sample.close();
      if (n === 0) continue;

      for (const frame of writer.push(planes.map((p) => quantize16(p, rng)))) {
        flac.write(frame, { at: flacAt });
        flacAt += frame.length;
      }
      peaks.push(planes);
      const [left, right] = planes;
      let mono = left ?? new Float32Array(n);
      if (right) mono = mono.map((v, i) => (v + (right[i] ?? 0)) / 2);
      await saveTiles(spectrogram.push(mono));

      frames += n;
      if (performance.now() - lastReport > 200) {
        lastReport = performance.now();
        post({ type: "progress", fraction: Math.min(1, frames / expected) });
      }
    }
    for (const frame of writer.finish()) {
      flac.write(frame, { at: flacAt });
      flacAt += frame.length;
    }
    flac.write(writer.header(), { at: 0 });
    flac.flush();
  } finally {
    flac.close();
  }
  await saveTiles(spectrogram.finish());
  await writeWhole(dir, paths.sourceIndex, Float64Array.from([...writer.index, flacAt]));
  await writeWhole(dir, paths.peaks(SOURCE_LAYER), serializePeaks(peaks.finish()));

  const project: Project = {
    version: 1,
    id: projectId,
    name,
    createdAt: new Date().toISOString(),
    engineVersion: ENGINE_VERSION,
    source: { fileName: file.name, sampleRate, channels, frames },
    layers: [
      {
        id: SOURCE_LAYER,
        label: "Source",
        data: { kind: "flac", path: paths.sourceFlac, index: paths.sourceIndex },
      },
    ],
    mix: { gains: {}, muted: [] },
  };
  await writeWhole(dir, paths.project, new TextEncoder().encode(JSON.stringify(project, null, 2)));
  return project;
}

addEventListener("message", (e: MessageEvent<ImportRequest>) => {
  importFile(e.data).then(
    (project) => post({ type: "done", project }),
    (err: unknown) => post({ type: "error", message: err instanceof Error ? err.message : String(err) }),
  );
});
