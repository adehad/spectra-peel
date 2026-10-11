import { ENGINE_VERSION, SPECTROGRAM } from "@spectra-peel/engine";
import { ALL_FORMATS, AudioSampleSink, BlobSource, Input } from "mediabunny";
import { SOURCE_LAYER, type Project } from "../project/schema";
import { paths, projectDir, projectsDir } from "../storage/opfs";
import { LayerWriter, writeWhole } from "./layer-writer";
import type { ImportMessage, ImportRequest } from "./protocol";

const post = (m: ImportMessage) => postMessage(m);

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
  const writer = await LayerWriter.create(
    dir,
    SOURCE_LAYER,
    paths.sourceFlac,
    paths.sourceIndex,
    sampleRate,
    channels,
  );

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

      await writer.push(planes);

      frames += n;
      if (performance.now() - lastReport > 200) {
        lastReport = performance.now();
        post({ type: "progress", fraction: Math.min(1, frames / expected) });
      }
    }
  } catch (e) {
    writer.abort();
    throw e;
  }
  await writer.finish();

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
        start: 0,
        frames,
        data: { kind: "flac", path: paths.sourceFlac, index: paths.sourceIndex },
      },
    ],
    ops: [],
    mix: { gains: {}, muted: [], solo: null, ab: "mix" },
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
