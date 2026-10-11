import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  DIARIZATION,
  bestAssignment,
  clusterVoices,
  decodePowerset,
  soloSpans,
  speakerActivity,
  windowStarts,
  type Window,
} from "./diarize";
import { Fbank } from "./fbank";

const dir = new URL("../fixtures/speakers/", import.meta.url);
const bytes = (name: string) => readFileSync(new URL(name, dir));
const f32 = (name: string) => {
  const b = bytes(name);
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4).slice();
};
const parsed: unknown = JSON.parse(bytes("meta.json").toString());
function field(key: string): number {
  const v: unknown = typeof parsed === "object" && parsed !== null ? Reflect.get(parsed, key) : undefined;
  if (typeof v !== "number") throw new Error(`fixture meta.json has no number ${key}`);
  return v;
}
const meta = {
  samples: field("samples"),
  windows: field("windows"),
  framesPerWindow: field("framesPerWindow"),
  embeddings: field("embeddings"),
  dimension: field("dimension"),
  speakers: field("speakers"),
};

describe("Fbank", () => {
  it("matches the NumPy reference, itself checked against torchaudio", () => {
    const want = f32("fbank.f32");
    const got = new Fbank().compute(f32("fbank_input.f32"));
    expect(got.length).toBe(want.length);
    let worst = 0;
    for (let i = 0; i < want.length; i++) worst = Math.max(worst, Math.abs((got[i] ?? 0) - (want[i] ?? 0)));
    expect(worst).toBeLessThan(1e-3);
  });
});

describe("diarization", () => {
  it("decodes powerset classes to speaker bit masks", () => {
    expect([...decodePowerset(Uint8Array.of(0, 1, 2, 3, 4, 5, 6))]).toEqual([0, 1, 2, 4, 3, 5, 6]);
  });

  it("finds where a local speaker talks alone", () => {
    const masks = new Uint8Array(10);
    masks.set([1, 1, 3, 1], 2);
    expect(soloSpans(masks, 0)).toEqual([
      [32_000, 64_000],
      [80_000, 96_000],
    ]);
  });

  it("pairs local speakers with voices for the highest similarity", () => {
    expect(
      bestAssignment(
        [
          [0.1, 0.9],
          [0.8, 0.7],
        ],
        2,
      ),
    ).toEqual([
      [0, 1],
      [1, 0],
    ]);
    expect(bestAssignment([[0.9], [0.95], [0.1]], 1)).toEqual([[1, 0]]);
  });

  it("places 10 s windows every 2 s", () => {
    expect(windowStarts(200_000)).toEqual([0, 32_000, 64_000]);
    expect(windowStarts(1000)).toEqual([0]);
  });

  it("matches the reference speaker activity, voices in order of appearance", () => {
    expect(windowStarts(meta.samples)).toHaveLength(meta.windows);
    const classes = bytes("classes.u8");
    const k16 = bytes("keys.u16");
    const keys = new Uint16Array(k16.buffer, k16.byteOffset, k16.byteLength / 2);
    const embs = f32("embeddings.f32");
    const windows: Window[] = windowStarts(meta.samples).map((start, w) => ({
      start,
      masks: decodePowerset(classes.subarray(w * meta.framesPerWindow, (w + 1) * meta.framesPerWindow)),
      embeddings: new Map(),
    }));
    const all: Float32Array[] = [];
    for (let e = 0; e < meta.embeddings; e++) {
      const v = embs.slice(e * meta.dimension, (e + 1) * meta.dimension);
      windows[keys[2 * e] ?? 0]?.embeddings.set(keys[2 * e + 1] ?? 0, v);
      all.push(v);
    }
    const active = speakerActivity(windows, clusterVoices(all, meta.speakers), meta.samples);

    const want = bytes("activity.u8");
    const frames = Math.floor(meta.samples / DIARIZATION.frame) + 1;
    expect(active).toHaveLength(meta.speakers);
    for (const [s, mask] of active.entries())
      expect([...mask]).toEqual([...want.subarray(s * frames, (s + 1) * frames)]);
  });
});
