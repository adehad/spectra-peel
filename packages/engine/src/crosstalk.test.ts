import { describe, expect, it } from "vitest";
import { frameGain, overlapPieces, projectToMixture, separateCrosstalk } from "./crosstalk";
import { DIARIZATION } from "./diarize";

const { frame: FRAME, sampleRate: SR } = DIARIZATION;
const SEGMENT = 6 * SR;
const embed = () => Promise.resolve(Float32Array.of(1, 1));

function masks(frames: number, ...spans: [number, number][][]): Uint8Array[] {
  return spans.map((list) => {
    const m = new Uint8Array(frames);
    for (const [a, b] of list) m.fill(1, a, b);
    return m;
  });
}

describe("overlapPieces", () => {
  it("merges crosstalk across short gaps and drops blips", () => {
    const active = masks(
      1000,
      [[0, 500]],
      [
        [100, 120],
        [130, 160],
        [400, 402],
      ],
    );
    expect(overlapPieces(active, SEGMENT)).toEqual([{ start: 100, end: 160, speakers: [0, 1] }]);
  });

  it("splits long crosstalk to leave a second of context each side", () => {
    const active = masks(2000, [[0, 2000]], [[0, 1000]]);
    const longest = Math.floor((SEGMENT - 2 * SR) / FRAME);
    const pieces = overlapPieces(active, SEGMENT);
    expect(pieces.map((p) => p.end - p.start)).toEqual([
      longest,
      longest,
      longest,
      longest,
      1000 - 4 * longest,
    ]);
  });

  it("picks the two most active of three voices", () => {
    const active = masks(400, [[0, 50]], [[0, 400]], [[0, 300]]);
    expect(overlapPieces(active, SEGMENT)[0]?.speakers).toEqual([1, 2]);
  });
});

describe("frameGain", () => {
  it("crossfades each switch over 20 ms, and complementary masks sum to one", () => {
    const a = new Uint8Array(100);
    a.fill(1, 50);
    const b = a.map((v) => 1 - v);
    const ga = new Float32Array(SR);
    const gb = new Float32Array(SR);
    frameGain(a, SR, 0, ga);
    frameGain(b, SR, 0, gb);
    const edge = 50 * FRAME;
    expect(ga[edge]).toBeCloseTo(0.5, 2);
    expect(ga[edge - 0.011 * SR]).toBe(0);
    expect(ga[edge + 0.011 * SR]).toBe(1);
    for (let i = 0; i < SR; i++) expect((ga[i] ?? 0) + (gb[i] ?? 0)).toBeCloseTo(1, 5);
  });

  it("gives the same curve at any rate", () => {
    const m = new Uint8Array(20).fill(1, 5, 12);
    const at16 = new Float32Array(16_000 / 10);
    const at48 = new Float32Array(48_000 / 10);
    frameGain(m, 16_000, 0, at16);
    frameGain(m, 48_000, 0, at48);
    for (let i = 0; i < at16.length; i++) expect(at48[3 * i]).toBeCloseTo(at16[i] ?? 0, 5);
  });
});

describe("projectToMixture", () => {
  it("rescales the outputs so they sum back to the mix", () => {
    const s1 = Float32Array.from({ length: 1000 }, (_, i) => Math.sin(i / 7));
    const s2 = Float32Array.from({ length: 1000 }, (_, i) => Math.sin(i / 3 + 1));
    const mix = s1.map((v, i) => v + (s2[i] ?? 0));
    const y: [Float32Array, Float32Array] = [s1.map((v) => 2 * v), s2.map((v) => 0.5 * v)];
    projectToMixture(y, mix);
    for (let i = 0; i < 1000; i++) expect((y[0][i] ?? 0) + (y[1][i] ?? 0)).toBeCloseTo(mix[i] ?? 0, 4);
  });
});

describe("separateCrosstalk", () => {
  it("gives each output to the voice that owns it where that voice talks alone", async () => {
    const frames = 900;
    const total = frames * FRAME;
    const tone = (hz: number, from: number, to: number) =>
      Float32Array.from({ length: total }, (_, i) =>
        i >= from * FRAME && i < to * FRAME ? 0.1 * Math.sin((2 * Math.PI * hz * i) / SR) : 0,
      );
    const a = tone(220, 0, 500);
    const b = tone(330, 300, 900);
    const mix = a.map((v, i) => v + (b[i] ?? 0));
    const active = masks(frames, [[0, 500]], [[300, 900]]);
    let c0 = 0;
    const model = {
      segment: SEGMENT,
      // An oracle that returns the voices in the wrong order.
      run(chunk: Float32Array) {
        const scale =
          (chunk.reduce((s, v, i) => s + v * (mix[c0 + i] ?? 0), 0) || 1) /
          (mix.slice(c0, c0 + SEGMENT).reduce((s, v) => s + v * v, 0) || 1);
        const out = new Float32Array(2 * SEGMENT);
        out.set(
          b.subarray(c0, c0 + SEGMENT).map((v) => v * scale),
          0,
        );
        out.set(
          a.subarray(c0, c0 + SEGMENT).map((v) => v * scale),
          SEGMENT,
        );
        return Promise.resolve(out);
      },
    };
    const mid = Math.floor((300 + 500) / 2) * FRAME;
    c0 = Math.min(Math.max(mid - SEGMENT / 2, 0), total - SEGMENT);
    const voices = { centroids: [Float64Array.of(1, 0), Float64Array.of(0, 1)], mean: Float64Array.of(0, 0) };
    const pieces = await separateCrosstalk(mix, active, model, embed, voices);
    expect(pieces.map((p) => p.speaker).toSorted((x, y) => x - y)).toEqual([0, 1]);
    for (const p of pieces) {
      const truth = p.speaker === 0 ? a : b;
      let err = 0;
      let sig = 0;
      for (const [i, v] of p.audio.entries()) {
        const t = truth[p.start + i] ?? 0;
        if (Math.abs(v) < 1e-9) continue;
        err += (v - t) ** 2;
        sig += t * t;
      }
      expect(err / sig).toBeLessThan(0.05);
    }
  });

  it("resumes after the pieces an earlier run separated", async () => {
    const frames = 1800;
    const total = frames * FRAME;
    const mix = Float32Array.from({ length: total }, (_, i) => 0.1 * Math.sin(i / 7));
    const active = masks(
      frames,
      [[0, 1800]],
      [
        [100, 300],
        [1300, 1500],
      ],
    );
    const runs: number[] = [];
    const model = {
      segment: SEGMENT,
      run(chunk: Float32Array) {
        runs.push(chunk.length);
        const out = new Float32Array(2 * SEGMENT);
        out.set(chunk.map((v) => v / 2));
        out.set(
          chunk.map((v) => v / 2),
          SEGMENT,
        );
        return Promise.resolve(out);
      },
    };
    const voices = { centroids: [Float64Array.of(1, 0), Float64Array.of(0, 1)], mean: Float64Array.of(0, 0) };
    const seen: number[] = [];
    const all = await separateCrosstalk(mix, active, model, embed, voices, {
      onPiece: (done, _total, outputs) => {
        seen.push(done, outputs.length);
      },
    });
    expect(seen).toEqual([1, 2, 2, 2]);
    runs.length = 0;
    const rest = await separateCrosstalk(mix, active, model, embed, voices, { from: 1 });
    expect(runs).toHaveLength(1);
    expect(rest.map((p) => p.start)).toEqual(all.slice(2).map((p) => p.start));
  });
});
