import { describe, expect, it } from "vitest";
import { OverlapAdder, planChunks, transitionWindow } from "./chunks";
import { SeparationStream, type SeparationModel } from "./separate";

const SEGMENT = 400;
const OPTIONS = { segment: SEGMENT, overlap: 0.25, sources: 4 };

function noise(length: number, seed: number): Float32Array {
  let s = seed;
  return Float32Array.from({ length }, () => {
    s = (s * 16807) % 2147483647;
    return s / 2147483647 - 0.5;
  });
}

/** Stem k is (k + 1) / 10 of the mix plus a term that depends on where in the segment a sample sits. */
class FakeModel implements SeparationModel {
  calls = 0;
  run(mix: Float32Array): Promise<Float32Array> {
    this.calls++;
    const out = new Float32Array(4 * 2 * SEGMENT);
    for (let k = 0; k < 4; k++) {
      for (let c = 0; c < 2; c++) {
        for (let i = 0; i < SEGMENT; i++) {
          out[(k * 2 + c) * SEGMENT + i] =
            ((k + 1) / 10) * (mix[c * SEGMENT + i] ?? 0) + 0.001 * k * (i / SEGMENT);
        }
      }
    }
    return Promise.resolve(out);
  }
}

async function stream(left: Float32Array, right: Float32Array, block: number) {
  const s = new SeparationStream(new FakeModel(), OPTIONS);
  const parts: Float32Array[][][] = [];
  for (let at = 0; at < left.length; at += block) {
    parts.push(await s.push(left.subarray(at, at + block), right.subarray(at, at + block)));
  }
  parts.push(await s.finish());
  return [0, 1, 2, 3].map((k) =>
    [0, 1].map((c) => {
      const pieces = parts.map((p) => p[k]?.[c] ?? new Float32Array(0));
      const out = new Float32Array(pieces.reduce((n, p) => n + p.length, 0));
      let at = 0;
      for (const p of pieces) {
        out.set(p, at);
        at += p.length;
      }
      return out;
    }),
  );
}

/** Whole-signal reference built from planChunks and OverlapAdder. */
async function reference(left: Float32Array, right: Float32Array) {
  const model = new FakeModel();
  const window = transitionWindow(SEGMENT);
  const adders = [0, 1, 2, 3].map(() => [new OverlapAdder(left.length), new OverlapAdder(left.length)]);
  for (const c of planChunks(left.length, SEGMENT, Math.floor(SEGMENT / 4))) {
    const len = c.end - c.start;
    const mix = new Float32Array(2 * SEGMENT);
    mix.set(left.subarray(c.start, c.end), 0);
    mix.set(right.subarray(c.start, c.end), SEGMENT);
    const out = await model.run(mix);
    for (let k = 0; k < 4; k++) {
      for (let ch = 0; ch < 2; ch++) {
        const base = (k * 2 + ch) * SEGMENT;
        adders[k]?.[ch]?.add(c.start, out.subarray(base, base + len), window.subarray(0, len));
      }
    }
  }
  return adders.map((pair) => pair.map((a) => a.finish()));
}

describe("SeparationStream", () => {
  it.each([50, 333, 10_000])("matches the whole-signal overlap-add with %i-sample pushes", async (block) => {
    const left = noise(2345, 1);
    const right = noise(2345, 2);
    const got = await stream(left, right, block);
    const want = await reference(left, right);
    for (let k = 0; k < 4; k++) {
      for (let c = 0; c < 2; c++) {
        const g = got[k]?.[c] ?? new Float32Array(0);
        const w = want[k]?.[c] ?? new Float32Array(0);
        expect(g.length).toBe(left.length);
        let worst = 0;
        for (let i = 0; i < g.length; i++) worst = Math.max(worst, Math.abs((g[i] ?? 0) - (w[i] ?? 0)));
        expect(worst).toBeLessThan(1e-5);
      }
    }
  });

  it("handles input shorter than one segment", async () => {
    const left = noise(100, 3);
    const got = await stream(left, left, 64);
    expect(got[3]?.[0]?.length).toBe(100);
  });
});
