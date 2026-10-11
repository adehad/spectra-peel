import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { concatFloat32 } from "./buffers";
import { DFN3, Dfn3Analysis, Dfn3Stream, erbWidths, type Dfn3Model } from "./dfn3";

const dir = new URL("../fixtures/dfn3/", import.meta.url);
function fixture(name: string): Float32Array {
  const b = readFileSync(new URL(`${name}.f32`, dir));
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4).slice();
}
const input = fixture("input");
const featErb = fixture("feat_erb");
const featSpec = fixture("feat_spec");
const erbMask = fixture("erb_mask");
const dfCoefs = fixture("df_coefs");
const output = fixture("output");
const T = featErb.length / DFN3.erbBands;

function snrDb(reference: Float32Array, test: Float32Array): number {
  let s = 0;
  let e = 0;
  for (let i = 0; i < reference.length; i++) {
    s += (reference[i] ?? 0) ** 2;
    e += ((reference[i] ?? 0) - (test[i] ?? 0)) ** 2;
  }
  return 10 * Math.log10(s / e);
}

/** Replays the reference graph outputs, checking it is fed the right frames and advanced by `keep`. */
class ReplayModel implements Dfn3Model {
  position = 0;
  calls: { frames: number; keep: number }[] = [];

  run(erb: Float32Array, spec: Float32Array, frames: number, keep: number) {
    const a = this.position;
    for (let t = 0; t < frames; t++) {
      for (let b = 0; b < DFN3.erbBands; b++) {
        const want = featErb[(a + t) * DFN3.erbBands + b] ?? NaN;
        if (Math.abs((erb[t * DFN3.erbBands + b] ?? NaN) - want) > 1e-4)
          throw new Error(`erb feature mismatch at frame ${a + t}`);
      }
      const re = spec[t * DFN3.dfBins + 5] ?? NaN;
      if (Math.abs(re - (featSpec[((a + t) * DFN3.dfBins + 5) * 2] ?? NaN)) > 1e-4)
        throw new Error(`spec feature mismatch at ${a + t}`);
    }
    const mask = erbMask.slice(a * DFN3.erbBands, (a + frames) * DFN3.erbBands);
    const coefs = new Float32Array(DFN3.dfOrder * frames * DFN3.dfBins * 2);
    for (let n = 0; n < DFN3.dfOrder; n++) {
      const from = (n * T + a) * DFN3.dfBins * 2;
      coefs.set(dfCoefs.subarray(from, from + frames * DFN3.dfBins * 2), n * frames * DFN3.dfBins * 2);
    }
    this.calls.push({ frames, keep });
    this.position += keep;
    return Promise.resolve({ erbMask: mask, dfCoefs: coefs });
  }
}

async function denoise(model: Dfn3Model, chunk: number, block: number): Promise<Float32Array> {
  const stream = new Dfn3Stream(model, chunk);
  const parts: Float32Array[] = [];
  for (let at = 0; at < input.length; at += block)
    parts.push(await stream.push(input.subarray(at, at + block)));
  parts.push(await stream.finish());
  return concatFloat32(parts);
}

describe("erbWidths", () => {
  it("covers all 481 bins with at least two bins per band", () => {
    const w = erbWidths();
    expect(w).toHaveLength(32);
    expect(w.reduce((a, b) => a + b, 0)).toBe(481);
    expect(Math.min(...w)).toBe(2);
  });
});

describe("Dfn3Analysis", () => {
  it("matches the NumPy reference features", () => {
    const a = new Dfn3Analysis();
    const padded = new Float32Array(input.length + DFN3.fftSize);
    padded.set(input);
    let worstErb = 0;
    let worstSpec = 0;
    for (let t = 0; t < T; t++) {
      const f = a.frame(padded.subarray(t * DFN3.hop, (t + 1) * DFN3.hop));
      for (let b = 0; b < DFN3.erbBands; b++) {
        worstErb = Math.max(worstErb, Math.abs((f.erb[b] ?? 0) - (featErb[t * DFN3.erbBands + b] ?? 0)));
      }
      for (let i = 0; i < DFN3.dfBins * 2; i++) {
        worstSpec = Math.max(
          worstSpec,
          Math.abs((f.cplx[i] ?? 0) - (featSpec[t * DFN3.dfBins * 2 + i] ?? 0)),
        );
      }
    }
    expect(worstErb).toBeLessThan(1e-4);
    expect(worstSpec).toBeLessThan(1e-4);
  });
});

describe("Dfn3Stream", () => {
  it.each([
    [500, 4096],
    [37, 1000],
    [7, 333],
  ])("reproduces the reference output with %i-frame chunks", async (chunk, block) => {
    const model = new ReplayModel();
    const out = await denoise(model, chunk, block);
    expect(out.length).toBe(input.length);
    expect(snrDb(output, out)).toBeGreaterThan(80);
    expect(model.position).toBe(T);
    for (const c of model.calls.slice(0, -1)) expect(c.frames).toBe(c.keep + DFN3.convLookahead);
  });
});
