import { describe, expect, it } from "vitest";
import { Xorshift32, quantize16 } from "./dither";
import { FlacWriter } from "./flac";
import { PeaksBuilder } from "./peaks";
import { Resampler } from "./resample";
import { SeparationStream, type SeparationModel } from "./separate";
import { SpectrogramBuilder } from "./spectrogram";

function noise(length: number, seed: number): Float32Array {
  let s = seed;
  return Float32Array.from({ length }, () => {
    s = (s * 16807) % 2147483647;
    return (s / 2147483647 - 0.5) * 0.8;
  });
}

/** `input` cut into uneven blocks, as a stream receives it. */
function blocks(input: Float32Array): Float32Array[] {
  const out: Float32Array[] = [];
  for (let i = 0, n = 0; i < input.length; i += 997 + 311 * (n++ % 5))
    out.push(input.subarray(i, Math.min(input.length, i + 997 + 311 * (n % 5))));
  return out;
}

const concat = (parts: readonly Float32Array[]) => {
  const out = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
};

/** A snapshot goes through structured cloning on its way to storage and back. */
const stored = <T>(state: T): T => structuredClone(state);

describe("snapshot and restore", () => {
  const input = noise(48_000, 7);
  const parts = blocks(input);
  const half = Math.floor(parts.length / 2);

  const resampled = (cut: boolean) => {
    let r = new Resampler(48_000, 44_100);
    const out = parts.map((p, i) => {
      if (cut && i === half) {
        const next = new Resampler(48_000, 44_100);
        next.restore(stored(r.snapshot()));
        r = next;
      }
      return r.push(p);
    });
    return concat([...out, r.finish()]);
  };

  it("resumes a resampler exactly", () => {
    expect(resampled(true)).toEqual(resampled(false));
  });

  const writers = (cut: boolean) => {
    let rng = new Xorshift32(0x5eed);
    let peaks = new PeaksBuilder();
    let spec = new SpectrogramBuilder(48_000);
    let flac = new FlacWriter({ sampleRate: 48_000, channels: 1 });
    const frames: Uint8Array[] = [];
    const tiles: Uint8Array[] = [];
    for (const [i, p] of parts.entries()) {
      if (cut && i === half) {
        const [r, pk, sp, fl] = [rng.snapshot(), peaks.snapshot(), spec.snapshot(), flac.snapshot()];
        rng = new Xorshift32(1);
        rng.restore(stored(r));
        peaks = new PeaksBuilder();
        peaks.restore(stored(pk));
        spec = new SpectrogramBuilder(48_000);
        spec.restore(stored(sp));
        flac = new FlacWriter({ sampleRate: 48_000, channels: 1 });
        flac.restore(stored(fl));
      }
      frames.push(...flac.push([quantize16(p, rng)]));
      peaks.push([p]);
      tiles.push(...spec.push(p).map((t) => t.data));
    }
    frames.push(...flac.finish());
    tiles.push(...spec.finish().map((t) => t.data));
    return { frames, tiles, index: [...flac.index], peaks: peaks.finish().levels };
  };

  it("resumes dither, peaks, spectrogram tiles and FLAC frames exactly", () => {
    expect(writers(true)).toEqual(writers(false));
  });

  it("resumes a separation stream exactly", async () => {
    const segment = 4000;
    const model: SeparationModel = {
      run: (mix) => {
        const out = new Float32Array(2 * 2 * segment);
        for (let i = 0; i < 2 * segment; i++) {
          out[i] = 0.3 * (mix[i] ?? 0) + 0.01 * Math.sin(i);
          out[2 * segment + i] = 0.7 * (mix[i] ?? 0);
        }
        return Promise.resolve(out);
      },
    };
    const options = { segment, overlap: 0.4, sources: 2 };
    const run = async (cut: boolean) => {
      let stream = new SeparationStream(model, options);
      const out: Float32Array[] = [];
      for (const [i, p] of parts.entries()) {
        if (cut && i === half) {
          const next = new SeparationStream(model, options);
          next.restore(stored(stream.snapshot()));
          stream = next;
        }
        out.push(
          ...(
            await stream.push(
              p,
              p.map((v) => -v),
            )
          ).flat(),
        );
      }
      out.push(...(await stream.finish()).flat());
      return concat(out);
    };
    expect(await run(true)).toEqual(await run(false));
  });
});
