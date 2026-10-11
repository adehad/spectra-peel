import { spawnSync } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { hasFfmpeg } from "./testing/ffmpeg";
import { LoudnessMeter, kWeighting, normalizationGain } from "./loudness";

function sine(rate: number, hz: number, seconds: number, amplitude: number): Float32Array {
  return Float32Array.from(
    { length: Math.round(rate * seconds) },
    (_, i) => amplitude * Math.sin((2 * Math.PI * hz * i) / rate),
  );
}

function measure(rate: number, planes: Float32Array[]) {
  const m = new LoudnessMeter(rate, planes.length);
  for (let at = 0; at < (planes[0]?.length ?? 0); at += 4096)
    m.push(planes.map((p) => p.subarray(at, at + 4096)));
  return m.result();
}

describe("kWeighting", () => {
  it("reproduces the BS.1770 coefficients at 48 kHz", () => {
    const [shelf, highPass] = kWeighting(48_000);
    expect(shelf.b[0]).toBeCloseTo(1.53512485958697, 8);
    expect(shelf.b[1]).toBeCloseTo(-2.69169618940638, 8);
    expect(shelf.a[0]).toBeCloseTo(-1.69065929318241, 8);
    expect(highPass.a[0]).toBeCloseTo(-1.99004745483398, 8);
    expect(highPass.a[1]).toBeCloseTo(0.99007225036621, 8);
  });
});

describe("LoudnessMeter", () => {
  it("reads a full-scale 997 Hz sine in both channels as 0 LUFS, and in one channel as -3 LUFS", () => {
    const s = sine(48_000, 997, 5, 1);
    expect(measure(48_000, [s, s]).integrated).toBeCloseTo(0, 1);
    expect(measure(48_000, [s]).integrated).toBeCloseTo(-3.01, 1);
  });

  it("ignores silence through the absolute gate, so a silent half does not halve the reading", () => {
    const tone = sine(44_100, 997, 4, 0.1);
    const withGap = new Float32Array(tone.length * 2);
    withGap.set(tone);
    // Blocks straddling the edge are partly silent and still pass both gates, so it reads a little lower.
    const drop = measure(44_100, [tone]).integrated - measure(44_100, [withGap]).integrated;
    expect(drop).toBeGreaterThanOrEqual(0);
    expect(drop).toBeLessThan(0.3);
  });

  it("finds inter-sample peaks above the sample peak", () => {
    // A quarter-rate sine sampled at 45 degrees never lands on its crest.
    const s = Float32Array.from({ length: 48_000 }, (_, i) => Math.sin((Math.PI / 2) * i + Math.PI / 4));
    const r = measure(48_000, [s]);
    expect(r.truePeak).toBeGreaterThan(-0.5);
    expect(r.truePeak).toBeLessThan(0.5);
  });

  it("returns -Infinity loudness for silence", () => {
    expect(measure(48_000, [new Float32Array(48_000)]).integrated).toBe(-Infinity);
  });

  it.skipIf(!hasFfmpeg)("agrees with ffmpeg's ebur128 on noisy music-like audio", () => {
    const rate = 44_100;
    let seed = 7;
    const n = rate * 8;
    const left = new Float32Array(n);
    const right = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      seed = (seed * 16807) % 2147483647;
      const t = i / rate;
      const env = 0.3 + 0.7 * Math.abs(Math.sin(t * 1.3));
      left[i] = env * (0.3 * Math.sin(2 * Math.PI * 220 * t) + 0.1 * (seed / 2147483647 - 0.5));
      right[i] = env * (0.25 * Math.sin(2 * Math.PI * 330 * t) + 0.1 * (seed / 2147483647 - 0.5));
    }
    const dir = mkdtempSync(join(tmpdir(), "spectra-lufs-"));
    const raw = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
      raw[2 * i] = left[i] ?? 0;
      raw[2 * i + 1] = right[i] ?? 0;
    }
    writeFileSync(join(dir, "a.f32"), raw);
    const run = spawnSync(
      "ffmpeg",
      [
        "-hide_banner",
        "-nostats",
        "-f",
        "f32le",
        "-ar",
        String(rate),
        "-ac",
        "2",
        "-i",
        join(dir, "a.f32"),
        "-af",
        "ebur128=peak=true",
        "-f",
        "null",
        "-",
      ],
      { encoding: "utf8" },
    );
    const summary = run.stderr.slice(run.stderr.lastIndexOf("Summary:"));
    const integrated = Number(/I:\s+(-?[\d.]+) LUFS/.exec(summary)?.[1]);
    const peak = Number(/Peak:\s+(-?[\d.]+) dBFS/.exec(summary)?.[1]);
    const ours = measure(rate, [left, right]);
    expect(Math.abs(ours.integrated - integrated)).toBeLessThan(0.15);
    expect(Math.abs(ours.truePeak - peak)).toBeLessThan(0.3);
  });
});

describe("normalizationGain", () => {
  it("stops at the peak ceiling when the target would clip", () => {
    expect(normalizationGain({ integrated: -20, truePeak: -3 }, -14, -1)).toBeCloseTo(2);
    expect(normalizationGain({ integrated: -20, truePeak: -10 }, -14, -1)).toBeCloseTo(6);
    expect(normalizationGain({ integrated: -Infinity, truePeak: -Infinity }, -14, -1)).toBe(0);
  });
});
