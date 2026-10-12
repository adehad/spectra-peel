import { describe, expect, it } from "vitest";
import { pack, unpack } from "./checkpoint";

describe("pack", () => {
  it("round-trips nested typed arrays, infinities and plain values", () => {
    const value = {
      at: 12_345,
      min: Infinity,
      max: -Infinity,
      flac: { index: Float64Array.of(42, 4096.5), block: [Int32Array.of(1, -2, 3), Int32Array.of(7)] },
      tiles: new Uint8Array([1, 2, 3]),
      pending: [[Float32Array.of(0.25, -0.5)], []],
      label: "Speech",
      nothing: null,
    };
    const bytes = pack(value);
    expect(bytes.byteLength % 8).toBe(0);
    expect(unpack(bytes.buffer)).toEqual(value);
  });
});
