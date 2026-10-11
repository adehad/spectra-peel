import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import { Sha256 } from "./sha256";

describe("Sha256", () => {
  it("matches node:crypto for lengths around the padding boundaries", () => {
    for (const n of [0, 1, 55, 56, 63, 64, 65, 119, 1000, 100_003]) {
      const data = Uint8Array.from({ length: n }, (_, i) => (i * 131 + 7) & 0xff);
      expect(new Sha256().update(data).hex()).toBe(createHash("sha256").update(data).digest("hex"));
    }
  });

  it("gives the same digest however the input is split", () => {
    const data = Uint8Array.from({ length: 5000 }, (_, i) => (i * 17) & 0xff);
    const h = new Sha256();
    for (let at = 0; at < data.length; at += 333) h.update(data.subarray(at, at + 333));
    expect(h.hex()).toBe(createHash("sha256").update(data).digest("hex"));
  });
});
