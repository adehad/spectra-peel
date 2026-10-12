import { describe, expect, it } from "vitest";
import prompts from "./extract-prompts.json";
import { EXTRACT_PROMPTS } from "./models";

describe("extract prompts", () => {
  it("has a unit-length 512-d CLAP embedding for each prompt, and no others", () => {
    expect(Object.keys(prompts).toSorted()).toEqual([...EXTRACT_PROMPTS].toSorted());
    for (const e of Object.values(prompts)) {
      expect(e).toHaveLength(512);
      expect(Math.hypot(...e)).toBeCloseTo(1, 5);
    }
  });
});
