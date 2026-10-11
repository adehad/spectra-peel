import { describe, expect, it } from "vitest";
import { stemModels } from "./models";

describe("stemModels", () => {
  it("runs only the fast model when no stem asks for its specialist", () => {
    expect(stemModels([])).toEqual(["htdemucs-stems"]);
  });

  it("adds a specialist per chosen stem after the fast model", () => {
    expect(stemModels([0, 3])).toEqual(["htdemucs-stems", "htdemucs-ft-drums", "htdemucs-ft-vocals"]);
  });

  it("skips the fast model when every stem has its specialist", () => {
    expect(stemModels([0, 1, 2, 3])).toEqual([
      "htdemucs-ft-drums",
      "htdemucs-ft-bass",
      "htdemucs-ft-other",
      "htdemucs-ft-vocals",
    ]);
  });
});
