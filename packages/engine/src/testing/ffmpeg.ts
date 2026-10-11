import { execFileSync } from "node:child_process";

/** Whether ffmpeg is on PATH, for cross-checks that skip without it. */
export const hasFfmpeg = (() => {
  try {
    execFileSync("ffmpeg", ["-version"], { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
})();
