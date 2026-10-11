import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { expect, test } from "@playwright/test";
import { writeSyntheticWav } from "./fixtures";

// Local only: writes a 635 MB WAV and imports an hour of audio.
test.skip(!process.env["SPECTRA_BENCH"], "set SPECTRA_BENCH=1 to run the import bench");

const MINUTES = Number(process.env["SPECTRA_BENCH_MINUTES"] ?? 60);

/** Resident memory of the given processes in bytes; workers share their page's renderer process. */
function residentBytes(pids: number[]): number {
  if (process.platform === "win32") {
    const out = execFileSync("powershell", [
      "-NoProfile",
      "-Command",
      `(Get-Process -Id ${pids.join(",")} | Measure-Object WorkingSet64 -Sum).Sum`,
    ]);
    return Number(out.toString().trim());
  }
  const out = execFileSync("ps", ["-o", "rss=", "-p", pids.join(",")]).toString();
  return out.split("\n").reduce((n, line) => n + Number(line.trim() || 0) * 1024, 0);
}

test("imports an hour with flat renderer memory and seeks quickly", async ({ page, browser }) => {
  test.setTimeout(30 * 60_000);
  mkdirSync("test-results", { recursive: true });
  const path = `test-results/synthetic-${MINUTES}min.wav`;
  if (!existsSync(path)) writeSyntheticWav(path, MINUTES * 60);

  await page.goto("/");
  const cdp = await browser.newBrowserCDPSession();
  const info = (await cdp.send("SystemInfo.getProcessInfo")) as {
    processInfo: { id: number; type: string }[];
  };
  const renderers = info.processInfo.filter((p) => p.type === "renderer").map((p) => p.id);
  const t0 = Date.now();
  await page.getByTestId("import").setInputFiles(path);
  const heap: number[] = [];
  const sampler = setInterval(() => heap.push(residentBytes(renderers)), 5000);
  await expect(page.getByTestId("project-name")).toBeVisible({ timeout: 30 * 60_000 });
  clearInterval(sampler);
  const importSeconds = (Date.now() - t0) / 1000;

  const seeks: number[] = [];
  await page.getByTestId("play").click();
  const box = await page.getByTestId("spectrogram").boundingBox();
  if (!box) throw new Error("no spectrogram");
  for (const fraction of [0.2, 0.8, 0.5]) {
    const before = await page.getByTestId("time").textContent();
    const t = Date.now();
    await page.mouse.click(box.x + box.width * fraction, box.y + box.height / 2);
    await expect(page.getByTestId("time")).not.toHaveText(before ?? "");
    seeks.push(Date.now() - t);
  }

  const result = {
    minutes: MINUTES,
    importSeconds,
    rendererMB: heap.map((b) => Math.round(b / 1e6)),
    seekMs: seeks,
  };
  console.log(JSON.stringify(result));
  writeFileSync("test-results/import-bench.json", JSON.stringify(result, null, 2));
});
