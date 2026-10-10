import { mkdirSync, writeFileSync } from "node:fs";
import { test } from "@playwright/test";
import type { RunRequest } from "../apps/web/spike/protocol";

// Local only: downloads ~490 MB of models into the browser profile's OPFS and takes minutes.
test.skip(!process.env["SPECTRA_BENCH"], "set SPECTRA_BENCH=1 to run the inference bench");

const RUNS: Omit<RunRequest, "modelFile">[] = [
  { model: "dfn3", ep: "wasm", seconds: 30 },
  { model: "dfn3", ep: "webgpu", seconds: 30 },
  { model: "demucs-vocals-fp16", ep: "webgpu", seconds: 30 },
  { model: "demucs-vocals-fp32", ep: "webgpu", seconds: 30 },
  { model: "demucs-vocals-fp32", ep: "wasm", seconds: 30 },
];

// SPECTRA_BENCH_MODEL=path/to/model.onnx benches a locally patched Demucs file instead of the downloads.
const LOCAL_MODEL = process.env["SPECTRA_BENCH_MODEL"];
const LOCAL_EPS = (process.env["SPECTRA_BENCH_EPS"] ?? "webgpu,wasm").split(",");
const LOCAL_SECONDS = process.env["SPECTRA_BENCH_SECONDS"] ?? "30";

test("local model bench", async ({ page }) => {
  test.skip(!LOCAL_MODEL, "set SPECTRA_BENCH_MODEL to bench a local .onnx");
  test.setTimeout(60 * 60_000);
  await page.goto("/spike/");
  const results = [];
  for (const ep of LOCAL_EPS) {
    await page.locator("#file").setInputFiles(LOCAL_MODEL ?? "");
    await page.locator("#model").selectOption("demucs-vocals-fp16");
    await page.locator("#ep").selectOption(ep);
    await page.locator("#seconds").fill(LOCAL_SECONDS);
    await page.locator("#run").click();
    await page.waitForFunction(
      () => window.spikeResult !== undefined && document.getElementById("status")?.textContent !== "idle",
      undefined,
      { timeout: 30 * 60_000 },
    );
    const r = await page.evaluate(() => {
      const res = window.spikeResult;
      delete window.spikeResult;
      return res;
    });
    console.log(JSON.stringify({ ep, report: r?.report, error: r?.error }));
    results.push({ ep, ...r });
  }
  mkdirSync("test-results", { recursive: true });
  writeFileSync("test-results/spike-bench-local.json", JSON.stringify(results, null, 2));
});

test("inference bench", async ({ page }) => {
  test.skip(LOCAL_MODEL !== undefined, "local model run requested");
  test.setTimeout(60 * 60_000);
  await page.goto("/spike/");
  const results = [];
  for (const req of RUNS) {
    const r = await page.evaluate((q) => window.runSpike(q), req);
    console.log(
      JSON.stringify({ req, report: r.report, error: r.error, peakMemoryBytes: r.peakMemoryBytes }),
    );
    results.push({ req, ...r });
  }
  mkdirSync("test-results", { recursive: true });
  writeFileSync("test-results/spike-bench.json", JSON.stringify(results, null, 2));
});
