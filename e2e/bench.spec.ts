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

// SPECTRA_BENCH_MODEL=a.onnx[,b.onnx] benches locally patched Demucs files instead of the downloads.
// With SPECTRA_BENCH_GRAPH_RUNS=n each file is any float32 graph with fixed input shapes, fed random data.
const LOCAL_MODELS = process.env["SPECTRA_BENCH_MODEL"]?.split(",");
const LOCAL_EPS = (process.env["SPECTRA_BENCH_EPS"] ?? "webgpu,wasm").split(",");
const LOCAL_SECONDS = Number(process.env["SPECTRA_BENCH_SECONDS"] ?? "30");
const GRAPH_RUNS = process.env["SPECTRA_BENCH_GRAPH_RUNS"];
const LAYOUT = process.env["SPECTRA_BENCH_LAYOUT"];
const GRAPH_CAPTURE = process.env["SPECTRA_BENCH_GRAPH_CAPTURE"];
// Writes ORT's profile events, GPU kernel times included, to test-results/spike-profile.log.
const PROFILE = process.env["SPECTRA_BENCH_PROFILE"] === "1";
const NO_SUBGROUPS = process.env["SPECTRA_BENCH_NO_SUBGROUPS"] === "1";

test("local model bench", async ({ page }) => {
  test.skip(!LOCAL_MODELS, "set SPECTRA_BENCH_MODEL to bench a local .onnx");
  test.setTimeout(3 * 60 * 60_000);
  const profileLines: string[] = [];
  page.on("worker", (w) => w.on("console", (m) => profileLines.push(m.text())));
  await page.goto("/spike/");
  const req: Omit<RunRequest, "ep" | "modelFile"> = {
    model: "demucs-vocals-fp16",
    seconds: LOCAL_SECONDS,
    ...(GRAPH_RUNS === undefined ? {} : { graphRuns: Number(GRAPH_RUNS) }),
    webgpu: {
      ...(LAYOUT === "NHWC" || LAYOUT === "NCHW" ? { preferredLayout: LAYOUT } : {}),
      ...(GRAPH_CAPTURE === undefined ? {} : { enableGraphCapture: GRAPH_CAPTURE === "1" }),
      profile: PROFILE,
      noSubgroups: NO_SUBGROUPS,
    },
  };
  const results = [];
  for (const file of LOCAL_MODELS ?? []) {
    await page.locator("#file").setInputFiles(file);
    for (const ep of LOCAL_EPS) {
      const r = await page.evaluate(
        (q) => {
          const modelFile = document.querySelector<HTMLInputElement>("#file")?.files?.[0];
          return window.runSpike({ ...q, ...(modelFile ? { modelFile } : {}) });
        },
        { ...req, ep: ep === "wasm" ? "wasm" : "webgpu" } as const,
      );
      console.log(JSON.stringify({ file, ep, report: r.report, error: r.error }));
      results.push({ file, ep, ...r });
    }
  }
  mkdirSync("test-results", { recursive: true });
  writeFileSync("test-results/spike-bench-local.json", JSON.stringify(results, null, 2));
  if (PROFILE) writeFileSync("test-results/spike-profile.log", profileLines.join("\n"));
});

test("inference bench", async ({ page }) => {
  test.skip(LOCAL_MODELS !== undefined, "local model run requested");
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
