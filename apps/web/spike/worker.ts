import { OverlapAdder, planChunks, transitionWindow } from "@spectra-peel/engine";
import * as ort from "onnxruntime-web/jspi";
import { MODELS, type ModelKey } from "./models";
import type { Report, RunRequest, WorkerMessage } from "./protocol";

const post = (m: WorkerMessage) => postMessage(m);

async function float32(t: ort.Tensor): Promise<Float32Array> {
  const data = await t.getData();
  if (!(data instanceof Float32Array)) throw new Error(`expected float32 output, got ${t.type}`);
  return data;
}
const progress = (text: string) => post({ type: "progress", text });

const DEMUCS_RATE = 44_100;
const DEMUCS_SEGMENT = Math.trunc(7.8 * DEMUCS_RATE);
const VOCALS = 3;

async function loadModel(
  key: ModelKey,
): Promise<{ bytes: Uint8Array; downloadMs: number; fromCache: boolean }> {
  const spec = MODELS[key];
  const name = spec.url.split("/").slice(-3).join("_");
  const dir = await (
    await navigator.storage.getDirectory()
  ).getDirectoryHandle("spike-models", { create: true });
  const t0 = performance.now();
  try {
    const file = await (await dir.getFileHandle(name)).getFile();
    if (file.size === spec.bytes) {
      return {
        bytes: new Uint8Array(await file.arrayBuffer()),
        downloadMs: performance.now() - t0,
        fromCache: true,
      };
    }
  } catch {
    // not cached yet
  }
  const res = await fetch(spec.url);
  if (!res.ok || !res.body) throw new Error(`model fetch ${res.status} ${res.statusText}`);
  const fileHandle = await dir.getFileHandle(name, { create: true });
  const writable = await fileHandle.createWritable();
  const reader = res.body.getReader();
  let received = 0;
  let lastReport = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    await writable.write(value);
    received += value.byteLength;
    if (performance.now() - lastReport > 500) {
      lastReport = performance.now();
      progress(`downloading ${(received / 1e6).toFixed(0)} / ${(spec.bytes / 1e6).toFixed(0)} MB`);
    }
  }
  await writable.close();
  const file = await fileHandle.getFile();
  return {
    bytes: new Uint8Array(await file.arrayBuffer()),
    downloadMs: performance.now() - t0,
    fromCache: false,
  };
}

// ORT picks its MatMulSubgroup and GemmSubgroup kernels whenever the adapter reports `subgroups`,
// and on desktop Ampere those run about 2x slower than the plain kernels. ORT has no option to opt out.
function hideSubgroups() {
  const descriptor = Object.getOwnPropertyDescriptor(GPUAdapter.prototype, "features");
  if (!descriptor) return;
  const original = Object.defineProperty({}, "features", descriptor);
  Object.defineProperty(GPUAdapter.prototype, "features", {
    ...descriptor,
    get(this: GPUAdapter) {
      const features: unknown = Reflect.get(original, "features", this);
      if (!(features instanceof GPUSupportedFeatures)) return features;
      return new Set([...features].filter((f) => !f.startsWith("subgroup")));
    },
  });
}

async function createSession(bytes: Uint8Array, ep: RunRequest["ep"], webgpu: RunRequest["webgpu"]) {
  const { enableGraphCapture, profile, noSubgroups, ...epOptions } = webgpu ?? {};
  const options: ort.InferenceSession.SessionOptions =
    ep === "webgpu"
      ? {
          executionProviders: [{ name: "webgpu", ...epOptions }],
          ...(enableGraphCapture === undefined ? {} : { enableGraphCapture }),
        }
      : { executionProviders: [ep] };
  if (ep === "webgpu" && noSubgroups) hideSubgroups();
  // Profile events go to the console when the session ends; on WebGPU the "Api" events are GPU kernel times.
  if (profile) options.enableProfiling = true;
  try {
    return { session: await ort.InferenceSession.create(bytes, options), epUsed: ep };
  } catch (e) {
    if (ep === "wasm") throw e;
    progress(`webgpu session failed, retrying on wasm: ${String(e)}`);
    const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
    return { session, epUsed: "wasm", epFallbackReason: String(e) };
  }
}

function syntheticStereo(seconds: number, rate: number): [Float32Array, Float32Array] {
  const n = Math.round(seconds * rate);
  const l = new Float32Array(n);
  const r = new Float32Array(n);
  let seed = 1;
  for (let i = 0; i < n; i++) {
    seed = (seed * 16807) % 2147483647;
    const t = i / rate;
    const voice = 0.3 * Math.sin(2 * Math.PI * (220 + 30 * Math.sin(t)) * t);
    const noise = 0.05 * (seed / 2147483647 - 0.5);
    l[i] = voice + noise;
    r[i] = 0.9 * voice + noise;
  }
  return [l, r];
}

async function runDemucs(session: ort.InferenceSession, seconds: number) {
  const [l, r] = syntheticStereo(seconds, DEMUCS_RATE);
  const total = l.length;
  const window = transitionWindow(DEMUCS_SEGMENT);
  const outL = new OverlapAdder(total);
  const outR = new OverlapAdder(total);
  const chunks = planChunks(total, DEMUCS_SEGMENT, Math.floor(DEMUCS_SEGMENT / 4));
  const perChunkMs: number[] = [];
  let peak = 0;
  for (const [i, c] of chunks.entries()) {
    const input = new Float32Array(2 * DEMUCS_SEGMENT);
    input.set(l.subarray(c.start, c.end), 0);
    input.set(r.subarray(c.start, c.end), DEMUCS_SEGMENT);
    const t0 = performance.now();
    const out = await session.run({ mix: new ort.Tensor("float32", input, [1, 2, DEMUCS_SEGMENT]) });
    const stems = out["stems"];
    if (!stems) throw new Error("model has no 'stems' output");
    const data = await float32(stems);
    stems.dispose();
    perChunkMs.push(performance.now() - t0);
    const len = c.end - c.start;
    const base = VOCALS * 2 * DEMUCS_SEGMENT;
    const vl = data.subarray(base, base + len);
    const vr = data.subarray(base + DEMUCS_SEGMENT, base + DEMUCS_SEGMENT + len);
    outL.add(c.start, vl, window.subarray(0, len));
    outR.add(c.start, vr, window.subarray(0, len));
    for (const v of vl) peak = Math.max(peak, Math.abs(v));
    progress(`chunk ${i + 1}/${chunks.length}: ${(perChunkMs.at(-1) ?? 0).toFixed(0)} ms`);
  }
  outL.finish();
  outR.finish();
  return { chunks: chunks.length, perChunkMs, peak };
}

async function runDfn3(session: ort.InferenceSession, seconds: number) {
  // DFN3 runs at 48 kHz with a 480-sample hop, so 100 frames per second.
  const frames = Math.round(seconds * 100);
  const erb = new Float32Array(frames * 32).map(() => -0.5 + Math.random() * 0.1);
  const spec = new Float32Array(2 * frames * 96).map(() => (Math.random() - 0.5) * 0.01);
  const t0 = performance.now();
  const out = await session.run({
    feat_erb: new ort.Tensor("float32", erb, [1, 1, frames, 32]),
    feat_spec: new ort.Tensor("float32", spec, [1, 2, frames, 96]),
  });
  const mask = out["erb_mask"];
  if (!mask) throw new Error("model has no 'erb_mask' output");
  const data = await float32(mask);
  for (const t of Object.values(out)) t.dispose();
  let peak = 0;
  for (const v of data) peak = Math.max(peak, Math.abs(v));
  return { chunks: 1, perChunkMs: [performance.now() - t0], peak };
}

async function runGraph(session: ort.InferenceSession, runs: number) {
  let seed = 1;
  const feeds: Record<string, ort.Tensor> = {};
  for (const meta of session.inputMetadata) {
    if (!meta.isTensor || meta.type !== "float32") throw new Error(`input ${meta.name} is not float32`);
    const dims = meta.shape.map((d) => {
      if (typeof d !== "number") throw new Error(`input ${meta.name} has symbolic dim ${d}`);
      return d;
    });
    const data = new Float32Array(dims.reduce((a, b) => a * b, 1)).map(() => {
      seed = (seed * 16807) % 2147483647;
      return seed / 2147483647 - 0.5;
    });
    feeds[meta.name] = new ort.Tensor("float32", data, dims);
  }
  const perChunkMs: number[] = [];
  let peak = 0;
  for (let i = 0; i < runs; i++) {
    const t0 = performance.now();
    const out = await session.run(feeds);
    for (const t of Object.values(out)) {
      const data = await float32(t);
      t.dispose();
      if (i === 0) for (const v of data) peak = Math.max(peak, Math.abs(v));
    }
    perChunkMs.push(performance.now() - t0);
    progress(`run ${i + 1}/${runs}: ${(perChunkMs.at(-1) ?? 0).toFixed(0)} ms`);
  }
  return { chunks: runs, perChunkMs, peak };
}

async function handle(req: RunRequest): Promise<void> {
  ort.env.wasm.numThreads = Math.max(1, Math.min(navigator.hardwareConcurrency, 8));
  progress("loading model");
  const { bytes, downloadMs, fromCache } = req.modelFile
    ? { bytes: new Uint8Array(await req.modelFile.arrayBuffer()), downloadMs: 0, fromCache: true }
    : await loadModel(req.model);
  progress("creating session");
  const t0 = performance.now();
  const { session, epUsed, epFallbackReason } = await createSession(bytes, req.ep, req.webgpu);
  const sessionCreateMs = performance.now() - t0;
  const result =
    req.graphRuns !== undefined
      ? await runGraph(session, req.graphRuns)
      : req.model === "dfn3"
        ? await runDfn3(session, req.seconds)
        : await runDemucs(session, req.seconds);
  if (req.webgpu?.profile) session.endProfiling();
  await session.release();
  const inferenceMs = result.perChunkMs.reduce((a, b) => a + b, 0);
  const report: Report = {
    model: req.model,
    epRequested: req.ep,
    epUsed,
    ...(epFallbackReason === undefined ? {} : { epFallbackReason }),
    modelBytes: bytes.byteLength,
    downloadMs,
    fromCache,
    sessionCreateMs,
    clipSeconds: req.seconds,
    chunks: result.chunks,
    inferenceMs,
    perChunkMs: result.perChunkMs.map((v) => Math.round(v)),
    realTimeFactor: inferenceMs / 1000 / req.seconds,
    outputPeak: result.peak,
  };
  post({ type: "done", report });
}

addEventListener("message", (e: MessageEvent<RunRequest>) => {
  handle(e.data).catch((err: unknown) => {
    post({
      type: "error",
      message: err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err),
    });
  });
});
