import type { Report, RunRequest, WorkerMessage } from "./protocol";

interface MeasurablePerformance {
  measureUserAgentSpecificMemory?: () => Promise<{ bytes: number }>;
}

function $<T extends HTMLElement>(id: string, type: new () => T): T {
  const el = document.getElementById(id);
  if (!(el instanceof type)) throw new Error(`#${id} missing or not a ${type.name}`);
  return el;
}

const MODEL_KEYS: readonly RunRequest["model"][] = ["demucs-vocals-fp32", "demucs-vocals-fp16", "dfn3"];
const EPS: readonly RunRequest["ep"][] = ["webgpu", "wasm"];

function pick<T extends string>(options: readonly T[], value: string): T {
  const found = options.find((o) => o === value);
  if (found === undefined) throw new Error(`unexpected option ${value}`);
  return found;
}

async function gpuInfo() {
  if (!("gpu" in navigator)) return { webgpu: false };
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return { webgpu: true, adapter: null };
  return {
    webgpu: true,
    vendor: adapter.info.vendor,
    architecture: adapter.info.architecture,
    shaderF16: adapter.features.has("shader-f16"),
    maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
  };
}

export interface SpikeResult {
  environment: Record<string, unknown>;
  report?: Report;
  error?: string;
  peakMemoryBytes?: number;
}

declare global {
  interface Window {
    spikeResult?: SpikeResult;
    runSpike: (req: RunRequest) => Promise<SpikeResult>;
  }
}

async function run(req: RunRequest): Promise<SpikeResult> {
  const status = $("status", HTMLElement);
  const out = $("out", HTMLPreElement);
  const environment = {
    userAgent: navigator.userAgent,
    crossOriginIsolated: globalThis.crossOriginIsolated,
    hardwareConcurrency: navigator.hardwareConcurrency,
    deviceMemoryGB: (navigator as Navigator & { deviceMemory?: number }).deviceMemory ?? null,
    jspi: "Suspending" in WebAssembly,
    gpu: await gpuInfo(),
  };
  out.textContent = JSON.stringify({ environment }, null, 2);

  // Counts the page and its workers but not GPU buffers.
  const perf = performance as Performance & MeasurablePerformance;
  let peakMemoryBytes: number | undefined;
  let sampling = crossOriginIsolated && typeof perf.measureUserAgentSpecificMemory === "function";
  const sample = async () => {
    while (sampling && perf.measureUserAgentSpecificMemory) {
      try {
        const m = await perf.measureUserAgentSpecificMemory();
        peakMemoryBytes = Math.max(peakMemoryBytes ?? 0, m.bytes);
      } catch {
        sampling = false;
      }
      await new Promise((r) => setTimeout(r, 1000));
    }
  };
  void sample();

  const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
  const result = await new Promise<SpikeResult>((resolve) => {
    worker.addEventListener("message", (e: MessageEvent<WorkerMessage>) => {
      const m = e.data;
      if (m.type === "progress") status.textContent = m.text;
      else if (m.type === "done") resolve({ environment, report: m.report });
      else resolve({ environment, error: m.message });
    });
    worker.addEventListener("error", (e) => {
      resolve({ environment, error: e.message });
    });
    worker.postMessage(req);
  });
  worker.terminate();
  sampling = false;
  if (peakMemoryBytes !== undefined) result.peakMemoryBytes = peakMemoryBytes;
  status.textContent = result.error === undefined ? "done" : "failed";
  out.textContent = JSON.stringify(result, null, 2);
  window.spikeResult = result;
  return result;
}

window.runSpike = run;

$("run", HTMLButtonElement).addEventListener("click", () => {
  const modelFile = $("file", HTMLInputElement).files?.[0];
  void run({
    model: pick(MODEL_KEYS, $("model", HTMLSelectElement).value),
    ep: pick(EPS, $("ep", HTMLSelectElement).value),
    seconds: Number($("seconds", HTMLInputElement).value),
    ...(modelFile ? { modelFile } : {}),
  });
});
