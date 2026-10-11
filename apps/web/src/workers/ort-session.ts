import * as ort from "onnxruntime-web/jspi";

// ORT 1.30 picks its MatMulSubgroup and GemmSubgroup kernels whenever the adapter reports `subgroups`,
// and on desktop Ampere those ran HT-Demucs about 2x slower than the plain kernels. ORT has no option to
// opt out, so the feature is hidden from it.
let subgroupsHidden = false;
export function hideSubgroups(): void {
  if (subgroupsHidden || typeof GPUAdapter === "undefined") return;
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
  subgroupsHidden = true;
}

export interface Session {
  session: ort.InferenceSession;
  provider: "webgpu" | "wasm";
}

/** WebGPU when the device has it, else wasm; a WebGPU session that fails to build falls back to wasm. */
export async function createSession(bytes: Uint8Array, preferGpu: boolean): Promise<Session> {
  ort.env.wasm.numThreads = Math.max(1, Math.min(navigator.hardwareConcurrency, 4));
  if (preferGpu && "gpu" in navigator) {
    hideSubgroups();
    try {
      return {
        session: await ort.InferenceSession.create(bytes, { executionProviders: ["webgpu"] }),
        provider: "webgpu",
      };
    } catch (e) {
      console.warn("WebGPU session failed, using wasm:", e);
    }
  }
  return {
    session: await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] }),
    provider: "wasm",
  };
}
