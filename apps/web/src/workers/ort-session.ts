import * as ort from "onnxruntime-web/jspi";
import { describeAdapter, requestGpu } from "../audio/gpu";

/**
 * A device without the `subgroups` feature. ORT 1.30 picks its MatMulSubgroup and GemmSubgroup kernels
 * whenever the device has it, and on desktop Ampere those ran HT-Demucs about 2x slower than the plain
 * kernels (docs/spike-results.md). The limits are the adapter's, since the model needs large buffers.
 */
async function createDevice(): Promise<{ device: GPUDevice; label: string } | null> {
  const adapter = await requestGpu();
  if (!adapter) return null;
  const l = adapter.limits;
  const device = await adapter.requestDevice({
    requiredFeatures: adapter.features.has("shader-f16") ? ["shader-f16"] : [],
    requiredLimits: {
      maxBufferSize: l.maxBufferSize,
      maxStorageBufferBindingSize: l.maxStorageBufferBindingSize,
      maxStorageBuffersPerShaderStage: l.maxStorageBuffersPerShaderStage,
      maxComputeWorkgroupStorageSize: l.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: l.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: l.maxComputeWorkgroupSizeX,
      maxComputeWorkgroupSizeY: l.maxComputeWorkgroupSizeY,
      maxComputeWorkgroupSizeZ: l.maxComputeWorkgroupSizeZ,
      maxComputeWorkgroupsPerDimension: l.maxComputeWorkgroupsPerDimension,
    },
  });
  return { device, label: describeAdapter(adapter) };
}

export interface Session {
  session: ort.InferenceSession;
  /** What ran it, for progress text: a GPU's vendor and architecture, or the CPU. */
  device: string;
}

/** On the GPU when asked and available, else wasm; a GPU session that fails to build falls back to wasm. */
export async function createSession(bytes: Uint8Array, useGpu: boolean): Promise<Session> {
  const threads = Math.max(1, Math.min(navigator.hardwareConcurrency, 8));
  ort.env.wasm.numThreads = threads;
  if (useGpu) {
    try {
      const gpu = await createDevice();
      if (gpu) {
        ort.env.webgpu.device = gpu.device;
        const session = await ort.InferenceSession.create(bytes, { executionProviders: ["webgpu"] });
        return { session, device: `${gpu.label} GPU` };
      }
    } catch (e) {
      console.warn("WebGPU session failed, using wasm:", e);
    }
  }
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  return { session, device: `CPU, ${threads} threads` };
}
