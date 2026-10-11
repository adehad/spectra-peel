/**
 * The WebGPU adapter a page would use. Chrome on Windows hands out the GPU the OS assigns to the browser
 * and ignores `powerPreference`, so on a laptop this is often the integrated one.
 */
export async function requestGpu(): Promise<GPUAdapter | null> {
  if (!("gpu" in navigator)) return null;
  return navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
}

export function describeAdapter(adapter: GPUAdapter): string {
  const { vendor, architecture } = adapter.info;
  return [vendor, architecture].filter(Boolean).join(" ") || "unknown";
}

/** Intel adapters are integrated graphics, which run HT-Demucs slower than the CPU path is likely to. */
export function isIntegrated(adapter: GPUAdapter): boolean {
  return adapter.info.vendor.toLowerCase() === "intel";
}
