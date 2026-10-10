import { ENGINE_VERSION } from "@spectra-peel/engine";

export function App() {
  return (
    <main>
      <h1>Spectra Peel</h1>
      <dl>
        <dt>Cross-origin isolated</dt>
        <dd data-testid="isolated">{String(globalThis.crossOriginIsolated)}</dd>
        <dt>WebGPU</dt>
        <dd data-testid="webgpu">{String("gpu" in navigator)}</dd>
        <dt>Engine</dt>
        <dd>{ENGINE_VERSION}</dd>
      </dl>
    </main>
  );
}
