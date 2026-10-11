# Spike results

Run with `SPECTRA_BENCH=1 bunx playwright test e2e/bench.spec.ts`, or by hand at `/spike/`.
Clips are 30 s of synthetic audio. Real-time factor (RTF) is inference time divided by clip length, so below 1 is faster than real time.

## Desktop, 2026-10-11

Edge 154 (headless, real-GPU flags), Windows 11, 16 threads, NVIDIA Ampere, `shader-f16` available, JSPI available, cross-origin isolated.

| Model | Provider | Session create | Inference (30 s clip) | RTF | Notes |
| --- | --- | --: | --: | --: | --- |
| DeepFilterNet3 graph | wasm | 1.4 s | 1.2 s | 0.041 | Graph only, random features |
| DeepFilterNet3 graph | webgpu | 2.0 s | 5.4 s | 0.18 | Slower than wasm: small GRU model, transfer cost dominates |
| htdemucs_ft vocals, StemSplitio export | any | fails | | | `ConstantOfShape(9)` with a float64 value in `/real_istft` has no ORT Web kernel |
| htdemucs_ft vocals fp16, float64 patched | webgpu | loads | ~71 s per 7.8 s chunk | ~9 | Second chunk timed; run stopped after 30 min with 2 of 6 chunks done |
| htdemucs_ft vocals fp16, float64 patched | wasm | loads | fails | | `std::bad_alloc` in `/encoder.1/dconv/layers.0/layers.0.3/Conv`: out of the 4 GB wasm32 heap on one chunk |
| htdemucs_ft vocals, patched and simplified (345 MB, float32) | webgpu | 2.2 s | 338 s | 11.3 | ~55 s per chunk, no better than unsimplified |
| htdemucs_ft vocals, patched and simplified (345 MB, float32) | wasm, 8 threads | 1.8 s | 46.6 s | 1.55 | ~7.5 s per 7.8 s chunk |

`performance.measureUserAgentSpecificMemory()` returned about 1 MB in headless Edge, which is not credible. Peak memory needs a headed run or the phone's `chrome://inspect` memory panel.

## Findings

- DFN3 should run on the wasm provider. WebGPU is 4x slower for it.
- The published HT-Demucs exports do not load in ORT Web. `reference/patch_float64.py` rewrites the float64 iSTFT path to float32; Python ORT output matches the original at 159 dB SNR (max abs diff 7.5e-9). Shipping that means hosting patched files, so our own Hugging Face model repo comes back into scope.
- Even patched, HT-Demucs is about 9x slower than real time on a desktop RTX-class GPU through WebGPU, against 0.2x on a native M4 CPU per StemSplitio. The graph is mostly shape bookkeeping (11,684 Constant, 2,968 Shape, 2,090 Slice, 684 ScatterND nodes), which points at CPU fallbacks and transfers. On wasm it runs out of memory. As exported, mobile Demucs is a no-go.
- `reference/simplify.py` fixes the input to one 7.8 s segment and runs onnxsim, which folds the graph from 24,917 nodes to 1,434 (no Shape or ScatterND left) at 120 dB SNR. The file grows to 345 MB because the fp16 weights are folded to float32.
- Simplified, wasm stops running out of memory and runs at RTF 1.55 on 8 desktop threads. WebGPU stays at ~55 s per chunk, so the cost there is one or more slow kernels, not shape ops. Every op left in the graph has a WebGPU kernel. `ort.env.webgpu.profiling` returned no data with the JSPI build, even with `--enable-dawn-features=allow_unsafe_apis`, so the slow kernel is not identified yet. ConvTranspose on long sequences is the first suspect.
- Hugging Face downloads work on a COEP `require-corp` page.

## Phone

Not run yet. Target is a Samsung Galaxy S25 (Snapdragon 8 Elite, Adreno 830, 12 GB). Needs the deployed `/spike/` page and its Chrome version.
