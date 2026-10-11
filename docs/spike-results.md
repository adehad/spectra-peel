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

## Desktop WebGPU kernels, 2026-10-11

Same machine. Single nodes and graph slices were benched with `SPECTRA_BENCH_GRAPH_RUNS`, which feeds random inputs, so their times include upload and readback. GPU kernel times come from `SPECTRA_BENCH_PROFILE=1`. Full-model rows are a 30 s clip at the steady-state chunk time, starting from the patched and simplified float32 model (345 MB).

| Measurement | webgpu | wasm, 8 threads |
| --- | --: | --: |
| One `/real_istft/ConvTranspose` node, [8, 2049, 340] to [8, 1, 351232], 4096 taps, stride 1024 (there are two) | 24.9 s | 0.6 s |
| Next-slowest single node (Softmax [8, 2688, 2688], transfer-bound) | 0.47 s | 0.12 s |
| Simplified model, per 7.8 s chunk | ~55 s | ~7.5 s |
| + `reference/rewrite_stft.py` without bin padding (STFT/iSTFT as MatMul + overlap-add) | 5.7 s | 6.4 s |
| + `subgroups` hidden from ORT | 5.6-6.1 s | |
| + frequency bins padded 2049 to 2052 (subgroups still on) | 5.7-6.9 s | |
| + padding and `subgroups` hidden: shipped `rewrite_stft.py` plus `noSubgroups` | 3.9 s (RTF 0.80) | 6.5 s (RTF 1.33) |
| Same, converted to fp16 with `onnxconverter-common` 1.16.0 | 2.6 s | |

GPU time for one chunk by kernel, rewritten model:

| Kernel | `subgroups` on, 2049 bins | `subgroups` hidden, 2052 bins |
| --- | --: | --: |
| All kernels | 5,449 ms | 3,639 ms |
| MatMulSubgroup + MatMul | 1,825 + 368 ms | 1,137 ms |
| of which the two iSTFT MatMuls | 966 ms | 413 ms |
| GemmSubgroup / Gemm | 1,305 ms | 553 ms |
| Conv2dMM | 549 ms | 552 ms |
| Transpose | 267 ms | 263 ms |
| ConvTranspose2D (decoder, 8 taps) | 242 ms | 241 ms |

## Kernel findings

- The two iSTFT ConvTranspose nodes took about 50 of the 55 s per chunk on WebGPU. Every other node runs at wasm speed or better, ignoring transfer cost. Verbose logging showed all nodes on `WebGpuExecutionProvider`, so no CPU fallback is involved.
- `reference/rewrite_stft.py` rewrites the STFT Conv and iSTFT ConvTranspose as framing, MatMul and overlap-add. It matches the simplified model at 119.2 dB SNR (max abs diff 1.1e-7) in Python ORT. WebGPU goes from ~55 s to 5.7 s per chunk.
- ORT 1.30 picks `MatMulSubgroup` and `GemmSubgroup` whenever the adapter reports `subgroups`. On this GPU they are about 2x slower than the plain kernels. A self-attention MatMul takes 104 ms against 55 ms, and Gemm [2688, 512] x [512, 2048] takes 59 ms against 25 ms. ORT has no option to turn them off, so the spike's `noSubgroups` option hides the feature on `GPUAdapter.prototype` before session creation.
- With subgroups hidden, the 2049-bin iSTFT MatMuls ran at ~53 GFLOP/s. Zero-padding to 2052 bins brings them to ~220 GFLOP/s, with the same output. Each change does little alone; together they give 3.9 s per chunk, RTF 0.80, against 6.5 s on wasm.
- `preferredLayout: "NHWC"` gave 5.66 s against 5.68 s for NCHW. `enableGraphCapture` needs GPU-resident inputs and outputs, and CPU dispatch was only 26 ms of a 914 ms slice, so it was not pursued.
- fp16 reaches 2.6 s per chunk, but a plain conversion gives 71.6 dB SNR overall and 49.3 dB on vocals. One constant (2,752,512) is clamped to 10,000 in the conversion. Keeping fp32 on the nodes that overflow, or on the STFT, norms and softmax, made it worse (2.1 dB and 55.6 dB). fp16 needs per-node work and a listening test before it is usable.
- Session-level `enableProfiling` works with the JSPI build, unlike `ort.env.webgpu.profiling`. Its "Api" events are per-program GPU times.
- What is left is plain fp32 GEMM throughput. Gemm [2688, 512] x [512, 2048] runs at ~220 GFLOP/s, a small fraction of an Ampere GPU's fp32 peak.
- Hiding subgroups is a desktop NVIDIA result. The phone's Adreno may behave differently, so measure it there before shipping the flag.

## Phone

Not run yet. Target is a Samsung Galaxy S25 (Snapdragon 8 Elite, Adreno 830, 12 GB). Needs the deployed `/spike/` page and its Chrome version.

## Import, desktop, 2026-10-11

`SPECTRA_BENCH=1 SPECTRA_BENCH_MINUTES=60 bunx playwright test e2e/import-bench.spec.ts`: a synthetic one-hour, 44.1 kHz stereo WAV (635 MB), same machine as above.

| Measure | Result |
| --- | --- |
| Import (decode, FLAC, peaks, spectrogram tiles) | 97 s |
| Renderer resident memory during import | 175 MB at start, 259 MB at the end, sampled every 5 s |
| Seek to a clicked point while playing | 41 to 214 ms until the clock moves |

Resident memory grows by about 85 MB over the hour. About 10 MB of that is the peaks pyramid being built, which has to stay in memory until the import ends. The rest may be garbage V8 has not collected yet, or a real leak. RSS cannot tell them apart. Reading the worker's live heap over CDP failed, because Playwright's own session owns the worker target. A heap snapshot in a headed run is the next check. Writing spectrogram tiles without awaiting each write let them queue behind OPFS: memory reached 341 MB and the import took 119 s, so the import worker now writes tiles one at a time.

## Denoise, desktop, 2026-10-11

Whole-file denoise of a synthetic 3-minute, 44.1 kHz stereo WAV in the app: 24.4 s, so 0.14x real time for both channels together, or about 8 minutes for an hour. Each channel goes 44.1 to 48 kHz, through DeepFilterNet3 on the wasm provider with 4 threads in 500-frame chunks, then back. The channels run one after the other in one worker, so a second worker would roughly halve this.

The synthetic tone fixture is a poor quality check: DeepFilterNet3 is a speech model and takes much of a steady tone along with the hiss. Quality needs real speech, which is what the VoiceBank+DEMAND floor in the plan is for.
