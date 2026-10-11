"""NumPy reference for the DeepFilterNet3 DSP around the ONNX graph, and golden fixtures for the engine.

Follows libDF v0.5.6: Vorbis window over the full 960-point frame, forward FFT scaled by 1/960, ERB
band powers in dB with exponential mean normalisation, unit normalisation of the first 96 bins,
an ERB gain on all bins, and an order-5 deep filter on the first 96 bins of the noisy spectrum
(frames t-2..t+2). The graph applies its own two-frame feature lookahead, so features go in unshifted.
official/check.py compares this against the official deepfilternet package.

    uv run python dfn3.py ../apps/web/public/models/deepfilter-stream.onnx ../packages/engine/fixtures/dfn3
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

from stream_dfn3 import GRUS, PADS

SR, N, H = 48_000, 960, 480
NB_ERB, NB_DF, ORDER, DF_LOOKAHEAD = 32, 96, 5, 2
ALPHA = 0.99


def window() -> np.ndarray:
    n = np.arange(N)
    return np.sin(0.5 * np.pi * np.sin(np.pi * (n + 0.5) / N) ** 2).astype(np.float32)


def erb_widths() -> np.ndarray:
    def f2e(f: float) -> float:
        return 9.265 * np.log1p(f / (24.7 * 9.265))

    def e2f(e: float) -> float:
        return 24.7 * 9.265 * (np.exp(e / 9.265) - 1)

    lo, hi = f2e(0.0), f2e(SR / 2)
    step = (hi - lo) / NB_ERB
    widths, prev, over = [], 0, 0
    for i in range(1, NB_ERB + 1):
        fb = int(round(e2f(lo + i * step) / (SR / N)))
        n = fb - prev - over
        if n < 2:
            over, n = 2 - n, 2
        else:
            over = 0
        widths.append(n)
        prev = fb
    widths[-1] += 1
    widths[-1] -= sum(widths) - (N // 2 + 1)
    return np.array(widths)


def analysis(x: np.ndarray) -> np.ndarray:
    w = window()
    mem = np.zeros(H, np.float32)
    frames = []
    for i in range(0, len(x) - H + 1, H):
        frames.append(np.fft.rfft(np.concatenate([mem, x[i : i + H]]) * w) / N)
        mem = x[i : i + H].copy()
    return np.array(frames).astype(np.complex64)


def features(spec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    widths = erb_widths()
    edges = np.concatenate([[0], np.cumsum(widths)])
    erb = np.zeros((len(spec), NB_ERB), np.float32)
    mean = np.linspace(-60, -90, NB_ERB).astype(np.float32)
    unit = np.linspace(0.001, 0.0001, NB_DF).astype(np.float32)
    cplx = np.zeros((len(spec), NB_DF), np.complex64)
    for t, frame in enumerate(spec):
        power = np.abs(frame) ** 2
        db = np.array([10 * np.log10(power[edges[b] : edges[b + 1]].mean() + 1e-10) for b in range(NB_ERB)])
        mean = (db * (1 - ALPHA) + mean * ALPHA).astype(np.float32)
        erb[t] = (db - mean) / 40
        unit = (np.abs(frame[:NB_DF]) * (1 - ALPHA) + unit * ALPHA).astype(np.float32)
        cplx[t] = frame[:NB_DF] / np.sqrt(unit)
    return erb, cplx


def apply(spec: np.ndarray, mask: np.ndarray, coefs: np.ndarray) -> np.ndarray:
    out = spec * np.repeat(mask, erb_widths(), axis=1)
    before = ORDER - 1 - DF_LOOKAHEAD
    padded = np.concatenate(
        [
            np.zeros((before, spec.shape[1]), spec.dtype),
            spec,
            np.zeros((DF_LOOKAHEAD, spec.shape[1]), spec.dtype),
        ]
    )
    c = coefs[..., 0] + 1j * coefs[..., 1]
    out[:, :NB_DF] = sum(padded[n : n + len(spec), :NB_DF] * c[n] for n in range(ORDER))
    return out


def synthesis(spec: np.ndarray) -> np.ndarray:
    w = window()
    mem = np.zeros(H, np.float32)
    out = []
    for frame in spec:
        x = np.fft.irfft(frame, N) * N * w
        out.append(x[:H] + mem)
        mem = x[H:]
    return np.concatenate(out).astype(np.float32)


def run_model(
    session: ort.InferenceSession, erb: np.ndarray, cplx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    feeds = {
        "feat_erb": erb[None, None],
        "feat_spec": np.stack([cplx.real, cplx.imag])[None].astype(np.float32),
        **{f"h_{n}": np.zeros((1, 1, 256), np.float32) for n in GRUS.values()},
        **{f"hist_{n}": np.zeros((1, c, k, w), np.float32) for n, k, c, w in PADS.values()},
    }
    mask, coefs = session.run(["erb_mask", "df_coefs"], feeds)
    return np.asarray(mask)[0, 0], np.asarray(coefs)[0]


def enhance(session: ort.InferenceSession, x: np.ndarray):
    spec = analysis(np.concatenate([x, np.zeros(N, np.float32)]))
    erb, cplx = features(spec)
    mask, coefs = run_model(session, erb, cplx)
    y = synthesis(apply(spec, mask, coefs))[N - H : N - H + len(x)]
    return y, erb, cplx, mask, coefs


def test_signal(seconds: float, seed: int = 0) -> np.ndarray:
    """A buzzy voice-like tone with a moving pitch and syllable-rate envelope, in white noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    phase = 2 * np.pi * np.cumsum(140 + 30 * np.sin(2 * np.pi * 0.7 * t)) / SR
    voice = sum(np.sin(k * phase) / k for k in range(1, 20)) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t) ** 2)
    return (0.2 * voice / np.abs(voice).max() + 0.03 * rng.standard_normal(len(t))).astype(np.float32)


def main(model: str, out_dir: str) -> None:
    session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    x = test_signal(1.5)
    y, erb, cplx, mask, coefs = enhance(session, x)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {
        "input": x,
        "feat_erb": erb,
        "feat_spec": np.stack([cplx.real, cplx.imag], axis=-1),
        "erb_mask": mask,
        "df_coefs": coefs,
        "output": y,
    }
    for name, a in arrays.items():
        np.ascontiguousarray(a, dtype="<f4").tofile(out / f"{name}.f32")
    meta = {"samples": len(x), "frames": len(erb), "shapes": {k: list(v.shape) for k, v in arrays.items()}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
