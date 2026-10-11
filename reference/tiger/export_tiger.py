"""Export TIGER (JusperLee/TIGER-speech, Apache-2.0) as a fixed 6 s, 16 kHz, two-speaker ONNX graph.

TIGER does its STFT and iSTFT with complex tensors, which do not export. TigerExport repeats
TIGER.forward with the STFT as framing plus MatMul and the iSTFT as MatMul plus a shifted overlap-add,
on real tensors and fixed shapes. Adaptive pooling becomes a MatMul with a constant matrix, since the
exporter only handles sizes that divide, and nearest upsampling becomes a Gather, since ONNX Runtime's
CPU Resize took over half of TIGER's run time. The graph is then saved after ONNX Runtime's basic
optimisation, which folds its 72k exported nodes to about 9k.

    git clone https://github.com/JusperLee/TIGER && git -C TIGER checkout 9f18d4a10a7137e1ce8052cfb62215179f1287b6
    uv run python export_tiger.py TIGER tiger-speech-6s.onnx
"""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download

SR = 16_000
SECONDS = 6
REPO = "JusperLee/TIGER-speech"
REVISION = "f0340340b2d9bbf72074edf8c076dcab59a10ba2"
TARGET_RMS = 0.05


def adaptive_avg_pool1d_matmul(x: torch.Tensor, output_size) -> torch.Tensor:
    size_in = int(x.shape[-1])
    size_out = int(output_size[0] if isinstance(output_size, (tuple, list)) else output_size)
    if size_in == size_out:
        return x
    pool = torch.zeros(size_in, size_out)
    for j in range(size_out):
        lo = (j * size_in) // size_out
        hi = -((-(j + 1) * size_in) // size_out)
        pool[lo:hi, j] = 1.0 / (hi - lo)
    return torch.matmul(x, pool.to(x.dtype))


_interpolate = F.interpolate


def interpolate_gather(x: torch.Tensor, size=None, scale_factor=None, mode="nearest", **kw) -> torch.Tensor:
    """1-D nearest upsampling with PyTorch's index rule (float32 scale, floor)."""
    if mode != "nearest" or x.dim() != 3 or size is None:
        return _interpolate(x, size=size, scale_factor=scale_factor, mode=mode, **kw)
    size_in = int(x.shape[-1])
    size_out = int(size[0] if isinstance(size, (tuple, list)) else size)
    scale = np.float32(size_in) / np.float32(size_out)
    idx = np.minimum(np.floor(np.arange(size_out, dtype=np.float32) * scale).astype(np.int64), size_in - 1)
    return x.index_select(-1, torch.from_numpy(idx))


class TigerExport(nn.Module):
    def __init__(self, model: nn.Module, n_samples: int):
        super().__init__()
        self.m = model
        self.n = n_samples
        win, hop = model.win, model.stride
        if win != 4 * hop:
            raise ValueError("the overlap-add below assumes win == 4 * hop")
        self.win, self.hop = win, hop
        self.n_frames = 1 + n_samples // hop
        nfreq = win // 2 + 1
        window = torch.hann_window(win, dtype=torch.float64)
        t = torch.arange(win, dtype=torch.float64)
        k = torch.arange(nfreq, dtype=torch.float64)
        ang = 2 * torch.pi * t[:, None] * k[None, :] / win
        self.register_buffer("fwd_cos", (window[:, None] * torch.cos(ang)).float())
        self.register_buffer("fwd_sin", (-window[:, None] * torch.sin(ang)).float())
        w = torch.full((nfreq,), 2.0, dtype=torch.float64)
        w[0] = 1
        w[-1] = 1
        self.register_buffer("inv_cos", (w[:, None] * torch.cos(ang.T) / win * window[None, :]).float())
        self.register_buffer("inv_sin", (-w[:, None] * torch.sin(ang.T) / win * window[None, :]).float())
        idx = torch.arange(self.n_frames)[:, None] * hop + torch.arange(win)[None, :]
        self.register_buffer("frame_idx", idx)
        env = self._ola(window.float()[None, None, :].pow(2).expand(1, self.n_frames, win))
        self.register_buffer("inv_env", 1.0 / env.clamp_min(1e-11))

    def _ola(self, frames: torch.Tensor) -> torch.Tensor:
        """Overlap-add [B, T, win] at hop = win / 4 as four shifted, padded adds."""
        b, n_frames, _ = frames.shape
        hop = self.hop
        out = None
        for j in range(4):
            part = frames[:, :, j * hop : (j + 1) * hop].reshape(b, n_frames * hop)
            part = F.pad(part, (j * hop, (3 - j) * hop))
            out = part if out is None else out + part
        return out

    def forward(self, mix: torch.Tensor) -> torch.Tensor:  # [1, n] -> [1, 2, n]
        m = self.m
        half = self.win // 2
        xp = F.pad(mix[:, None, :], (half, half), mode="reflect")[:, 0]
        frames = xp[:, self.frame_idx]
        re = torch.matmul(frames, self.fwd_cos).transpose(1, 2)
        im = torch.matmul(frames, self.fwd_sin).transpose(1, 2)
        spec_ri = torch.stack([re, im], 1)

        feats, bands = [], []
        idx = 0
        for i, bw in enumerate(m.band_width):
            band = spec_ri[:, :, idx : idx + bw]
            bands.append(band)
            feats.append(m.BN[i](band.reshape(1, bw * 2, -1)))
            idx += bw
        feat = torch.stack(feats, 1)
        sep = m.separator(feat.view(1, m.nband, m.feature_dim, -1)).view(1, m.nband, m.feature_dim, -1)

        k = m.num_output
        out_re, out_im = [], []
        for i, bw in enumerate(m.band_width):
            o = m.mask[i](sep[:, i]).view(1, 2, 2, k, bw, -1)
            mask = o[:, 0] * torch.sigmoid(o[:, 1])
            mr, mi = mask[:, 0], mask[:, 1]
            mr = mr - (mr.sum(1, keepdim=True) - 1) / k
            mi = mi - mi.sum(1, keepdim=True) / k
            sr, si = bands[i][:, 0:1], bands[i][:, 1:2]
            out_re.append(sr * mr - si * mi)
            out_im.append(sr * mi + si * mr)
        est_re = torch.cat(out_re, 2).transpose(2, 3)
        est_im = torch.cat(out_im, 2).transpose(2, 3)
        frames_out = torch.matmul(est_re, self.inv_cos) + torch.matmul(est_im, self.inv_sin)
        y = self._ola(frames_out.reshape(k, self.n_frames, self.win)) * self.inv_env
        return y[:, half : half + self.n][None]


def snr_db(reference: np.ndarray, test: np.ndarray) -> float:
    return float(10 * np.log10((reference**2).sum() / ((reference - test) ** 2).sum()))


def main(tiger_repo: str, out_path: str) -> None:
    sys.path.insert(0, str(Path(tiger_repo).resolve()))
    import look2hear.models  # noqa: PLC0415

    weights = snapshot_download(REPO, revision=REVISION)
    base = look2hear.models.TIGER.from_pretrained(weights).eval()
    n = SECONDS * SR
    t = np.arange(n) / SR
    rng = np.random.default_rng(0)
    mix = (np.sin(2 * np.pi * (140 + 60 * np.sin(3 * t)) * t) + np.sin(2 * np.pi * (230 + 400 * t / SECONDS) * t)
           + 0.1 * rng.standard_normal(n)).astype(np.float32)
    mix *= TARGET_RMS / np.sqrt(np.mean(mix**2))
    x = torch.from_numpy(mix)[None]
    wrapped = TigerExport(base, n).eval()
    with torch.no_grad():
        ref = base(x[None]).numpy()
        F.adaptive_avg_pool1d = adaptive_avg_pool1d_matmul
        F.interpolate = interpolate_gather
        print(f"wrapper vs TIGER: {snr_db(ref, wrapped(x).numpy()):.1f} dB")
        raw = Path(out_path).with_suffix(".raw.onnx")
        torch.onnx.export(wrapped, (x,), str(raw), input_names=["mix"], output_names=["sources"], opset_version=17,
                          do_constant_folding=True)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = out_path
    ort.InferenceSession(str(raw), so, providers=["CPUExecutionProvider"])
    raw.unlink()
    got = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"]).run(None, {"mix": mix[None]})[0]
    print(f"ONNX vs TIGER: {snr_db(ref, got):.1f} dB, {Path(out_path).stat().st_size} bytes")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
