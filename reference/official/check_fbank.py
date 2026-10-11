"""Compare ../speakers.py's Kaldi fbank with torchaudio's, as WeSpeaker computes it.

    cd reference/official && uv run python check_fbank.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import speakers  # noqa: E402

rng = np.random.default_rng(1)
t = np.arange(3 * speakers.SR) / speakers.SR
wav = (0.3 * np.sin(2 * np.pi * (200 + 900 * t) * t) + 0.05 * rng.standard_normal(t.size)).astype(np.float32)
ours = speakers.fbank(wav)
x = torch.from_numpy(wav)[None] * (1 << 15)
theirs = kaldi.fbank(x, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0,
                     sample_frequency=speakers.SR, window_type="hamming")
theirs = (theirs - theirs.mean(0, keepdim=True)).numpy()
print("frames", ours.shape, theirs.shape, "max abs diff", float(np.abs(ours - theirs).max()))
