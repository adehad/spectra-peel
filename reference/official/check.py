"""Compare ../dfn3.py with the official deepfilternet 0.5.6 package on the same noisy clip.

Kept apart from the main reference project because it pulls in PyTorch. The official package
downloads its checkpoint on first use.

    cd reference/official && uv run python check.py ../../apps/web/public/models/deepfilter-stream.onnx
"""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from df.enhance import enhance, init_df

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dfn3  # noqa: E402


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    scale = np.dot(estimate, reference) / np.dot(reference, reference)
    target = scale * reference
    return float(10 * np.log10(np.sum(target**2) / np.sum((estimate - target) ** 2)))


def main(model: str) -> None:
    x = dfn3.test_signal(4.0, seed=0)
    net, state, _ = init_df()
    official = enhance(net, state, torch.from_numpy(x[None])).numpy()[0]
    ours, *_ = dfn3.enhance(ort.InferenceSession(model, providers=["CPUExecutionProvider"]), x)
    score = si_sdr(official, ours)
    print(f"SI-SDR against official deepfilternet 0.5.6: {score:.1f} dB")
    if score < 60:
        sys.exit("reference DSP drifted from the official implementation")


if __name__ == "__main__":
    main(sys.argv[1])
