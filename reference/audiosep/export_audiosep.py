"""Export AudioSep's separator (Audio-AGI/AudioSep, MIT) as a fixed 5.11 s, 32 kHz mono ONNX graph,
and the CLAP text embeddings of the app's Extract prompts.

The separator is the ResUNet30 from the base checkpoint. The prompt enters it only as a 512-d CLAP
text embedding, so the app ships a table of embeddings instead of the 125M-parameter text encoder.

torchlibrosa computes the STFT as a 2048-tap Conv1d at hop 320 and the iSTFT with torch.flip and
F.fold, neither of which suits ORT Web. 2048 and 320 share a 64-sample block, so here the framing is
32 strided Slices of a [blocks, 64] view, the DFT is one MatMul, and overlap-add is Pads and Adds.

    git clone https://github.com/Audio-AGI/AudioSep && git -C AudioSep checkout 944583f18b84589dc965de3ad77525c945334252
    uv run python export_audiosep.py AudioSep audiosep-base-5s.onnx ../../apps/web/src/extract-prompts.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import onnxsim
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, snapshot_download

SPACE = "Audio-AGI/AudioSep"
SPACE_REVISION = "0807a713283035d8b0d0628f1b8e3d6580459f7b"
CHECKPOINT = "checkpoint/audiosep_base_4M_steps.ckpt"
ROBERTA = "FacebookAI/roberta-base"
ROBERTA_REVISION = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"

SR = 32_000
N, HOP, BLOCK = 2048, 320, 64
TAPS, STEP = N // BLOCK, HOP // BLOCK
BINS = N // 2 + 1
# 512 STFT frames, a multiple of the UNet's 32-frame alignment, so it needs no padding.
LENGTH = HOP * (512 - 1)

# Label shown in the app, and the text CLAP embeds for it. Each tested well on the base checkpoint;
# "background noise" and "traffic noise" did not, so they are left out.
PROMPTS = {
    "Speech": "speech",
    "Music": "music",
    "Piano": "piano",
    "Strings": "string instruments",
    "Dog barking": "dog barking",
}


def hann() -> np.ndarray:
    n = np.arange(N)
    return 0.5 - 0.5 * np.cos(2 * np.pi * n / N)


class MatStft(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        n, k = np.arange(N)[:, None], np.arange(BINS)[None, :]
        ang = 2 * np.pi * n * k / N
        w = hann()[:, None]
        basis = np.concatenate([w * np.cos(ang), -w * np.sin(ang)], axis=1)
        self.register_buffer("basis", torch.from_numpy(basis.astype(np.float32)))

    def forward(self, x: torch.Tensor):
        """x [B, L] -> real, imag [B, 1, T, BINS], matching torchlibrosa STFT(center=True, reflect)."""
        b, length = x.shape
        frames_num = length // HOP + 1
        x = F.pad(x[:, None, :], (N // 2, N // 2), mode="reflect")[:, 0, :]
        blocks = x.reshape(b, -1, BLOCK)
        span = STEP * (frames_num - 1) + 1
        framed = torch.cat([blocks[:, j : j + span : STEP, :] for j in range(TAPS)], dim=-1)
        spec = framed @ self.basis
        return spec[:, None, :, :BINS], spec[:, None, :, BINS:]


class MatIstft(torch.nn.Module):
    def __init__(self, length: int) -> None:
        super().__init__()
        frames_num = length // HOP + 1
        k, n = np.arange(BINS)[:, None], np.arange(N)[None, :]
        ang = 2 * np.pi * k * n / N
        c = np.full((BINS, 1), 2.0)
        c[0] = c[-1] = 1.0
        w = hann()[None, :]
        self.register_buffer("cos_basis", torch.from_numpy((c * np.cos(ang) / N * w).astype(np.float32)))
        self.register_buffer("sin_basis", torch.from_numpy((-c * np.sin(ang) / N * w).astype(np.float32)))
        total = (frames_num - 1) * HOP + N
        wsum = np.zeros(total)
        for t in range(frames_num):
            wsum[t * HOP : t * HOP + N] += hann() ** 2
        inv = 1.0 / np.clip(wsum[N // 2 : N // 2 + length], 1e-11, None)
        self.register_buffer("inv_wsum", torch.from_numpy(inv.astype(np.float32)))
        self.length = length

    def forward(self, real: torch.Tensor, imag: torch.Tensor, length: int) -> torch.Tensor:
        b, _, t, _ = real.shape
        frames = real[:, 0] @ self.cos_basis + imag[:, 0] @ self.sin_basis
        blocks = frames.reshape(b, t, TAPS, BLOCK)
        q_max = (TAPS - 1) // STEP
        classes = []
        for r in range(STEP):
            acc = None
            for q in range((TAPS - 1 - r) // STEP + 1):
                part = F.pad(blocks[:, :, r + STEP * q, :], (0, 0, q, q_max - q))
                acc = part if acc is None else acc + part
            classes.append(acc)
        y = torch.stack(classes, dim=2).reshape(b, -1)
        return y[:, N // 2 : N // 2 + self.length] * self.inv_wsum


class Separator(torch.nn.Module):
    """mixture [1, 1, L] at 32 kHz mono, condition [1, 512] CLAP embedding -> target [1, 1, L]."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        model.base.stft = MatStft()
        model.base.istft = MatIstft(LENGTH)

    def forward(self, mixture: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.model({"mixture": mixture, "condition": condition})["waveform"]


class ClapText(torch.nn.Module):
    """CLAP's RoBERTa text tower: pooler output, two-layer projection, L2 normalisation."""

    def __init__(self, roberta: str) -> None:
        super().__init__()
        from transformers import RobertaConfig, RobertaModel

        self.text_branch = RobertaModel(RobertaConfig.from_pretrained(roberta))
        self.text_projection = torch.nn.Sequential(
            torch.nn.Linear(768, 512), torch.nn.ReLU(), torch.nn.Linear(512, 512)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pooled = self.text_branch(input_ids=input_ids, attention_mask=attention_mask)["pooler_output"]
        return F.normalize(self.text_projection(pooled), dim=-1)


def build_separator(src: Path, state: dict[str, torch.Tensor]) -> torch.nn.Module:
    sys.path.insert(0, str(src))
    from models.resunet import ResUNet30

    model = ResUNet30(input_channels=1, output_channels=1, condition_size=512)
    sub = {k.removeprefix("ss_model."): v for k, v in state.items() if k.startswith("ss_model.")}
    missing, unexpected = model.load_state_dict(sub, strict=False)
    missing = [k for k in missing if not k.startswith(("base.stft.", "base.istft."))]
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    return model.eval()


def embed_prompts(state: dict[str, torch.Tensor]) -> dict[str, list[float]]:
    from transformers import RobertaTokenizer

    roberta = snapshot_download(ROBERTA, revision=ROBERTA_REVISION, allow_patterns=["*.json", "*.txt"])
    model = ClapText(roberta)
    prefix = "query_encoder.model."
    sub = {
        k.removeprefix(prefix): v
        for k, v in state.items()
        if k.startswith((prefix + "text_branch.", prefix + "text_projection."))
    }
    missing, unexpected = model.load_state_dict(sub, strict=False)
    # position_ids is a constant buffer that only some transformers versions save.
    missing = [k for k in missing if "position_ids" not in k]
    unexpected = [k for k in unexpected if "position_ids" not in k]
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    tokenizer = RobertaTokenizer.from_pretrained(roberta)
    out = {}
    with torch.inference_mode():
        for label, text in PROMPTS.items():
            t = tokenizer(text, return_tensors="pt")
            e = model.eval()(t["input_ids"], t["attention_mask"])[0]
            out[label] = [round(float(v), 7) for v in e]
    return out


def snr_db(reference: np.ndarray, test: np.ndarray) -> float:
    return float(10 * np.log10((reference**2).sum() / ((reference - test) ** 2).sum()))


def main(src: str, onnx_path: str, prompts_path: str) -> None:
    torch.set_num_threads(8)
    ckpt = hf_hub_download(SPACE, CHECKPOINT, repo_type="space", revision=SPACE_REVISION)
    # The pickle holds only tensors, so the safe loader takes it.
    state = torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"]

    reference = build_separator(Path(src), state)
    wrapped = Separator(build_separator(Path(src), state)).eval()
    rng = np.random.default_rng(0)
    mix = torch.from_numpy((rng.standard_normal((1, 1, LENGTH)) * 0.1).astype(np.float32))
    cond = F.normalize(torch.from_numpy(rng.standard_normal((1, 512)).astype(np.float32)), dim=-1)
    with torch.inference_mode():
        expected = reference({"mixture": mix, "condition": cond})["waveform"].numpy()

    raw = Path(onnx_path).with_suffix(".raw.onnx")
    torch.onnx.export(
        wrapped,
        (mix, cond),
        str(raw),
        input_names=["mixture", "condition"],
        output_names=["target"],
        opset_version=17,
        dynamo=False,
    )
    simplified, ok = onnxsim.simplify(onnx.load(str(raw)))
    assert ok
    onnx.save(simplified, onnx_path)
    raw.unlink()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    got = session.run(None, {"mixture": mix.numpy(), "condition": cond.numpy()})[0]
    print(f"ONNX vs PyTorch: {snr_db(expected, got):.1f} dB SNR, {Path(onnx_path).stat().st_size:,} bytes")

    Path(prompts_path).write_text(json.dumps(embed_prompts(state), indent=1) + "\n")
    print(f"wrote {len(PROMPTS)} prompt embeddings to {prompts_path}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
