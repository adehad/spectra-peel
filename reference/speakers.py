"""NumPy reference for speaker diarization around the segmentation and embedding graphs, and golden
fixtures for the engine.

Kaldi fbank follows torchaudio.compliance.kaldi.fbank with WeSpeaker's settings (80 bins, 25/10 ms,
Hamming, no dither); official/check_fbank.py compares it with torchaudio. Diarization follows the
speaker separation pipeline: pyannote segmentation-3.0 on 10 s windows every 2 s, WeSpeaker ECAPA
embeddings of each window's single-speaker frames, average-linkage clustering then spherical k-means,
and an assignment of each window's local speakers to the clusters.

    uv run python speakers.py SEGMENTATION.onnx EMBEDDING.onnx MIX_16K.wav ../packages/engine/fixtures/speakers

The committed fixtures come from the first minute of a two-person conversation mixed from LibriSpeech
dev-clean (CC BY 4.0); only labels and embeddings are kept, not the audio. The fbank fixture is a chirp.
"""

import itertools
import json
import sys
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.cluster.hierarchy import fcluster, linkage

SR = 16_000
WINDOW = 160_000
STEP = 32_000
# segmentation-3.0's receptive-field shift, in samples: the global frame grid.
FRAME = 270
MIN_EMBEDDING_SECONDS = 1.0
MAX_LINKAGE = 1000

POWERSET = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.uint8
)


def mel_banks(bins: int = 80, fft: int = 512, low: float = 20.0, high: float = SR / 2) -> np.ndarray:
    mel = lambda f: 1127.0 * np.log(1.0 + f / 700.0)  # noqa: E731
    lo, hi = mel(low), mel(high)
    delta = (hi - lo) / (bins + 1)
    freqs = mel(np.arange(fft // 2) * SR / fft)
    banks = np.zeros((bins, fft // 2 + 1))
    for b in range(bins):
        left, centre, right = lo + b * delta, lo + (b + 1) * delta, lo + (b + 2) * delta
        up = (freqs - left) / (centre - left)
        down = (right - freqs) / (right - centre)
        banks[b, : fft // 2] = np.maximum(0, np.minimum(up, down))
    return banks


def fbank(wav: np.ndarray) -> np.ndarray:
    """[frames, 80] log mel energies of 16 kHz audio in [-1, 1], mean-normalised over frames."""
    x = wav.astype(np.float64) * 32768
    n = 1 + (x.size - 400) // 160
    frames = np.stack([x[i * 160 : i * 160 + 400] for i in range(n)])
    frames -= frames.mean(1, keepdims=True)
    frames = np.concatenate([frames[:, :1] * 0.03, frames[:, 1:] - 0.97 * frames[:, :-1]], axis=1)
    frames *= 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(400) / 399)
    power = np.abs(np.fft.rfft(frames, 512)) ** 2
    feats = np.log(np.maximum(power @ mel_banks().T, np.finfo(np.float32).eps))
    return (feats - feats.mean(0)).astype(np.float32)


def window_starts(total: int) -> list[int]:
    return list(range(0, max(1, total - WINDOW + STEP), STEP))


def frame_span(frames: int, f: int) -> tuple[int, int]:
    return f * WINDOW // frames, (f + 1) * WINDOW // frames


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def centre(e: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Embeddings are compared at unit length, less the mean of all of them."""
    return unit(unit(e) - mean)


def cluster(embs: np.ndarray, k: int, iters: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Unit centroids of k voices in order of first embedding, and the mean every embedding is centred on."""
    mean = unit(embs).mean(0)
    x = centre(embs, mean)
    pick = np.linspace(0, len(x) - 1, min(len(x), MAX_LINKAGE)).round().astype(int)
    sample = x[pick]
    cl = fcluster(linkage(sample, method="average", metric="cosine"), k, criterion="maxclust") - 1
    sizes = np.bincount(cl, minlength=k)
    if sizes.min() < 0.05 * len(sample):
        c0 = unit(sample[cl == int(np.argmax(sizes))].mean(0))
        cents = [c0]
        for _ in range(k - 1):
            cents.append(x[int(np.argmin(np.max(x @ np.stack(cents).T, axis=1)))])
        centroids = np.stack(cents)
    else:
        centroids = np.stack([sample[cl == c].mean(0) for c in range(k)])
    for _ in range(iters):
        centroids = unit(centroids)
        a = np.argmax(x @ centroids.T, axis=1)
        centroids = np.stack([x[a == c].mean(0) if np.any(a == c) else centroids[c] for c in range(k)])
    centroids = unit(centroids)
    nearest = np.argmax(x @ centroids.T, 1)
    first = [int(np.flatnonzero(nearest == c)[0]) if np.any(nearest == c) else len(x) for c in range(k)]
    return centroids[np.argsort(first, kind="stable")], mean


def best_assignment(sim: np.ndarray) -> list[tuple[int, int]]:
    """Rows to distinct columns maximising total similarity; at most 3 x 3, so exhaustive."""
    rows, cols = sim.shape
    best, pairs = -np.inf, []
    if rows <= cols:
        for p in itertools.permutations(range(cols), rows):
            s = sum(sim[r, p[r]] for r in range(rows))
            if s > best:
                best, pairs = s, [(r, p[r]) for r in range(rows)]
    else:
        for p in itertools.permutations(range(rows), cols):
            s = sum(sim[p[c], c] for c in range(cols))
            if s > best:
                best, pairs = s, sorted((p[c], c) for c in range(cols))
    return pairs


def activity(labels: np.ndarray, starts: list[int], local: list[dict[int, np.ndarray]], centroids: np.ndarray,
             mean: np.ndarray, total: int) -> np.ndarray:
    """[k, frames] who speaks on the global frame grid, from per-window powerset labels."""
    k = len(centroids)
    n_fr = labels.shape[1]
    frames = total // FRAME + 1
    acc = np.zeros((k, frames))
    cnt = np.zeros(frames)
    count_acc = np.zeros(frames)
    for w, s in enumerate(starts):
        f0 = s // FRAME
        m = min(n_fr, frames - f0)
        cnt[f0 : f0 + m] += 1
        count_acc[f0 : f0 + m] += labels[w, :m].sum(-1)
        js = sorted(local[w])
        if not js:
            continue
        sim = centre(np.stack([local[w][j] for j in js]), mean) @ centroids.T
        for r, c in best_assignment(sim):
            acc[c, f0 : f0 + m] += labels[w, :m, js[r]]
    score = acc / np.maximum(cnt, 1)
    count = np.floor(count_acc / np.maximum(cnt, 1) + 0.5).astype(int)
    active = np.zeros((k, frames), dtype=np.uint8)
    order = np.argsort(-score, axis=0, kind="stable")
    for t in range(frames):
        for r in range(min(count[t], k)):
            if score[order[r, t], t] > 0:
                active[order[r, t], t] = 1
    return active


def read_wav(path: str) -> np.ndarray:
    with wave.open(path) as f:
        if f.getframerate() != SR or f.getnchannels() != 1 or f.getsampwidth() != 2:
            raise SystemExit("expects 16 kHz mono 16-bit WAV")
        return np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float32) / 32768


def main(seg_path: str, emb_path: str, wav_path: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mix = read_wav(wav_path)[: 60 * SR]
    t = np.arange(3 * SR) / SR
    noise = np.random.default_rng(1).standard_normal(t.size)
    clip = (0.3 * np.sin(2 * np.pi * (200 + 900 * t) * t) + 0.05 * noise).astype(np.float32)
    clip.tofile(out / "fbank_input.f32")
    fbank(clip).tofile(out / "fbank.f32")

    seg = ort.InferenceSession(seg_path, providers=["CPUExecutionProvider"])
    emb = ort.InferenceSession(emb_path, providers=["CPUExecutionProvider"])
    starts = window_starts(mix.size)
    windows = np.stack([np.pad(mix[s : s + WINDOW], (0, max(0, WINDOW - mix[s : s + WINDOW].size))) for s in starts])
    classes = np.concatenate([seg.run(None, {"x": windows[i : i + 8, None]})[0].argmax(-1) for i in range(0, len(windows), 8)])
    labels = POWERSET[classes]
    n_fr = labels.shape[1]
    single = labels * (labels.sum(-1, keepdims=True) < 2)
    local: list[dict[int, np.ndarray]] = [{} for _ in starts]
    keys, embs = [], []
    for w, s in enumerate(starts):
        for j in range(3):
            fr = np.flatnonzero(single[w, :, j])
            idx = np.concatenate([np.arange(*frame_span(n_fr, f)) for f in fr]) + s if fr.size else np.array([], int)
            idx = idx[idx < mix.size]
            if idx.size < MIN_EMBEDDING_SECONDS * SR:
                continue
            e = emb.run(None, {"feats": fbank(mix[idx])[None]})[0][0]
            local[w][j] = e
            keys.append([w, j])
            embs.append(e)
    embs = np.stack(embs).astype(np.float32)
    centroids, mean = cluster(embs, 2)
    active = activity(labels, starts, local, centroids, mean, mix.size)

    classes.astype(np.uint8).tofile(out / "classes.u8")
    np.asarray(keys, dtype=np.uint16).tofile(out / "keys.u16")
    embs.tofile(out / "embeddings.f32")
    active.tofile(out / "activity.u8")
    (out / "meta.json").write_text(json.dumps({
        "samples": int(mix.size), "windows": len(starts), "framesPerWindow": int(n_fr),
        "embeddings": len(keys), "dimension": int(embs.shape[1]), "speakers": 2,
        "fbankFrames": int(1 + (clip.size - 400) // 160),
    }, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:5])
