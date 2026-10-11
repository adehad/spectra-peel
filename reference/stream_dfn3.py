"""Make the DeepFilterNet3 ONNX graph resumable, so a long file can run in chunks.

The published graph starts its five GRUs from zero and pads its three time convolutions with
zeros, so every run starts cold. A cold GRU takes well over 15 s of audio to converge, so cutting a
file into chunks changes the output audibly. This rewrites the graph so the caller carries that state:

- each GRU's initial state becomes an input (h_*), and its per-step output is exposed (y_*);
- each causal time pad becomes a concat with history passed in (hist_*), and the tensor it padded
  is exposed (pre_*).

For a chunk of output frames [a, b), feed features [a, b + 2) (the graph looks two frames ahead),
keep the first b - a output frames, and take the next state at step b - a - 1 of each y_*, and
history from the last k steps before b - a of each pre_*. Zero state reproduces a fresh run.

    uv run python stream_dfn3.py deepfilter.onnx deepfilter-stream.onnx
"""

import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

LOOKAHEAD = 2
HIDDEN = 256
GRUS: dict[str, str] = {
    "/enc/emb_gru/gru/GRU": "enc",
    "/erb_dec/emb_gru/gru/GRU": "erb0",
    "/erb_dec/emb_gru/gru/GRU_1": "erb1",
    "/df_dec/df_gru/gru/GRU": "df0",
    "/df_dec/df_gru/gru/GRU_1": "df1",
}
# Pad node -> (state name, history frames, channels, width)
PADS: dict[str, tuple[str, int, int, int]] = {
    "/enc/erb_conv0/erb_conv0.0/Pad": ("erb", 2, 1, 32),
    "/enc/df_conv0/df_conv0.0/Pad": ("df", 2, 2, 96),
    "/df_dec/df_convp/df_convp.0/Pad": ("dfp", 4, 64, 96),
}


def make_streaming(model: onnx.ModelProto) -> onnx.ModelProto:
    g = model.graph
    for node in g.node:
        if node.name in GRUS:
            name = GRUS[node.name]
            g.input.append(helper.make_tensor_value_info(f"h_{name}", TensorProto.FLOAT, [1, 1, HIDDEN]))
            node.input[5] = f"h_{name}"
            g.output.append(helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, None))
    for i, node in enumerate(g.node):
        if node.name in PADS:
            name, k, c, w = PADS[node.name]
            hist = f"hist_{name}"
            g.input.append(helper.make_tensor_value_info(hist, TensorProto.FLOAT, [1, c, k, w]))
            g.output.append(helper.make_tensor_value_info(node.input[0], TensorProto.FLOAT, None))
            g.node[i].CopyFrom(
                helper.make_node("Concat", [hist, node.input[0]], [node.output[0]], name=node.name, axis=2)
            )
    return model


def state_names(model: onnx.ModelProto) -> tuple[dict[str, str], dict[str, str]]:
    """Exposed tensor name of each GRU output and each pre-pad tensor, keyed by state name."""
    y = {GRUS[n.name]: n.output[0] for n in model.graph.node if n.name in GRUS}
    pre = {PADS[n.name][0]: n.input[1] for n in model.graph.node if n.name in PADS}
    return y, pre


def run_chunked(
    session: ort.InferenceSession, y: dict, pre: dict, erb: np.ndarray, spec: np.ndarray, chunk: int
):
    frames = erb.shape[2]
    state = {f"h_{n}": np.zeros((1, 1, HIDDEN), np.float32) for n in GRUS.values()}
    for name, k, c, w in PADS.values():
        state[f"hist_{name}"] = np.zeros((1, c, k, w), np.float32)
    masks, coefs = [], []
    for a in range(0, frames, chunk):
        b = min(frames, a + chunk)
        end = min(frames, b + LOOKAHEAD)
        feeds = {"feat_erb": erb[:, :, a:end], "feat_spec": spec[:, :, a:end], **state}
        names = ["erb_mask", "df_coefs", *y.values(), *pre.values()]
        out = {n: np.asarray(v) for n, v in zip(names, session.run(names, feeds))}
        masks.append(out["erb_mask"][:, :, : b - a])
        coefs.append(out["df_coefs"][:, :, : b - a])
        state = {f"h_{n}": out[t][b - a - 1] for n, t in y.items()}
        for name, k, _, _ in PADS.values():
            state[f"hist_{name}"] = out[pre[name]][:, :, b - a - k : b - a]
    return np.concatenate(masks, axis=2), np.concatenate(coefs, axis=2)


def main(src: str, dst: str) -> None:
    streaming = make_streaming(onnx.load(src))
    onnx.save(streaming, dst)

    rng = np.random.default_rng(0)
    frames = 1234
    erb = (rng.standard_normal((1, 1, frames, 32)) * 0.3).astype(np.float32)
    spec = (rng.standard_normal((1, 2, frames, 96)) * 0.5).astype(np.float32)
    whole = [
        np.asarray(v)
        for v in ort.InferenceSession(src, providers=["CPUExecutionProvider"]).run(
            None, {"feat_erb": erb, "feat_spec": spec}
        )
    ]
    session = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    y, pre = state_names(streaming)
    for chunk in (97, 400):
        mask, coefs = run_chunked(session, y, pre, erb, spec, chunk)
        dm = np.abs(mask - whole[0]).max()
        dc = np.abs(coefs - whole[1]).max()
        print(f"chunk {chunk}: max abs diff mask {dm:.2e}, coefs {dc:.2e}")
        if dm > 1e-4 or dc > 1e-4:
            sys.exit("chunked run diverges from the whole-file run")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
