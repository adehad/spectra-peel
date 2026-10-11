"""Rewrite the STFT and iSTFT convolutions of a simplified HT-Demucs export as MatMuls.

The export computes its STFT as a Conv and its iSTFT as a ConvTranspose, each with a
4096-tap kernel, 1024 stride and 2049 channels. ONNX Runtime Web's WebGPU ConvTranspose
takes about 25 s per iSTFT node on one segment, against 0.6 s on wasm. Framing plus MatMul
plus overlap-add computes the same thing with kernels the WebGPU provider handles well.
Run it on the output of simplify.py.

    uv run python rewrite_stft.py simplified.onnx rewritten.onnx
"""

import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

from patch_float64 import vocals_check

PAD_TO = 4


class Builder:
    def __init__(self, model: onnx.ModelProto) -> None:
        self.model = model
        self.nodes: list[onnx.NodeProto] = []
        self.count = 0

    def name(self, base: str) -> str:
        self.count += 1
        return f"{base}/stft_rewrite_{self.count}"

    def const(self, base: str, value: np.ndarray) -> str:
        name = self.name(base)
        self.model.graph.initializer.append(numpy_helper.from_array(value, name))
        return name

    def op(self, base: str, op_type: str, inputs: list[str], output: str | None = None, **attrs) -> str:
        out = output or self.name(base)
        self.nodes.append(helper.make_node(op_type, inputs, [out], name=self.name(base), **attrs))
        return out

    def ints(self, base: str, values: list[int]) -> str:
        return self.const(base, np.array(values, dtype=np.int64))


def attrs_of(node: onnx.NodeProto) -> dict:
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def is_strided_frame_op(node: onnx.NodeProto, weights: dict[str, np.ndarray]) -> bool:
    if node.op_type not in ("Conv", "ConvTranspose") or len(node.input) != 2 or node.input[1] not in weights:
        return False
    a = attrs_of(node)
    kernel, stride = a.get("kernel_shape", [0]), a.get("strides", [1])
    return (
        len(kernel) == 1
        and kernel[0] > stride[0]
        and kernel[0] % stride[0] == 0
        and a.get("group", 1) == 1
        and not any(a.get("pads", [0, 0]))
        and a.get("dilations", [1]) == [1]
        and not any(a.get("output_padding", [0]))
    )


def padded(channels: int) -> int:
    """ORT's WebGPU MatMul runs about 4x slower with 2049 frequency bins than with 2052 (zero-padded)."""
    return -(-channels // PAD_TO) * PAD_TO


def stft(b: Builder, node: onnx.NodeProto, w: np.ndarray, shape: list[int]) -> None:
    """Conv(x [N, 1, L], w [C, 1, K], stride H) -> frames [N, T, K] @ w.T -> [N, C, T]."""
    n, cin, length = shape
    c, _, k = w.shape
    hop = attrs_of(node)["strides"][0]
    assert cin == 1 and length % hop == 0, (node.name, shape)
    blocks, taps = length // hop, k // hop
    frames = (length - k) // hop + 1
    base = node.name
    x = b.op(base, "Reshape", [node.input[0], b.ints(base, [n, blocks, hop])])
    parts = [
        b.op(base, "Slice", [x, b.ints(base, [j]), b.ints(base, [j + frames]), b.ints(base, [1])])
        for j in range(taps)
    ]
    framed = b.op(base, "Concat", parts, axis=2)
    basis = np.zeros((k, padded(c)), dtype=w.dtype)
    basis[:, :c] = w[:, 0, :].T
    spec = b.op(base, "MatMul", [framed, b.const(base, basis)])
    if padded(c) != c:
        spec = b.op(base, "Slice", [spec, b.ints(base, [0]), b.ints(base, [c]), b.ints(base, [2])])
    b.op(base, "Transpose", [spec], node.output[0], perm=[0, 2, 1])


def istft(b: Builder, node: onnx.NodeProto, w: np.ndarray, shape: list[int]) -> None:
    """ConvTranspose(x [N, C, T], w [C, 1, K], stride H) -> x.T @ w, then overlap-add -> [N, 1, L]."""
    n, c, frames = shape
    _, cout, k = w.shape
    hop = attrs_of(node)["strides"][0]
    assert cout == 1, (node.name, w.shape)
    taps = k // hop
    base = node.name
    xt = b.op(base, "Transpose", [node.input[0]], perm=[0, 2, 1])
    if padded(c) != c:
        xt = b.op(base, "Pad", [xt, b.ints(base, [0, 0, 0, 0, 0, padded(c) - c])])
    basis = np.zeros((padded(c), k), dtype=w.dtype)
    basis[:c] = w[:, 0, :]
    framed = b.op(base, "MatMul", [xt, b.const(base, basis)])
    split = b.op(base, "Reshape", [framed, b.ints(base, [n, frames, taps, hop])])
    total = None
    for j in range(taps):
        part = b.op(base, "Slice", [split, b.ints(base, [j]), b.ints(base, [j + 1]), b.ints(base, [2])])
        part = b.op(base, "Reshape", [part, b.ints(base, [n, frames, hop])])
        part = b.op(base, "Pad", [part, b.ints(base, [0, j, 0, 0, taps - 1 - j, 0])])
        total = part if total is None else b.op(base, "Add", [total, part])
    assert total is not None
    b.op(base, "Reshape", [total, b.ints(base, [n, 1, (frames + taps - 1) * hop])], node.output[0])


def rewrite(model: onnx.ModelProto) -> int:
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes = {
        vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim]
        for vi in list(inferred.graph.value_info) + list(inferred.graph.input)
    }
    weights = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    for n in model.graph.node:
        if n.op_type == "Constant" and n.attribute[0].name == "value":
            weights[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
    b = Builder(model)
    rewritten = 0
    for node in model.graph.node:
        if not is_strided_frame_op(node, weights):
            b.nodes.append(node)
            continue
        w, shape = weights[node.input[1]], shapes[node.input[0]]
        (stft if node.op_type == "Conv" else istft)(b, node, w, shape)
        rewritten += 1
    used = {i for n in b.nodes for i in n.input} | {o.name for o in model.graph.output}
    del model.graph.node[:]
    model.graph.node.extend(n for n in b.nodes if n.op_type != "Constant" or n.output[0] in used)
    return rewritten


def main(src: str, dst: str) -> None:
    model = onnx.load(src)
    print("rewrote", rewrite(model), "nodes")
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, dst)

    if vocals_check(src, dst) < 100:
        sys.exit("rewritten model diverges from the original")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
