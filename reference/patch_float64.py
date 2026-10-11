"""Rewrite float64 tensors in an ONNX graph to float32 and check the outputs still match.

ONNX Runtime Web ships no float64 kernels for some ops (ConstantOfShape among them), so
StemSplitio's HT-Demucs exports fail to load in the browser on their iSTFT. Only that path
uses float64, and float32 reproduces the original stems within float rounding.

    uv run python patch_float64.py in.onnx out.onnx
"""

import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, numpy_helper

DOUBLE, FLOAT = TensorProto.DOUBLE, TensorProto.FLOAT
SEGMENT = int(7.8 * 44_100)


def to_float32(t: onnx.TensorProto) -> onnx.TensorProto:
    return numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float32), t.name)


def patch(model: onnx.ModelProto) -> dict[str, int]:
    counts = {"tensor_attributes": 0, "casts": 0, "initializers": 0}
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == DOUBLE:
                attr.t.CopyFrom(to_float32(attr.t))
                counts["tensor_attributes"] += 1
            if node.op_type == "Cast" and attr.name == "to" and attr.i == DOUBLE:
                attr.i = FLOAT
                counts["casts"] += 1
    for init in model.graph.initializer:
        if init.data_type == DOUBLE:
            init.CopyFrom(to_float32(init))
            counts["initializers"] += 1
    for vi in model.graph.value_info:
        if vi.type.tensor_type.elem_type == DOUBLE:
            vi.type.tensor_type.elem_type = FLOAT
    return counts


def snr_db(reference: np.ndarray, test: np.ndarray) -> float:
    noise = float(((reference - test) ** 2).sum())
    return float("inf") if noise == 0 else 10 * np.log10(float((reference**2).sum()) / noise)


def main(src: str, dst: str) -> None:
    model = onnx.load(src)
    print("patched", patch(model))
    onnx.save(model, dst)

    x = (np.random.default_rng(0).standard_normal((1, 2, SEGMENT)) * 0.1).astype(np.float32)
    before = ort.InferenceSession(src, providers=["CPUExecutionProvider"]).run(None, {"mix": x})[0]
    after = ort.InferenceSession(dst, providers=["CPUExecutionProvider"]).run(None, {"mix": x})[0]
    snr = snr_db(before, after)
    print(f"max abs diff {np.abs(before - after).max():.3e}, SNR {snr:.1f} dB")
    if snr < 100:
        sys.exit("patched model diverges from the original")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
