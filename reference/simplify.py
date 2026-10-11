"""Fold an HT-Demucs export to a fixed input shape and check the outputs still match.

The published exports keep the segment length dynamic, so most of their nodes are shape
bookkeeping (Constant, Shape, Slice, ScatterND). Fixing the input to one 7.8 s segment lets
onnxsim fold those away, leaving about 1,430 nodes. Without this, ORT Web's wasm provider
runs out of its 4 GB heap on a single segment. Run it on the output of patch_float64.py,
on a machine with plenty of free RAM.

    uv run python simplify.py patched.onnx simplified.onnx
"""

import sys
from collections import Counter

import onnx
import onnxsim

from patch_float64 import SEGMENT, vocals_check


def main(src: str, dst: str) -> None:
    model = onnx.load(src)
    simplified, ok = onnxsim.simplify(model, overwrite_input_shapes={"mix": [1, 2, SEGMENT]})
    if not ok:
        sys.exit("onnxsim could not validate the simplified model")
    print("nodes", len(model.graph.node), "->", len(simplified.graph.node))
    print(Counter(n.op_type for n in simplified.graph.node).most_common(10))
    onnx.save(simplified, dst)

    if vocals_check(src, dst) < 100:
        sys.exit("simplified model drifted from the original")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
