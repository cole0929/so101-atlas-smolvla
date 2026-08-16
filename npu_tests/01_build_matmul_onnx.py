#!/opt/smolvla_npu_test/bin/python
"""Build a deterministic ONNX graph for the Atlas NPU smoke test."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


OUT = Path("/root/lerobot_project/npu_tests/matmul")
OUT.mkdir(parents=True, exist_ok=True)

x = np.asarray([[1.0, -2.0, 0.5, 3.0]], dtype=np.float32)
w = np.asarray(
    [[0.25, -1.0, 2.0], [1.5, 0.5, -0.25], [-2.0, 3.0, 0.75], [0.1, -0.2, 1.25]],
    dtype=np.float32,
)
b = np.asarray([0.5, -0.75, 1.0], dtype=np.float32)
y = x @ w + b

graph = helper.make_graph(
    nodes=[
        helper.make_node("MatMul", ["input", "weight"], ["matmul"]),
        helper.make_node("Add", ["matmul", "bias"], ["output"]),
    ],
    name="atlas_npu_matmul_smoke",
    inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
    outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])],
    initializer=[numpy_helper.from_array(w, "weight"), numpy_helper.from_array(b, "bias")],
)
model = helper.make_model(
    graph,
    producer_name="atlas_smolvla_deployment",
    opset_imports=[helper.make_opsetid("", 11)],
)
model.ir_version = 7
onnx.checker.check_model(model)
onnx.save(model, OUT / "matmul.onnx")
np.save(OUT / "input.npy", x)
np.save(OUT / "expected.npy", y)
print(f"ONNX_SMOKE_BUILT path={OUT / 'matmul.onnx'} expected={y.tolist()}")
