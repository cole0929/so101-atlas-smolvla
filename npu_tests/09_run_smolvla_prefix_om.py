#!/opt/smolvla_npu_test/bin/python
"""Run the SmolVLA prefix prefill OM and compare all 16 KV layers."""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/usr/local/Ascend/thirdpart/aarch64/acllite")
from acllite_model import AclLiteModel  # noqa: E402
from acllite_resource import AclLiteResource  # noqa: E402


ROOT = Path("/root/lerobot_project/npu_tests/prefix")
REPEATS = 10


def main() -> None:
    model_filename = os.environ.get("PREFIX_OM", "prefix.om")
    variant = Path(model_filename).stem
    prefix = np.ascontiguousarray(np.load(ROOT / "input.npy"), dtype=np.float32)
    with np.load(ROOT / "expected_cpu.npz") as archive:
        expected = [archive[f"output_{i:02d}"] for i in range(len(archive.files))]
    resource = AclLiteResource(device_id=0)
    resource.init()
    model = AclLiteModel(str(ROOT / model_filename))
    latencies = []
    output = None
    for index in range(REPEATS + 1):
        started = time.perf_counter()
        output = model.execute([prefix])
        elapsed = (time.perf_counter() - started) * 1000.0
        if output is None or len(output) != len(expected):
            raise RuntimeError(f"expected {len(expected)} KV outputs, got {None if output is None else len(output)}")
        if index > 0:
            latencies.append(elapsed)
    layer_errors = []
    for actual, reference in zip(output, expected, strict=True):
        error = np.abs(actual.astype(np.float32) - reference.astype(np.float32))
        layer_errors.append({"max": float(error.max()), "mean": float(error.mean())})
    max_error = max(item["max"] for item in layer_errors)
    mean_error = float(np.mean([item["mean"] for item in layer_errors]))
    finite = all(np.isfinite(item).all() for item in output)
    passed = finite and max_error <= 0.2 and mean_error <= 0.02
    output_name = "output_npu.npz" if variant == "prefix" else f"output_{variant}_npu.npz"
    np.savez_compressed(ROOT / output_name, **{f"output_{i:02d}": x for i, x in enumerate(output)})
    report = {
        "status": "PASS" if passed else "FAIL",
        "subgraph": "SmolVLA 16-layer prefix KV prefill",
        "checkpoint": "020000",
        "model": model_filename,
        "input_shape": list(prefix.shape),
        "outputs": len(output),
        "finite": finite,
        "max_abs_error": max_error,
        "mean_abs_error_across_outputs": mean_error,
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_min_ms": float(np.min(latencies)),
        "latency_max_ms": float(np.max(latencies)),
        "per_output_error": layer_errors,
    }
    report_name = "report.json" if variant == "prefix" else f"report_{variant}.json"
    (ROOT / report_name).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    model.destroy()
    del model
    del resource
    gc.collect()
    print(json.dumps({key: value for key, value in report.items() if key != "per_output_error"}, indent=2))
    print("SMOLVLA_PREFIX_NPU_PASS" if passed else "SMOLVLA_PREFIX_NPU_FAIL")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
