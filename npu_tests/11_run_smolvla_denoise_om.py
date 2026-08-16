#!/opt/smolvla_npu_test/bin/python
"""Run one SmolVLA denoising step OM against its CPU reference."""

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


ROOT = Path("/root/lerobot_project/npu_tests/denoise")
PREFIX = Path("/root/lerobot_project/npu_tests/prefix")
REPEATS = 10


def main() -> None:
    model_filename = os.environ.get("DENOISE_OM", "denoise_origin.om")
    x_t = np.ascontiguousarray(np.load(ROOT / "input_xt.npy"), dtype=np.float32)
    timestep = np.ascontiguousarray(np.load(ROOT / "input_timestep.npy"), dtype=np.float32)
    expected = np.load(ROOT / "expected_cpu.npy")
    with np.load(PREFIX / "output_prefix_origin_npu.npz") as archive:
        kv = [np.ascontiguousarray(archive[f"output_{i:02d}"], dtype=np.float32) for i in range(32)]
    inputs = [x_t, timestep, *kv]
    resource = AclLiteResource(device_id=0)
    resource.init()
    model = AclLiteModel(str(ROOT / model_filename))
    latencies = []
    output = None
    for index in range(REPEATS + 1):
        started = time.perf_counter()
        result = model.execute(inputs)
        elapsed = (time.perf_counter() - started) * 1000.0
        if result is None or len(result) != 1:
            raise RuntimeError("denoise OM returned no output")
        output = result[0]
        if index > 0:
            latencies.append(elapsed)
    assert output is not None
    error = np.abs(output.astype(np.float32) - expected.astype(np.float32))
    finite = bool(np.isfinite(output).all())
    max_error = float(error.max())
    mean_error = float(error.mean())
    passed = finite and max_error <= 0.02 and mean_error <= 0.002
    np.save(ROOT / "output_npu.npy", output)
    report = {
        "status": "PASS" if passed else "FAIL",
        "subgraph": "SmolVLA action expert single denoise step",
        "model": model_filename,
        "output_shape": list(output.shape),
        "finite": finite,
        "max_abs_error": max_error,
        "mean_abs_error": mean_error,
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_min_ms": float(np.min(latencies)),
        "latency_max_ms": float(np.max(latencies)),
    }
    (ROOT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    model.destroy()
    del model
    del resource
    gc.collect()
    print(json.dumps(report, indent=2))
    print("SMOLVLA_DENOISE_NPU_PASS" if passed else "SMOLVLA_DENOISE_NPU_FAIL")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
