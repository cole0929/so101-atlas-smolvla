#!/opt/smolvla_npu_test/bin/python
"""Run the trained SmolVLA state-projection OM and compare with CPU."""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/usr/local/Ascend/thirdpart/aarch64/acllite")
from acllite_model import AclLiteModel  # noqa: E402
from acllite_resource import AclLiteResource  # noqa: E402


ROOT = Path("/root/lerobot_project/npu_tests/state_proj")
REPEATS = 100


def main() -> None:
    input_array = np.ascontiguousarray(np.load(ROOT / "input.npy"), dtype=np.float32)
    expected = np.load(ROOT / "expected_cpu.npy")
    resource = AclLiteResource(device_id=0)
    resource.init()
    model = AclLiteModel(str(ROOT / "state_proj.om"))
    latencies_ms = []
    output = None
    for index in range(REPEATS + 1):
        started = time.perf_counter()
        result = model.execute([input_array])
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if result is None or len(result) != 1:
            raise RuntimeError("ACL state projection returned no output")
        output = result[0]
        if index > 0:
            latencies_ms.append(elapsed_ms)
    assert output is not None
    abs_error = np.abs(output.astype(np.float32) - expected.astype(np.float32))
    max_abs_error = float(abs_error.max())
    mean_abs_error = float(abs_error.mean())
    finite = bool(np.isfinite(output).all())
    max_tolerance = 2e-2
    mean_tolerance = 2e-3
    passed = (
        finite
        and output.shape == expected.shape
        and max_abs_error <= max_tolerance
        and mean_abs_error <= mean_tolerance
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "subgraph": "SmolVLA.model.state_proj",
        "checkpoint": "020000",
        "soc": "Ascend310B1",
        "input_shape": list(input_array.shape),
        "output_shape": list(output.shape),
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "max_tolerance": max_tolerance,
        "mean_tolerance": mean_tolerance,
        "finite": finite,
        "repeats": REPEATS,
        "latency_mean_ms": float(np.mean(latencies_ms)),
        "latency_min_ms": float(np.min(latencies_ms)),
        "latency_max_ms": float(np.max(latencies_ms)),
    }
    np.save(ROOT / "output_npu.npy", output)
    (ROOT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    model.destroy()
    del model
    del resource
    gc.collect()
    print(json.dumps(report, indent=2))
    print("SMOLVLA_STATE_PROJ_NPU_PASS" if passed else "SMOLVLA_STATE_PROJ_NPU_FAIL")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
