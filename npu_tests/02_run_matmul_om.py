#!/opt/smolvla_npu_test/bin/python
"""Execute the deterministic smoke-test OM through pyACL/AclLite."""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

ACLLITE = Path("/usr/local/Ascend/thirdpart/aarch64/acllite")
sys.path.insert(0, str(ACLLITE))

from acllite_model import AclLiteModel  # noqa: E402
from acllite_resource import AclLiteResource  # noqa: E402


ROOT = Path("/root/lerobot_project/npu_tests/matmul")
REPEATS = 100


def main() -> None:
    input_array = np.ascontiguousarray(np.load(ROOT / "input.npy"), dtype=np.float32)
    expected = np.load(ROOT / "expected.npy")
    resource = AclLiteResource(device_id=0)
    resource.init()
    model = AclLiteModel(str(ROOT / "matmul.om"))

    latencies_ms = []
    output = None
    for index in range(REPEATS + 1):
        started = time.perf_counter()
        result = model.execute([input_array])
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if result is None or len(result) != 1:
            raise RuntimeError("ACL model execution returned no output")
        output = result[0]
        if index > 0:
            latencies_ms.append(elapsed_ms)

    assert output is not None
    max_abs_error = float(np.max(np.abs(output - expected)))
    finite = bool(np.isfinite(output).all())
    tolerance = 1e-3
    passed = finite and output.shape == expected.shape and max_abs_error <= tolerance
    report = {
        "status": "PASS" if passed else "FAIL",
        "backend": "pyACL/AclLite",
        "soc": "Ascend310B1",
        "repeats": REPEATS,
        "output": output.tolist(),
        "expected": expected.tolist(),
        "max_abs_error": max_abs_error,
        "tolerance": tolerance,
        "finite": finite,
        "latency_mean_ms": float(np.mean(latencies_ms)),
        "latency_min_ms": float(np.min(latencies_ms)),
        "latency_max_ms": float(np.max(latencies_ms)),
    }
    (ROOT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.save(ROOT / "output_npu.npy", output)
    model.destroy()
    del model
    del resource
    gc.collect()
    print(json.dumps(report, indent=2))
    print("NPU_SMOKE_PASS" if passed else "NPU_SMOKE_FAIL")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
