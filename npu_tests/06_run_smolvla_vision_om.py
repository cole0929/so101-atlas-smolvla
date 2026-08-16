#!/opt/smolvla_npu_test/bin/python
"""Run both fixed camera frames through the SmolVLA vision OM."""

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


ROOT = Path("/root/lerobot_project/npu_tests/vision")
REPEATS = 20


def main() -> None:
    model_filename = os.environ.get("VISION_OM", "vision.om")
    variant = Path(model_filename).stem
    resource = AclLiteResource(device_id=0)
    resource.init()
    model = AclLiteModel(str(ROOT / model_filename))
    camera_reports = {}
    overall_pass = True
    for camera in ("front", "handeye"):
        image = np.ascontiguousarray(np.load(ROOT / f"input_{camera}.npy"), dtype=np.float32)
        expected = np.load(ROOT / f"expected_{camera}_cpu.npy")
        latencies_ms = []
        output = None
        for index in range(REPEATS + 1):
            started = time.perf_counter()
            result = model.execute([image])
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if result is None or len(result) != 1:
                raise RuntimeError(f"ACL vision execution failed for {camera}")
            output = result[0]
            if index > 0:
                latencies_ms.append(elapsed_ms)
        assert output is not None
        abs_error = np.abs(output.astype(np.float32) - expected.astype(np.float32))
        max_abs_error = float(abs_error.max())
        mean_abs_error = float(abs_error.mean())
        finite = bool(np.isfinite(output).all())
        passed = finite and output.shape == expected.shape and max_abs_error <= 0.15 and mean_abs_error <= 0.02
        overall_pass &= passed
        output_name = f"output_{camera}_npu.npy" if variant == "vision" else f"output_{camera}_{variant}_npu.npy"
        np.save(ROOT / output_name, output)
        camera_reports[camera] = {
            "status": "PASS" if passed else "FAIL",
            "input_shape": list(image.shape),
            "output_shape": list(output.shape),
            "max_abs_error": max_abs_error,
            "mean_abs_error": mean_abs_error,
            "finite": finite,
            "latency_mean_ms": float(np.mean(latencies_ms)),
            "latency_min_ms": float(np.min(latencies_ms)),
            "latency_max_ms": float(np.max(latencies_ms)),
        }
    report = {
        "status": "PASS" if overall_pass else "FAIL",
        "subgraph": "SmolVLA vision_model + connector",
        "checkpoint": "020000",
        "soc": "Ascend310B1",
        "model": model_filename,
        "repeats_per_camera": REPEATS,
        "cameras": camera_reports,
    }
    report_name = "report.json" if variant == "vision" else f"report_{variant}.json"
    (ROOT / report_name).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    model.destroy()
    del model
    del resource
    gc.collect()
    print(json.dumps(report, indent=2))
    print("SMOLVLA_VISION_NPU_PASS" if overall_pass else "SMOLVLA_VISION_NPU_FAIL")
    if not overall_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
