#!/opt/smolvla_npu_test/bin/python
"""Run the complete fixed golden SmolVLA path on Atlas NPU."""

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


ROOT = Path("/root/lerobot_project")
NPU = ROOT / "npu_tests"
GOLDEN = ROOT / "golden_case/cpu_020000_ep000_frame000"


def execute_one(model: AclLiteModel, inputs: list[np.ndarray]) -> tuple[list[np.ndarray], float]:
    started = time.perf_counter()
    outputs = model.execute(inputs)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if outputs is None:
        raise RuntimeError("ACL model returned no outputs")
    return outputs, elapsed_ms


def main() -> None:
    resource = AclLiteResource(device_id=0)
    resource.init()
    vision = AclLiteModel(str(NPU / "vision/vision.om"))
    state_proj = AclLiteModel(str(NPU / "state_proj/state_proj.om"))
    prefix_model = AclLiteModel(str(NPU / "prefix/prefix_origin.om"))
    denoise = AclLiteModel(str(NPU / "denoise/denoise_origin.om"))

    latencies: dict[str, float | list[float]] = {}
    front = np.ascontiguousarray(np.load(NPU / "vision/input_front.npy"), dtype=np.float32)
    handeye = np.ascontiguousarray(np.load(NPU / "vision/input_handeye.npy"), dtype=np.float32)
    state = np.ascontiguousarray(np.load(NPU / "state_proj/input.npy"), dtype=np.float32)
    front_emb, front_ms = execute_one(vision, [front])
    handeye_emb, handeye_ms = execute_one(vision, [handeye])
    state_emb, state_ms = execute_one(state_proj, [state])

    # Keep the fixed 48-token task embedding and replace all observation-dependent fields.
    prefix = np.ascontiguousarray(np.load(NPU / "prefix/input.npy"), dtype=np.float32)
    prefix[:, 0:64, :] = front_emb[0].astype(np.float32) * np.sqrt(960.0)
    prefix[:, 64:128, :] = handeye_emb[0].astype(np.float32) * np.sqrt(960.0)
    prefix[:, 176:177, :] = state_emb[0].astype(np.float32)[:, None, :]
    kv, prefix_ms = execute_one(prefix_model, [prefix])
    kv = [np.ascontiguousarray(value, dtype=np.float32) for value in kv]

    x_t = np.ascontiguousarray(np.load(NPU / "denoise/input_xt.npy"), dtype=np.float32)
    denoise_ms = []
    for step in range(10):
        timestep = np.ascontiguousarray(np.array([1.0 - step / 10.0], dtype=np.float32))
        velocity, elapsed_ms = execute_one(denoise, [x_t, timestep, *kv])
        x_t = np.ascontiguousarray(x_t - 0.1 * velocity[0].astype(np.float32))
        denoise_ms.append(elapsed_ms)

    np.save(NPU / "full_raw32_npu.npy", x_t)
    raw6 = x_t[0, :, :6]
    np.save(NPU / "full_raw6_npu.npy", raw6)
    expected_same_prefix = np.load(NPU / "denoise/expected_euler10_cpu.npy")
    same_prefix_error = np.abs(x_t - expected_same_prefix)
    original_cpu = np.load(GOLDEN / "raw_actions_cpu.npy")
    original_error = np.abs(raw6 - original_cpu)
    latencies.update(
        vision_front_ms=front_ms,
        vision_handeye_ms=handeye_ms,
        state_ms=state_ms,
        prefix_ms=prefix_ms,
        denoise_each_ms=denoise_ms,
    )
    total_ms = front_ms + handeye_ms + state_ms + prefix_ms + sum(denoise_ms)
    report = {
        "status": "PASS" if np.isfinite(x_t).all() and same_prefix_error.max() <= 2e-4 else "FAIL",
        "finite": bool(np.isfinite(x_t).all()),
        "raw_action_shape": list(raw6.shape),
        "same_npu_prefix_cpu_max_abs_error": float(same_prefix_error.max()),
        "same_npu_prefix_cpu_mean_abs_error": float(same_prefix_error.mean()),
        "original_policy_cpu_max_abs_error": float(original_error.max()),
        "original_policy_cpu_mean_abs_error": float(original_error.mean()),
        "raw6_min": float(raw6.min()),
        "raw6_max": float(raw6.max()),
        "latencies_ms": latencies,
        "total_ms": total_ms,
    }
    (NPU / "full_npu_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

    for model in (denoise, prefix_model, state_proj, vision):
        model.destroy()
    del denoise, prefix_model, state_proj, vision, resource
    gc.collect()
    if report["status"] != "PASS":
        raise SystemExit(2)
    print("SMOLVLA_FULL_NPU_PASS")


if __name__ == "__main__":
    main()
