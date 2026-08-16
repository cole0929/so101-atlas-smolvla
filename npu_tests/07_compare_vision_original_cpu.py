#!/opt/smolvla_npu_test/bin/python
"""Compare vision OM outputs with the unmodified BF16 SmolVLA vision path."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from safetensors.torch import load_file

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import get_policy_class


PROJECT = Path("/root/lerobot_project")
MODEL = (PROJECT / "models/smolvla_yellow_block_atlas_v2_clean_a100/latest").resolve()
ASSETS = PROJECT / "models/HuggingFaceTB_SmolVLM2-500M-Video-Instruct_assets"
CPU_CASE = PROJECT / "golden_case/cpu_020000_ep000_frame000/preprocessed_inputs.npz"
ROOT = PROJECT / "npu_tests/vision"


def main() -> None:
    torch.set_num_threads(4)
    cfg = PreTrainedConfig.from_pretrained(MODEL, local_files_only=True)
    cfg.device = "cpu"
    cfg.use_amp = False
    cfg.compile_model = False
    cfg.load_vlm_weights = False
    cfg.vlm_model_name = str(ASSETS)
    cfg.pretrained_path = str(MODEL)
    policy = get_policy_class(cfg.type)(cfg)
    checkpoint = load_file(MODEL / "model.safetensors", device="cpu")
    policy.load_state_dict(checkpoint, strict=True, assign=True)
    del checkpoint
    policy.eval()
    with np.load(CPU_CASE) as case:
        batch = {
            "observation.images.front": torch.from_numpy(case["observation.images.front"]),
            "observation.images.handeye": torch.from_numpy(case["observation.images.handeye"]),
        }
    prepared, _ = policy.prepare_images(batch)
    reports = {}
    for camera, image in zip(("front", "handeye"), prepared, strict=True):
        with torch.inference_mode():
            original = policy.model.vlm_with_expert.embed_image(image).float().cpu().numpy()
        npu = np.load(ROOT / f"output_{camera}_npu.npy").astype(np.float32)
        error = np.abs(npu - original)
        np.save(ROOT / f"expected_{camera}_original_bf16_cpu.npy", original)
        reports[camera] = {
            "original_dtype": str(policy.model.vlm_with_expert.get_vlm_model().vision_model.dtype),
            "shape": list(original.shape),
            "max_abs_error": float(error.max()),
            "mean_abs_error": float(error.mean()),
            "reference_abs_mean": float(np.abs(original).mean()),
            "finite": bool(np.isfinite(npu).all()),
        }
    report = {"checkpoint": "020000", "reference": "original BF16 SmolVLA embed_image", "cameras": reports}
    (ROOT / "original_bf16_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print("VISION_ORIGINAL_BF16_COMPARISON_FINISHED")


if __name__ == "__main__":
    main()
