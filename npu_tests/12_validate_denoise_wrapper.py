#!/opt/smolvla_npu_test/bin/python
"""Prove that the export wrapper matches LeRobot's native denoise_step."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from transformers import DynamicCache

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import get_policy_class

from importlib.util import module_from_spec, spec_from_file_location


PROJECT = Path("/root/lerobot_project")
MODEL = (PROJECT / "models/smolvla_yellow_block_atlas_v2_clean_a100/latest").resolve()
ASSETS = PROJECT / "models/HuggingFaceTB_SmolVLM2-500M-Video-Instruct_assets"
PREFIX = PROJECT / "npu_tests/prefix"
DENOISE = PROJECT / "npu_tests/denoise"


def main() -> None:
    cfg = PreTrainedConfig.from_pretrained(MODEL, local_files_only=True)
    cfg.device = "cpu"
    cfg.use_amp = False
    cfg.compile_model = False
    cfg.load_vlm_weights = False
    cfg.vlm_model_name = str(ASSETS)
    cfg.pretrained_path = str(MODEL)
    policy = get_policy_class(cfg.type).from_pretrained(
        MODEL, config=cfg, local_files_only=True, strict=True
    ).eval()

    # The filename starts with a number, so load it directly by path.
    spec = spec_from_file_location("smolvla_denoise_export", PROJECT / "npu_tests/10_export_smolvla_denoise.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load denoise export module")
    export_module = module_from_spec(spec)
    spec.loader.exec_module(export_module)
    prefix_pad = torch.from_numpy(np.load(PREFIX / "pad_mask.npy")).bool()
    wrapper = export_module.DenoiseStep(policy.model, prefix_pad).float().cpu().eval()
    with np.load(PREFIX / "output_prefix_origin_npu.npz") as archive:
        kv = tuple(torch.from_numpy(archive[f"output_{i:02d}"]).float() for i in range(32))

    cache = DynamicCache()
    for layer in range(16):
        cache.update(kv[layer * 2], kv[layer * 2 + 1], layer)
    x_t = torch.from_numpy(np.load(DENOISE / "input_xt.npy")).float()
    timestep = torch.from_numpy(np.load(DENOISE / "input_timestep.npy")).float()
    with torch.inference_mode():
        native = policy.model.denoise_step(prefix_pad, cache, x_t, timestep)
        exported = wrapper(x_t, timestep, *kv)
    delta = (native.float() - exported.float()).abs()
    print(
        "DENOISE_WRAPPER_MATCH "
        f"max_abs={delta.max().item():.9g} mean_abs={delta.mean().item():.9g} "
        f"native_shape={tuple(native.shape)}"
    )
    if delta.max().item() > 1e-5:
        raise SystemExit("wrapper does not match native LeRobot denoise_step")

    # Save an end-to-end golden action chunk using LeRobot's exact Euler schedule.
    x_cpu = x_t.clone()
    with torch.inference_mode():
        for step in range(10):
            current_t = torch.tensor([1.0 - step / 10.0], dtype=torch.float32)
            x_cpu = x_cpu - 0.1 * wrapper(x_cpu, current_t, *kv)
    np.save(DENOISE / "expected_euler10_cpu.npy", x_cpu.numpy())
    print(
        "DENOISE_EULER10_CPU_SAVED "
        f"range=[{x_cpu.min().item():.6f},{x_cpu.max().item():.6f}]"
    )


if __name__ == "__main__":
    main()
