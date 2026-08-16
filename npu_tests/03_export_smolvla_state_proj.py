#!/opt/smolvla_npu_test/bin/python
"""Export the trained SmolVLA state projection as the first real ONNX subgraph."""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import get_policy_class


PROJECT = Path("/root/lerobot_project")
MODEL = (PROJECT / "models/smolvla_yellow_block_atlas_v2_clean_a100/latest").resolve()
ASSETS = PROJECT / "models/HuggingFaceTB_SmolVLM2-500M-Video-Instruct_assets"
CPU_CASE = PROJECT / "golden_case/cpu_020000_ep000_frame000/preprocessed_inputs.npz"
OUT = PROJECT / "npu_tests/state_proj"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cfg = PreTrainedConfig.from_pretrained(MODEL, local_files_only=True)
    cfg.device = "cpu"
    cfg.use_amp = False
    cfg.compile_model = False
    cfg.load_vlm_weights = False
    cfg.vlm_model_name = str(ASSETS)
    cfg.pretrained_path = str(MODEL)

    started = time.perf_counter()
    policy = get_policy_class(cfg.type).from_pretrained(
        MODEL, config=cfg, local_files_only=True, strict=True
    )
    load_seconds = time.perf_counter() - started

    with np.load(CPU_CASE) as case:
        state6 = case["observation.state"].astype(np.float32)
    state32 = np.zeros((state6.shape[0], cfg.max_state_dim), dtype=np.float32)
    state32[:, : state6.shape[1]] = state6
    state_tensor = torch.from_numpy(state32)
    layer = policy.model.state_proj.float().cpu().eval()
    with torch.inference_mode():
        expected = layer(state_tensor).numpy()

    np.save(OUT / "input.npy", state32)
    np.save(OUT / "expected_cpu.npy", expected)
    torch.onnx.export(
        layer,
        (state_tensor,),
        OUT / "state_proj.onnx",
        input_names=["state"],
        output_names=["state_embedding"],
        opset_version=11,
        dynamo=False,
    )
    print(
        f"SMOLVLA_STATE_PROJ_EXPORTED load_s={load_seconds:.3f} "
        f"input={state32.shape} output={expected.shape} "
        f"range=[{expected.min():.6f},{expected.max():.6f}]"
    )


if __name__ == "__main__":
    main()
