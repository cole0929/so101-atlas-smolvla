#!/opt/smolvla_npu_test/bin/python
"""Export the trained SmolVLA vision encoder and connector on the Atlas board."""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import get_policy_class


PROJECT = Path("/root/lerobot_project")
MODEL = (PROJECT / "models/smolvla_yellow_block_atlas_v2_clean_a100/latest").resolve()
ASSETS = PROJECT / "models/HuggingFaceTB_SmolVLM2-500M-Video-Instruct_assets"
CPU_CASE = PROJECT / "golden_case/cpu_020000_ep000_frame000/preprocessed_inputs.npz"
OUT = PROJECT / "npu_tests/vision"


class VisionWithConnector(nn.Module):
    def __init__(self, vision_model: nn.Module, connector: nn.Module):
        super().__init__()
        self.patch_embedding = vision_model.embeddings.patch_embedding
        self.position_embedding = vision_model.embeddings.position_embedding
        self.encoder = vision_model.encoder
        self.post_layernorm = vision_model.post_layernorm
        self.connector = connector
        self.encoder.config._attn_implementation = "eager"
        for layer in self.encoder.layers:
            layer.self_attn.config._attn_implementation = "eager"
        self.register_buffer("position_ids", torch.arange(1024, dtype=torch.long).unsqueeze(0))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # The deployment input is fixed at 512x512 with 16x16 patches and no
        # padding.  Static position ids and full eager attention avoid the
        # dynamic Transformers mask builder, which is not ONNX traceable.
        hidden = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        hidden = hidden + self.position_embedding(self.position_ids)
        hidden = self.encoder(inputs_embeds=hidden, attention_mask=None).last_hidden_state
        hidden = self.post_layernorm(hidden)
        return self.connector(hidden)


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
    ).eval()
    load_seconds = time.perf_counter() - started

    with np.load(CPU_CASE) as case:
        batch = {
            "observation.images.front": torch.from_numpy(case["observation.images.front"]),
            "observation.images.handeye": torch.from_numpy(case["observation.images.handeye"]),
        }
    prepared, _ = policy.prepare_images(batch)
    front, handeye = (tensor.float().cpu().contiguous() for tensor in prepared)
    vlm_model = policy.model.vlm_with_expert.get_vlm_model()
    wrapper = VisionWithConnector(vlm_model.vision_model, vlm_model.connector).float().cpu().eval()
    with torch.inference_mode():
        expected_front = wrapper(front).numpy()
        expected_handeye = wrapper(handeye).numpy()

    np.save(OUT / "input_front.npy", front.numpy())
    np.save(OUT / "input_handeye.npy", handeye.numpy())
    np.save(OUT / "expected_front_cpu.npy", expected_front)
    np.save(OUT / "expected_handeye_cpu.npy", expected_handeye)
    export_started = time.perf_counter()
    torch.onnx.export(
        wrapper,
        (front,),
        OUT / "vision.onnx",
        input_names=["pixel_values"],
        output_names=["image_embeddings"],
        opset_version=11,
        dynamo=False,
        do_constant_folding=True,
    )
    export_seconds = time.perf_counter() - export_started
    print(
        f"SMOLVLA_VISION_EXPORTED load_s={load_seconds:.3f} export_s={export_seconds:.3f} "
        f"input={tuple(front.shape)} output={expected_front.shape} "
        f"range=[{expected_front.min():.6f},{expected_front.max():.6f}]"
    )


if __name__ == "__main__":
    main()
