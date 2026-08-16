#!/opt/smolvla_npu_test/bin/python
"""Export the 16-layer SmolVLA language prefix prefill to ONNX."""

from __future__ import annotations

import math
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
from lerobot.policies.common.vla_utils import make_att_2d_masks


PROJECT = Path("/root/lerobot_project")
MODEL = (PROJECT / "models/smolvla_yellow_block_atlas_v2_clean_a100/latest").resolve()
ASSETS = PROJECT / "models/HuggingFaceTB_SmolVLM2-500M-Video-Instruct_assets"
CPU_CASE = PROJECT / "golden_case/cpu_020000_ep000_frame000/preprocessed_inputs.npz"
VISION = PROJECT / "npu_tests/vision"
STATE = PROJECT / "npu_tests/state_proj"
OUT = PROJECT / "npu_tests/prefix"


class PrefixPrefill(nn.Module):
    def __init__(self, text_model: nn.Module, pad_mask: torch.Tensor, attention_mask: torch.Tensor):
        super().__init__()
        self.layers = text_model.layers
        self.final_norm = text_model.norm
        self.num_heads = 15
        self.num_kv_heads = 5
        self.head_dim = 64
        self.groups = self.num_heads // self.num_kv_heads
        positions = torch.cumsum(pad_mask, dim=1) - 1
        d_half = self.head_dim // 2
        exponents = (2.0 / self.head_dim) * torch.arange(d_half, dtype=torch.float32)
        timescale = 10_000**exponents
        radians = positions[..., None].float() / timescale[None, None, :]
        self.register_buffer("rope_sin", torch.sin(radians)[:, :, None, :])
        self.register_buffer("rope_cos", torch.cos(radians)[:, :, None, :])
        self.register_buffer("attention_mask", attention_mask)

    def rope(self, value: torch.Tensor) -> torch.Tensor:
        left, right = value.float().split(self.head_dim // 2, dim=-1)
        return torch.cat(
            [left * self.rope_cos - right * self.rope_sin, right * self.rope_cos + left * self.rope_sin],
            dim=-1,
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        caches = []
        for layer in self.layers:
            residual = hidden_states
            normalized = layer.input_layernorm(hidden_states)
            shape = (*normalized.shape[:-1], -1, self.head_dim)
            query = self.rope(layer.self_attn.q_proj(normalized).view(shape))
            key = self.rope(layer.self_attn.k_proj(normalized).view(*normalized.shape[:-1], -1, self.head_dim))
            value = layer.self_attn.v_proj(normalized).view(*normalized.shape[:-1], -1, self.head_dim)
            caches.extend((key.transpose(1, 2), value.transpose(1, 2)))
            batch, length = key.shape[:2]
            key_rep = key[:, :, :, None, :].expand(
                batch, length, self.num_kv_heads, self.groups, self.head_dim
            ).reshape(batch, length, self.num_heads, self.head_dim)
            value_rep = value[:, :, :, None, :].expand(
                batch, length, self.num_kv_heads, self.groups, self.head_dim
            ).reshape(batch, length, self.num_heads, self.head_dim)
            weights = torch.matmul(
                query.transpose(1, 2), key_rep.transpose(1, 2).transpose(2, 3)
            ) * (self.head_dim**-0.5)
            weights = torch.where(
                self.attention_mask[:, None, :, :], weights, torch.finfo(torch.float32).min
            )
            probs = torch.softmax(weights.float(), dim=-1)
            attended = torch.matmul(probs, value_rep.permute(0, 2, 1, 3))
            attended = attended.permute(0, 2, 1, 3).reshape(batch, length, -1)
            hidden_states = layer.self_attn.o_proj(attended) + residual
            after_attention = hidden_states.clone()
            hidden_states = layer.mlp(layer.post_attention_layernorm(hidden_states)) + after_attention
        # The normalized prefix output is not consumed by denoising; evaluate it
        # to keep this wrapper identical through the final layer, but caches are outputs.
        _ = self.final_norm(hidden_states)
        return tuple(caches)


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
        tokens = torch.from_numpy(case["observation.language.tokens"]).long()
        language_mask = torch.from_numpy(case["observation.language.attention_mask"]).bool()
    text_model = policy.model.vlm_with_expert.get_vlm_model().text_model.float().cpu().eval()
    with torch.inference_mode():
        language = text_model.get_input_embeddings()(tokens).float() * math.sqrt(960)
    front = torch.from_numpy(np.load(VISION / "output_front_npu.npy")).float() * math.sqrt(960)
    handeye = torch.from_numpy(np.load(VISION / "output_handeye_npu.npy")).float() * math.sqrt(960)
    state = torch.from_numpy(np.load(STATE / "output_npu.npy")).float()[:, None, :]
    prefix = torch.cat([front, handeye, language, state], dim=1).contiguous()
    image_mask = torch.ones((1, front.shape[1] + handeye.shape[1]), dtype=torch.bool)
    state_mask = torch.ones((1, 1), dtype=torch.bool)
    pad_mask = torch.cat([image_mask, language_mask, state_mask], dim=1)
    block_mask = torch.cat(
        [torch.zeros((1, prefix.shape[1] - 1), dtype=torch.bool), torch.ones((1, 1), dtype=torch.bool)],
        dim=1,
    )
    attention_mask = make_att_2d_masks(pad_mask, block_mask)
    wrapper = PrefixPrefill(text_model, pad_mask, attention_mask).float().cpu().eval()
    with torch.inference_mode():
        expected = wrapper(prefix)
    np.save(OUT / "input.npy", prefix.numpy())
    np.save(OUT / "pad_mask.npy", pad_mask.numpy())
    np.savez_compressed(OUT / "expected_cpu.npz", **{f"output_{i:02d}": x.numpy() for i, x in enumerate(expected)})
    names = [f"kv_{i:02d}" for i in range(len(expected))]
    export_started = time.perf_counter()
    torch.onnx.export(
        wrapper,
        (prefix,),
        OUT / "prefix.onnx",
        input_names=["prefix_embeddings"],
        output_names=names,
        opset_version=11,
        dynamo=False,
        do_constant_folding=True,
    )
    print(
        f"SMOLVLA_PREFIX_EXPORTED load_s={load_seconds:.3f} "
        f"export_s={time.perf_counter() - export_started:.3f} input={tuple(prefix.shape)} outputs={len(expected)}"
    )


if __name__ == "__main__":
    main()
