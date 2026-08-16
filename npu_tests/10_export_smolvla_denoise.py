#!/opt/smolvla_npu_test/bin/python
"""Export one SmolVLA action-expert denoising step with prefix KV inputs."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import get_policy_class
from lerobot.policies.common.vla_utils import make_att_2d_masks


PROJECT = Path("/root/lerobot_project")
MODEL = (PROJECT / "models/smolvla_yellow_block_atlas_v2_clean_a100/latest").resolve()
ASSETS = PROJECT / "models/HuggingFaceTB_SmolVLM2-500M-Video-Instruct_assets"
PREFIX = PROJECT / "npu_tests/prefix"
OUT = PROJECT / "npu_tests/denoise"


class DenoiseStep(nn.Module):
    def __init__(self, flow_model: nn.Module, prefix_pad_mask: torch.Tensor):
        super().__init__()
        self.action_in_proj = flow_model.action_in_proj
        self.time_mlp_in = flow_model.action_time_mlp_in
        self.time_mlp_out = flow_model.action_time_mlp_out
        self.layers = flow_model.vlm_with_expert.lm_expert.layers
        self.final_norm = flow_model.vlm_with_expert.lm_expert.norm
        self.action_out_proj = flow_model.action_out_proj
        self.num_heads = 15
        self.num_kv_heads = 5
        self.groups = 3
        self.head_dim = 64
        self.prefix_len = prefix_pad_mask.shape[1]
        self.suffix_len = 50

        suffix_pad = torch.ones((1, self.suffix_len), dtype=torch.bool)
        suffix_blocks = torch.ones((1, self.suffix_len), dtype=torch.bool)
        suffix_mask = make_att_2d_masks(suffix_pad, suffix_blocks)
        prefix_to_suffix = prefix_pad_mask[:, None, :].expand(1, self.suffix_len, self.prefix_len)
        self.register_buffer("full_mask", torch.cat([prefix_to_suffix, suffix_mask], dim=2))
        self.register_buffer("cross_mask", prefix_to_suffix)

        prefix_count = prefix_pad_mask.sum(dim=1, keepdim=True)
        absolute_positions = prefix_count + torch.arange(self.suffix_len)[None, :]
        relative_positions = torch.arange(self.suffix_len)[None, :]
        self._register_rope("absolute", absolute_positions)
        self._register_rope("relative", relative_positions)

        fraction = torch.linspace(0.0, 1.0, 720 // 2, dtype=torch.float64)
        period = 0.004 * (4.0 / 0.004) ** fraction
        self.register_buffer("time_scale", (2 * math.pi / period).float())

    def _register_rope(self, name: str, positions: torch.Tensor) -> None:
        exponents = (2.0 / self.head_dim) * torch.arange(self.head_dim // 2, dtype=torch.float32)
        timescale = 10_000**exponents
        radians = positions[..., None].float() / timescale[None, None, :]
        self.register_buffer(f"{name}_sin", torch.sin(radians)[:, :, None, :])
        self.register_buffer(f"{name}_cos", torch.cos(radians)[:, :, None, :])

    def rope(self, value: torch.Tensor, name: str) -> torch.Tensor:
        left, right = value.float().split(self.head_dim // 2, dim=-1)
        sin = getattr(self, f"{name}_sin")
        cos = getattr(self, f"{name}_cos")
        return torch.cat([left * cos - right * sin, right * cos + left * sin], dim=-1)

    def attend(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch, key_len = key.shape[:2]
        key = key[:, :, :, None, :].expand(
            batch, key_len, self.num_kv_heads, self.groups, self.head_dim
        ).reshape(batch, key_len, self.num_heads, self.head_dim)
        value = value[:, :, :, None, :].expand(
            batch, key_len, self.num_kv_heads, self.groups, self.head_dim
        ).reshape(batch, key_len, self.num_heads, self.head_dim)
        weights = torch.matmul(query.transpose(1, 2), key.transpose(1, 2).transpose(2, 3))
        weights = weights * (self.head_dim**-0.5)
        weights = torch.where(mask[:, None, :, :], weights.float(), torch.finfo(torch.float32).min)
        probs = torch.softmax(weights, dim=-1)
        output = torch.matmul(probs, value.permute(0, 2, 1, 3))
        return output.permute(0, 2, 1, 3).reshape(batch, self.suffix_len, -1)

    def forward(self, x_t: torch.Tensor, timestep: torch.Tensor, *kv: torch.Tensor) -> torch.Tensor:
        action = self.action_in_proj(x_t)
        angle = timestep[:, None] * self.time_scale[None, :]
        time_emb = torch.cat([torch.sin(angle), torch.cos(angle)], dim=1)
        time_emb = time_emb[:, None, :].expand_as(action)
        hidden = self.time_mlp_out(F.silu(self.time_mlp_in(torch.cat([action, time_emb], dim=2))))

        for index, layer in enumerate(self.layers):
            residual = hidden
            normalized = layer.input_layernorm(hidden)
            query = layer.self_attn.q_proj(normalized).view(1, self.suffix_len, -1, self.head_dim)
            prefix_key = kv[index * 2].transpose(1, 2)
            prefix_value = kv[index * 2 + 1].transpose(1, 2)
            if index % 2 == 0:
                query = self.rope(query, "absolute")
                suffix_key = self.rope(
                    layer.self_attn.k_proj(normalized).view(1, self.suffix_len, -1, self.head_dim),
                    "absolute",
                )
                suffix_value = layer.self_attn.v_proj(normalized).view(
                    1, self.suffix_len, -1, self.head_dim
                )
                key = torch.cat([prefix_key, suffix_key], dim=1)
                value = torch.cat([prefix_value, suffix_value], dim=1)
                attended = self.attend(query, key, value, self.full_mask)
            else:
                query = self.rope(query, "relative")
                key = layer.self_attn.k_proj(prefix_key.reshape(1, self.prefix_len, -1)).view(
                    1, self.prefix_len, -1, self.head_dim
                )
                value = layer.self_attn.v_proj(prefix_value.reshape(1, self.prefix_len, -1)).view(
                    1, self.prefix_len, -1, self.head_dim
                )
                attended = self.attend(query, key, value, self.cross_mask)
            hidden = layer.self_attn.o_proj(attended) + residual
            after_attention = hidden.clone()
            hidden = layer.mlp(layer.post_attention_layernorm(hidden)) + after_attention

        hidden = self.final_norm(hidden).float()
        return self.action_out_proj(hidden)


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
    prefix_pad = torch.from_numpy(np.load(PREFIX / "pad_mask.npy")).bool()
    wrapper = DenoiseStep(policy.model, prefix_pad).float().cpu().eval()
    with np.load(PREFIX / "output_prefix_origin_npu.npz") as archive:
        kv = tuple(torch.from_numpy(archive[f"output_{i:02d}"]).float() for i in range(32))
    generator = torch.Generator(device="cpu").manual_seed(1000)
    x_t = torch.randn((1, 50, 32), generator=generator, dtype=torch.float32)
    timestep = torch.ones((1,), dtype=torch.float32)
    with torch.inference_mode():
        expected = wrapper(x_t, timestep, *kv)
    np.save(OUT / "input_xt.npy", x_t.numpy())
    np.save(OUT / "input_timestep.npy", timestep.numpy())
    np.save(OUT / "expected_cpu.npy", expected.numpy())
    names = ["x_t", "timestep"] + [f"kv_{i:02d}" for i in range(32)]
    export_started = time.perf_counter()
    torch.onnx.export(
        wrapper,
        (x_t, timestep, *kv),
        OUT / "denoise.onnx",
        input_names=names,
        output_names=["velocity"],
        opset_version=11,
        dynamo=False,
        do_constant_folding=True,
    )
    print(
        f"SMOLVLA_DENOISE_EXPORTED load_s={load_seconds:.3f} "
        f"export_s={time.perf_counter() - export_started:.3f} output={tuple(expected.shape)}"
    )


if __name__ == "__main__":
    main()
