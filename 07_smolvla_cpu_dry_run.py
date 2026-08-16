#!/opt/lerobot061/bin/python
"""Offline, CPU-only SmolVLA inference check for the Atlas board.

This program deliberately imports no robot or serial modules and never opens a
camera.  It reads one recorded dataset frame and predicts an action chunk.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import get_policy_class, make_pre_post_processors


PROJECT = Path("/root/lerobot_project")
DEFAULT_MODEL = PROJECT / "models/smolvla_yellow_block_atlas_v2_clean_a100/latest"
DEFAULT_DATASET = PROJECT / "datasets/admin/smolvla_yellow_block_atlas_v2_clean"
DEFAULT_VLM_ASSETS = PROJECT / "models/HuggingFaceTB_SmolVLM2-500M-Video-Instruct_assets"
DEFAULT_OUTPUT = PROJECT / "golden_case/cpu_020000_ep000_frame000"
REQUIRED_KEYS = (
    "observation.images.front",
    "observation.images.handeye",
    "observation.state",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--vlm-assets", type=Path, default=DEFAULT_VLM_ASSETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0, help="Frame offset within the selected episode")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def tensor_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def numeric_tensors(batch: dict) -> dict[str, np.ndarray]:
    return {key: tensor_numpy(value) for key, value in batch.items() if isinstance(value, torch.Tensor)}


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.threads < 1:
        raise ValueError("--threads must be at least 1")

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.model.resolve(strict=True)
    dataset_path = args.dataset.resolve(strict=True)
    vlm_assets = args.vlm_assets.resolve(strict=True)

    print("SAFETY: offline dataset inference only; no robot, serial port, or camera is opened.", flush=True)
    print(f"model={model_path}", flush=True)
    print(f"dataset={dataset_path}", flush=True)
    print(f"device=cpu threads={args.threads} seed={args.seed}", flush=True)

    t0 = time.perf_counter()
    dataset = LeRobotDataset(
        repo_id="admin/smolvla_yellow_block_atlas_v2_clean",
        root=dataset_path,
        episodes=[args.episode],
        video_backend="pyav",
    )
    if not 0 <= args.frame < len(dataset):
        raise IndexError(f"frame offset {args.frame} is outside selected episode length {len(dataset)}")
    sample = dataset[args.frame]
    missing = [key for key in REQUIRED_KEYS if key not in sample]
    if missing:
        raise KeyError(f"missing required input keys: {missing}")
    task = str(sample["task"])
    raw_input = {
        "front": tensor_numpy(sample["observation.images.front"]),
        "handeye": tensor_numpy(sample["observation.images.handeye"]),
        "state": tensor_numpy(sample["observation.state"]),
        "task": np.asarray(task),
        "episode": np.asarray(args.episode, dtype=np.int64),
        "frame": np.asarray(args.frame, dtype=np.int64),
    }
    np.savez_compressed(args.output / "raw_input.npz", **raw_input)
    (args.output / "task.txt").write_text(task + "\n", encoding="utf-8")
    dataset_seconds = time.perf_counter() - t0
    print(f"dataset_frame_loaded_s={dataset_seconds:.3f}", flush=True)

    cfg = PreTrainedConfig.from_pretrained(model_path, local_files_only=True)
    cfg.device = "cpu"
    cfg.use_amp = False
    cfg.compile_model = False
    # Avoid loading the original 2 GB base checkpoint before immediately replacing
    # all parameters with the fine-tuned checkpoint.  Local config/tokenizer assets
    # are still used to construct the exact same architecture and text inputs.
    cfg.load_vlm_weights = False
    cfg.vlm_model_name = str(vlm_assets)
    cfg.pretrained_path = str(model_path)

    policy_class = get_policy_class(cfg.type)
    load_started = time.perf_counter()
    policy = policy_class.from_pretrained(
        model_path,
        config=cfg,
        local_files_only=True,
        strict=True,
    )
    policy.to("cpu")
    policy.eval()
    load_seconds = time.perf_counter() - load_started
    print(f"model_loaded_s={load_seconds:.3f} peak_rss_mib={rss_mib():.1f}", flush=True)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(model_path),
        preprocessor_overrides={
            "device_processor": {"device": "cpu"},
            "tokenizer_processor": {"tokenizer_name": str(vlm_assets)},
        },
    )
    observation = {key: sample[key] for key in REQUIRED_KEYS}
    observation["task"] = task
    processed = preprocessor(observation)
    np.savez_compressed(args.output / "preprocessed_inputs.npz", **numeric_tensors(processed))

    latencies = []
    repeat_max_abs_diff = []
    first_latency = None
    raw_actions = None
    actions = None
    for index in range(args.repeats + 1):
        torch.manual_seed(args.seed)
        started = time.perf_counter()
        with torch.inference_mode():
            raw = policy.predict_action_chunk(processed)
            final = postprocessor(raw)
        elapsed = time.perf_counter() - started
        if index == 0:
            first_latency = elapsed
            raw_actions = tensor_numpy(raw)
            actions = tensor_numpy(final)
            print(f"first_inference_s={elapsed:.3f} peak_rss_mib={rss_mib():.1f}", flush=True)
        else:
            latencies.append(elapsed)
            repeat_max_abs_diff.append(float(np.max(np.abs(tensor_numpy(final) - actions))))
            print(f"repeat={index}/{args.repeats} latency_s={elapsed:.3f}", flush=True)

    assert raw_actions is not None and actions is not None
    raw_actions = np.squeeze(raw_actions, axis=0)
    actions = np.squeeze(actions, axis=0)
    finite = bool(np.isfinite(actions).all())
    expected_shape = (cfg.n_action_steps, cfg.output_features["action"].shape[0])
    shape_ok = actions.shape == expected_shape
    np.save(args.output / "raw_actions_cpu.npy", raw_actions)
    np.save(args.output / "actions_postprocessed_cpu.npy", actions)

    report = {
        "status": "PASS" if finite and shape_ok else "FAIL",
        "model": str(model_path),
        "dataset": str(dataset_path),
        "episode": args.episode,
        "frame": args.frame,
        "task": task,
        "seed": args.seed,
        "device": "cpu",
        "torch_version": torch.__version__,
        "threads": args.threads,
        "dataset_frame_loaded_s": dataset_seconds,
        "model_loaded_s": load_seconds,
        "first_inference_s": first_latency,
        "steady_inference_mean_s": float(np.mean(latencies)),
        "steady_inference_min_s": float(np.min(latencies)),
        "steady_inference_max_s": float(np.max(latencies)),
        "repeat_max_abs_diff": repeat_max_abs_diff,
        "deterministic": bool(max(repeat_max_abs_diff, default=0.0) == 0.0),
        "peak_rss_mib": rss_mib(),
        "raw_action_shape": list(raw_actions.shape),
        "action_shape": list(actions.shape),
        "expected_action_shape": list(expected_shape),
        "shape_ok": shape_ok,
        "finite": finite,
        "action_min": float(actions.min()),
        "action_max": float(actions.max()),
        "per_joint_min": actions.min(axis=0).tolist(),
        "per_joint_max": actions.max(axis=0).tolist(),
    }
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

    print(
        f"action_shape={actions.shape} min={actions.min():.6f} max={actions.max():.6f} "
        f"finite={finite} shape_ok={shape_ok}",
        flush=True,
    )
    print(f"steady_mean_s={np.mean(latencies):.3f} peak_rss_mib={rss_mib():.1f}", flush=True)
    print(f"artifacts={args.output}", flush=True)
    print("CPU_DRY_RUN_PASS" if report["status"] == "PASS" else "CPU_DRY_RUN_FAIL", flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"CPU_DRY_RUN_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
