#!/opt/smolvla_npu_test/bin/python
"""Run repeated, fully recorded SmolVLA evaluations on the Atlas board.

The NPU models, robot and cameras are opened once for the whole session.  Each
trial records two videos, command/feedback telemetry, every predicted action
chunk, inference timings and a human-readable result file.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import select
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT = Path("/root/lerobot_project")
LIVE_SCRIPT = PROJECT / "14_smolvla_atlas_live.py"
EVALUATIONS = PROJECT / "evaluations"
CHECKPOINT = "020000"


def load_live_module() -> Any:
    spec = importlib.util.spec_from_file_location("smolvla_atlas_live", LIVE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {LIVE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


live = load_live_module()
MOTOR_KEYS = live.MOTOR_KEYS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect repeated Atlas SmolVLA rollout evaluations")
    parser.add_argument("--motion-key", default="", help="must equal ENABLE_ATLAS_MOTION")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--duration", type=float, default=60.0, help="policy rollout seconds per trial")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--chunk-actions", type=int, default=50)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--max-relative-target", type=float, default=10.0)
    parser.add_argument("--action-filter", choices=("none", "kalman"), default="none")
    parser.add_argument("--kalman-process-noise", type=float, default=1.0)
    parser.add_argument("--kalman-measurement-noise", type=float, default=4.0)
    parser.add_argument("--return-duration", type=float, default=3.0)
    parser.add_argument("--return-tolerance", type=float, default=8.0, help="deg; gripper is included")
    parser.add_argument("--dashboard-port", type=int, default=8080)
    parser.add_argument("--dashboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evaluation-name", default="")
    parser.add_argument("--skip-labels", action="store_true", help="write null manual labels without prompting")
    return parser.parse_args()


def unique_directory(base: Path) -> Path:
    if not base.exists():
        base.mkdir(parents=True)
        return base
    for index in range(2, 1000):
        candidate = base.with_name(f"{base.name}_{index:02d}")
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
    raise RuntimeError(f"cannot allocate a unique directory beside {base}")


def json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def yes_no(prompt: str) -> bool | None:
    while True:
        try:
            value = input(prompt).strip().lower()
        except EOFError:
            return None
        if value in ("y", "yes", "1", "是"):
            return True
        if value in ("n", "no", "0", "否"):
            return False
        if value in ("", "u", "unknown", "未知"):
            return None
        print("Please enter y or n (直接回车表示未知).")


def collect_labels(policy_return_auto: bool | None, skip: bool) -> dict[str, Any]:
    if skip:
        return {
            "grasp_success": None,
            "place_inside_box": None,
            "release_success": None,
            "policy_return_success": policy_return_auto,
            "collision": None,
            "notes": "",
        }
    print("\n请根据刚才这一轮填写结果（y/n，直接回车=未知）：")
    labels = {
        "grasp_success": yes_no("  1. 成功抓起黄色积木？ [y/n] "),
        "place_inside_box": yes_no("  2. 成功放进黑盒？ [y/n] "),
        "release_success": yes_no("  3. 成功松爪释放？ [y/n] "),
        "policy_return_success": yes_no(
            f"  4. 策略自身成功回到起点？ [y/n]（姿态自动判断={policy_return_auto}） "
        ),
        "collision": yes_no("  5. 是否发生碰撞/危险动作？ [y/n] "),
    }
    try:
        labels["notes"] = input("  6. 备注（可直接回车）：").strip()
    except EOFError:
        labels["notes"] = ""
    return labels


def make_video(path: Path, frame: np.ndarray, fps: float) -> cv2.VideoWriter:
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {path}")
    return writer


def sent_pose(sent: dict[str, float] | np.ndarray) -> np.ndarray:
    if isinstance(sent, dict):
        return np.asarray([sent[key] for key in MOTOR_KEYS], dtype=np.float32)
    return np.asarray(sent, dtype=np.float32)


def poll_trial_command() -> str | None:
    """Read a terminal command without blocking the control loop (Linux board)."""
    if not sys.stdin.isatty():
        return None
    readable, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not readable:
        return None
    line = sys.stdin.readline()
    if line == "":
        return None
    return line.strip().lower()


def telemetry_fields() -> list[str]:
    fields = ["elapsed_s", "inference_index", "action_index", "inference_ms", "clamped"]
    for prefix in ("actual", "model_target", "filtered_target", "sent_target"):
        fields.extend(f"{prefix}.{key}" for key in MOTOR_KEYS)
    return fields


def add_pose(row: dict[str, Any], prefix: str, values: np.ndarray) -> None:
    for key, value in zip(MOTOR_KEYS, values, strict=True):
        row[f"{prefix}.{key}"] = f"{float(value):.6f}"


def write_summary(evaluation_dir: Path, results: list[dict[str, Any]]) -> None:
    fields = [
        "trial", "status", "full_task_success", "grasp_success", "place_inside_box",
        "release_success", "policy_return_success", "collision", "sent_actions",
        "inference_chunks", "mean_inference_ms", "action_filter", "clamped_actions", "max_clamp_delta_deg",
        "max_filter_delta_deg",
        "policy_final_max_error_deg", "recovery_final_max_error_deg", "trial_dir", "notes",
    ]
    with (evaluation_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            metrics = result.get("metrics", {})
            labels = result.get("labels", {})
            writer.writerow({
                "trial": result.get("trial"),
                "status": result.get("status"),
                "full_task_success": result.get("full_task_success"),
                "grasp_success": labels.get("grasp_success"),
                "place_inside_box": labels.get("place_inside_box"),
                "release_success": labels.get("release_success"),
                "policy_return_success": labels.get("policy_return_success"),
                "collision": labels.get("collision"),
                "sent_actions": metrics.get("sent_actions"),
                "inference_chunks": metrics.get("inference_chunks"),
                "mean_inference_ms": metrics.get("mean_inference_ms"),
                "action_filter": result.get("parameters", {}).get("action_filter"),
                "clamped_actions": metrics.get("clamped_actions"),
                "max_clamp_delta_deg": metrics.get("max_clamp_delta_deg"),
                "max_filter_delta_deg": metrics.get("max_filter_delta_deg"),
                "policy_final_max_error_deg": metrics.get("policy_final_max_error_deg"),
                "recovery_final_max_error_deg": metrics.get("recovery_final_max_error_deg"),
                "trial_dir": result.get("trial_dir"),
                "notes": labels.get("notes", ""),
            })
    successful = sum(result.get("full_task_success") is True for result in results)
    json_write(evaluation_dir / "summary.json", {
        "updated_at": iso_now(),
        "completed_trials": len(results),
        "successful_trials": successful,
        "success_rate": successful / len(results) if results else None,
        "results": results,
    })


def run_trial(
    trial_number: int,
    trial_dir: Path,
    args: argparse.Namespace,
    backend: Any,
    robot: Any,
    dashboard: Any | None,
) -> tuple[dict[str, Any], bool]:
    trial_dir.mkdir(parents=True, exist_ok=False)
    started_wall = iso_now()
    started = time.monotonic()
    front_writer: cv2.VideoWriter | None = None
    handeye_writer: cv2.VideoWriter | None = None
    telemetry_handle = None
    initial: np.ndarray | None = None
    policy_final: np.ndarray | None = None
    recovery_final: np.ndarray | None = None
    chunks: list[np.ndarray] = []
    inference_records: list[dict[str, Any]] = []
    sent_count = 0
    clamped_count = 0
    max_clamp_delta = 0.0
    max_filter_delta = 0.0
    error_text: str | None = None
    stopped = False
    ended_early = False

    try:
        observation = robot.get_observation()
        initial = live.pose(observation).copy()
        front_writer = make_video(trial_dir / "front.mp4", observation["front"], args.fps)
        handeye_writer = make_video(trial_dir / "handeye.mp4", observation["handeye"], args.fps)
        telemetry_handle = (trial_dir / "telemetry.csv").open("w", newline="", encoding="utf-8-sig")
        telemetry = csv.DictWriter(telemetry_handle, fieldnames=telemetry_fields())
        telemetry.writeheader()

        if dashboard is not None:
            dashboard.update_observation(observation)
            dashboard.set_status(f"Trial {trial_number}/{args.trials}: initial Atlas NPU inference")

        inference_started = time.monotonic() - started
        chunk, timings = backend.infer(observation, num_steps=args.num_steps)
        chunks.append(chunk.copy())
        inference_records.append({"index": 1, "generated_at_s": inference_started, **timings})
        chunk_index = 0
        inference_index = 1
        target_filter = (
            live.JointTargetKalmanFilter(args.kalman_process_noise, args.kalman_measurement_noise)
            if args.action_filter == "kalman" else None
        )
        deadline = time.monotonic() + args.duration
        print("任务提前完成时按 Enter 结束当前轮；输入 q 再按 Enter 可结束整组测试。")

        while time.monotonic() < deadline and (dashboard is None or not dashboard.stop_event.is_set()):
            tick_started = time.perf_counter()
            command = poll_trial_command()
            if command is not None:
                if command == "q":
                    stopped = True
                    print("Operator requested the end of the complete evaluation session.")
                    break
                if command in ("", "n", "next"):
                    ended_early = True
                    print("EARLY_TRIAL_COMPLETION_REQUESTED")
                    break
                print("Unknown command. Press Enter for next trial, or enter q to stop all trials.")
            observation = robot.get_observation()
            actual = live.pose(observation)
            front_writer.write(cv2.cvtColor(observation["front"], cv2.COLOR_RGB2BGR))
            handeye_writer.write(cv2.cvtColor(observation["handeye"], cv2.COLOR_RGB2BGR))
            if dashboard is not None:
                dashboard.update_observation(observation)

            usable_actions = min(args.chunk_actions, len(chunk))
            if chunk_index >= usable_actions:
                if dashboard is not None:
                    dashboard.set_status(f"Trial {trial_number}/{args.trials}: replanning on Atlas NPU")
                inference_started = time.monotonic() - started
                chunk, timings = backend.infer(observation, num_steps=args.num_steps)
                chunks.append(chunk.copy())
                inference_index += 1
                inference_records.append({"index": inference_index, "generated_at_s": inference_started, **timings})
                chunk_index = 0
                usable_actions = min(args.chunk_actions, len(chunk))

            model_target = np.asarray(chunk[chunk_index], dtype=np.float32)
            filtered_target = target_filter.update(model_target) if target_filter is not None else model_target
            max_filter_delta = max(max_filter_delta, float(np.max(np.abs(filtered_target - model_target))))
            if dashboard is not None:
                dashboard.update_target(filtered_target)
            sent = sent_pose(robot.send_action(live.action_dict(filtered_target)))
            clamp_delta = float(np.max(np.abs(sent - filtered_target)))
            was_clamped = clamp_delta > 1e-4
            clamped_count += int(was_clamped)
            max_clamp_delta = max(max_clamp_delta, clamp_delta)

            row: dict[str, Any] = {
                "elapsed_s": f"{time.monotonic() - started:.6f}",
                "inference_index": inference_index,
                "action_index": chunk_index,
                "inference_ms": f"{timings['total_ms']:.3f}",
                "clamped": int(was_clamped),
            }
            add_pose(row, "actual", actual)
            add_pose(row, "model_target", model_target)
            add_pose(row, "filtered_target", filtered_target)
            add_pose(row, "sent_target", sent)
            telemetry.writerow(row)
            telemetry_handle.flush()

            sent_count += 1
            chunk_index += 1
            if dashboard is not None:
                dashboard.set_status(
                    f"Trial {trial_number}/{args.trials}: action {sent_count}, "
                    f"chunk {chunk_index}/{usable_actions}, inference {timings['total_ms']:.0f} ms",
                    timings,
                )
            time.sleep(max(0.0, 1.0 / args.fps - (time.perf_counter() - tick_started)))

        stopped = stopped or (dashboard is not None and dashboard.stop_event.is_set())
        final_observation = robot.get_observation()
        policy_final = live.pose(final_observation).copy()
    except BaseException as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        (trial_dir / "error_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        if front_writer is not None:
            front_writer.release()
        if handeye_writer is not None:
            handeye_writer.release()
        if telemetry_handle is not None:
            telemetry_handle.flush()
            telemetry_handle.close()
        if chunks:
            np.savez_compressed(
                trial_dir / "predicted_chunks.npz",
                chunks=np.stack(chunks),
                generated_at_s=np.asarray([x["generated_at_s"] for x in inference_records], np.float64),
                inference_ms=np.asarray([x["total_ms"] for x in inference_records], np.float32),
            )

        if initial is not None and robot.is_connected:
            try:
                live.return_to_initial_position(
                    robot, initial, dashboard, duration_s=args.return_duration
                )
                recovery_final = live.pose(robot.get_observation()).copy()
            except Exception as recovery_exc:
                recovery_error = f"{type(recovery_exc).__name__}: {recovery_exc}"
                error_text = f"{error_text}; recovery failed: {recovery_error}" if error_text else f"recovery failed: {recovery_error}"

    inference_values = [float(x["total_ms"]) for x in inference_records]
    policy_error = float(np.max(np.abs(policy_final - initial))) if initial is not None and policy_final is not None else None
    recovery_error = float(np.max(np.abs(recovery_final - initial))) if initial is not None and recovery_final is not None else None
    policy_return_auto = policy_error <= args.return_tolerance if policy_error is not None else None
    labels = collect_labels(policy_return_auto, args.skip_labels) if error_text is None else {
        "grasp_success": None, "place_inside_box": None, "release_success": None,
        "policy_return_success": policy_return_auto, "collision": None, "notes": error_text,
    }
    required = (
        labels.get("grasp_success"), labels.get("place_inside_box"),
        labels.get("release_success"), labels.get("policy_return_success"),
    )
    full_success = all(value is True for value in required) and labels.get("collision") is False and error_text is None
    if any(value is None for value in required) or labels.get("collision") is None:
        full_success = None

    result = {
        "trial": trial_number,
        "trial_dir": str(trial_dir),
        "status": "error" if error_text else ("stopped" if stopped else "completed"),
        "ended_early_by_operator": ended_early,
        "started_at": started_wall,
        "finished_at": iso_now(),
        "task": live.TASK,
        "checkpoint": CHECKPOINT,
        "parameters": {
            "duration_s": args.duration, "fps": args.fps, "chunk_actions": args.chunk_actions,
            "num_steps": args.num_steps, "max_relative_target": args.max_relative_target,
            "action_filter": args.action_filter,
            "kalman_process_noise": args.kalman_process_noise,
            "kalman_measurement_noise": args.kalman_measurement_noise,
            "return_tolerance_deg": args.return_tolerance,
        },
        "poses": {
            "initial": None if initial is None else initial.tolist(),
            "policy_final_before_forced_recovery": None if policy_final is None else policy_final.tolist(),
            "recovery_final": None if recovery_final is None else recovery_final.tolist(),
        },
        "metrics": {
            "sent_actions": sent_count,
            "inference_chunks": len(inference_records),
            "mean_inference_ms": float(np.mean(inference_values)) if inference_values else None,
            "min_inference_ms": float(np.min(inference_values)) if inference_values else None,
            "max_inference_ms": float(np.max(inference_values)) if inference_values else None,
            "clamped_actions": clamped_count,
            "max_clamp_delta_deg": max_clamp_delta,
            "max_filter_delta_deg": max_filter_delta,
            "policy_final_max_error_deg": policy_error,
            "policy_return_auto": policy_return_auto,
            "recovery_final_max_error_deg": recovery_error,
        },
        "inferences": inference_records,
        "labels": labels,
        "full_task_success": full_success,
        "error": error_text,
        "files": ["front.mp4", "handeye.mp4", "telemetry.csv", "predicted_chunks.npz", "result.json"],
    }
    json_write(trial_dir / "result.json", result)
    return result, stopped or error_text is not None


def main() -> None:
    args = parse_args()
    if args.motion_key != "ENABLE_ATLAS_MOTION":
        raise SystemExit("motion refused: pass --motion-key ENABLE_ATLAS_MOTION")
    if args.trials < 1 or args.duration <= 0 or args.fps <= 0 or args.chunk_actions < 1:
        raise SystemExit("trials, duration, fps and chunk-actions must be positive")

    name = args.evaluation_name.strip() or f"smolvla_{CHECKPOINT}_{time.strftime('%Y%m%d_%H%M%S')}"
    evaluation_dir = unique_directory(EVALUATIONS / name)
    session = {
        "created_at": iso_now(), "hostname": socket.gethostname(), "task": live.TASK,
        "checkpoint": CHECKPOINT, "requested_trials": args.trials,
        "command_parameters": vars(args), "evaluation_dir": str(evaluation_dir),
    }
    json_write(evaluation_dir / "session.json", session)
    print(f"\n[EVALUATION] Results: {evaluation_dir}")
    if args.dashboard:
        print(f"[DASHBOARD] http://192.168.0.2:{args.dashboard_port}")

    dashboard = live.AtlasDashboard(args.dashboard_port, evaluation_dir / "dashboard") if args.dashboard else None
    backend = None
    robot = None
    results: list[dict[str, Any]] = []
    try:
        if dashboard is not None:
            dashboard.start()
            dashboard.set_status("Loading SmolVLA models once for the 10-trial evaluation...")
        backend = live.AtlasSmolVLANPU()
        robot = live.make_robot(args.max_relative_target)
        robot.connect()
        for trial_number in range(1, args.trials + 1):
            print(f"\n{'=' * 68}\nTRIAL {trial_number}/{args.trials}\n{'=' * 68}")
            try:
                ready = input("摆好黄色积木和黑盒、确认工作区无人后按 Enter；输入 q 结束：").strip().lower()
            except EOFError:
                ready = ""
            if ready == "q":
                print("Evaluation stopped by operator before the next trial.")
                break
            result, must_stop = run_trial(
                trial_number, evaluation_dir / f"trial_{trial_number:03d}",
                args, backend, robot, dashboard,
            )
            results.append(result)
            write_summary(evaluation_dir, results)
            print(
                f"TRIAL_{trial_number:03d}_{result['status'].upper()} "
                f"full_success={result['full_task_success']} dir={result['trial_dir']}"
            )
            if must_stop:
                print("Stopping the session because safe-stop was requested or a hardware/runtime error occurred.")
                break
    except KeyboardInterrupt:
        print("\nCtrl-C received; returning and disconnecting safely.")
    finally:
        if robot is not None and robot.is_connected:
            robot.disconnect()
        if backend is not None:
            backend.close()
        if dashboard is not None:
            dashboard.close()
        write_summary(evaluation_dir, results)
        print(f"\nEVALUATION_FINISHED completed={len(results)}/{args.trials} results={evaluation_dir}")


if __name__ == "__main__":
    main()
