#!/opt/lerobot061/bin/python
"""Safe prototype for SO-ARM101 hand-guiding with automatic position hold.

The default ``monitor`` mode is read-only.  ``auto`` mode requires an explicit
``--enable-control`` flag and defaults to one low-risk joint (wrist_flex).
It is an automatic clutch/hold controller, not model-based torque feed-forward
gravity compensation: STS3215 exposes no commanded-current/torque mode.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

from atlas_runner import FOLLOWER_SERIAL, serial_port
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


ALL_MOTORS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("monitor", "auto", "release"), default="monitor")
    parser.add_argument("--motors", nargs="+", default=["wrist_flex"], choices=ALL_MOTORS)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--release-error-deg", type=float, default=3.0)
    parser.add_argument("--release-frames", type=int, default=2)
    parser.add_argument("--still-speed-deg-s", type=float, default=2.0)
    parser.add_argument("--still-time-s", type=float, default=0.40)
    parser.add_argument("--min-free-time-s", type=float, default=0.50)
    parser.add_argument("--settle-time-s", type=float, default=0.75)
    parser.add_argument("--max-temperature-c", type=float, default=60.0)
    parser.add_argument("--enable-control", action="store_true")
    parser.add_argument(
        "--release-on-exit",
        action="store_true",
        help="Disable all torque on exit. The arm can fall; physically support it first.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.duration_s <= 0 or args.duration_s > 300:
        raise ValueError("--duration-s must be in (0, 300]")
    if args.fps <= 0 or args.fps > 50:
        raise ValueError("--fps must be in (0, 50]")
    if args.release_error_deg < 1.0:
        raise ValueError("--release-error-deg below 1 degree is unsafe/noise-sensitive")
    if args.release_frames < 2:
        raise ValueError("--release-frames must be at least 2")
    if args.still_time_s < 0.25 or args.min_free_time_s < 0.25:
        raise ValueError("still/free times below 0.25 s are unsafe")
    if args.max_temperature_c > 65:
        raise ValueError("--max-temperature-c may not exceed 65 C")
    if args.mode in {"auto", "release"} and not args.enable_control:
        raise SystemExit(f"[SAFE REFUSAL] {args.mode} mode requires --enable-control")


def read_positions(bus, motors: list[str], *, raw: bool) -> dict[str, float]:
    return bus.sync_read("Present_Position", motors, normalize=not raw, num_retry=2)


def capture_and_hold(bus, motors: list[str]) -> dict[str, float]:
    """Set Goal=Present while torque is off, then enable; avoids startup snap."""
    bus.disable_torque(motors, num_retry=2)
    raw = read_positions(bus, motors, raw=True)
    bus.sync_write("Goal_Position", raw, normalize=False, num_retry=2)
    bus.enable_torque(motors, num_retry=2)
    return read_positions(bus, motors, raw=False)


def print_telemetry(bus, motors: list[str], state: str) -> float:
    pos = read_positions(bus, motors, raw=False)
    velocity = bus.sync_read("Present_Velocity", motors, normalize=False, num_retry=2)
    load = bus.sync_read("Present_Load", motors, normalize=False, num_retry=2)
    current = bus.sync_read("Present_Current", motors, normalize=False, num_retry=2)
    temperatures = bus.sync_read("Present_Temperature", motors, normalize=False, num_retry=2)
    fields = []
    for motor in motors:
        fields.append(
            f"{motor}:pos={pos[motor]:7.2f}deg vel={velocity[motor]:5} "
            f"load={load[motor]:5} current={current[motor]:5} temp={temperatures[motor]:2}C"
        )
    print(f"[{state}] " + " | ".join(fields), flush=True)
    return max(float(value) for value in temperatures.values())


def run_monitor(bus, motors: list[str], duration_s: float) -> None:
    print("[READ ONLY] No motor register will be written.", flush=True)
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        print_telemetry(bus, motors, "MONITOR")
        time.sleep(0.25)


def run_auto(bus, args: argparse.Namespace, stop_requested) -> None:
    selected = list(dict.fromkeys(args.motors))
    period = 1.0 / args.fps

    # Hold every joint at its measured raw position first. This avoids an arm
    # segment falling while only one selected joint enters hand-guiding mode.
    goals = capture_and_hold(bus, list(ALL_MOTORS))
    selected_goals = {motor: goals[motor] for motor in selected}
    print("[HOLD] All joints captured at their measured positions.", flush=True)
    print(f"[AUTO] Hand-guiding joints: {', '.join(selected)}", flush=True)

    state = "HOLD"
    release_count = 0
    state_since = time.monotonic()
    still_since: float | None = None
    previous = read_positions(bus, selected, raw=False)
    previous_time = time.monotonic()
    last_telemetry = 0.0
    deadline = time.monotonic() + args.duration_s

    while time.monotonic() < deadline and not stop_requested():
        loop_started = time.monotonic()
        current = read_positions(bus, selected, raw=False)
        now = time.monotonic()
        dt = max(now - previous_time, 1e-6)
        speed = {motor: abs(current[motor] - previous[motor]) / dt for motor in selected}

        if state == "HOLD":
            max_error = max(abs(current[motor] - selected_goals[motor]) for motor in selected)
            settled = now - state_since >= args.settle_time_s
            release_count = release_count + 1 if settled and max_error >= args.release_error_deg else 0
            if release_count >= args.release_frames:
                bus.disable_torque(selected, num_retry=2)
                state = "FREE"
                state_since = now
                still_since = None
                release_count = 0
                print(f"[FREE] External displacement detected ({max_error:.2f} deg). Support the arm.", flush=True)
        else:
            max_speed = max(speed.values())
            if now - state_since >= args.min_free_time_s and max_speed <= args.still_speed_deg_s:
                still_since = still_since or now
                if now - still_since >= args.still_time_s:
                    selected_goals = capture_and_hold(bus, selected)
                    state = "HOLD"
                    state_since = time.monotonic()
                    still_since = None
                    print("[HOLD] Motion stopped; captured current position without a goal jump.", flush=True)
            else:
                still_since = None

        if now - last_telemetry >= 0.5:
            max_temp = print_telemetry(bus, selected, state)
            last_telemetry = time.monotonic()
            if max_temp >= args.max_temperature_c:
                bus.disable_torque(list(ALL_MOTORS), num_retry=2)
                raise RuntimeError(
                    f"motor temperature {max_temp:.0f} C reached limit; torque disabled, support the arm"
                )

        previous = current
        previous_time = now
        time.sleep(max(0.0, period - (time.monotonic() - loop_started)))


def main() -> int:
    args = parse_args()
    validate_args(args)
    port = serial_port(FOLLOWER_SERIAL, "follower")
    robot = SO101Follower(
        SO101FollowerConfig(
            port=port,
            id="my_awesome_follower_arm",
            cameras={},
            use_degrees=True,
        )
    )
    if not robot.calibration or set(robot.calibration) != set(ALL_MOTORS):
        raise RuntimeError(f"valid six-motor calibration is required: {robot.calibration_fpath}")

    stop = False

    def request_stop(_signum=None, _frame=None):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    bus = robot.bus
    control_started = False
    faulted = False
    bus.connect()
    try:
        print(f"[INFO] Connected to follower on {port}; calibration={robot.calibration_fpath}", flush=True)
        if args.mode == "monitor":
            run_monitor(bus, list(dict.fromkeys(args.motors)), args.duration_s)
        elif args.mode == "release":
            print("[SAFETY] Physically support the whole arm; torque will turn off in 3 seconds.", flush=True)
            for seconds in (3, 2, 1):
                print(f"[RELEASE] {seconds}...", flush=True)
                time.sleep(1)
            bus.disable_torque(list(ALL_MOTORS), num_retry=2)
            print("[RELEASED] All six motor torques are disabled. Keep supporting the arm.", flush=True)
        else:
            control_started = True
            print("[SAFETY] Support the arm, clear its workspace, and keep power cutoff within reach.", flush=True)
            run_auto(bus, args, lambda: stop)
    except BaseException:
        faulted = True
        raise
    finally:
        if bus.is_connected:
            if control_started and not faulted and not args.release_on_exit:
                try:
                    capture_and_hold(bus, list(ALL_MOTORS))
                    print("[EXIT] Current pose captured; servo hold remains enabled.", flush=True)
                except Exception as exc:
                    print(f"[EXIT WARNING] Could not capture hold pose: {exc}", file=sys.stderr, flush=True)
            elif control_started:
                try:
                    bus.disable_torque(list(ALL_MOTORS), num_retry=2)
                    reason = "fault" if faulted else "--release-on-exit"
                    print(f"[EXIT] Torque disabled ({reason}); physically support the arm.", flush=True)
                except Exception as exc:
                    print(f"[EXIT WARNING] Could not disable torque: {exc}", file=sys.stderr, flush=True)
            bus.disconnect(disable_torque=False)
    print("HAND_GUIDING_TEST_FINISHED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
