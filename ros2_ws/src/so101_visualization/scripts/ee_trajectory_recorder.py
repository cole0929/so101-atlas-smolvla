#!/usr/bin/env python3
"""SO-ARM101 end-effector trajectory recorder, visualizer and analyzer.

Subscribes to joint states, resolves the end-effector pose through TF, and
publishes:

  /ee_path   (nav_msgs/Path)          full end-effector path
  /ee_trail  (visualization_msgs/MarkerArray) color-coded trail (one strip per
             detected circle), current tip sphere, live stats text
  /ee_stats  (std_msgs/String)        JSON statistics: arc length, speed,
             acceleration, curvature smoothness, circle count, per-circle
             radius and repeat error vs. the first circle

Exposes a service /ee_trail/reset (std_srvs/Empty) to clear the recording.

Circle detection is heuristic: a new circle is closed when the tip returns
within ``circle-close-tol`` of the current circle start after travelling at
least ``circle-min-arc`` meters.  Every closed circle is resampled to
``align-samples`` points of equal arc length and compared against the first
circle to compute the repeat error (RMS point distance).

Optional CSV logging with ``--out-file``.

Run inside the WSL ROS2 environment together with
``so101_description/display.launch.py`` (robot_state_publisher must be up):

  ros2 run so101_visualization ee_trajectory_recorder.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import deque
from typing import List, Optional, Tuple

import rclpy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Empty
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


def hsv_to_rgb(h: float, s: float = 0.9, v: float = 0.95) -> Tuple[float, float, float]:
    """Convert HSV (h in [0,1]) to RGB floats in [0,1]."""
    i = int(h * 6.0) % 6
    f = h * 6.0 - math.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    return {
        0: (v, t, p),
        1: (q, v, p),
        2: (p, v, t),
        3: (p, q, v),
        4: (t, p, v),
        5: (v, p, q),
    }[i]


class EETrajectoryRecorder(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("ee_trajectory_recorder")
        self.args = args

        # TF buffer so we can resolve the tip pose from joint states.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Recent samples: (stamp_ns, x, y, z)
        self.samples: List[Tuple[int, float, float, float]] = []
        # Closed circles: list of point lists (each point is (x, y, z))
        self.circles: List[List[Tuple[float, float, float]]] = []
        # Current circle start index into self.samples
        self.circle_start_index = 0
        self.last_sample_ns: Optional[int] = None

        self.stats_pub = self.create_publisher(String, "/ee_stats", 10)
        self.path_pub = self.create_publisher(Path, "/ee_path", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/ee_trail", 10)
        self.sub = self.create_subscription(
            JointState, args.topic, self.joint_state_cb, 10
        )
        self.reset_srv = self.create_service(Empty, "/ee_trail/reset", self.reset_cb)

        self.timer = self.create_timer(1.0 / args.publish_rate, self.publish_loop)
        self._csv_handle = None
        self._csv_writer = None
        if args.out_file:
            self._csv_handle = open(args.out_file, "w", newline="")
            self._csv_writer = csv.writer(self._csv_handle)
            self._csv_writer.writerow(
                ["t_s", "x_m", "y_m", "z_m", "vx", "vy", "vz", "speed", "accel", "curvature"]
            )

    # ------------------------------------------------------------------ #
    # sampling
    # ------------------------------------------------------------------ #
    def joint_state_cb(self, msg: JointState) -> None:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.args.frame_id,
                self.args.target_frame,
                Time(),
                timeout=Duration(seconds=0.05),
            )
        except Exception:
            return  # TF not ready yet; skip this sample.

        t = transform.transform.translation
        now = self.get_clock().now().nanoseconds
        if self.last_sample_ns is not None:
            dt_s = (now - self.last_sample_ns) * 1e-9
            if dt_s < self.args.min_sample_dt:
                return  # throttle sampling
        self.last_sample_ns = now
        self.samples.append((now, t.x, t.y, t.z))

    def reset_cb(self, request, response) -> Empty.Response:  # noqa: N803
        self.samples.clear()
        self.circles.clear()
        self.circle_start_index = 0
        self.last_sample_ns = None
        self.get_logger().info("trajectory cleared")
        return response

    # ------------------------------------------------------------------ #
    # analysis
    # ------------------------------------------------------------------ #
    def _positions(self, pts: List[Tuple[int, float, float, float]]) -> List[Tuple[float, float, float]]:
        return [(p[1], p[2], p[3]) for p in pts]

    def _velocities(self, pts: List[Tuple[int, float, float, float]]) -> List[Tuple[float, float, float]]:
        vel = []
        for i in range(1, len(pts)):
            dt = (pts[i][0] - pts[i - 1][0]) * 1e-9
            if dt <= 0:
                continue
            vel.append(
                (
                    (pts[i][1] - pts[i - 1][1]) / dt,
                    (pts[i][2] - pts[i - 1][2]) / dt,
                    (pts[i][3] - pts[i - 1][3]) / dt,
                )
            )
        return vel

    @staticmethod
    def _smooth(vecs: List[Tuple[float, float, float]], window: int) -> List[Tuple[float, float, float]]:
        if not vecs:
            return []
        out = []
        n = len(vecs)
        for i in range(n):
            lo = max(0, i - window // 2)
            hi = min(n, i + window // 2 + 1)
            seg = vecs[lo:hi]
            out.append(tuple(sum(v[k] for v in seg) / len(seg) for k in range(3)))  # type: ignore[index]
        return out

    @staticmethod
    def _norm(v: Tuple[float, float, float]) -> float:
        return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

    @staticmethod
    def _cross(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    def _arc_length(self, pts: List[Tuple[float, float, float]]) -> float:
        total = 0.0
        for i in range(1, len(pts)):
            total += math.dist(pts[i - 1], pts[i])
        return total

    def _analyze(self) -> dict:
        pts = self.samples
        stats: dict = {
            "points": len(pts),
            "circles_closed": len(self.circles),
            "circle_radius_m": [],
            "circle_error_rms_m": [],
            "circle_error_max_m": [],
        }
        if len(pts) < 3:
            stats["arc_length_m"] = 0.0
            stats["speed_mean_mps"] = 0.0
            stats["speed_max_mps"] = 0.0
            stats["accel_mean_mps2"] = 0.0
            stats["accel_max_mps2"] = 0.0
            stats["curvature_mean"] = 0.0
            stats["curvature_max"] = 0.0
            stats["duration_s"] = 0.0
            return stats

        stats["duration_s"] = (pts[-1][0] - pts[0][0]) * 1e-9
        stats["arc_length_m"] = self._arc_length(self._positions(pts))

        vel = self._smooth(self._velocities(pts), self.args.speed_window)
        speeds = [self._norm(v) for v in vel]
        stats["speed_mean_mps"] = sum(speeds) / len(speeds) if speeds else 0.0
        stats["speed_max_mps"] = max(speeds) if speeds else 0.0

        acc = []
        for i in range(1, len(vel)):
            dt = (pts[i + 1][0] - pts[i][0]) * 1e-9
            if dt <= 0:
                continue
            acc.append(
                (
                    (vel[i][0] - vel[i - 1][0]) / dt,
                    (vel[i][1] - vel[i - 1][1]) / dt,
                    (vel[i][2] - vel[i - 1][2]) / dt,
                )
            )
        acc_norms = [self._norm(a) for a in acc]
        stats["accel_mean_mps2"] = sum(acc_norms) / len(acc_norms) if acc_norms else 0.0
        stats["accel_max_mps2"] = max(acc_norms) if acc_norms else 0.0

        # Curvature kappa = |v x a| / |v|^3 ; smoothness = small curvature.
        curv = []
        for i in range(1, min(len(vel), len(acc) + 1)):
            v = vel[i]
            a = acc[i - 1]
            vn = self._norm(v)
            if vn < 1e-6:
                continue
            curv.append(self._norm(self._cross(v, a)) / (vn ** 3))
        stats["curvature_mean"] = sum(curv) / len(curv) if curv else 0.0
        stats["curvature_max"] = max(curv) if curv else 0.0

        # Per-circle radius and repeat error vs the first closed circle.
        if self.circles:
            first = self._resample(self.circles[0], self.args.align_samples)
            for circle in self.circles:
                center = (
                    sum(p[0] for p in circle) / len(circle),
                    sum(p[1] for p in circle) / len(circle),
                    sum(p[2] for p in circle) / len(circle),
                )
                radii = [math.dist(p, center) for p in circle]
                stats["circle_radius_m"].append(round(sum(radii) / len(radii), 4))
                rs = self._resample(circle, self.args.align_samples)
                errs = [math.dist(rs[i], first[i]) for i in range(len(first))]
                stats["circle_error_rms_m"].append(round(math.sqrt(sum(e * e for e in errs) / len(errs)), 4))
                stats["circle_error_max_m"].append(round(max(errs), 4))
        return stats

    @staticmethod
    def _resample(pts: List[Tuple[float, float, float]], n: int) -> List[Tuple[float, float, float]]:
        """Resample a polyline to n points of equal arc length (closed loop assumed)."""
        if len(pts) < 2:
            return list(pts)
        seg_lens = [math.dist(pts[i - 1], pts[i]) for i in range(1, len(pts))]
        total = sum(seg_lens)
        if total <= 0:
            return [pts[0]] * n
        out = []
        target = 0.0
        step = total / n
        acc = 0.0
        idx = 0
        out.append(pts[0])
        for _ in range(1, n):
            target += step
            while idx < len(seg_lens) - 1 and acc + seg_lens[idx] < target:
                acc += seg_lens[idx]
                idx += 1
            if idx >= len(seg_lens):
                out.append(pts[-1])
                continue
            seg_len = seg_lens[idx]
            frac = (target - acc) / seg_len if seg_len > 0 else 0.0
            p0, p1 = pts[idx], pts[idx + 1]
            out.append(
                (
                    p0[0] + (p1[0] - p0[0]) * frac,
                    p0[1] + (p1[1] - p0[1]) * frac,
                    p0[2] + (p1[2] - p0[2]) * frac,
                )
            )
        return out

    # ------------------------------------------------------------------ #
    # circle closing
    # ------------------------------------------------------------------ #
    def _update_circles(self) -> None:
        """Close a circle when the tip returns near the circle start."""
        if len(self.samples) < 10:
            return
        start = self.samples[self.circle_start_index]
        tip = self.samples[-1]
        arc = self._arc_length(self._positions(self.samples[self.circle_start_index:]))
        if arc < self.args.circle_min_arc:
            return
        dist_to_start = math.dist((start[1], start[2], start[3]), (tip[1], tip[2], tip[3]))
        if dist_to_start > self.args.circle_close_tol:
            return
        closed = [(p[1], p[2], p[3]) for p in self.samples[self.circle_start_index:]]
        self.circles.append(closed)
        self.circle_start_index = len(self.samples) - 1
        self.get_logger().info(
            f"circle closed #{len(self.circles)}  arc={arc:.3f} m  close_dist={dist_to_start*1000:.1f} mm"
        )

    # ------------------------------------------------------------------ #
    # publishing
    # ------------------------------------------------------------------ #
    def publish_loop(self) -> None:
        self._update_circles()
        self._publish_path()
        self._publish_markers()
        self._publish_stats()
        self._write_csv()

    def _publish_path(self) -> None:
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.args.frame_id
        for sample in self.samples:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = sample[1]
            pose.pose.position.y = sample[2]
            pose.pose.position.z = sample[3]
            path.poses.append(pose)
        self.path_pub.publish(path)

    def _publish_markers(self) -> None:
        arr = MarkerArray()
        # One LINE_STRIP per closed circle, hue rotates with circle index.
        for ci, circle in enumerate(self.circles):
            m = Marker()
            m.header.frame_id = self.args.frame_id
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "circle"
            m.id = ci
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.006
            r, g, b = hsv_to_rgb((ci % 12) / 12.0)
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 0.95
            m.pose.orientation.w = 1.0
            for p in circle:
                pt = Point()
                pt.x, pt.y, pt.z = p
                m.points.append(pt)
            arr.markers.append(m)

        # Current unfinished trail in white.
        if len(self.samples) > 1:
            m = Marker()
            m.header.frame_id = self.args.frame_id
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "current"
            m.id = 0
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.004
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 1.0, 1.0, 0.6
            m.pose.orientation.w = 1.0
            for sample in self.samples:
                pt = Point()
                pt.x, pt.y, pt.z = sample[1], sample[2], sample[3]
                m.points.append(pt)
            arr.markers.append(m)

        # Tip sphere.
        if self.samples:
            tip = self.samples[-1]
            m = Marker()
            m.header.frame_id = self.args.frame_id
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "tip"
            m.id = 0
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.scale.x = m.scale.y = m.scale.z = 0.02
            m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.9, 0.3, 1.0
            m.pose.position.x, m.pose.position.y, m.pose.position.z = tip[1], tip[2], tip[3]
            m.pose.orientation.w = 1.0
            arr.markers.append(m)

            # Live stats text above the tip.
            stats = self._analyze()
            m = Marker()
            m.header.frame_id = self.args.frame_id
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "stats"
            m.id = 0
            m.type = Marker.TEXT_VIEW_FACING
            m.action = Marker.ADD
            m.scale.z = 0.025
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 1.0, 1.0, 1.0
            m.pose.position.x, m.pose.position.y, m.pose.position.z = tip[1], tip[2], tip[3] + 0.05
            m.pose.orientation.w = 1.0
            m.text = (
                f"L={stats['arc_length_m']:.2f}m  "
                f"v={stats['speed_mean_mps']:.2f}m/s  "
                f"circles={len(self.circles)}"
            )
            arr.markers.append(m)

        self.marker_pub.publish(arr)

    def _publish_stats(self) -> None:
        stats = self._analyze()
        msg = String()
        msg.data = json.dumps(stats, ensure_ascii=False)
        self.stats_pub.publish(msg)

    def _write_csv(self) -> None:
        if self._csv_writer is None or len(self.samples) < 2:
            return
        vel = self._smooth(self._velocities(self.samples), self.args.speed_window)
        acc = []
        for i in range(1, len(vel)):
            dt = (self.samples[i + 1][0] - self.samples[i][0]) * 1e-9
            if dt > 0:
                acc.append(
                    (
                        (vel[i][0] - vel[i - 1][0]) / dt,
                        (vel[i][1] - vel[i - 1][1]) / dt,
                        (vel[i][2] - vel[i - 1][2]) / dt,
                    )
                )
        curv = []
        for i in range(1, min(len(vel), len(acc) + 1)):
            v = vel[i]
            vn = self._norm(v)
            if vn > 1e-6:
                curv.append(self._norm(self._cross(v, acc[i - 1])) / (vn ** 3))
        t0 = self.samples[0][0]
        for i, s in enumerate(self.samples):
            v = vel[i - 1] if 0 < i <= len(vel) else (0.0, 0.0, 0.0)
            a = acc[i - 2] if 1 < i <= len(acc) + 1 else (0.0, 0.0, 0.0)
            c = curv[i - 2] if 1 < i <= len(curv) + 1 else 0.0
            self._csv_writer.writerow(
                [
                    round((s[0] - t0) * 1e-9, 4),
                    round(s[1], 5), round(s[2], 5), round(s[3], 5),
                    round(v[0], 5), round(v[1], 5), round(v[2], 5),
                    round(self._norm(v), 5),
                    round(self._norm(a), 5),
                    round(c, 5),
                ]
            )
        self._csv_handle.flush()

    def destroy_node(self) -> None:
        if self._csv_handle is not None:
            self._csv_handle.close()
        super().destroy_node()


def parse_args(argv=None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[1:]
    # launch_ros appends "--ros-args -r __node:=..." after our arguments;
    # strip everything from "--ros-args" onward so argparse is happy.
    if "--ros-args" in argv:
        argv = argv[: argv.index("--ros-args")]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/joint_states_local",
                        help="joint states topic to trigger sampling (default: /joint_states_local)")
    parser.add_argument("--frame-id", default="base_link",
                        help="reference frame for the trajectory (default: base_link)")
    parser.add_argument("--target-frame", default="gripper_frame_link",
                        help="end-effector frame tracked (default: gripper_frame_link)")
    parser.add_argument("--publish-rate", type=float, default=10.0,
                        help="visualization publish rate in Hz (default: 10)")
    parser.add_argument("--min-sample-dt", type=float, default=0.02,
                        help="minimum seconds between samples (default: 0.02 -> 50 Hz cap)")
    parser.add_argument("--circle-close-tol", type=float, default=0.02,
                        help="meters from circle start that closes a circle (default: 0.02)")
    parser.add_argument("--circle-min-arc", type=float, default=0.15,
                        help="minimum arc length in m before a circle can close (default: 0.15)")
    parser.add_argument("--align-samples", type=int, default=120,
                        help="resample points per circle for error alignment (default: 120)")
    parser.add_argument("--speed-window", type=int, default=5,
                        help="moving-average window for velocity smoothing (default: 5)")
    parser.add_argument("--out-file", default=None,
                        help="optional CSV output path for samples")
    return parser.parse_args(argv)


def main() -> None:
    rclpy.init()
    node = EETrajectoryRecorder(parse_args())
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
