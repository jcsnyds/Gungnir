#!/usr/bin/env python3
"""Home pose, reach map and grasp check for the gungnir scene. Nothing moves; needs a running launch (mock is enough).

  view_pose.py               home pose for the camera pose make_world.py stored in worlds/gungnir_world_truth.yaml,
                             then a grasp check of every object. --write stores it as config/initial_positions.yaml
  view_pose.py X Y           home pose for a camera looking at table point (X, Y) instead
  view_pose.py --map         which table positions the arm can image (home view) AND grasp from above;
                             suggests --center-x for make_world.py

View options (defaults match make_world.py): --pitch 45 (deg below the horizon, 90 = straight down),
--view-dist 0.45 (m, camera to target), --heading 0 (0 = along world +X), --target-z 0.05.
World frame: origin = robot base, table top at z = 0."""
import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from rclpy.node import Node
from rclpy.time import Time
from rclpy.utilities import remove_ros_args
from tf2_ros import Buffer, TransformException, TransformListener

from robot_info import JOINTS, joint_limits_deg
from scene_geometry import view_from

HERE = Path(__file__).resolve().parent
TRUTH = HERE / "worlds" / "gungnir_world_truth.yaml"
INITIAL = HERE / "config" / "initial_positions.yaml"
CAM, TCP, GROUP = "camera_depth_optical_frame", "tcp_frame", "manipulator"
CHECK_LINKS = ["J3_link", "J4_link", "J5_link", "J6_link", "gripper", "camera_link"]
MIN_LINK_Z = 0.05      # m above the table: rejects elbow-down solutions that would pass through the table


def quat_to_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return (0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s)
    if R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return ((R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s)
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return ((R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s)


def down_rotation(yaw_deg=0.0):
    """Camera looking straight down; image 'up' = (cos yaw, sin yaw, 0) in world (yaw 0: up = +X, right = -Y)."""
    a = math.radians(yaw_deg)
    z = np.array([0.0, 0.0, -1.0])
    y = -np.array([math.cos(a), math.sin(a), 0.0])
    return np.column_stack((np.cross(y, z), y, z))


class Solver:
    def __init__(self, node):
        self.node = node
        self.limits = joint_limits_deg(node)
        self.buf = Buffer()
        self.listener = TransformListener(self.buf, node)
        self.ik = node.create_client(GetPositionIK, "/compute_ik")
        self.fk = node.create_client(GetPositionFK, "/compute_fk")
        for cli, name in ((self.ik, "/compute_ik"), (self.fk, "/compute_fk")):
            if not cli.wait_for_service(timeout_sec=10.0):
                sys.exit(f"{name} not available (is a launch running?)")
        self.T_cam_tcp = self._tcp_in_camera()

    def _tcp_in_camera(self):
        t0 = time.monotonic()
        while True:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            try:
                tf = self.buf.lookup_transform(CAM, TCP, Time()).transform
                break
            except TransformException:
                if time.monotonic() - t0 > 5.0:
                    sys.exit(f"No TF from {CAM} to {TCP}")
        T = np.eye(4)
        T[:3, :3] = quat_to_mat((tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w))
        T[:3, 3] = (tf.translation.x, tf.translation.y, tf.translation.z)
        return T

    def pose_for_camera(self, position, R):
        """TCP pose (world) that puts the camera at `position` with rotation R."""
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = position
        return T @ self.T_cam_tcp

    def grasp_pose(self, x, y, z, yaw_deg=0.0):
        """TCP pose (world) with the tool pointing straight down, TCP at (x, y, z)."""
        T = np.eye(4)
        T[:3, :3] = down_rotation(yaw_deg) @ self.T_cam_tcp[:3, :3]
        T[:3, 3] = (x, y, z)
        return T

    def solve_ik(self, T, seed_deg=None):
        pose = PoseStamped()
        pose.header.frame_id = "world"
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(float, T[:3, 3])
        (pose.pose.orientation.x, pose.pose.orientation.y,
         pose.pose.orientation.z, pose.pose.orientation.w) = mat_to_quat(T[:3, :3])
        req = GetPositionIK.Request()
        r = req.ik_request
        r.group_name, r.ik_link_name, r.avoid_collisions = GROUP, TCP, True
        r.timeout = Duration(sec=0, nanosec=300_000_000)
        r.pose_stamped = pose
        if seed_deg is not None:                  # different seeds reach different arm configurations
            r.robot_state.joint_state.name = list(JOINTS)
            r.robot_state.joint_state.position = [math.radians(v) for v in seed_deg]
        fut = self.ik.call_async(req)
        rclpy.spin_until_future_complete(self.node, fut, timeout_sec=5.0)
        res = fut.result()
        if res is None or res.error_code.val != 1:
            return None
        pos = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
        return [math.degrees(pos[j]) for j in JOINTS] if all(j in pos for j in JOINTS) else None

    def usable(self, q):
        if not 20 <= abs(q[4]) <= 160:            # wrist singular when J5 is near 0 or 180
            return False
        return all(v - self.limits[j][0] >= 10 and self.limits[j][1] - v >= 10 for j, v in zip(JOINTS, q))

    def above_table(self, q):
        req = GetPositionFK.Request()
        req.header.frame_id = "world"
        req.fk_link_names = CHECK_LINKS
        req.robot_state.joint_state.name = list(JOINTS)
        req.robot_state.joint_state.position = [math.radians(v) for v in q]
        fut = self.fk.call_async(req)
        rclpy.spin_until_future_complete(self.node, fut, timeout_sec=5.0)
        res = fut.result()
        if res is None or res.error_code.val != 1:
            return True                           # cannot check: do not reject
        return all(p.pose.position.z >= MIN_LINK_Z for p in res.pose_stamped)

    def solve(self, T, attempts):
        """-> (most compact usable solution in degrees or None, IK successes, usable solutions)"""
        found, good = 0, []
        for i in range(attempts):
            seed = None if i == 0 else [random.uniform(0.8 * self.limits[j][0], 0.8 * self.limits[j][1]) for j in JOINTS]
            q = self.solve_ik(T, seed)
            if q is None:
                continue
            found += 1
            if self.usable(q):
                good.append(q)
        good.sort(key=lambda q: sum(abs(v) for v in q))
        for q in good:
            if self.above_table(q):
                return q, found, len(good)
        return None, found, len(good)


def run_map(solver, view, xs, ys, attempts, spread_x):
    pitch, dist, heading, target_z = view
    print(f"Reach map. Grasp = tool straight down at table height 0.05 m and 0.25 m (approach) over each y.")
    print(f"Home view = camera {dist:.2f} m from the table point (x, 0), pitch {pitch:g} deg, heading {heading:g} deg.\n")
    print(f"{'x (m)':>7}" + "".join(f"{f'grasp y={y:+.1f}':>14}" for y in ys) + f"{'home view':>12}")
    ok_x = []
    for x in xs:
        cells = [all(solver.solve(solver.grasp_pose(x, y, z), attempts)[0] is not None for z in (0.05, 0.25)) for y in ys]
        pos, R = view_from((x, 0.0), dist, pitch, heading, target_z)
        cells.append(solver.solve(solver.pose_for_camera(pos, R), attempts)[0] is not None)
        print(f"{x:7.2f}" + "".join(f"{'OK' if c else '--':>14}" for c in cells[:-1])
              + f"{'OK' if cells[-1] else '--':>12}", flush=True)
        if all(cells):
            ok_x.append(x)

    runs, cur = [], []
    for x in xs:
        if x in ok_x:
            cur.append(x)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    if not runs:
        print("\nNo x where everything works. Try --pitch 55, a smaller --view-dist, or send me this table.")
        return
    best = max(runs, key=len)
    margin = spread_x + 0.05
    lo, hi = best[0] + margin, best[-1] - margin
    flags = f"--pitch {pitch:g} --view-dist {dist:g} --heading {heading:g}"
    print(f"\nWorkable zone: x = {best[0]:.2f} .. {best[-1]:.2f} m")
    if lo <= hi:
        print(f"Objects fit with --center-x between {lo:.2f} (nearest the robot) and {hi:.2f}.")
        print(f"Suggested: python3 scripts/make_world.py --center-x {(lo + hi) / 2:.2f} {flags}")
    else:
        print("The zone is narrow. Use a smaller spread:")
        print(f"Suggested: python3 scripts/make_world.py --center-x {(best[0] + best[-1]) / 2:.2f} --spread-x 0.05 {flags}")


def check_objects(solver, objects, attempts):
    print("\nGrasp check: tool straight down over each object (grasp at mid-height, approach 0.20 m above its top).")
    print(f"{'object':16}{'x':>8}{'y':>8}{'top cm':>8}{'grasp':>8}{'approach':>10}")
    bad = 0
    for o in objects:
        x, y = o["world_xy_m"]
        top = o["top_height_m"]
        g = solver.solve(solver.grasp_pose(x, y, max(0.5 * top, 0.02)), attempts)[0] is not None
        a = solver.solve(solver.grasp_pose(x, y, top + 0.20), attempts)[0] is not None
        bad += not (g and a)
        print(f"{o['name']:16}{x:8.3f}{y:8.3f}{top * 100:8.1f}{'OK' if g else '--':>8}{'OK' if a else '--':>10}")
    print("All objects can be grasped." if not bad else
          f"{bad} object(s) cannot be reached from above: move --center-x (see --map) or use another --seed.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("xy", nargs="*", type=float)
    ap.add_argument("--pitch", type=float, default=45.0)
    ap.add_argument("--view-dist", type=float, default=0.45)
    ap.add_argument("--heading", type=float, default=0.0)
    ap.add_argument("--target-z", type=float, default=0.05)
    ap.add_argument("--spread-x", type=float, default=0.11, help="used by --map to size the suggestion")
    ap.add_argument("--write", action="store_true", help="write the home pose to config/initial_positions.yaml")
    ap.add_argument("--map", action="store_true")
    ap.add_argument("--map-x", nargs=3, type=float, default=[0.4, 1.5, 0.1], metavar=("MIN", "MAX", "STEP"))
    ap.add_argument("--attempts", type=int, default=None)
    args = ap.parse_args(remove_ros_args(sys.argv)[1:])
    if len(args.xy) not in (0, 2):
        sys.exit("Give X and Y together, or neither")

    random.seed(1)
    rclpy.init(args=sys.argv)
    node = Node("view_pose")
    solver = Solver(node)

    if args.map:
        lo, hi, step = args.map_x
        xs = [round(float(v), 3) for v in np.arange(lo, hi + 1e-9, step)]
        run_map(solver, (args.pitch, args.view_dist, args.heading, args.target_z), xs, [-0.3, 0.0, 0.3],
                args.attempts or 4, args.spread_x)
        return

    objects = []
    if len(args.xy) == 2:
        pos, R = view_from(args.xy, args.view_dist, args.pitch, args.heading, args.target_z)
        what = (f"looking at ({args.xy[0]:.2f}, {args.xy[1]:.2f}, {args.target_z:.2f}), pitch {args.pitch:g} deg, "
                f"heading {args.heading:g} deg")
    else:
        if not TRUTH.exists():
            sys.exit(f"{TRUTH} not found: run scripts/make_world.py first (or give X Y).")
        truth = yaml.safe_load(TRUTH.read_text())
        c = truth.get("camera", {})
        if "position" not in c:
            sys.exit("The ground-truth file has no camera pose (made by an older make_world.py): re-run scripts/make_world.py.")
        pos, R = np.array(c["position"], float), np.array(c["rotation"], float)
        what = (f"as chosen by make_world.py: looking at ({c['center_xy'][0]:.2f}, {c['center_xy'][1]:.2f}, "
                f"{c['target_z']:.2f}), pitch {c['pitch_deg']:g} deg, heading {c['heading_deg']:g} deg")
        objects = truth["objects"]

    attempts = args.attempts or 8
    q, found, good = solver.solve(solver.pose_for_camera(pos, R), attempts)
    print(f"Home camera at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) m, {what}.")
    print(f"IK: {found} solutions from {attempts} attempts, {good} usable "
          f"(J5 away from 0/180, no joint within 10 deg of a limit, links above the table).")
    if q is None:
        print("No usable home pose for this view. Try --pitch 55, a smaller --view-dist, or another --center-x (see --map).")
        sys.exit(1)
    print("\nJ1..J6 (deg): " + "  ".join(f"{v:.1f}" for v in q))
    text = ("# Start (home) pose for mock and sim modes, in radians (written by view_pose.py).\n"
            f"# Camera at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) m, {what}.\n"
            "initial_positions:\n" + "".join(f"  {j}: {math.radians(v):.4f}\n" for j, v in zip(JOINTS, q)))
    if args.write:
        INITIAL.write_text(text)
        print(f"\nWrote {INITIAL}. Restart the launch to use it.")
    else:
        print("\nRe-run with --write to store it as the start pose of mock and sim, or put this in config/initial_positions.yaml:\n")
        print(text)
    if objects:
        check_objects(solver, objects, attempts)


if __name__ == "__main__":
    main()
