#!/usr/bin/env python3
"""Which 10 cm Cartesian moves does the planner complete from the current pose?
Collisions on/off, and where the furthest reachable point sits relative to the joint limits.
Run right after a fresh launch (zero pose). Optional argument: distance in metres."""
import math
import sys
import time

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.srv import GetCartesianPath
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener
from robot_info import joint_limits_deg

FRAME, EEF, GROUP = "camera_depth_optical_frame", "tcp_frame", "manipulator"
LIMITS = {"J1": (-200, 200), "J2": (-179.9, 179.9), "J3": (-68, 248),
          "J4": (-190, 190), "J5": (-179.9, 179.9), "J6": (-225, 225)}
D = float(sys.argv[1]) if len(sys.argv) > 1 else 0.10
MOVES = {"+z fwd ": (0, 0, D), "-z back": (0, 0, -D), "+x     ": (D, 0, 0),
         "-x     ": (-D, 0, 0), "+y down": (0, D, 0), "-y up  ": (0, -D, 0)}


def main():
    rclpy.init()
    node = Node("cartesian_diag")
    LIMITS.update(joint_limits_deg(node))
    buf = Buffer()
    TransformListener(buf, node)
    cli = node.create_client(GetCartesianPath, "/compute_cartesian_path")
    if not cli.wait_for_service(timeout_sec=10.0):
        sys.exit("compute_cartesian_path not available")

    t0 = time.monotonic()
    while True:
        rclpy.spin_once(node, timeout_sec=0.1)
        try:
            tf = buf.lookup_transform(FRAME, EEF, Time()).transform
            break
        except TransformException:
            if time.monotonic() - t0 > 5.0:
                sys.exit(f"No TF from {FRAME} to {EEF}")

    def plan(delta, collisions):
        pose = Pose()
        pose.position.x = tf.translation.x + delta[0]
        pose.position.y = tf.translation.y + delta[1]
        pose.position.z = tf.translation.z + delta[2]
        pose.orientation = tf.rotation
        req = GetCartesianPath.Request()
        req.header.frame_id = FRAME
        req.group_name = GROUP
        req.link_name = EEF
        req.waypoints = [pose]
        req.max_step = 0.01
        req.jump_threshold = 0.0
        req.avoid_collisions = collisions
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=20.0)
        return fut.result()

    print(f"{D * 100:.0f} cm moves in {FRAME}")
    print(f"{'move':8} {'collisions':10} {'fraction':9} max joint step (deg), furthest reachable joints (deg)")
    for name, delta in MOVES.items():
        for coll in (True, False):
            r = plan(delta, coll)
            if r is None:
                print(f"{name} {str(coll):10} no response")
                continue
            traj = r.solution.joint_trajectory
            if not traj.points:
                print(f"{name} {str(coll):10} {r.fraction:<9.2f} (no points)")
                continue
            q = [math.degrees(v) for v in traj.points[-1].positions]
            near = [n for n, v in zip(traj.joint_names, q)
                    if v - LIMITS.get(n, (-1e9, 1e9))[0] < 5 or LIMITS.get(n, (-1e9, 1e9))[1] - v < 5]
            pts = traj.points
            step = max((abs(b - a) for p0, p1 in zip(pts, pts[1:])
                        for a, b in zip(p0.positions, p1.positions)), default=0.0) * 180.0 / math.pi
            joints = " ".join(f"{n}={v:7.1f}" for n, v in zip(traj.joint_names, q))
            print(f"{name} {str(coll):10} {r.fraction:<9.2f} step={step:5.1f}  {joints}" +
                  (f"   NEAR LIMIT: {','.join(near)}" if near else ""))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
