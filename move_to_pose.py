#!/usr/bin/env python3
"""
Move the robot so that a chosen frame (default: the tool flange) reaches a pose
given as a 4x4 homogeneous matrix expressed in the CAMERA frame.

    T_cam_target : pose of the target in the camera frame (target -> camera);
                   e.g. the output of an ArUco / object-pose detector.
    T_world_cam  : looked up from TF (world <- camera optical frame)
    T_world_goal = T_world_cam @ T_cam_target @ T_target_goal

T_target_goal optionally adds a stand-off along the target's +Z and a 180 deg flip
about X. Both assume the target's +Z is the surface normal pointing OUT of the surface
towards the camera (the ArUco/OpenCV marker convention): --standoff then backs the tool
off from the surface, and --flip makes the tool's +Z point INTO the surface.

The motion is planned and executed by MoveIt 2 (action /move_action), which runs
IK + collision-aware planning and sends the trajectory to the ros2_control
joint_trajectory_controller. Keep the robot still while the matrix is captured:
the camera pose is read from TF at call time.

Examples
  ros2 run crx_gz_sim move_to_pose.py --matrix '[[1,0,0,0.05],[0,1,0,0.02],[0,0,1,0.45],[0,0,0,1]]'
  ros2 run crx_gz_sim move_to_pose.py --matrix-file target.npy --standoff 0.10 --flip
  ros2 run crx_gz_sim move_to_pose.py --matrix-file target.npy --print-only
"""
import argparse
import json
import math
import sys
import time

import numpy as np


# ----------------------------------------------------------------------------- math
def parse_matrix(text):
    s = text.strip()
    if s.startswith("["):
        arr = np.array(json.loads(s), dtype=float)
    else:
        arr = np.array([float(x) for x in s.replace(",", " ").split()], dtype=float)
    if arr.size != 16:
        raise ValueError(f"expected 16 numbers for a 4x4 matrix, got {arr.size}")
    return arr.reshape(4, 4)


def load_matrix_file(path):
    if path.endswith(".npy"):
        arr = np.load(path)
    elif path.endswith(".json"):
        with open(path, encoding="utf-8") as fh:
            arr = np.array(json.load(fh), dtype=float)
    else:
        arr = np.loadtxt(path, delimiter="," if path.endswith(".csv") else None)
    arr = np.asarray(arr, dtype=float)
    if arr.size != 16:
        raise ValueError(f"{path}: expected 16 numbers, got {arr.size}")
    return arr.reshape(4, 4)


def validate_transform(T, tol=1e-3):
    """Check it is a rigid transform; snap the rotation block to an exact rotation."""
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError("matrix must be 4x4")
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"bottom row must be [0 0 0 1], got {T[3]}")
    R = T[:3, :3]
    if np.abs(R @ R.T - np.eye(3)).max() > tol or abs(np.linalg.det(R) - 1.0) > tol:
        raise ValueError("rotation block is not a proper rotation (scaled, skewed or mirrored?)")
    U, _, Vt = np.linalg.svd(R)
    out = T.copy()
    out[:3, :3] = U @ Vt
    return out


def rot_to_quat(R):
    """3x3 rotation -> quaternion (x, y, z, w)."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def quat_to_rot(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def make_transform(t, q_xyzw):
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(*q_xyzw)
    T[:3, 3] = t
    return T


def compose_goal(T_world_cam, T_cam_target, standoff=0.0, flip=False):
    """T_world_goal = T_world_cam @ T_cam_target @ [Rflip | (0,0,standoff)]."""
    T_tg = np.eye(4)
    T_tg[2, 3] = standoff
    if flip:
        T_tg[:3, :3] = np.diag([1.0, -1.0, -1.0])      # 180 deg about X
    return T_world_cam @ T_cam_target @ T_tg


# ----------------------------------------------------------------------------- ROS
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--matrix", help="4x4 as JSON [[..],[..],[..],[..]] or 16 numbers (row-major)")
    g.add_argument("--matrix-file", help=".npy / .json / .csv / whitespace text file with a 4x4 matrix")
    ap.add_argument("--invert", action="store_true",
                    help="the matrix is T_target_cam (camera pose in the target frame); invert it first")
    ap.add_argument("--camera-frame", default="camera_color_optical_frame",
                    help="frame the matrix is expressed in (default: L515 optical frame, z forward)")
    ap.add_argument("--world-frame", default="world")
    ap.add_argument("--ee-frame", default="fanuc_flange",
                    help="robot frame that should reach the pose (e.g. your gripper TCP link)")
    ap.add_argument("--standoff", type=float, default=0.0, help="metres to back off along target +Z (assumed to point out of the surface, towards the camera)")
    ap.add_argument("--flip", action="store_true", help="rotate goal 180 deg about target X so tool +Z points into the surface (target +Z = surface normal)")
    ap.add_argument("--group", default="manipulator")
    ap.add_argument("--vel", type=float, default=0.2, help="max velocity scaling 0..1")
    ap.add_argument("--acc", type=float, default=0.2, help="max acceleration scaling 0..1")
    ap.add_argument("--pos-tol", type=float, default=0.001, help="goal position tolerance (m)")
    ap.add_argument("--ori-tol", type=float, default=0.01, help="goal orientation tolerance (rad)")
    ap.add_argument("--planning-time", type=float, default=5.0)
    ap.add_argument("--plan-only", action="store_true", help="plan but do not execute")
    ap.add_argument("--print-only", action="store_true", help="only print the computed goal pose, no planning")
    ap.add_argument("--no-sim-time", action="store_true", help="do not use /clock (real robot)")
    args, _ros_args = ap.parse_known_args()

    try:
        T_cam_target = parse_matrix(args.matrix) if args.matrix else load_matrix_file(args.matrix_file)
        T_cam_target = validate_transform(T_cam_target)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Invalid matrix: {exc}")
    if args.invert:
        T_cam_target = np.linalg.inv(T_cam_target)

    import rclpy
    from geometry_msgs.msg import Pose
    from moveit_msgs.action import MoveGroup
    from moveit_msgs.msg import (Constraints, MoveItErrorCodes, OrientationConstraint,
                                 PositionConstraint)
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.time import Time
    from shape_msgs.msg import SolidPrimitive
    from tf2_ros import Buffer, TransformListener

    rclpy.init()
    node = Node("move_to_pose",
                parameter_overrides=[Parameter("use_sim_time", value=not args.no_sim_time)])
    log = node.get_logger()
    try:
        # --- camera pose in the world from TF ---------------------------------
        buf = Buffer()
        TransformListener(buf, node)
        t0 = time.time()
        tf = None
        while time.time() - t0 < 10.0 and tf is None:
            rclpy.spin_once(node, timeout_sec=0.1)
            if buf.can_transform(args.world_frame, args.camera_frame, Time()):
                tf = buf.lookup_transform(args.world_frame, args.camera_frame, Time())
        if tf is None:
            sys.exit(f"TF {args.world_frame} <- {args.camera_frame} not available after 10 s. Is the sim running?")
        tr, ro = tf.transform.translation, tf.transform.rotation
        T_world_cam = make_transform([tr.x, tr.y, tr.z], [ro.x, ro.y, ro.z, ro.w])

        T_goal = compose_goal(T_world_cam, T_cam_target, args.standoff, args.flip)
        pos = T_goal[:3, 3]
        quat = rot_to_quat(T_goal[:3, :3])
        np.set_printoptions(precision=4, suppress=True)
        print(f"T_{args.world_frame}_{args.camera_frame} (from TF):\n{T_world_cam}")
        print(f"Goal for '{args.ee_frame}' in '{args.world_frame}':\n{T_goal}")
        print(f"  position xyz = {pos}\n  quaternion xyzw = {quat}")
        if args.print_only:
            return

        # --- MoveIt goal -------------------------------------------------------
        client = ActionClient(node, MoveGroup, "move_action")
        if not client.wait_for_server(timeout_sec=15.0):
            sys.exit("MoveGroup action server '/move_action' not found. Launch with moveit:=true.")

        pc = PositionConstraint()
        pc.header.frame_id = args.world_frame
        pc.link_name = args.ee_frame
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [args.pos_tol]
        pc.constraint_region.primitives.append(sphere)
        p = Pose()
        p.position.x, p.position.y, p.position.z = (float(v) for v in pos)
        p.orientation.w = 1.0
        pc.constraint_region.primitive_poses.append(p)
        pc.weight = 1.0

        oc = OrientationConstraint()
        oc.header.frame_id = args.world_frame
        oc.link_name = args.ee_frame
        oc.orientation.x, oc.orientation.y, oc.orientation.z, oc.orientation.w = (float(v) for v in quat)
        oc.absolute_x_axis_tolerance = oc.absolute_y_axis_tolerance = oc.absolute_z_axis_tolerance = args.ori_tol
        oc.weight = 1.0

        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = args.group
        if hasattr(req, "pipeline_id"):
            req.pipeline_id = "ompl"
        req.planner_id = "RRTConnectkConfigDefault"
        req.num_planning_attempts = 10
        req.allowed_planning_time = args.planning_time
        req.max_velocity_scaling_factor = args.vel
        req.max_acceleration_scaling_factor = args.acc
        req.start_state.is_diff = True
        req.goal_constraints.append(Constraints(position_constraints=[pc], orientation_constraints=[oc]))
        goal.planning_options.plan_only = args.plan_only
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        log.info("Sending goal to MoveIt (planning%s)..." % ("" if args.plan_only else " + execution"))
        send = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, send)
        handle = send.result()
        if not handle or not handle.accepted:
            sys.exit("MoveIt rejected the goal.")
        res_fut = handle.get_result_async()
        rclpy.spin_until_future_complete(node, res_fut)
        code = res_fut.result().result.error_code.val
        names = {v: k for k, v in vars(MoveItErrorCodes).items() if k.isupper() and isinstance(v, int)}
        if code == MoveItErrorCodes.SUCCESS:
            log.info("Done: SUCCESS")
        else:
            sys.exit(f"MoveIt failed: {names.get(code, code)} "
                     "(NO_IK_SOLUTION / GOAL_VIOLATES_PATH_CONSTRAINTS usually means the pose is unreachable)")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
