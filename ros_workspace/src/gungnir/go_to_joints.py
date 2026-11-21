#!/usr/bin/env python3
"""Plan (with collision checking) and execute a joint-space move through MoveIt.
Usage: go_to_joints.py J1 J2 J3 J4 J5 J6   (degrees)   [--speed 0.2]"""
import argparse
import math
import sys

import rclpy
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

JOINTS = ["J1", "J2", "J3", "J4", "J5", "J6"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("degrees", nargs=6, type=float)
    ap.add_argument("--group", default="manipulator")
    ap.add_argument("--speed", type=float, default=0.2)
    args = ap.parse_args(remove_ros_args(sys.argv)[1:])

    rclpy.init(args=sys.argv)
    node = Node("go_to_joints")
    client = ActionClient(node, MoveGroup, "/move_action")
    if not client.wait_for_server(timeout_sec=10.0):
        sys.exit("/move_action not available")

    goal = MoveGroup.Goal()
    req = goal.request
    req.group_name = args.group
    req.start_state.is_diff = True          # start from the current state
    req.num_planning_attempts = 5
    req.allowed_planning_time = 5.0
    req.max_velocity_scaling_factor = args.speed
    req.max_acceleration_scaling_factor = args.speed
    c = Constraints()
    for name, deg in zip(JOINTS, args.degrees):
        jc = JointConstraint()
        jc.joint_name = name
        jc.position = math.radians(deg)
        jc.tolerance_above = jc.tolerance_below = 0.001
        jc.weight = 1.0
        c.joint_constraints.append(jc)
    req.goal_constraints = [c]
    goal.planning_options.plan_only = False

    fut = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=15.0)
    handle = fut.result()
    if handle is None or not handle.accepted:
        sys.exit("Goal rejected")
    res = handle.get_result_async()
    rclpy.spin_until_future_complete(node, res, timeout_sec=120.0)
    ok = res.done() and res.result().result.error_code.val == MoveItErrorCodes.SUCCESS
    print("SUCCESS" if ok else f"FAILED (error code {res.result().result.error_code.val if res.done() else 'timeout'})")
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
