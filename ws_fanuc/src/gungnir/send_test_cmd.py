#!/usr/bin/env python3
"""
AI Use Statement:
ChatGPT (OpenAI) was used to assist in modifying and debugging the Python code
for Cartesian path planning in MoveIt. AI assistance was used to implement
conversion from roll-pitch-yaw Euler angles to quaternion orientation,
generate Cartesian trajectories from user-defined poses, and execute
trajectories only when the planned path was sufficiently complete.

The code was adapted to use ROS 2 interfaces, including rclpy, the
GetCartesianPath service, and the ExecuteTrajectory action. The final
code was reviewed and incorporated into the project by the author.

Claude (Anthropic) was later used to add command-line arguments and a
relative-move mode that keeps the current TCP orientation (read from TF).
"""

import argparse
import math
import sys
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
from rclpy.utilities import remove_ros_args

from geometry_msgs.msg import Pose
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetCartesianPath
from tf2_ros import Buffer, TransformException, TransformListener


def rpy_to_quaternion(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy,   # x
            cr * sp * cy + sr * cp * sy,   # y
            cr * cp * sy - sr * sp * cy,   # z
            cr * cp * cy + sr * sp * sy)   # w


def make_pose(x, y, z, roll, pitch, yaw):
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = x, y, z
    (pose.orientation.x, pose.orientation.y,
     pose.orientation.z, pose.orientation.w) = rpy_to_quaternion(roll, pitch, yaw)
    return pose


def fmt(pose):
    p, q = pose.position, pose.orientation
    return (f"xyz=({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) "
            f"quat=({q.x:.3f}, {q.y:.3f}, {q.z:.3f}, {q.w:.3f})")


class CartesianPathPlanner(Node):

    def __init__(self, group, eef_link, reference_frame, speed):
        super().__init__("cartesian_path_planner")
        self.group_name = group
        self.eef_link = eef_link
        self.reference_frame = reference_frame
        self.eef_step = 0.01
        self.jump_threshold = 0.0
        self.velocity_scaling = speed
        self.acceleration_scaling = speed

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cartesian_client = self.create_client(GetCartesianPath, "/compute_cartesian_path")
        self.execute_client = ActionClient(self, ExecuteTrajectory, "/execute_trajectory")

    def current_eef_pose(self):
        """Pose of the end-effector link expressed in the reference frame (latest TF)."""
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            try:
                t = self.tf_buffer.lookup_transform(
                    self.reference_frame, self.eef_link, Time()).transform
            except TransformException:
                continue
            pose = Pose()
            pose.position.x = t.translation.x
            pose.position.y = t.translation.y
            pose.position.z = t.translation.z
            pose.orientation = t.rotation
            return pose
        raise RuntimeError(f"No TF from '{self.reference_frame}' to '{self.eef_link}'")

    def plan_cartesian_path(self, target_pose):
        request = GetCartesianPath.Request()
        # The header stamp is left at 0 on purpose: it is not needed and avoids
        # wall-clock / sim-clock mismatches between modes.
        request.header.frame_id = self.reference_frame
        request.group_name = self.group_name
        request.link_name = self.eef_link
        request.waypoints = [target_pose]
        request.max_step = self.eef_step
        request.jump_threshold = self.jump_threshold
        request.avoid_collisions = True
        request.max_velocity_scaling_factor = self.velocity_scaling
        request.max_acceleration_scaling_factor = self.acceleration_scaling

        if not self.cartesian_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Cartesian path service unavailable")
            return None, 0.0

        self.get_logger().info("Computing Cartesian path...")
        future = self.cartesian_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
        if not future.done() or future.result() is None:
            self.get_logger().error("Cartesian path planning failed or timed out")
            return None, 0.0

        response = future.result()
        self.get_logger().info(
            f"error code {response.error_code.val}, fraction {response.fraction:.3f}, "
            f"{len(response.solution.joint_trajectory.points)} points "
            f"(group={self.group_name}, eef={self.eef_link}, frame={self.reference_frame})")
        if response.fraction == 0.0:
            self.get_logger().error(
                "No Cartesian path could be calculated. Check IK, reachability, collisions, robot state.")
        return response.solution, response.fraction

    def execute_plan(self, plan):
        if not self.execute_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Trajectory execution action server unavailable")
            return False

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = plan
        self.get_logger().info("Executing Cartesian trajectory...")
        future = self.execute_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if not future.done():
            self.get_logger().error("Execution request timed out")
            return False
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Trajectory execution rejected")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=120.0)
        if not result_future.done():
            self.get_logger().error("Trajectory execution timed out")
            return False

        code = result_future.result().result.error_code.val
        if code == MoveItErrorCodes.SUCCESS:
            self.get_logger().info("Trajectory executed successfully")
            return True
        self.get_logger().error(f"Trajectory execution failed with error code: {code}")
        return False


def max_joint_step_deg(trajectory):
    """Largest single-joint change (degrees) between consecutive trajectory points."""
    pts = trajectory.joint_trajectory.points
    return max((abs(b - a) for p0, p1 in zip(pts, pts[1:])
                for a, b in zip(p0.positions, p1.positions)), default=0.0) * 180.0 / math.pi


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Cartesian move of the TCP.",
        epilog=("no numbers : move 1 cm along the reference frame's Z, keeping the current orientation\n"
                "3 numbers  : dx dy dz (m) in the reference frame, keeping the current orientation\n"
                "6 numbers  : absolute pose x y z roll pitch yaw (m, rad) in the reference frame"),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("values", nargs="*", type=float)
    p.add_argument("--group", default="manipulator")
    p.add_argument("--eef", default="tcp_frame")
    p.add_argument("--frame", default="camera_depth_optical_frame")
    p.add_argument("--speed", type=float, default=0.2, help="velocity/acceleration scaling (0-1)")
    p.add_argument("--max-joint-step", type=float, default=10.0,
                   help="refuse paths with a larger joint change (deg) between points")
    return p.parse_args(argv)


def main():
    args = parse_args(remove_ros_args(sys.argv)[1:])
    if len(args.values) not in (0, 3, 6):
        sys.exit("Give 0, 3 or 6 numbers (see --help)")

    rclpy.init(args=sys.argv)
    planner = CartesianPathPlanner(args.group, args.eef, args.frame, args.speed)
    success = False
    try:
        if len(args.values) == 6:
            target = make_pose(*args.values)
        else:
            dx, dy, dz = args.values if args.values else (0.0, 0.0, 0.01)
            target = planner.current_eef_pose()
            planner.get_logger().info(f"Current TCP in {args.frame}: {fmt(target)}")
            target.position.x += dx
            target.position.y += dy
            target.position.z += dz
        planner.get_logger().info(f"Target TCP in {args.frame}: {fmt(target)}")

        plan, fraction = planner.plan_cartesian_path(target)
        print(f"Path completion: {fraction * 100:.1f}%")

        step = max_joint_step_deg(plan) if plan is not None else 0.0
        print(f"Largest joint step between points: {step:.1f} deg")
        if plan is not None and fraction >= 0.999 and step > args.max_joint_step:
            planner.get_logger().error(
                f"Path jumps {step:.1f} deg between two points (limit {args.max_joint_step:.0f}): "
                "probably an IK branch flip near a singularity. Execution canceled.")
        elif plan is not None and fraction >= 0.999:
            success = planner.execute_plan(plan)
        else:
            planner.get_logger().error("Cartesian path planning was incomplete. Execution canceled.")
        print("Execution successful:", success)
    except Exception as error:  # noqa: BLE001
        planner.get_logger().error(f"Error: {error}")
    finally:
        planner.destroy_node()
        rclpy.shutdown()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
