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

Claude (Anthropic; Claude Sonnet 5.5 via claude.ai), November 2026, renamed this
file from send_test_cmd.py to Cartesian_Path_Planner.py (the name that
Gungnir_Client_ROS2.py starts after the user approves a move), added this
description of the script, and added the line-by-line comments and function
docstrings. Those edits changed comments and docstrings only: Claude checked
that the executable code is identical to send_test_cmd.py as committed in
jcsnyds/Gungnir on 2026-11-22 by comparing the two files' syntax trees with the
docstrings removed. Claude has NOT run this file with ROS 2, MoveIt or a robot.
Reviewed and tested by: not yet reviewed or tested by the author after the
November 2026 rename and commenting.

----------------------------------------------------------------------------------------
WHAT THIS SCRIPT DOES
----------------------------------------------------------------------------------------
Moves the robot's tool centre point (TCP) in a straight line in Cartesian space with
MoveIt. It asks MoveIt's move_group node for a straight-line path (the
/compute_cartesian_path service), checks that the whole path was found and does not
contain a sudden joint jump, and only then executes it (the /execute_trajectory
action). THIS SCRIPT MOVES THE ROBOT: try it in simulation (mode:=sim) first.

It is used in two ways:
  * By the Gungnir client. After the user approves a grasp in the web page, the client
    runs
        <python> Cartesian_Path_Planner.py x y z roll pitch yaw --frame F --eef E --group G --speed S
    and treats exit code 0 as "the robot reached the target" and any other code as failure.
  * By hand, as a test tool:
        ros2 run gungnir Cartesian_Path_Planner.py                    # 1 cm along the reference frame's Z
        ros2 run gungnir Cartesian_Path_Planner.py 0.05 0 0           # 5 cm along X, same orientation
        ros2 run gungnir Cartesian_Path_Planner.py x y z roll pitch yaw   # absolute pose (m, rad)
    (`ros2 run` works because CMakeLists.txt installs this file; see `--help` for options.)

Requirements: the ROS 2 workspace is sourced, and move_group is running (started by
gungnir.launch.py). The reference frame (default camera_depth_optical_frame) and the TCP
link (default tcp_frame) must exist in the robot's TF tree.
"""

import argparse  # reads the command-line options and numbers
import math  # sine/cosine for the angle conversion, pi for radians <-> degrees
import sys  # sys.argv (the command line) and sys.exit (the process exit code)
import time  # time.monotonic(): a clock that cannot jump, used for the 5 s TF wait

import rclpy  # the ROS 2 Python client library
from rclpy.action import ActionClient  # lets a node send a goal to an action server
from rclpy.node import Node  # base class of every ROS 2 node
from rclpy.time import Time  # Time() is time zero, which TF treats as "the latest transform"
from rclpy.utilities import remove_ros_args  # strips ROS-specific arguments (--ros-args ...) from argv

from geometry_msgs.msg import Pose  # a position (x, y, z) plus an orientation quaternion
from moveit_msgs.action import ExecuteTrajectory  # action: "run this trajectory on the robot"
from moveit_msgs.msg import MoveItErrorCodes  # numeric result codes (SUCCESS = 1)
from moveit_msgs.srv import GetCartesianPath  # service: "plan a straight line through these waypoints"
from tf2_ros import Buffer, TransformException, TransformListener  # reads the robot's TF frames


def rpy_to_quaternion(roll, pitch, yaw):
    """Convert roll, pitch, yaw (radians) to a quaternion (x, y, z, w).

    Uses the usual ROS/URDF convention: rotations about the FIXED x, y and z axes,
    i.e. R = Rz(yaw) * Ry(pitch) * Rx(roll). Gungnir_Client_ROS2.py contains the exact
    inverse of this function, so a change here must be mirrored there.
    """
    # Half angles: a quaternion encodes a rotation by angle a with sin(a/2) and cos(a/2).
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    # The product of the three single-axis quaternions, written out term by term.
    return (sr * cp * cy - cr * sp * sy,   # x
            cr * sp * cy + sr * cp * sy,   # y
            cr * cp * sy - sr * sp * cy,   # z
            cr * cp * cy + sr * sp * sy)   # w


def make_pose(x, y, z, roll, pitch, yaw):
    """Build a geometry_msgs/Pose from a position (metres) and roll/pitch/yaw (radians)."""
    pose = Pose()
    # Position, in metres.
    pose.position.x, pose.position.y, pose.position.z = x, y, z
    # Orientation: convert the three angles to a quaternion and store its four parts.
    (pose.orientation.x, pose.orientation.y,
     pose.orientation.z, pose.orientation.w) = rpy_to_quaternion(roll, pitch, yaw)
    return pose


def fmt(pose):
    """Short human-readable text for a pose, used in log messages."""
    p, q = pose.position, pose.orientation
    return (f"xyz=({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) "
            f"quat=({q.x:.3f}, {q.y:.3f}, {q.z:.3f}, {q.w:.3f})")


class CartesianPathPlanner(Node):
    """A ROS 2 node that plans and executes straight-line TCP moves through MoveIt."""

    def __init__(self, group, eef_link, reference_frame, speed):
        """Create the node and its connections to move_group.

        group:           MoveIt planning group, e.g. "manipulator".
        eef_link:        the link that is moved in a straight line (the TCP), e.g. "tcp_frame".
        reference_frame: the TF frame in which target poses are given, e.g. "camera_depth_optical_frame".
        speed:           velocity and acceleration scaling, 0 < speed <= 1 (1 = the robot's limits).
        """
        # Register this node with ROS under the name "cartesian_path_planner".
        super().__init__("cartesian_path_planner")
        # What to move and in which frame.
        self.group_name = group
        self.eef_link = eef_link
        self.reference_frame = reference_frame
        # Distance between interpolated points along the straight line: 0.01 m = 1 cm.
        self.eef_step = 0.01
        # 0.0 turns off MoveIt's own joint-jump check; this script does its own check instead
        # (max_joint_step_deg, limited by --max-joint-step).
        self.jump_threshold = 0.0
        # Speed limits for the trajectory MoveIt returns.
        self.velocity_scaling = speed
        self.acceleration_scaling = speed

        # TF: the buffer stores the robot's frame transforms; the listener fills it from /tf and /tf_static.
        # The buffer only fills while this node is spinning (see current_eef_pose).
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # Client for the service that plans the straight-line path (provided by move_group).
        self.cartesian_client = self.create_client(GetCartesianPath, "/compute_cartesian_path")
        # Client for the action that executes a trajectory (also provided by move_group).
        self.execute_client = ActionClient(self, ExecuteTrajectory, "/execute_trajectory")

    def current_eef_pose(self):
        """Pose of the end-effector link expressed in the reference frame (latest TF)."""
        # Give TF up to 5 seconds to arrive: a freshly started node has received nothing yet.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            # Process incoming messages for up to 0.1 s so the TF buffer can fill.
            rclpy.spin_once(self, timeout_sec=0.1)
            try:
                # "Where is eef_link, seen from reference_frame?" Time() asks for the latest known transform.
                t = self.tf_buffer.lookup_transform(
                    self.reference_frame, self.eef_link, Time()).transform
            except TransformException:
                # Not available yet (or the frames are not connected): keep waiting until the deadline.
                continue
            # Copy the transform's translation and rotation into a Pose.
            pose = Pose()
            pose.position.x = t.translation.x
            pose.position.y = t.translation.y
            pose.position.z = t.translation.z
            pose.orientation = t.rotation
            return pose
        # The loop ended without a transform: report which frames were being looked up.
        raise RuntimeError(f"No TF from '{self.reference_frame}' to '{self.eef_link}'")

    def plan_cartesian_path(self, target_pose):
        """Ask move_group for a straight-line path to target_pose.

        Returns (trajectory, fraction): the planned trajectory and the fraction (0.0 to 1.0) of the
        requested line that could be planned. (None, 0.0) if the service failed or timed out.
        """
        # Fill in the service request.
        request = GetCartesianPath.Request()
        # The header stamp is left at 0 on purpose: it is not needed and avoids
        # wall-clock / sim-clock mismatches between modes.
        request.header.frame_id = self.reference_frame  # the frame the waypoint below is expressed in
        request.group_name = self.group_name  # which joints may move
        request.link_name = self.eef_link  # which link follows the straight line
        request.waypoints = [target_pose]  # one waypoint: the target (the line starts at the current pose)
        request.max_step = self.eef_step  # spacing of the interpolated points
        request.jump_threshold = self.jump_threshold  # 0.0 = MoveIt's own jump check is off
        request.avoid_collisions = True  # the path must not collide with the planning scene
        request.max_velocity_scaling_factor = self.velocity_scaling
        request.max_acceleration_scaling_factor = self.acceleration_scaling

        # Wait up to 5 s for move_group's service to exist (it is missing if move_group is not running).
        if not self.cartesian_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Cartesian path service unavailable")
            return None, 0.0

        self.get_logger().info("Computing Cartesian path...")
        # Send the request without blocking, then spin this node until the answer arrives (at most 30 s).
        future = self.cartesian_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
        if not future.done() or future.result() is None:
            self.get_logger().error("Cartesian path planning failed or timed out")
            return None, 0.0

        response = future.result()
        # Log what came back: MoveIt's error code, how much of the line was planned, and the point count.
        self.get_logger().info(
            f"error code {response.error_code.val}, fraction {response.fraction:.3f}, "
            f"{len(response.solution.joint_trajectory.points)} points "
            f"(group={self.group_name}, eef={self.eef_link}, frame={self.reference_frame})")
        # A fraction of exactly 0 means not even the first step could be planned.
        if response.fraction == 0.0:
            self.get_logger().error(
                "No Cartesian path could be calculated. Check IK, reachability, collisions, robot state.")
        return response.solution, response.fraction

    def execute_plan(self, plan):
        """Run a planned trajectory on the robot. Returns True only if MoveIt reports success."""
        # Wait up to 5 s for move_group's execution action server.
        if not self.execute_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Trajectory execution action server unavailable")
            return False

        # Package the trajectory as an action goal and send it (non-blocking).
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = plan
        self.get_logger().info("Executing Cartesian trajectory...")
        future = self.execute_client.send_goal_async(goal)
        # Wait up to 10 s for the server to accept or reject the goal.
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if not future.done():
            self.get_logger().error("Execution request timed out")
            return False
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Trajectory execution rejected")
            return False

        # The goal was accepted: now wait for the motion itself to finish (at most 120 s).
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=120.0)
        if not result_future.done():
            self.get_logger().error("Trajectory execution timed out")
            return False

        # MoveIt's result code: SUCCESS means the robot followed the trajectory.
        code = result_future.result().result.error_code.val
        if code == MoveItErrorCodes.SUCCESS:
            self.get_logger().info("Trajectory executed successfully")
            return True
        self.get_logger().error(f"Trajectory execution failed with error code: {code}")
        return False


def max_joint_step_deg(trajectory):
    """Largest single-joint change (degrees) between consecutive trajectory points."""
    # All the points of the joint trajectory, in order.
    pts = trajectory.joint_trajectory.points
    # Compare each point with the next one, joint by joint, and keep the biggest absolute change.
    # default=0.0 covers a trajectory with fewer than two points. The result is converted from radians.
    # A large step between two neighbouring points (1 cm apart) usually means the solver flipped to a
    # different arm configuration, e.g. near a singularity.
    return max((abs(b - a) for p0, p1 in zip(pts, pts[1:])
                for a, b in zip(p0.positions, p1.positions)), default=0.0) * 180.0 / math.pi


def parse_args(argv):
    """Read the command line (argv without the program name) into an argparse namespace."""
    p = argparse.ArgumentParser(
        description="Cartesian move of the TCP.",
        # The three accepted amounts of numbers, shown at the end of --help (line breaks kept as written).
        epilog=("no numbers : move 1 cm along the reference frame's Z, keeping the current orientation\n"
                "3 numbers  : dx dy dz (m) in the reference frame, keeping the current orientation\n"
                "6 numbers  : absolute pose x y z roll pitch yaw (m, rad) in the reference frame"),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # Zero or more numbers; main() checks that there are 0, 3 or 6 of them.
    p.add_argument("values", nargs="*", type=float)
    # Which MoveIt planning group and which link to move (defaults match the Gungnir robot).
    p.add_argument("--group", default="manipulator")
    p.add_argument("--eef", default="tcp_frame")
    # The TF frame the numbers (and the straight line) are expressed in.
    p.add_argument("--frame", default="camera_depth_optical_frame")
    p.add_argument("--speed", type=float, default=0.2, help="velocity/acceleration scaling (0-1)")
    # Safety limit for the joint-jump check done in main().
    p.add_argument("--max-joint-step", type=float, default=10.0,
                   help="refuse paths with a larger joint change (deg) between points")
    return p.parse_args(argv)


def main():
    """Plan the requested move, execute it if the plan is safe, and exit 0 (success) or 1 (anything else)."""
    # remove_ros_args drops ROS-only arguments; [1:] drops the program name.
    args = parse_args(remove_ros_args(sys.argv)[1:])
    # Only 0, 3 or 6 numbers have a meaning (see --help); stop before touching ROS or the robot otherwise.
    if len(args.values) not in (0, 3, 6):
        sys.exit("Give 0, 3 or 6 numbers (see --help)")

    # Start ROS and create the planner node.
    rclpy.init(args=sys.argv)
    planner = CartesianPathPlanner(args.group, args.eef, args.frame, args.speed)
    success = False
    try:
        if len(args.values) == 6:
            # Absolute mode: the six numbers are the whole target pose.
            target = make_pose(*args.values)
        else:
            # Relative mode: offset (dx, dy, dz) from where the TCP is now; no numbers = 1 cm up the Z axis.
            dx, dy, dz = args.values if args.values else (0.0, 0.0, 0.01)
            # Start from the TCP's current pose (read from TF) so its orientation is kept.
            target = planner.current_eef_pose()
            planner.get_logger().info(f"Current TCP in {args.frame}: {fmt(target)}")
            target.position.x += dx
            target.position.y += dy
            target.position.z += dz
        planner.get_logger().info(f"Target TCP in {args.frame}: {fmt(target)}")

        # Ask MoveIt for the straight-line path.
        plan, fraction = planner.plan_cartesian_path(target)
        print(f"Path completion: {fraction * 100:.1f}%")

        # How big is the largest single-joint step along the path? (0.0 if there is no plan.)
        step = max_joint_step_deg(plan) if plan is not None else 0.0
        print(f"Largest joint step between points: {step:.1f} deg")
        # 0.999 instead of 1.0 tolerates rounding: the whole line must have been planned.
        if plan is not None and fraction >= 0.999 and step > args.max_joint_step:
            # Complete path, but with a suspicious jump: do not move.
            planner.get_logger().error(
                f"Path jumps {step:.1f} deg between two points (limit {args.max_joint_step:.0f}): "
                "probably an IK branch flip near a singularity. Execution canceled.")
        elif plan is not None and fraction >= 0.999:
            # Complete and smooth path: execute it.
            success = planner.execute_plan(plan)
        else:
            # Incomplete path (blocked, unreachable, ...): do not move.
            planner.get_logger().error("Cartesian path planning was incomplete. Execution canceled.")
        print("Execution successful:", success)
    except Exception as error:  # noqa: BLE001
        # Any unexpected error (for example no TF) is logged and counts as failure; the robot is not moved
        # by a failure that happens before execute_plan().
        planner.get_logger().error(f"Error: {error}")
    finally:
        # Always release the node and shut ROS down, whatever happened above.
        planner.destroy_node()
        rclpy.shutdown()
    # The exit code is how the Gungnir client learns the result: 0 = success, 1 = failure.
    sys.exit(0 if success else 1)


# Run main() only when the file is started as a program (not when it is imported).
if __name__ == "__main__":
    main()