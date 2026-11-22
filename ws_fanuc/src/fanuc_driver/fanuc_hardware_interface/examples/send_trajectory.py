# SPDX-FileCopyrightText: 2025, FANUC America Corporation
# SPDX-FileCopyrightText: 2025, FANUC CORPORATION
#
# SPDX-License-Identifier: Apache-2.0

import rclpy
import math
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class SineTrajectoryPublisher(Node):
    def __init__(self):
        super().__init__("sine_trajectory_publisher")
        self.future = rclpy.task.Future()
        self.joint_names = []
        self.initial_positions = []

        self.joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self.joint_state_callback, 10
        )

        self.traj_pub = self.create_publisher(
            JointTrajectory, "/joint_trajectory_controller/joint_trajectory", 10
        )

        self.timer = self.create_timer(1.0, self.publish_trajectory_once)

    def joint_state_callback(self, msg):
        self.joint_names = list(msg.name)
        self.initial_positions = list(msg.position)

    def publish_trajectory_once(self):
        if len(self.initial_positions) == 0:
            self.get_logger().info("Waiting for initial joint states...")
            return

        self.get_logger().info(
            f"Captured initial joint states: {self.initial_positions}"
        )

        # Parameters
        amplitude_rad = math.radians(20)
        period = 3.0  # seconds
        steps = 1000  # discretization

        msg = JointTrajectory()
        msg.joint_names = self.joint_names

        for i in range(steps + 1):
            t = i * period / steps
            phase = (2.0 * math.pi / period) * t
            point = JointTrajectoryPoint()
            point.time_from_start.sec = int(t)
            point.time_from_start.nanosec = int((t - int(t)) * 1e9)
            point.positions = [
                pos + amplitude_rad * (1.0 - math.cos(phase))
                for pos in self.initial_positions
            ]
            point.velocities = [
                amplitude_rad * (2.0 * math.pi / period) * math.sin(phase)
                for _ in self.initial_positions
            ]
            msg.points.append(point)

        self.traj_pub.publish(msg)
        self.get_logger().info("Published sinusoidal trajectory.")
        self.timer.cancel()  # Cancel timer to prevent sending more than one trajectory
        self.future.set_result(True)


def main():
    rclpy.init()
    node = SineTrajectoryPublisher()
    rclpy.spin_until_future_complete(node, node.future)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
