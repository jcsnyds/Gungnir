#!/usr/bin/env python3
"""Helpers shared by the command-line tools: joint limits read from the robot description that is running."""
import math
import xml.etree.ElementTree as ET

import rclpy
from rcl_interfaces.srv import GetParameters

JOINTS = ("J1", "J2", "J3", "J4", "J5", "J6")


def joint_limits_deg(node, joints=JOINTS, timeout=5.0):
    """{joint: (lower, upper)} in degrees, from /robot_state_publisher's robot_description (+-360 if unavailable)."""
    limits = {j: (-360.0, 360.0) for j in joints}
    cli = node.create_client(GetParameters, "/robot_state_publisher/get_parameters")
    if not cli.wait_for_service(timeout_sec=timeout):
        node.get_logger().warn("robot_state_publisher not found: joint limits unknown, using +-360 deg")
        return limits
    req = GetParameters.Request()
    req.names = ["robot_description"]
    fut = cli.call_async(req)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=timeout)
    try:
        root = ET.fromstring(fut.result().values[0].string_value)
    except Exception as exc:                                    # noqa: BLE001
        node.get_logger().warn(f"cannot read robot_description ({exc}): using +-360 deg")
        return limits
    for joint in root.findall("joint"):
        lim = joint.find("limit")
        if joint.get("name") in limits and lim is not None and lim.get("lower") is not None:
            limits[joint.get("name")] = (math.degrees(float(lim.get("lower"))), math.degrees(float(lim.get("upper"))))
    return limits
