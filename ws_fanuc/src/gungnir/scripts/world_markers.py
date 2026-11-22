#!/usr/bin/env python3
"""Draw the Gazebo world's models in RViz: reads the SDF and publishes /sim_world_markers (MarkerArray, 1 Hz).
Supports box / cylinder / sphere visuals (what make_world.py generates). Parameter: world_file."""
import sys
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def pose_matrix(elem):
    vals = [float(v) for v in (elem.text.split() if elem is not None and elem.text else [])]
    vals += [0.0] * (6 - len(vals))
    T = np.eye(4)
    T[:3, :3] = rpy_matrix(*vals[3:6])
    T[:3, 3] = vals[:3]
    return T


def mat_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return (0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s)
    if R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return ((R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s)
    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return ((R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s)


def floats(elem):
    return [float(v) for v in elem.text.split()]


class WorldMarkers(Node):
    def __init__(self):
        super().__init__("world_markers")
        path = self.declare_parameter("world_file", "").value
        try:
            self.markers = self.build(path)
        except Exception as exc:                                  # noqa: BLE001
            self.get_logger().error(f"Cannot read world file '{path}': {exc}")
            sys.exit(1)
        self.get_logger().info(f"{len(self.markers.markers)} world markers from {path}")
        self.pub = self.create_publisher(MarkerArray, "/sim_world_markers", 1)
        self.create_timer(1.0, lambda: self.pub.publish(self.markers))

    def build(self, path):
        world = ET.parse(path).getroot().find("world")
        out, n = MarkerArray(), 0
        for model in world.findall("model"):
            Tm = pose_matrix(model.find("pose"))
            for link in model.findall("link"):
                Tl = Tm @ pose_matrix(link.find("pose"))
                for visual in link.findall("visual"):
                    T = Tl @ pose_matrix(visual.find("pose"))
                    geom = visual.find("geometry")
                    m = Marker()
                    if geom.find("box/size") is not None:
                        m.type = Marker.CUBE
                        m.scale.x, m.scale.y, m.scale.z = floats(geom.find("box/size"))
                    elif geom.find("cylinder") is not None:
                        m.type = Marker.CYLINDER
                        r, length = float(geom.find("cylinder/radius").text), float(geom.find("cylinder/length").text)
                        m.scale.x, m.scale.y, m.scale.z = 2 * r, 2 * r, length
                    elif geom.find("sphere") is not None:
                        m.type = Marker.SPHERE
                        r = float(geom.find("sphere/radius").text)
                        m.scale.x = m.scale.y = m.scale.z = 2 * r
                    else:
                        continue
                    col = visual.find("material/diffuse")
                    if col is None:
                        col = visual.find("material/ambient")
                    rgba = (floats(col) + [1.0] * 4)[:4] if col is not None and col.text else [0.7, 0.7, 0.7, 1.0]
                    m.color.r, m.color.g, m.color.b, m.color.a = rgba
                    m.header.frame_id = "world"
                    m.ns, m.id, m.action = model.get("name", "model"), n, Marker.ADD
                    n += 1
                    m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, T[:3, 3])
                    m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = \
                        map(float, mat_to_quat(T[:3, :3]))
                    out.markers.append(m)
        return out


def main():
    rclpy.init()
    node = WorldMarkers()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
