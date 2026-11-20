#!/usr/bin/env python3
"""Visualisation helpers, identical for the simulated and the physical camera.

Subscribes: /camera/depth/image_rect_raw (32FC1 m or 16UC1 mm), /camera/color/image_raw (rgb8/bgr8),
            /camera/color/camera_info
Publishes : /camera/depth/color/points      coloured point cloud in the optical frame
            /camera/depth/image_colorized   colour-coded depth (rgb8), for RViz
Parameters: rate (Hz, 5), stride (cloud subsampling, 2), depth_min/depth_max (colour scale, m; 0.25/0.60),
            cloud_max (largest depth kept in the cloud, m; 9.0)."""
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField


def decode(msg):
    """Depth -> float32 metres; colour -> RGB uint8."""
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    if msg.encoding == "32FC1":
        return np.ascontiguousarray(raw[:, : msg.width * 4]).view(np.float32).reshape(msg.height, msg.width)
    if msg.encoding == "16UC1":
        mm = np.ascontiguousarray(raw[:, : msg.width * 2]).view(np.uint16).reshape(msg.height, msg.width)
        return mm.astype(np.float32) * 0.001
    if msg.encoding in ("rgb8", "bgr8"):
        img = raw[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
        return np.ascontiguousarray(img if msg.encoding == "rgb8" else img[..., ::-1])
    raise ValueError(f"unsupported image encoding {msg.encoding}")


def jet(t):
    r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


class CameraViz(Node):
    def __init__(self):
        super().__init__("camera_viz")
        param = lambda name, default: self.declare_parameter(name, default).value  # noqa: E731
        rate = float(param("rate", 5.0))
        self.stride = max(1, int(param("stride", 2)))
        self.dmin, self.dmax = float(param("depth_min", 0.25)), float(param("depth_max", 0.60))
        self.cloud_max = float(param("cloud_max", 9.0))
        self.depth_msg = self.color_msg = self.k = None
        self.fresh = False
        self.create_subscription(Image, "/camera/depth/image_rect_raw", self.on_depth, 2)
        self.create_subscription(Image, "/camera/color/image_raw", self.on_color, 2)
        self.create_subscription(CameraInfo, "/camera/color/camera_info", self.on_info, 2)
        self.pub_cloud = self.create_publisher(PointCloud2, "/camera/depth/color/points", 2)
        self.pub_vis = self.create_publisher(Image, "/camera/depth/image_colorized", 2)
        self.create_timer(1.0 / max(rate, 0.1), self.tick)

    def on_depth(self, msg):
        self.depth_msg, self.fresh = msg, True

    def on_color(self, msg):
        self.color_msg = msg

    def on_info(self, msg):
        self.k = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def tick(self):
        if not self.fresh or self.depth_msg is None or self.color_msg is None or self.k is None:
            return
        self.fresh = False
        try:
            depth, color = decode(self.depth_msg), decode(self.color_msg)
        except ValueError as exc:
            self.get_logger().error(str(exc), throttle_duration_sec=10.0)
            return
        if color.shape[:2] != depth.shape:
            self.get_logger().warn("colour and depth sizes differ; skipping", throttle_duration_sec=10.0)
            return
        header = self.depth_msg.header
        h, w = depth.shape

        # colour-coded depth
        valid = np.isfinite(depth) & (depth > 0)
        t = np.clip((np.where(valid, depth, 0.0) - self.dmin) / max(self.dmax - self.dmin, 1e-6), 0, 1)
        vis = jet(t)
        vis[~valid] = 0
        img = Image()
        img.header = header
        img.height, img.width, img.encoding, img.is_bigendian, img.step = h, w, "rgb8", 0, w * 3
        img.data = np.ascontiguousarray(vis).tobytes()
        self.pub_vis.publish(img)

        # coloured point cloud (subsampled)
        s = self.stride
        d, c = depth[::s, ::s], color[::s, ::s]
        v, u = np.mgrid[0:h:s, 0:w:s]
        ok = np.isfinite(d) & (d > 0) & (d <= self.cloud_max)
        z = d[ok]
        fx, fy, cx, cy = self.k[0, 0], self.k[1, 1], self.k[0, 2], self.k[1, 2]
        px = c[ok].astype(np.uint32)
        cloud = np.empty(z.size, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")])
        cloud["x"] = (u[ok] - cx) * z / fx
        cloud["y"] = (v[ok] - cy) * z / fy
        cloud["z"] = z
        cloud["rgb"] = (px[:, 0] << 16) | (px[:, 1] << 8) | px[:, 2]
        pc = PointCloud2()
        pc.header = header
        pc.height, pc.width = 1, int(z.size)
        pc.fields = [PointField(name=n, offset=o, datatype=PointField.FLOAT32, count=1)
                     for n, o in (("x", 0), ("y", 4), ("z", 8), ("rgb", 12))]
        pc.is_bigendian, pc.point_step, pc.row_step, pc.is_dense = False, 16, 16 * int(z.size), True
        pc.data = cloud.tobytes()
        self.pub_cloud.publish(pc)


def main():
    rclpy.init()
    node = CameraViz()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
