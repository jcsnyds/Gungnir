#!/usr/bin/env python3
"""Makes the Gazebo camera look like the real L515 wrapper (l515.py): depth as float32 metres with 0 = no return,
colour rgb8, K from the colour stream, everything in camera_depth_optical_frame, plus an optional depth-noise model.

Publishes
  /camera/depth/image_rect_raw  32FC1 m   depth WITH the noise model (identical to the clean frame if noise is off)
  /camera/depth/image_clean     32FC1 m   the same frame without noise (ground truth, for evaluation)
  /camera/color/image_raw, /camera/depth/camera_info, /camera/color/camera_info

Noise model (parameters in config/sim_camera_noise.yaml; the defaults are placeholders, not measured L515 values):
  1. depth edges: pixels at depth discontinuities are dropped or become 'flying pixels' between foreground and background
  2. Gaussian axial noise, sigma = sigma_base + sigma_slope * depth
  3. grazing-angle dropout: surfaces seen at a steep angle return nothing (probability ramps up between two angles)
  4. random single-pixel dropout"""
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


def neighbours(a):
    """The 8 shifted copies of a 2D array (edge-padded, so the image border creates no false edges)."""
    h, w = a.shape
    p = np.pad(a, 1, mode="edge")
    for dy in range(3):
        for dx in range(3):
            if dy != 1 or dx != 1:
                yield p[dy:dy + h, dx:dx + w]


class DepthNoise:
    def __init__(self, node):
        param = lambda name, default: node.declare_parameter(name, default).value  # noqa: E731
        self.enabled = bool(param("noise_enabled", True))
        seed = int(param("seed", 42))
        self.rng = np.random.default_rng(None if seed < 0 else seed)
        self.sigma_base = float(param("sigma_base", 0.001))
        self.sigma_slope = float(param("sigma_slope", 0.004))
        self.dropout = float(param("dropout_prob", 0.002))
        self.g0 = float(param("grazing_start_deg", 75.0))
        self.g1 = float(param("grazing_end_deg", 88.0))
        self.edge_jump = float(param("edge_jump", 0.02))
        self.edge_drop = float(param("edge_dropout_prob", 0.5))
        self.fly = float(param("flying_pixel_prob", 0.3))
        log = node.get_logger()
        if self.enabled:
            log.info(f"Depth noise ON: sigma = {self.sigma_base * 1000:.1f} mm + {self.sigma_slope * 1000:.1f} mm/m * depth, "
                     f"dropout {self.dropout:.3f}, grazing {self.g0:.0f}-{self.g1:.0f} deg, "
                     f"edges (drop {self.edge_drop:.2f}, flying {self.fly:.2f}), seed {seed}")
        else:
            log.info("Depth noise OFF (perfect depth)")

    def apply(self, z, K, zmin, zmax):
        """z: float32 HxW metres, 0 = no return (not modified). Returns a noisy copy."""
        if not self.enabled:
            return z
        rng, shape = self.rng, z.shape
        valid = z > 0
        out = z.copy()
        drop = np.zeros(shape, bool)

        # 1. depth edges: dropped, or a flying pixel somewhere between the local minimum and maximum depth
        if self.edge_jump > 0 and (self.edge_drop > 0 or self.fly > 0):
            lo = np.where(valid, z, np.float32(np.inf))
            hi = np.where(valid, z, np.float32(0.0))
            lo_min, hi_max = lo.copy(), hi.copy()
            for nb in neighbours(lo):
                np.minimum(lo_min, nb, out=lo_min)
            for nb in neighbours(hi):
                np.maximum(hi_max, nb, out=hi_max)
            edge = valid & ((hi_max - lo_min) > self.edge_jump)
            r = rng.random(shape, dtype=np.float32)
            drop |= edge & (r < self.edge_drop)
            fly = edge & (r >= self.edge_drop) & (r < self.edge_drop + self.fly)
            if fly.any():
                u = rng.random(int(fly.sum()), dtype=np.float32)
                out[fly] = lo_min[fly] + u * (hi_max[fly] - lo_min[fly])

        # 2. Gaussian axial noise
        out += rng.standard_normal(shape, dtype=np.float32) * (self.sigma_base + self.sigma_slope * z)

        # 3. grazing-angle dropout (surface normals from the depth gradient)
        if K is not None and self.g1 > self.g0:
            h, w = shape
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            zz = np.where(valid, z, np.float32(np.nan))
            dzdv, dzdu = np.gradient(zz)
            xo = ((np.arange(w, dtype=np.float32) - cx) / fx)[None, :]
            yo = ((np.arange(h, dtype=np.float32) - cy) / fy)[:, None]
            Pu = np.stack((zz / fx + xo * dzdu, yo * dzdu, dzdu), axis=-1)
            Pv = np.stack((xo * dzdv, zz / fy + yo * dzdv, dzdv), axis=-1)
            P = np.stack((xo * zz, yo * zz, zz), axis=-1)
            n = np.cross(Pu, Pv)
            with np.errstate(invalid="ignore", divide="ignore"):
                cosang = np.abs(np.sum(n * P, axis=-1)) / (np.linalg.norm(n, axis=-1) * np.linalg.norm(P, axis=-1))
                theta = np.degrees(np.arccos(np.clip(cosang, 0.0, 1.0)))
                p = np.clip((theta - self.g0) / (self.g1 - self.g0), 0.0, 1.0)
            drop |= rng.random(shape, dtype=np.float32) < np.nan_to_num(p, nan=0.0)

        # 4. random dropout
        if self.dropout > 0:
            drop |= rng.random(shape, dtype=np.float32) < self.dropout

        out = out.astype(np.float32, copy=False)
        out[drop | ~valid | (out < zmin) | (out > zmax)] = 0.0
        return out


class SimCameraAdapter(Node):
    def __init__(self):
        super().__init__("sim_camera_adapter")
        self.frame_id = self.declare_parameter("frame_id", "camera_depth_optical_frame").value
        self.min_depth = float(self.declare_parameter("min_depth", 0.25).value)
        self.max_depth = float(self.declare_parameter("max_depth", 9.0).value)
        self.noise = DepthNoise(self)
        self.K = None
        self._warned = False

        self.pub_depth = self.create_publisher(Image, "/camera/depth/image_rect_raw", 5)
        self.pub_clean = self.create_publisher(Image, "/camera/depth/image_clean", 5)
        self.pub_color = self.create_publisher(Image, "/camera/color/image_raw", 5)
        self.pub_info_d = self.create_publisher(CameraInfo, "/camera/depth/camera_info", 5)
        self.pub_info_c = self.create_publisher(CameraInfo, "/camera/color/camera_info", 5)

        self.create_subscription(Image, "/sim_camera/depth_image", self.on_depth, 5)
        self.create_subscription(Image, "/sim_camera/image", self.on_color, 5)
        self.create_subscription(CameraInfo, "/sim_camera/camera_info", self.on_info, 5)

    def make_msg(self, depth, header):
        out = Image()
        out.header.stamp = header.stamp
        out.header.frame_id = self.frame_id
        out.height, out.width = depth.shape
        out.encoding = "32FC1"
        out.is_bigendian = 0
        out.step = out.width * 4
        out.data = depth.tobytes()
        return out

    def on_depth(self, msg):
        if msg.encoding != "32FC1":
            if not self._warned:
                self.get_logger().warn(f"Unexpected depth encoding {msg.encoding}; expected 32FC1")
                self._warned = True
            return
        depth = np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(msg.height, msg.width)
        invalid = ~np.isfinite(depth) | (depth < self.min_depth) | (depth > self.max_depth)
        clean = np.where(invalid, np.float32(0.0), depth)
        t0 = time.perf_counter()
        noisy = self.noise.apply(clean, self.K, self.min_depth, self.max_depth)
        if self.noise.enabled:
            self.get_logger().info(f"noise model: {(time.perf_counter() - t0) * 1000:.0f} ms per frame",
                                   throttle_duration_sec=30.0)
        self.pub_clean.publish(self.make_msg(clean, msg.header))
        self.pub_depth.publish(self.make_msg(noisy, msg.header))

    def on_color(self, msg):
        msg.header.frame_id = self.frame_id
        self.pub_color.publish(msg)

    def on_info(self, msg):
        msg.header.frame_id = self.frame_id
        self.K = np.array(msg.k, dtype=np.float32).reshape(3, 3)
        self.pub_info_d.publish(msg)
        self.pub_info_c.publish(msg)


def main():
    rclpy.init()
    node = SimCameraAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
