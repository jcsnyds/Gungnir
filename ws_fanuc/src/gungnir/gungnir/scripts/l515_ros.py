#!/usr/bin/env python3
"""Drop-in replacement for l515.L515 that reads the SIMULATED camera over ROS 2.

Same interface as the real wrapper for what CameraFeed uses:
  capture_image(), depth_image (float32 metres, 0 = no return), color_image (BGR uint8),
  get_k(), get_depth(), get_color() (RGB), get_RGBD(), get_pointcloud(min, max), close().
Needs the sim running (ros2 launch gungnir gungnir.launch.py mode:=sim) and ROS sourced."""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np


class L515Error(RuntimeError):
    """Raised when no simulated camera data is available."""


def _decode(msg) -> np.ndarray:
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    if msg.encoding == "32FC1":
        return np.ascontiguousarray(raw[:, : msg.width * 4]).view(np.float32).reshape(
            msg.height, msg.width).copy()
    if msg.encoding in ("rgb8", "bgr8"):
        img = raw[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
        if msg.encoding == "rgb8":
            img = img[..., ::-1]               # the real wrapper delivers BGR
        return np.ascontiguousarray(img)
    raise L515Error(f"Unsupported image encoding: {msg.encoding}")


class L515:
    def __init__(self, clipping_distance: float = 9.0, width: int = 640, height: int = 480, fps: int = 30,
                 library_path=None, timeout: float = 5.0,
                 depth_topic: str = "/camera/depth/image_rect_raw",
                 color_topic: str = "/camera/color/image_raw",
                 info_topic: str = "/camera/color/camera_info") -> None:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import QoSProfile
        from sensor_msgs.msg import CameraInfo, Image

        self.clipping_distance = float(clipping_distance)
        self.width, self.height, self.fps = int(width), int(height), int(fps)
        self.depth_scale = 1.0                 # sim depth is already in metres
        self.depth_image: Optional[np.ndarray] = None
        self.color_image: Optional[np.ndarray] = None
        self.running = False
        self.pipeline = None
        self._timeout = float(timeout)
        self._cond = threading.Condition()
        self._depth = self._color = self._k = None
        self._depth_seq = self._served = 0

        self._rclpy = rclpy
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init()
        self._node = rclpy.create_node(f"sim_l515_{id(self) & 0xFFFF}")
        qos = QoSProfile(depth=2)
        self._node.create_subscription(Image, depth_topic, self._on_depth, qos)
        self._node.create_subscription(Image, color_topic, self._on_color, qos)
        self._node.create_subscription(CameraInfo, info_topic, self._on_info, qos)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True, name="sim_l515_spin")
        self._thread.start()
        self.running = True
        self.pipeline = self

    # -- ROS callbacks (executor thread) ------------------------------------------------
    def _on_depth(self, msg) -> None:
        depth = _decode(msg)
        with self._cond:
            self._depth = depth
            self._depth_seq += 1
            self._cond.notify_all()

    def _on_color(self, msg) -> None:
        color = _decode(msg)
        with self._cond:
            self._color = color

    def _on_info(self, msg) -> None:
        k = np.array(msg.k, dtype=np.float32).reshape(3, 3)
        with self._cond:
            self._k = k

    # -- camera API -----------------------------------------------------------------------
    def capture_image(self) -> bool:
        """Block until a new depth frame (plus colour and camera_info) is available."""
        if not self.running:
            raise L515Error("Simulated camera is closed.")
        with self._cond:
            ready = lambda: (self._depth is not None and self._color is not None  # noqa: E731
                             and self._k is not None and self._depth_seq != self._served)
            if not self._cond.wait_for(ready, timeout=self._timeout):
                raise L515Error(
                    "No frames from the simulated camera. Is the sim running "
                    "(gungnir.launch.py mode:=sim) and /camera/depth/image_rect_raw publishing?")
            self._served = self._depth_seq
            self.depth_image, self.color_image = self._depth, self._color
        self.height, self.width = self.depth_image.shape
        return True

    def get_depth(self) -> np.ndarray:
        if self.depth_image is None:
            raise L515Error("No depth image has been captured yet.")
        return self.depth_image

    def get_color(self) -> np.ndarray:
        """Most recent colour image in RGB order (same convention as the real wrapper)."""
        if self.color_image is None:
            raise L515Error("No color image has been captured yet.")
        return self.color_image[..., ::-1]

    def get_k(self) -> np.ndarray:
        with self._cond:
            if self._k is None:
                raise L515Error("No camera_info received yet.")
            return self._k.copy()

    def get_RGBD(self) -> np.ndarray:
        return np.dstack((self.color_image[..., ::-1], self.depth_image))

    def get_pointcloud(self, MIN_DEPTH, MAX_DEPTH) -> np.ndarray:
        """Nx3 points (x right, y down, z forward, metres) in the optical frame."""
        h, w = self.depth_image.shape
        K = self.get_k()
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        valid = (self.depth_image > MIN_DEPTH) & (self.depth_image < MAX_DEPTH)
        z = self.depth_image[valid]
        x = (u[valid] - cx) * z / fx
        y = (v[valid] - cy) * z / fy
        return np.stack((x, y, z), axis=-1).astype(np.float32)

    def close(self) -> None:
        if not self.running:
            return
        self.running = False
        self.pipeline = None
        try:
            self._executor.shutdown()
            self._thread.join(timeout=2.0)
            self._node.destroy_node()
        finally:
            if self._owns_rclpy and self._rclpy.ok():
                self._rclpy.shutdown()

    def __enter__(self) -> "L515":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
