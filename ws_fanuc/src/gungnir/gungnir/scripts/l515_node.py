#!/usr/bin/env python3
"""Publish the physical L515 (through your l515.L515 wrapper) on the same topics as the simulated camera.

  /camera/depth/image_rect_raw  32FC1 metres, 0 = no return     /camera/depth/camera_info
  /camera/color/image_raw       rgb8                            /camera/color/camera_info   (K of the colour stream)
All in camera_depth_optical_frame.

Parameters: frame_id, max_rate (Hz, default 10), swap_rb (default false, same meaning as in CameraFeed),
            width/height/fps (default 640/480/30), l515_path (folder containing l515.py, optional),
            fake_dir (replay depth.npy/k.npy/rgb.npy from a saved capture instead of opening the camera)."""
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

try:
    from rclpy.executors import ExternalShutdownException
except ImportError:                                   # pragma: no cover
    class ExternalShutdownException(Exception):
        pass


class FakeL515:
    """Replays a capture saved by camera_snapshot.py. Same interface as the parts of L515 used here."""

    def __init__(self, directory, fps):
        d = Path(directory).expanduser()
        self.depth_image = np.load(d / "depth.npy").astype(np.float32)
        self._k = np.load(d / "k.npy").astype(np.float32)
        rgb_file = d / "rgb.npy"
        if rgb_file.exists():
            self.color_image = np.ascontiguousarray(np.load(rgb_file)[..., ::-1])   # BGR, like the real wrapper
        else:
            self.color_image = np.full((*self.depth_image.shape, 3), 160, np.uint8)  # no colour saved: grey
        self._period = 1.0 / max(fps, 0.1)

    def capture_image(self):
        time.sleep(self._period)
        return True

    def get_k(self):
        return self._k.copy()

    def close(self):
        pass


class L515Node(Node):
    def __init__(self):
        super().__init__("l515_node")
        param = lambda name, default: self.declare_parameter(name, default).value  # noqa: E731
        self.frame_id = str(param("frame_id", "camera_depth_optical_frame"))
        self.max_rate = float(param("max_rate", 10.0))
        self.swap_rb = bool(param("swap_rb", False))
        self.l515_path = str(param("l515_path", ""))
        self.fake_dir = str(param("fake_dir", ""))
        self.width, self.height, self.fps = int(param("width", 640)), int(param("height", 480)), int(param("fps", 30))
        self.fatal = False

        self.pub_depth = self.create_publisher(Image, "/camera/depth/image_rect_raw", 2)
        self.pub_color = self.create_publisher(Image, "/camera/color/image_raw", 2)
        self.pub_info_d = self.create_publisher(CameraInfo, "/camera/depth/camera_info", 2)
        self.pub_info_c = self.create_publisher(CameraInfo, "/camera/color/camera_info", 2)

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="l515_capture")
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5.0)

    def _open(self):
        if self.fake_dir:
            self.get_logger().info(f"Replaying capture from {self.fake_dir} (no hardware)")
            return FakeL515(self.fake_dir, self.max_rate)
        if self.l515_path:
            sys.path.insert(0, os.path.expanduser(self.l515_path))
        from l515 import L515                      # the wrapper from IntelRealsense_Driver
        return L515(width=self.width, height=self.height, fps=self.fps)

    def _run(self):
        log = self.get_logger()
        cam, k, failures, last_pub = None, None, 0, 0.0
        min_interval = 1.0 / max(self.max_rate, 0.1)
        while not self._stop.is_set():
            try:
                if cam is None:
                    cam, k = self._open(), None
                    log.info("Camera opened")
                cam.capture_image()
                now = time.monotonic()
                if now - last_pub < min_interval:      # keep capturing, publish at most max_rate
                    continue
                last_pub = now
                depth = np.nan_to_num(np.asarray(cam.depth_image, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                raw = np.asarray(cam.color_image)
                rgb = np.ascontiguousarray(raw if self.swap_rb else raw[..., ::-1])    # same logic as CameraFeed
                if k is None:
                    k = np.asarray(cam.get_k(), dtype=np.float32)
                self._publish(self.get_clock().now().to_msg(), depth, rgb, k)
                failures = 0
            except ImportError as exc:
                log.error(f"Cannot import l515 ({exc}). Pass l515_path:=<folder containing l515.py>, or install "
                          "the IntelRealsense_Driver package for the python ROS uses (/usr/bin/python3).")
                self.fatal = True
                break
            except Exception as exc:               # noqa: BLE001
                failures += 1
                log.warn(f"{type(exc).__name__}: {exc}  (failure {failures}). Is the camera connected, not open in "
                         "another program, and is L515_NATIVE_PATH set?", throttle_duration_sec=5.0)
                if cam is not None and failures >= 5:
                    try:
                        cam.close()
                    except Exception:              # noqa: BLE001
                        pass
                    cam = None
                time.sleep(min(2.0, 0.2 * failures))
        if cam is not None:
            try:
                cam.close()
            except Exception:                      # noqa: BLE001
                pass
        if self.fatal and rclpy.ok():
            rclpy.shutdown()

    def _publish(self, stamp, depth, rgb, k):
        h, w = depth.shape
        d = Image()
        d.header.stamp, d.header.frame_id = stamp, self.frame_id
        d.height, d.width, d.encoding, d.is_bigendian, d.step = h, w, "32FC1", 0, w * 4
        d.data = depth.tobytes()
        c = Image()
        c.header.stamp, c.header.frame_id = stamp, self.frame_id
        c.height, c.width, c.encoding, c.is_bigendian, c.step = h, w, "rgb8", 0, w * 3
        c.data = rgb.tobytes()
        fx, fy, cx, cy = (float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]))
        info = CameraInfo()
        info.header.stamp, info.header.frame_id = stamp, self.frame_id
        info.height, info.width, info.distortion_model = h, w, "plumb_bob"
        info.d = [0.0] * 5
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.pub_depth.publish(d)
        self.pub_color.publish(c)
        self.pub_info_d.publish(info)
        self.pub_info_c.publish(info)


def main():
    rclpy.init()
    node = L515Node()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(1 if node.fatal else 0)


if __name__ == "__main__":
    main()
