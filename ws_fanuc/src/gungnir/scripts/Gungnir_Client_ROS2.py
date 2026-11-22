#!/usr/bin/env python3
"""Gungnir_Client: browser UI for the L515 + Contact-GraspNet pipeline. Runs on the camera computer.

    python Gungnir_Client.py <gpu-computer-ip>        # then open http://localhost:8000

Start ContactGraspnet_Server.py on the GPU computer first (it prints this exact command). To view the page
from your own computer: ssh -L 8000:localhost:8000 <user>@<camera-computer>, then browse to http://localhost:8000.

The page shows the live RGB and depth streams, a prompt box (saved with each capture, not used by the model
yet), and a Capture & Run button. The model output appears next to the stream and in a separate output window
(http://localhost:8000/?view=results) that follows the latest result.

Grasp approval and robot motion: after every result the client picks the highest-scoring grasp, draws it in
magenta on top of the other grasps (2D overlay and 3D scene), and shows a "Proposed robot move" panel with the
target pose. Nothing moves until the user presses "Approve & move robot" (and confirms a browser dialog). On
approval this program starts the Cartesian path planner script (--planner-script, see Cartesian_Path_Planner.py)
as a separate process and passes it the target pose as six numbers: x y z roll pitch yaw (metres / radians)
expressed in --move-frame. The planner script itself is NOT modified. A proposal can be approved only once, and
it expires after --approval-timeout seconds, because the camera may be mounted on the robot and a grasp pose is
only valid for the arm position at the moment of capture.

Architecture: this process owns the camera and the FastSAM segmentation model and never imports TensorFlow.
Each capture is uploaded over HTTP to ContactGraspnet_Server (which keeps the model loaded on the GPU
computer); the result is downloaded and drawn here. The camera can only be opened by one process: stop Odin.py
before starting this. There is no authentication on the web page: the default bind address is localhost; use
--host 0.0.0.0 only on a network you trust. Because the page can now MOVE A ROBOT, never expose it on an
untrusted network, and keep the emergency stop within reach.

Protocol with ContactGraspnet_Server (full description at the top of ContactGraspnet_Server.py): plain HTTP.
The client polls GET /health once a second to learn the server's state (starting / ready / busy), and sends
each capture as POST /infer with the input .npz as the body and the depth crop in X-Z-Min / X-Z-Max headers;
the reply body is the result .npz. POST /restart asks the server to restart its model. The class ServerLink
below implements the client side.

Server address: the first argument (or $GUNGNIR_SERVER); "192.168.1.50", "192.168.1.50:8765" and
"http://192.168.1.50:8765" all work. If the server was started with a token, pass --token (or $GUNGNIR_TOKEN).

AI use: This file was originally developed with assistance from Claude (Anthropic; Claude Sonnet 5.5 via
claude.ai) in October-November 2026, based on the author's requirements and existing Gungnir code. In a
later revision, ChatGPT (OpenAI; GPT-6) modified only the camera input path: the former direct L515 capture
thread was replaced with ROS 2 subscriptions to the image and camera-info topics published by l515_node. The
ServerLink class, HTTP protocol, Flask routes, segmentation, inference data format, and browser interface
were intentionally left unchanged in that revision.
ChatGPT (OpenAI; GPT-6) later corrected the ROS executor member to avoid colliding with rclpy Node.executor,
and hardened startup cleanup if camera or server initialization fails.
In a later revision, Claude (Anthropic; Claude Sonnet 5.5 via claude.ai) added: selection of the
best-scoring grasp; its highlighting in the 2D overlay and 3D scene; the "Proposed robot move" approval panel
and its /api/move/* routes; the conversion of a grasp pose (4x4 matrix) into the x y z roll pitch yaw
arguments of Cartesian_Path_Planner.py (the inverse of that script's rpy_to_quaternion); and the code that
launches the planner as a subprocess after the user approves. Claude also fixed one existing bug in
draw_overlay() (the segmentation blend result was discarded). The author gave the requirements and is
responsible for reviewing and testing this revision; no claim is made that it has been tested on the real
robot or hardware.
Claude later made a documentation-only edit: the stale "SAM2" mentions in the docstring and comments now say
FastSAM, which is the model SegmentationModel actually loads. No code behaviour was changed by that edit.
Claude (Anthropic; Claude Sonnet 5.5 via claude.ai), later in November 2026, added the approach move: the
--approach-distance and --move-fixed-frame options, their pass-through to Cartesian_Path_Planner.py in
run_move(), and one sentence in the approval panel saying the robot will first stop that far back from the
grasp. Claude checked the edited file compiles and that the planner's command line accepts the arguments this
file builds (with stand-in ROS modules); it has NOT run the client with ROS 2, a camera, the model server or a
robot.
Reviewed and tested by: Ryan Poliquin (revisions up to and including the ROS 2 camera change). The
grasp-approval / robot-motion revision and the approach-move addition have NOT yet been reviewed or tested by
the author.
"""

# Store type annotations as text instead of evaluating them when the file is loaded, so annotations such as
# `ServerLink` used before the class is fully defined work on any Python 3 version.
from __future__ import annotations

import argparse  # reads the command-line options (server address, --port, --min-depth, ...)
import json  # encodes/decodes JSON messages and the saved meta.json files
import logging  # controls how much the web server prints to the terminal
import math  # asin / atan2 for turning a rotation matrix into roll-pitch-yaw angles
import os  # environment variables ($GUNGNIR_SERVER, $GUNGNIR_TOKEN)
import re  # regular expressions, used to validate run ids
import signal  # lets us react to SIGTERM / SIGHUP so cleanup runs on shutdown
import subprocess  # launches Cartesian_Path_Planner.py as a separate process after the user approves a move
import sys  # the error stream (sys.stderr) and the Python interpreter path (sys.executable)
import threading  # background threads, locks, and events
import time  # timestamps, sleeping, and run-id generation
import urllib.error  # HTTPError: raised by urllib for 4xx/5xx replies
import urllib.request  # the HTTP client used to talk to ContactGraspnet_Server (standard library)
from pathlib import Path  # path objects that behave the same on any operating system
from urllib.parse import urlparse  # splits an address such as "192.168.1.50:8765" into its parts

import cv2  # OpenCV: JPEG encoding, colour maps, and drawing lines/text on images
import numpy as np  # arrays: images, point clouds, and grasp matrices
import torch  # needed by the FastSAM segmentation model
import rclpy  # ROS 2 Python client library used to receive the L515 node's topics
from rclpy.node import Node  # ROS 2 base class for the camera subscriber node
from rclpy.executors import SingleThreadedExecutor  # spins the camera subscriptions in one ROS callback thread
from sensor_msgs.msg import CameraInfo, Image  # ROS 2 image and camera-calibration message types
from cv_bridge import CvBridge  # converts ROS Image messages into NumPy arrays
from flask import Flask, Response, abort, jsonify, request, send_file  # the web framework and its helpers

# Folder that contains this script: resolve() makes the path absolute (following any symlinks) and .parent
# removes the file name. The default output folder is located relative to it.
ROOT = Path(__file__).resolve().parent
# ContactGraspnet_Server's default TCP port.
DEFAULT_PORT = 8765
# Seconds to wait for one inference (upload + model + download) before giving up.
INFER_TIMEOUT = 300
# JPEG quality (0-100) of the live-view streams; higher = bigger, sharper images.
JPEG_QUALITY = 70
# A valid run id looks like 20261017-143005 or 20261017-143005-2 (date-time, optional counter). Used to reject
# anything else, so a request can never point at a different folder (path traversal).
RUN_ID_RE = re.compile(r"\d{8}-\d{6}(-\d+)?")
# Same tiers as visualization_utils.visualize_grasps: gold = top 10%, green = top 30%, the rest faint grey.
# Colours are in OpenCV's BGR order (blue, green, red), each channel 0-255.
GOLD_BGR, GREEN_BGR, GREY_BGR = (0, 215, 255), (50, 205, 50), (170, 170, 170)
# The single best grasp (the one proposed to the robot) is drawn in magenta so it cannot be confused with a tier.
BEST_BGR = (255, 0, 255)
# Maximum seconds the planner subprocess may run (the planner has its own internal timeouts that add up to
# roughly 170 s; this is a safety net that kills a hung process).
MOVE_TIMEOUT = 300
# How many of the planner's output lines are kept and shown in the page.
MOVE_LOG_LINES = 40
# Pose of the robot's TCP (tool centre point) expressed in the Contact-GraspNet grasp frame.
# The planner moves the TCP frame (--move-eef) to   T_target = T_grasp @ TCP_IN_GRASP.
#
# Contact-GraspNet's grasp frame (see build_6d_grasp in contact_grasp_estimator.py):
#   * origin = BASE of the Panda gripper it was trained with (not the fingertips)
#   * +z     = approach direction (the way the fingers travel toward the object)
#   * +x     = finger closing direction (the line between the two contact points)
#   * the point midway between the fingertips sits gripper_depth = 0.1034 m along +z from the origin
#
# Gungnir's tcp_frame (gungnir_geometry.urdf.xacro):
#   * origin = fingertip centre of the gripper
#   * +z     = tool / approach axis, +x = jaw closing direction (checked in RViz 2026-11-22)
#
# Axes already agree, so no rotation: only shift the TCP 0.1034 m forward along the approach axis,
# from the gripper base to the fingertip centre. Validated in sim: with the identity matrix the TCP
# stopped exactly 10.3 cm short of the object.
# AI USE: this offset and its derivation were worked out with Claude (Anthropic); verified in sim by the author.
GRIPPER_DEPTH = 0.1034          # m, Contact-GraspNet's base-to-fingertip-centre distance (Panda hand)
TCP_IN_GRASP = np.array([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, GRIPPER_DEPTH],   # translate along grasp +z (approach)
                         [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)

# ======================================================================================
# Camera
# ======================================================================================
class CameraFeed(Node):
    """Subscribes to l515_node and keeps the latest frames plus pre-encoded browser JPEGs."""

    def __init__(self, min_depth, max_depth, stream_fps, swap_rb):
        # Initialise this object as a ROS 2 node; it does not open or control the physical camera.
        super().__init__("gungnir_camera_feed")
        # Depth range (metres) used for the depth colour map.
        self.min_depth, self.max_depth = float(min_depth), float(max_depth)
        # Minimum seconds between two encoded stream frames (1 / fps); fps is never allowed below 1.
        self.min_interval = 1.0 / max(1.0, float(stream_fps))
        # True if the incoming RGB image should have its red and blue channels swapped.
        self.swap_rb = bool(swap_rb)
        # CvBridge converts sensor_msgs/Image messages into NumPy arrays.
        self.bridge = CvBridge()
        # A condition variable protects the latest frame data and wakes browser stream readers.
        self.cond = threading.Condition()
        # Counts encoded frames; stream readers use it to detect newly prepared JPEGs.
        self.seq = 0
        # Latest depth image as float32 HxW in metres (0 = no return).
        self.depth = None
        # Latest colour image as uint8 HxWx3 in RGB channel order.
        self.rgb = None
        # Latest 3x3 float32 camera intrinsic matrix from /camera/color/camera_info.
        self.K = None
        # The latest frames already compressed to JPEG bytes, ready to stream to browsers.
        self.rgb_jpeg = self.depth_jpeg = None
        # Text of the most recent camera error (None when everything is fine).
        self.error = None
        # Measured frames per second, updated every ~2 seconds.
        self.fps = 0.0
        # Wall-clock time of the last complete depth/RGB/calibration update.
        self.last_frame_time = 0.0
        # Image size as (width, height), populated after the first valid image pair.
        self.size = None
        # Store the most recent depth image and camera matrix independently until all three are available.
        self._latest_depth = None
        self._latest_k = None
        # Track encoding and FPS intervals without running a camera-capture loop.
        self._last_encode = 0.0
        self._fps_window_start = time.time()
        self._fps_frame_count = 0
        # The segmentation model is unchanged; it processes RGB frames before browser JPEG encoding.
        self.seg_model = SegmentationModel()
        # Subscribe to the exact topics published by the supplied l515_node.
        self.depth_subscription = self.create_subscription(
            Image, "/camera/depth/image_rect_raw", self._depth_callback, 10
        )
        self.color_subscription = self.create_subscription(
            Image, "/camera/color/image_raw", self._color_callback, 10
        )
        self.camera_info_subscription = self.create_subscription(
            CameraInfo, "/camera/color/camera_info", self._camera_info_callback, 10
        )
        # Keep our executor in a private attribute; Node.executor is managed by rclpy when a node is added.
        self._executor = SingleThreadedExecutor()
        # Register this node with the executor so it can process the three camera subscriptions.
        self._executor.add_node(self)
        # Flask runs in the main thread, so ROS callbacks need a separate executor-spin thread.
        self._spin_thread = None

    # -- ROS 2 callbacks -------------------------------------------------------------
    def start(self):
        """Start processing ROS 2 subscriptions in the executor's background thread."""
        # Avoid creating more than one executor thread if start() is called more than once.
        if self._spin_thread is not None and self._spin_thread.is_alive():
            return
        # Spin services and subscriptions until stop() shuts down the executor.
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True, name="camera-ros-spin")
        self._spin_thread.start()

    def _depth_callback(self, msg: Image):
        """Convert and cache the newest depth topic message."""
        try:
            # Preserve the published encoding; the supplied l515_node publishes depth as 32FC1 metres.
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            # Ensure depth has the float32 representation expected by the existing pipeline.
            if msg.encoding == "16UC1":
                depth = depth.astype(np.float32) * 0.001
            else:
                depth = depth.astype(np.float32, copy=False)
            # Replace the cached array rather than mutating it after another thread may have read it.
            with self.cond:
                self._latest_depth = np.ascontiguousarray(depth)
                self.error = None
        except Exception as exc:  # noqa: BLE001
            # Report conversion failures through the existing camera status endpoint.
            self.error = f"Depth topic conversion failed: {type(exc).__name__}: {exc}"
            print(f"[camera] {self.error}", file=sys.stderr)

    def _camera_info_callback(self, msg: CameraInfo):
        """Convert the colour CameraInfo K array into the 3x3 intrinsic matrix."""
        try:
            # CameraInfo.k is a row-major 3x3 projection intrinsic matrix.
            k = np.asarray(msg.k, dtype=np.float32).reshape(3, 3)
            # Cache a copy so the message's underlying storage is never retained.
            with self.cond:
                self._latest_k = k.copy()
                self.error = None
        except Exception as exc:  # noqa: BLE001
            # Report calibration-message failures through the existing camera status endpoint.
            self.error = f"CameraInfo conversion failed: {type(exc).__name__}: {exc}"
            print(f"[camera] {self.error}", file=sys.stderr)

    def _color_callback(self, msg: Image):
        """Convert each incoming colour frame and update the shared camera snapshot."""
        try:
            # Convert the ROS colour message without forcing an encoding conversion.
            raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            # The l515_node publishes rgb8; also accept bgr8 if the publisher configuration changes.
            if msg.encoding.lower() == "bgr8":
                rgb = np.ascontiguousarray(raw[..., ::-1])
            elif msg.encoding.lower() in ("rgb8", "8uc3"):
                rgb = np.ascontiguousarray(raw)
            else:
                raise ValueError(f"Unsupported colour encoding: {msg.encoding}")
            # Preserve the original --swap-rb option's purpose for the incoming RGB stream.
            if self.swap_rb:
                rgb = np.ascontiguousarray(rgb[..., ::-1])
            # Copy the most recent depth and calibration references while holding the shared-data lock.
            with self.cond:
                depth = self._latest_depth
                k = self._latest_k
            # Wait for the depth and CameraInfo topics to deliver their first messages.
            if depth is None or k is None:
                return
            # The l515_node publishes aligned colour and depth images, so their dimensions must agree.
            if depth.shape[:2] != rgb.shape[:2]:
                raise ValueError(f"RGB/depth size mismatch: RGB={rgb.shape[:2]}, depth={depth.shape[:2]}")
            # Use wall-clock time for the existing health and stream-FPS calculations.
            now = time.time()
            # Prepare JPEGs only at the requested browser stream rate; snapshots still update on every callback.
            rgb_jpeg = depth_jpeg = None
            if now - self._last_encode >= self.min_interval:
                # OpenCV JPEG encoding uses BGR channel order, so convert the RGB frame before encoding.
                params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                # Segment the RGB image using the existing segmentation model.
                seg_map = self.seg_model.segment(rgb)
                # Blend the object-id segmentation colours onto the live RGB image.
                segmented_rgb = overlay_segmentation(rgb, seg_map)
                # Encode the segmented colour image as JPEG bytes for the browser stream.
                rgb_jpeg = cv2.imencode(".jpg", segmented_rgb[..., ::-1], params)[1].tobytes()
                # Colourise the depth image and encode it as JPEG bytes for the browser stream.
                depth_jpeg = cv2.imencode(".jpg", self.colorize_depth(depth), params)[1].tobytes()
                # Record when the last JPEG pair was generated.
                self._last_encode = now
            # Update the exact fields consumed by snapshot(), mjpeg(), /api/status, and /api/capture.
            with self.cond:
                self.depth = depth
                self.rgb = rgb
                self.K = k
                # Image size is (width, height); NumPy shape is (rows, columns) = (height, width).
                self.size = (int(depth.shape[1]), int(depth.shape[0]))
                # Publish new JPEGs and increment the sequence only when a new encoded pair is available.
                if rgb_jpeg is not None:
                    self.rgb_jpeg, self.depth_jpeg = rgb_jpeg, depth_jpeg
                    self.seq += 1
                # Record a successful complete frame and clear any earlier conversion error.
                self.last_frame_time = now
                self.error = None
                # Wake any browser stream generators waiting for a newer JPEG pair.
                self.cond.notify_all()
            # Count complete colour frames for the measured incoming frame rate.
            self._fps_frame_count += 1
            # Update the measured FPS every two seconds, then begin a new measurement window.
            if now - self._fps_window_start >= 2.0:
                self.fps = self._fps_frame_count / (now - self._fps_window_start)
                self._fps_frame_count = 0
                self._fps_window_start = now
        except Exception as exc:  # noqa: BLE001
            # Store and print callback failures without terminating the ROS executor.
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"[camera] {self.error}", file=sys.stderr)

    # -- helpers ---------------------------------------------------------------------
    def colorize_depth(self, depth: np.ndarray) -> np.ndarray:
        """Turbo colormap over [min_depth, max_depth]; black where the model ignores the pixel."""
        # Rescale so min_depth maps to 0 and max_depth to 1 (values outside the range are clipped to 0 or 1).
        scaled = np.clip((depth - self.min_depth) / (self.max_depth - self.min_depth), 0.0, 1.0)
        # Apply the colour map (0-1 is first converted to 0-255 integers); the result is a BGR colour image.
        bgr = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        # Paint black every pixel outside [min_depth, max_depth] or with no depth (the ~ inverts the mask).
        bgr[~((depth >= self.min_depth) & (depth <= self.max_depth))] = 0
        return bgr

    def snapshot(self):
        """Latest (depth, rgb, K), or None before the first complete frame."""
        # Hold the lock so the three values come from one completed callback update.
        with self.cond:
            if self.depth is None or self.rgb is None or self.K is None:
                return None
            # The callbacks replace arrays instead of editing them, so returning these references is safe.
            return self.depth, self.rgb, self.K

    @property
    def ok(self) -> bool:
        # True once a frame has been encoded and the last complete frame is less than 3 seconds old.
        return self.seq > 0 and (time.time() - self.last_frame_time) < 3.0

    def stop(self):
        # Stop the ROS executor so no more subscription callbacks are processed.
        self._executor.shutdown(timeout_sec=5.0)
        # Wait briefly for the executor thread to finish before the node is destroyed by main().
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=5.0)

    # -- MJPEG -----------------------------------------------------------------------
    def mjpeg(self, which: str):
        """Generator yielding an endless MJPEG stream ("rgb" or "depth") for one browser connection."""
        # The last frame number this viewer has been sent (-1 = none yet).
        last = -1
        # Keep going until the browser disconnects (Flask then stops iterating this generator).
        while True:
            with self.cond:
                # Sleep until a newer frame exists, but wake up at least every 2 seconds.
                self.cond.wait_for(lambda: self.seq != last, timeout=2.0)
                # If still nothing new after waiting, there is nothing to send this round.
                if self.seq == last:
                    frame = None
                else:
                    # Remember which frame we are about to send, and pick the matching JPEG.
                    last = self.seq
                    frame = self.rgb_jpeg if which == "rgb" else self.depth_jpeg
            # No new frame: go back and wait again.
            if frame is None:
                continue
            # One multipart chunk: the boundary marker, headers for this JPEG (type and length), a blank line,
            # the JPEG bytes, and a line break. The browser replaces the displayed image each time.
            yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(frame) + frame + b"\r\n"

# ======================================================================================
# Link to ContactGraspnet_Server (on the GPU computer)
# ======================================================================================
def server_url(spec: str) -> str:
    """Accept "host", "host:port" or a full URL and return "http://host:port"."""
    spec = spec.strip()
    # urlparse needs a scheme to split the address correctly, so add "http://" if there is none.
    parsed = urlparse(spec if "://" in spec else "http://" + spec)
    if not parsed.hostname:
        raise SystemExit(f"Bad server address: {spec!r}")
    # Rebuild a clean address, using the default port if none was given.
    return f"{parsed.scheme}://{parsed.hostname}:{parsed.port or DEFAULT_PORT}"


class ServerLink:
    """Talks HTTP to ContactGraspnet_Server: polls /health for its status and uploads captures to /infer."""

    POLL_INTERVAL = 1.0  # seconds between status checks
    # Failed status checks in a row before the state becomes "error" (a server restart takes a few seconds).
    UNREACHABLE_AFTER = 5

    def __init__(self, url, token, on_done, on_error):
        # Base address such as http://192.168.1.50:8765, and the shared secret (None = send none).
        self.url, self.token = url, token
        # Functions to call when a request finishes / fails (provided by make_callbacks).
        self.on_done, self.on_error = on_done, on_error
        # A re-entrant lock protecting the fields below (the same thread may take it twice).
        self.lock = threading.RLock()
        self.state = "starting"  # starting | ready | busy | error | stopped
        # Human-readable status text shown in the UI.
        self.message = f"Connecting to ContactGraspnet_Server at {url} ..."
        # Run id of the request currently in flight (None when idle).
        self.pending = None
        # How many server restarts have been seen.
        self.restarts = 0
        # The model's batch size, as reported by the server (None until the first answer).
        self.forward_passes = None
        # _log: the server's last log lines; _boot_id: identifies the server process (a change = it restarted);
        # _misses: consecutive failed status checks.
        self._log, self._boot_id, self._misses = [], None, 0
        # Set by stop() to end the polling loop.
        self._halt = threading.Event()

    # -- HTTP ----------------------------------------------------------------------
    def _open(self, path, data=None, headers=None, timeout=5.0):
        """Open one HTTP request: GET if data is None, otherwise POST with data as the body. Raises on failure."""
        req = urllib.request.Request(self.url + path, data=data)
        # Attach the token if there is one.
        if self.token:
            req.add_header("Authorization", "Bearer " + self.token)
        # Attach any extra headers (e.g. the depth crop).
        for key, value in (headers or {}).items():
            req.add_header(key, value)
        # Send it. `timeout` is how long to wait for the server's answer; it raises HTTPError for 4xx/5xx replies
        # and URLError / OSError if the server cannot be reached.
        return urllib.request.urlopen(req, timeout=timeout)

    @staticmethod
    def _http_error(exc: urllib.error.HTTPError) -> str:
        """Turn an HTTP error reply into a sentence for the UI."""
        if exc.code == 401:
            return "ContactGraspnet_Server rejected the token (HTTP 401): use the same --token on both computers."
        try:
            # The server sends {"error": "..."} with every error reply.
            detail = json.loads(exc.read().decode()).get("error")
        except Exception:  # noqa: BLE001
            detail = None
        return detail or f"ContactGraspnet_Server returned HTTP {exc.code}."

    # -- lifecycle -------------------------------------------------------------------
    def start(self):
        """Start polling the server's status in a background thread."""
        threading.Thread(target=self._poll, daemon=True, name="server-health").start()

    def stop(self):
        """Stop polling. The remote server keeps running; it belongs to the other computer."""
        self._halt.set()
        with self.lock:
            self.state = "stopped"

    def restart(self):
        """Ask the server to restart its model (the 'Restart model' button, e.g. after a CUDA error)."""
        try:
            # An empty body makes this a POST.
            with self._open("/restart", data=b""):
                pass
        except urllib.error.HTTPError as exc:
            note = self._http_error(exc)
        except Exception as exc:  # noqa: BLE001
            note = f"Restart request failed: {exc}"
        else:
            # Accepted: show "starting" until the next status check reports the new state.
            with self.lock:
                self.state, self.message = "starting", "Restart requested; waiting for ContactGraspnet_Server..."
            return
        # Failed: show why.
        with self.lock:
            self.message = note

    def log_tail(self, n=20):
        """The last n lines of the server's log, as last reported by /health."""
        with self.lock:
            return self._log[-n:]

    # -- status polling ----------------------------------------------------------------
    def _poll(self):
        """Ask the server for its status every POLL_INTERVAL seconds until stop() is called."""
        while not self._halt.is_set():
            try:
                with self._open("/health", timeout=3.0) as reply:
                    self._apply(json.loads(reply.read().decode()))
            except urllib.error.HTTPError as exc:
                # The server answered, but with an error (usually a wrong token): report it at once.
                self._miss(self._http_error(exc), immediately=True)
            except Exception as exc:  # noqa: BLE001
                # No answer at all (server down, cable unplugged, wrong address...).
                self._miss(f"Cannot reach ContactGraspnet_Server at {self.url}: {exc}")
            # Wait for the next check (returns early if stop() is called).
            self._halt.wait(self.POLL_INTERVAL)

    def _apply(self, info):
        """Update our fields from one successful /health reply."""
        with self.lock:
            self._misses = 0
            self._log = list(info.get("log", []))
            if info.get("forward_passes") is not None:
                self.forward_passes = info["forward_passes"]
            # A different boot id means the server process was replaced since the last check.
            boot = info.get("boot_id")
            if self._boot_id is not None and boot != self._boot_id:
                self.restarts += 1
            self._boot_id = boot
            remote = info.get("state", "starting")
            if self.pending is not None and remote != "starting":
                # Our own request is in flight: stay "busy" until it finishes, whatever the server says.
                self.state, self.message = "busy", "Running inference..."
            else:
                self.state, self.message = remote, info.get("message", "")

    def _miss(self, message, immediately=False):
        """Record one failed status check; switch to "error" once it has kept failing."""
        with self.lock:
            self._misses += 1
            if immediately or self._misses >= self.UNREACHABLE_AFTER:
                self.state, self.message = "error", message

    # -- requests ----------------------------------------------------------------------
    def submit(self, run_id, input_path, output_path, z_range) -> bool:
        """Start one inference request. Returns True if it was started, False if the server is not ready."""
        with self.lock:
            # Only accept work when the model is loaded and idle.
            if self.state != "ready":
                return False
            # Mark busy and remember which request is in flight.
            self.state, self.message, self.pending = "busy", "Running inference...", run_id
        # The upload, inference and download take a while, so they run in their own thread.
        threading.Thread(target=self._run, args=(run_id, input_path, output_path, z_range),
                         daemon=True, name="infer-request").start()
        return True

    def _run(self, run_id, input_path, output_path, z_range):
        """Upload the input file, wait for the result, and save it. Runs in its own thread."""
        started = time.time()
        try:
            # The depth crop travels in headers; the body is the raw bytes of the input .npz file.
            headers = {"Content-Type": "application/octet-stream", "X-Run-Id": run_id,
                       "X-Z-Min": str(z_range[0]), "X-Z-Max": str(z_range[1])}
            with self._open("/infer", data=Path(input_path).read_bytes(), headers=headers,
                            timeout=INFER_TIMEOUT) as reply:
                # The reply body is the result .npz file; the summary numbers come back in headers.
                payload = reply.read()
                info = {"num_grasps": int(reply.headers.get("X-Num-Grasps", 0)),
                        "seconds": float(reply.headers.get("X-Seconds", round(time.time() - started, 2)))}
            # Save the reply as result.npz, where build_views() expects it.
            Path(output_path).write_bytes(payload)
            error = None
        except urllib.error.HTTPError as exc:
            error = self._http_error(exc)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        with self.lock:
            # The request is over, whatever happened.
            self.pending = None
            if error is None:
                self.state, self.message = "ready", "Model ready"
            else:
                # The server usually restarts itself after an error; the next status check says when it is back.
                self.state, self.message = "starting", "Inference failed; waiting for ContactGraspnet_Server..."
        # Tell the app, outside the lock, because the callbacks do slow work (drawing the views).
        if error is None:
            self.on_done(run_id, info)
        else:
            self.on_error(run_id, error)


# ======================================================================================
# Segmentation model
# ======================================================================================
class SegmentationModel:
    """FastSAM: turns an RGB image into an instance-id map."""

    def __init__(self):
        # Imported here so the (slow) Ultralytics import only happens when the model is created.
        from ultralytics import FastSAM

        # Weights are downloaded automatically on first use if not found locally.
        # "FastSAM-s.pt" is the small model, "FastSAM-x.pt" the larger, more accurate one.
        self.model = FastSAM("FastSAM-s.pt")
        # Use the GPU if PyTorch can see one, otherwise the CPU.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def segment(self, image):
        """Return an array the size of the image: 0 = background, 1, 2, ... = object ids (largest object = 1)."""
        # Instance id map, one 16-bit id per pixel.
        instance_map = np.zeros(image.shape[:2], dtype=np.uint16)

        # Ultralytics treats numpy arrays as BGR, so flip the RGB input.
        results = self.model.predict(
            np.ascontiguousarray(image[..., ::-1]),
            device=self.device,
            imgsz=1024,          # inference resolution (FastSAM default)
            conf=0.4,            # drop low-confidence masks (roughly like pred_iou_thresh in SAM2)
            iou=0.9,             # NMS IoU threshold; high so that distinct-but-overlapping objects are kept
            retina_masks=True,   # return masks at the original image resolution
            max_det=300,         # maximum number of instances
            verbose=False,
        )
        result = results[0]
        if result.masks is None:  # nothing detected
            return instance_map

        # Binary masks, shape (N, H, W).
        masks = result.masks.data > 0.5
        areas = masks.flatten(1).sum(dim=1)

        # Drop tiny regions (same role as min_mask_region_area=100 before).
        keep = areas >= 100
        masks, areas = masks[keep], areas[keep]

        # Sort by area, largest first.
        order = torch.argsort(areas, descending=True)
        masks = masks[order].cpu().numpy()

        for object_id, mask in enumerate(masks, start=1):
            # Only assign pixels that have not been assigned yet, so a large mask is not overwritten by a
            # later overlapping one.
            instance_map[mask & (instance_map == 0)] = object_id
        return instance_map


# ======================================================================================
# Grasp pose -> robot target (used for the "Proposed robot move")
# ======================================================================================
def rotation_to_rpy(rot: np.ndarray):
    """3x3 rotation matrix -> (roll, pitch, yaw) in radians.

    This is the exact inverse of rpy_to_quaternion() in Cartesian_Path_Planner.py. That function builds the
    orientation  R = Rz(yaw) @ Ry(pitch) @ Rx(roll)  (the common "ZYX" / aerospace convention), so the matrix
    elements are:  R[2,0] = -sin(pitch),  R[2,1] = cos(pitch) sin(roll),  R[2,2] = cos(pitch) cos(roll),
                   R[1,0] =  cos(pitch) sin(yaw),  R[0,0] = cos(pitch) cos(yaw).
    Feeding the returned angles back into the planner therefore reproduces the same orientation.
    """
    # sin(pitch) = -R[2,0]; clip guards against tiny numerical errors pushing the value just outside [-1, 1].
    sin_pitch = float(np.clip(-rot[2, 0], -1.0, 1.0))
    pitch = math.asin(sin_pitch)
    if abs(sin_pitch) < 1.0 - 1e-6:
        # Normal case: roll and yaw come from the entries that were divided by cos(pitch).
        roll = math.atan2(rot[2, 1], rot[2, 2])
        yaw = math.atan2(rot[1, 0], rot[0, 0])
    else:
        # Gimbal lock (pitch = +-90 degrees): roll and yaw rotate about the same axis, so only their combination
        # is defined. Pick roll = 0 and put the whole rotation into yaw (any valid split gives the same pose).
        roll = 0.0
        yaw = math.atan2(-rot[0, 1], rot[1, 1])
    return roll, pitch, yaw


def grasp_to_target(grasp: np.ndarray):
    """One 4x4 grasp pose (camera frame) -> ([x, y, z] metres, [roll, pitch, yaw] radians) for the planner."""
    # Shift from the grasp frame to the robot's TCP frame (identity unless TCP_IN_GRASP was edited).
    pose = np.asarray(grasp, dtype=np.float64) @ TCP_IN_GRASP
    # Translation column of the matrix = where the TCP should be.
    position = [float(v) for v in pose[:3, 3]]
    # Rotation block of the matrix = how the TCP should be oriented, expressed as roll-pitch-yaw.
    rpy = [float(v) for v in rotation_to_rpy(pose[:3, :3])]
    return position, rpy


# ======================================================================================
# Output views (3D scene + 2D overlay)
# ======================================================================================
def grasp_points(grasps: np.ndarray, gripper_points: np.ndarray) -> np.ndarray:
    """Gripper wireframe points for every grasp in camera coordinates: (N,4,4),(K,3) -> (N,K,3)."""
    # Split each 4x4 pose into its 3x3 rotation block and its 3-element translation vector.
    rot, trans = grasps[:, :3, :3], grasps[:, :3, 3]
    # For every grasp n and wireframe point k compute rotation * point + translation. einsum "nij,kj->nki" sums
    # over j of rot[n,i,j] * point[k,j], giving a (grasps, points, 3) array.
    return np.einsum("nij,kj->nki", rot, gripper_points) + trans[:, None, :]


def score_tiers(scores: np.ndarray):
    """Boolean masks (gold, green, grey): top 10% / next 20% / rest."""
    # No grasps: return three empty masks.
    if scores.size == 0:
        empty = np.zeros(0, dtype=bool)
        return empty, empty, empty
    # The 70th and 90th percentile score values.
    p70, p90 = np.percentile(scores, [70, 90])
    # Gold: at or above the 90th percentile.
    gold = scores >= p90
    # Green: at or above the 70th percentile but not gold.
    green = (scores >= p70) & ~gold
    # Grey: everything that is neither gold nor green.
    return gold, green, ~(gold | green)


def overlay_segmentation(img, segmap):
    num_objects = np.max(segmap)
    # A fixed random colour per object id (fixed seed = same colours every time).
    colors = np.random.default_rng(42).integers(0, 256, size=(num_objects + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # background = black
    # colors[segmap] looks up a color for every pixel; the result is 80% original image + 20% segmentation colors.
    img = cv2.addWeighted(img, 0.8, colors[segmap], 0.2, 0)
    return img


def draw_overlay(rgb, grasps, scores, K, gripper_points, segments, segmap=None, best_index=None) -> bytes:
    """Grasp wireframes projected onto the captured RGB image (JPEG bytes).

    best_index: index (into `grasps`) of the grasp proposed to the robot; it is drawn last, thicker, in magenta.
    """
    # OpenCV draws in BGR, so convert the RGB image (this also makes a copy, leaving the original untouched).
    img = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    # Number of grasps to draw.
    n = len(grasps)
    # If a segmentation map is given, colorize it and blend it over the image. overlay_segmentation() RETURNS a new
    # image, so the result must be assigned back (the earlier version discarded it, so the blend never appeared).
    if segmap is not None:
        img = overlay_segmentation(img, segmap)
    if n:
        # 3D wireframe points of every grasp in camera coordinates: shape (N, K, 3).
        pts = grasp_points(grasps, gripper_points)
        # Depth (z) of each point.
        z = pts[..., 2]
        # Points behind (or on) the camera plane cannot be projected; mark which ones are usable.
        ok = z > 1e-3
        # Replace unusable depths by 1.0 so the divisions below never divide by zero.
        safe_z = np.where(ok, z, 1.0)
        # Pinhole projection: pixel column u = fx * x / z + cx, clipped to a sane range so the integer
        # conversion below cannot overflow.
        u = np.clip(K[0, 0] * pts[..., 0] / safe_z + K[0, 2], -1e4, 1e4)
        # Pixel row v = fy * y / z + cy.
        v = np.clip(K[1, 1] * pts[..., 1] / safe_z + K[1, 2], -1e4, 1e4)

        def draw(canvas, indices, color, thickness):
            # Draw every wireframe segment of every grasp listed in `indices` onto `canvas`.
            for i in indices:
                for a, b in segments:
                    # Only draw a line if both of its end points are in front of the camera.
                    if ok[i, a] and ok[i, b]:
                        # Pixel coordinates must be whole numbers; LINE_AA gives smooth anti-aliased lines.
                        cv2.line(canvas, (int(round(u[i, a])), int(round(v[i, a]))),
                                 (int(round(u[i, b])), int(round(v[i, b]))), color, thickness, cv2.LINE_AA)

        # Split the grasps into score tiers.
        gold, green, grey = score_tiers(scores)
        # Draw the low-scoring grasps on a copy, then blend it in at 40% so they look faint.
        faint = img.copy()
        draw(faint, np.flatnonzero(grey), GREY_BGR, 1)
        img = cv2.addWeighted(faint, 0.4, img, 0.6, 0)
        # Draw the better grasps on top at full strength (gold last, so it is never hidden).
        draw(img, np.flatnonzero(green), GREEN_BGR, 2)
        draw(img, np.flatnonzero(gold), GOLD_BGR, 2)
        # Finally draw the proposed (best) grasp on top of everything, thicker and in magenta, so it stands out
        # from the gold tier it belongs to.
        if best_index is not None:
            draw(img, [best_index], BEST_BGR, 4)

    # Caption: the grasp count, plus the best score if there are any grasps.
    label = f"{n} grasps" + (f" | best score {float(scores.max()):.2f} (magenta)" if n else " | none found")
    # Draw it twice: a thick black outline first, then thin white text on top, so it is readable on any background.
    cv2.putText(img, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    # Encode the finished image as a high-quality JPEG and return its bytes.
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


def scene_figure_json(pc, colors, grasps, scores, gripper_points, segments, max_points=30000, best_index=None) -> str:
    """Interactive 3D scene as Plotly JSON. All grasps of one tier share ONE trace (not 6 per grasp).

    best_index: index (into `grasps`) of the grasp proposed to the robot; it gets its own thick magenta trace.
    """
    # Imported here so the program starts even before Plotly is needed.
    import plotly.graph_objects as go
    import plotly.io as pio

    # The list of 3D drawing layers (point cloud, grasp lines, camera axes) built below.
    traces = []
    if len(pc):
        # Indices of the points to show; by default all of them.
        keep = np.arange(len(pc))
        # Too many points make the browser slow: pick a random subset (fixed seed = same subset each time).
        if len(pc) > max_points:
            keep = np.random.default_rng(0).choice(len(pc), max_points, replace=False)
        pc = pc[keep]
        # Default marker style: small light-grey dots.
        marker = dict(size=1.6, color="lightgray")
        # If the model returned per-point colours, use them.
        if len(colors) > 0:
            c = np.asarray(colors)
            # Only use them if they are an (N, 3) array covering every point we kept.
            if c.ndim == 2 and c.shape[1] == 3 and len(c) >= keep.max() + 1:
                c = c[keep]  # same subset as the points
                # Floating-point colours in 0-1 are scaled to 0-255.
                if c.dtype.kind == "f" and c.size and c.max() <= 1.0:
                    c = c * 255
                # Clamp to the valid range and convert to whole numbers.
                c = np.clip(c, 0, 255).astype(np.uint8)
                # Plotly wants one "rgb(r,g,b)" text per point.
                marker["color"] = [f"rgb({r},{g},{b})" for r, g, b in c]
        # Add the point cloud layer (hoverinfo="skip" disables hover tooltips for speed).
        traces.append(go.Scatter3d(x=pc[:, 0], y=pc[:, 1], z=pc[:, 2], mode="markers",
                                   marker=marker, name="Point cloud", hoverinfo="skip"))

    if len(grasps):
        # 3D wireframe points of every grasp, shape (N, K, 3).
        pts = grasp_points(grasps, gripper_points)

        def lines(indices):
            # Start and end points of every segment of the selected grasps, shapes (m, segments, 3).
            a, b = pts[indices][:, segments[:, 0], :], pts[indices][:, segments[:, 1], :]
            # For each segment emit: start, end, NaN. The NaN row makes Plotly lift the pen, so thousands of
            # separate segments fit in one layer. reshape flattens everything into a list of 3D points.
            return np.stack([a, b, np.full_like(a, np.nan)], axis=2).reshape(-1, 3)

        # Split the grasps into score tiers.
        gold, green, grey = score_tiers(scores)
        # One layer per tier: (mask of grasps, legend name, line colour, line width).
        for mask, name, color, width in ((grey, "other grasps", "rgba(130,130,130,0.35)", 3),
                                         (green, "top 30%", "limegreen", 4),
                                         (gold, "top 10%", "gold", 4)):
            idx = np.flatnonzero(mask)  # indices of the grasps in this tier
            # Skip empty tiers.
            if idx.size:
                xyz = lines(idx)
                traces.append(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines",
                                           line=dict(color=color, width=width), name=name, hoverinfo="skip"))
        # The proposed (best) grasp gets its own layer, added last so it is drawn on top, with a thicker magenta
        # line and a legend entry that says it is the one offered to the robot.
        if best_index is not None:
            xyz = lines(np.array([best_index]))
            traces.append(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines",
                                       line=dict(color="magenta", width=10), name="best grasp (proposed move)",
                                       hoverinfo="skip"))

    # Draw three short lines at the origin showing the camera's x (red), y (green), z (blue) axes.
    for axis, color in enumerate(("red", "green", "blue")):
        # A 10 cm line along the given axis, starting at the origin.
        end = np.zeros(3)
        end[axis] = 0.1
        traces.append(go.Scatter3d(x=[0, end[0]], y=[0, end[1]], z=[0, end[2]], mode="lines",
                                   line=dict(color=color, width=6), showlegend=False, hoverinfo="skip"))

    # Combine the layers into one figure and style it.
    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_dark",  # dark theme to match the page
        margin=dict(l=0, r=0, t=0, b=0),  # no margins, so the plot fills its box
        legend=dict(orientation="h", y=1.0),  # horizontal legend along the top
        scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",  # axis labels in metres
                   aspectmode="data",  # equal scale on all three axes, so shapes are not stretched
                   # OpenCV camera frame has y pointing down: flip "up" and view from behind the camera.
                   camera=dict(up=dict(x=0, y=-1, z=0), eye=dict(x=0.0, y=-0.4, z=-1.6))))
    # Serialise the figure to a JSON string the browser's Plotly library can draw.
    return pio.to_json(fig)


def build_views(run_dir: Path) -> dict:
    """Create overlay.jpg and scene.json for one finished run; returns summary numbers (incl. the best grasp)."""
    # Open the model's result file and the frame that was sent to it.
    result, captured = np.load(run_dir / "result.npz"), np.load(run_dir / "input.npz")
    # Grasp poses and scores.
    grasps, scores = result["grasps"], result["scores"]
    # Gripper wireframe points and the point pairs to connect.
    gp, seg = result["gripper_points"], result["gripper_segments"]

    # Pick the best grasp = the one with the highest score (None when the model found no grasps).
    best_index = int(np.argmax(scores)) if scores.size else None
    best_grasp = None
    if best_index is not None:
        # Convert its 4x4 pose into the x y z roll pitch yaw numbers the Cartesian planner expects.
        position, rpy = grasp_to_target(grasps[best_index])
        best_grasp = {"index": best_index, "score": float(scores[best_index]), "position": position, "rpy": rpy}

    # Make the 2D overlay and save it as a JPEG file: grasps drawn on the captured RGB image, with the segmentation
    # map (captured["seg_map"], saved in input.npz at capture time) blended in underneath. The best grasp is magenta.
    (run_dir / "overlay.jpg").write_bytes(
        draw_overlay(captured["rgb"], grasps, scores, captured["K"], gp, seg, captured["seg_map"], best_index))
    # Make the 3D scene description and save it as JSON text (the best grasp is a separate magenta layer).
    (run_dir / "scene.json").write_text(
        scene_figure_json(result["pc_full"], result["pc_colors"], grasps, scores, gp, seg, best_index=best_index))
    # Summary numbers for the status line (best_score is None when there are no grasps).
    return {"num_grasps": int(len(grasps)), "best_score": float(scores.max()) if scores.size else None,
            "best_grasp": best_grasp}


# ======================================================================================
# Web app
# ======================================================================================
def create_app(feed: CameraFeed, server: ServerLink, state: dict, args) -> Flask:
    """Build the Flask web application and register all of its URL routes."""
    # The Flask application object; __name__ tells Flask where this module lives.
    app = Flask(__name__)
    # Folder where every capture gets its own sub-folder.
    out_root = Path(args.output_dir)
    # Lock that keeps status reads and capture requests from interleaving.
    lock = threading.Lock()
    # Lock that protects state["proposal"] (the grasp move awaiting approval / being executed). It is created in
    # main() because make_callbacks() needs the same lock. Lock order everywhere: `lock` first, then `move_lock`.
    move_lock = state["move_lock"]
    # Small dictionary used to cache the (large) Plotly JavaScript text after the first request.
    plotly_js = {}
    # Load the  segmentation model (slow; this is why the camera and server link are started before it).
    seg_model = SegmentationModel()

    def run_dir_for(run_id: str) -> Path:
        # Reject any run id that does not look like a real one (blocks "../" tricks): HTTP 404.
        if not RUN_ID_RE.fullmatch(run_id):
            abort(404)
        path = out_root / run_id
        # If the folder does not exist, also 404.
        if not path.is_dir():
            abort(404)
        return path

    def no_store(response):
        # Tell the browser never to cache this response, so it always sees the latest data.
        response.headers["Cache-Control"] = "no-store"
        return response

    # -- robot move proposal -----------------------------------------------------------
    def proposal_snapshot():
        """A JSON-safe copy of the current move proposal for the UI (None when there is none).

        Also turns an unanswered proposal into "expired" once --approval-timeout seconds have passed.
        """
        with move_lock:
            p = state["proposal"]
            if p is None:
                return None
            # Too old to trust: the arm (and a wrist-mounted camera) may have moved since the capture.
            if p["status"] == "awaiting" and time.time() > p["expires_at"]:
                p["status"] = "expired"
                p["message"] = "This proposal expired. Press Capture & run again to get a fresh grasp."
            # Copy everything except the internal deadline, and add the remaining time for the countdown.
            view = {k: v for k, v in p.items() if k != "expires_at"}
            view["log"] = list(p["log"])
            view["seconds_left"] = max(0, int(p["expires_at"] - time.time())) if p["status"] == "awaiting" else 0
            return view

    def run_move(proposal):
        """Run Cartesian_Path_Planner.py for one approved proposal and record how it went. Runs in its own thread."""
        # The planner's command line: six numbers (x y z roll pitch yaw) = an absolute pose in --frame, followed by
        # its options. Numbers use fixed-point text (.6f): the planner's argparse would mistake exponent notation
        # such as "-1e-05" for an option and fail.
        numbers = [f"{v:.6f}" for v in (*proposal["position"], *proposal["rpy"])]
        cmd = [sys.executable, str(args.planner_script), *numbers,
               "--frame", args.move_frame, "--eef", args.move_eef,
               "--group", args.move_group, "--speed", str(args.move_speed)]
        # Approach move: tell the planner to stop this many metres short of the grasp (along the approach axis)
        # before the final move. --fixed-frame is a frame that does not move with the arm, in which the planner
        # pins both points down before the wrist camera moves. 0 leaves the planner's single straight move.
        if args.approach_distance > 0.0:
            cmd += ["--approach-distance", f"{args.approach_distance:.4f}", "--fixed-frame", args.move_fixed_frame]
        status, note = "failed", ""
        try:
            # Start the planner. stderr is merged into stdout because ROS 2 loggers write to stderr and we want
            # both in one log. It inherits this process's environment, so the sourced ROS 2 setup applies to it too.
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            # Safety net: kill the planner if it has not finished within MOVE_TIMEOUT seconds.
            watchdog = threading.Timer(MOVE_TIMEOUT, proc.kill)
            watchdog.start()
            try:
                # Read the planner's output line by line while it runs, so the page shows live progress.
                for line in proc.stdout:
                    with move_lock:
                        proposal["log"] = (proposal["log"] + [line.rstrip()])[-MOVE_LOG_LINES:]
                # The output stream ends when the process exits; collect its exit code (0 = success).
                code = proc.wait()
            finally:
                watchdog.cancel()
            if code == 0:
                status, note = "succeeded", "The robot reached the target (the planner reported success)."
            elif code < 0:
                note = (f"The planner was stopped (signal {-code}), probably by the {MOVE_TIMEOUT} s time limit. "
                        "Check the robot's actual state before doing anything else.")
            else:
                note = ("The planner reported a failure, so the robot may not have moved (it refuses incomplete "
                        "paths and large joint jumps). See the planner output below.")
        except Exception as exc:  # noqa: BLE001
            note = f"Could not run the planner: {type(exc).__name__}: {exc}"
        # Record the outcome. The proposal is now used up: its status is final, so it cannot be approved again.
        with move_lock:
            proposal["status"], proposal["message"] = status, note

    @app.get("/")
    def index():
        # The main page: the HTML text defined at the bottom of this file.
        return no_store(Response(INDEX_HTML, mimetype="text/html"))

    @app.get("/plotly.min.js")
    def plotly_bundle():  # served locally so the browser needs no internet access
        # Load the JavaScript from the installed plotly package once, then reuse it.
        if "js" not in plotly_js:
            from plotly.offline import get_plotlyjs
            plotly_js["js"] = get_plotlyjs()
        response = Response(plotly_js["js"], mimetype="application/javascript")
        # This file never changes, so let the browser cache it for a day.
        response.headers["Cache-Control"] = "public, max-age=86400"
        return response

    @app.get("/stream/<which>")
    def stream(which):
        # Only "rgb" and "depth" streams exist.
        if which not in ("rgb", "depth"):
            abort(404)
        # Return the endless MJPEG stream; the boundary name must match the "--frame" marker in mjpeg().
        return no_store(Response(feed.mjpeg(which), mimetype="multipart/x-mixed-replace; boundary=frame"))

    @app.get("/api/status")
    def status():
        # Read everything under the lock so the pieces are consistent with each other.
        with lock:
            payload = {
                # Camera health: working?, measured fps, last error text, image size.
                "camera": {"ok": feed.ok, "fps": round(feed.fps, 1), "error": feed.error, "size": feed.size},
                # Model server state, its status text, restart count, and recent log lines.
                "server": {"state": server.state, "message": server.message, "restarts": server.restarts,
                           "log": server.log_tail(20)},
                # Id of the capture being processed (None when idle).
                "busy": state["pending"],
                # Text of the last failure, shown to the user.
                "last_error": state["last_error"],
                # Summary of the newest finished result (None before the first one).
                "result": state["latest"],
                # The best-grasp robot move awaiting approval / running / finished (None when there is none).
                "proposal": proposal_snapshot(),
                # Settings the page displays (forward_passes is reported by the server).
                "config": {"min_depth": args.min_depth, "max_depth": args.max_depth,
                           "forward_passes": server.forward_passes},
            }
        # Convert the dictionary to a JSON response.
        return no_store(jsonify(payload))

    @app.post("/api/capture")
    def capture():
        # Read the prompt text from the JSON body (empty if absent); limit it to 2000 characters.
        prompt = str((request.get_json(silent=True) or {}).get("prompt", ""))[:2000]
        with lock:
            # Refuse (HTTP 409 Conflict) if a capture is running or the model is not ready.
            if state["pending"] is not None or server.state != "ready":
                return jsonify(error="The model is not ready (or is already busy)."), 409
            # Refuse while the robot is executing an approved move: the camera view (and any wrist-mounted camera)
            # is changing, and a new proposal must not replace the one being executed.
            with move_lock:
                moving = state["proposal"] is not None and state["proposal"]["status"] == "running"
            if moving:
                return jsonify(error="The robot is moving. Wait until the move has finished."), 409
            # Grab the latest camera frame.
            frame = feed.snapshot()
            # No frame yet: HTTP 503 Service Unavailable.
            if frame is None:
                return jsonify(error="No camera frame available yet."), 503
            depth, rgb, K = frame
            # TODO: Add image cropping based on "prompt"

            # Segment the RGB image so the model can tell which regions belong to different objects.
            seg_map = seg_model.segment(rgb)
            # Run id = current date and time, e.g. 20261017-143005.
            run_id = time.strftime("%Y%m%d-%H%M%S")
            run_dir, n = out_root / run_id, 1
            # If two captures happen in the same second, add a counter (-2, -3, ...) to keep ids unique.
            while run_dir.exists():
                n += 1
                run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{n}"
                run_dir = out_root / run_id
            # Create the run's folder (and parents).
            run_dir.mkdir(parents=True)
            # Same keys as Odin.py's input_pc.npz, so inference.py can be pointed at any saved run.
            np.savez(run_dir / "input.npz", depth=depth.astype(np.float32), K=K.astype(np.float32), rgb=rgb, seg_map=seg_map)
            # Information saved alongside the capture.
            meta = {"id": run_id, "prompt": prompt, "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "z_range": [args.min_depth, args.max_depth], "forward_passes": server.forward_passes}
            # Write it as indented JSON.
            (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
            # Keep it in memory too, so the result handler can reuse it.
            state["meta"][run_id] = meta
            # A new capture clears the previous error and marks this run as in progress.
            state["last_error"] = None
            state["pending"] = run_id
            # A new capture also withdraws the old move proposal: its grasp belongs to an older camera view and
            # must not be approvable while (or after) the new result arrives.
            with move_lock:
                state["proposal"] = None
            # Hand the request to the server link; if that fails, undo the "in progress" mark.
            if not server.submit(run_id, run_dir / "input.npz", run_dir / "result.npz",
                                 z_range=(args.min_depth, args.max_depth)):
                state["pending"] = None
                return jsonify(error="ContactGraspnet_Server is not ready."), 503
        # Success: tell the page which run was started.
        return jsonify(id=run_id)

    @app.post("/api/move/approve")
    def move_approve():
        """The user approved the proposed move: start the Cartesian planner (once) for exactly that proposal."""
        # The page sends the run id it is showing, so an approval can never apply to a different (newer) proposal.
        run_id = str((request.get_json(silent=True) or {}).get("id", ""))
        # Fail early, with a clear message, if the planner script is missing (nothing has been changed yet).
        if not Path(args.planner_script).is_file():
            return jsonify(error=f"Planner script not found: {args.planner_script} (use --planner-script)."), 500
        with move_lock:
            proposal = state["proposal"]
            # No proposal, or the page is showing an older one that a newer capture has replaced.
            if proposal is None or proposal["run_id"] != run_id:
                return jsonify(error="That proposal is no longer current. Look at the latest result."), 409
            # Only a proposal that is still waiting for an answer can be approved (this makes it single-use).
            if proposal["status"] != "awaiting":
                return jsonify(error=f"This proposal can no longer be approved (it is {proposal['status']})."), 409
            # Too old: the arm may have moved since the capture.
            if time.time() > proposal["expires_at"]:
                proposal["status"] = "expired"
                proposal["message"] = "This proposal expired. Press Capture & run again to get a fresh grasp."
                return jsonify(error=proposal["message"]), 409
            # Mark it running INSIDE the lock, so a second click cannot start a second move.
            proposal["status"], proposal["message"], proposal["log"] = "running", "Planning and moving the robot...", []
        # The planner takes a while, so it runs in its own thread and this request returns at once.
        threading.Thread(target=run_move, args=(proposal,), daemon=True, name="robot-move").start()
        return jsonify(ok=True)

    @app.post("/api/move/decline")
    def move_decline():
        """The user rejected the proposed move: nothing is sent to the robot."""
        run_id = str((request.get_json(silent=True) or {}).get("id", ""))
        with move_lock:
            proposal = state["proposal"]
            # Only an unanswered proposal can be declined; anything else is left untouched.
            if proposal is not None and proposal["run_id"] == run_id and proposal["status"] == "awaiting":
                proposal["status"], proposal["message"] = "declined", "Move declined. The robot was not moved."
        return jsonify(ok=True)

    @app.post("/api/server/restart")
    def restart_server():
        # Restart in a background thread so this request returns immediately (a restart takes a while).
        threading.Thread(target=server.restart, daemon=True).start()
        return jsonify(ok=True)

    @app.get("/api/result/<run_id>/scene.json")
    def result_scene(run_id):
        # Validate the id and locate the run's folder, then the file.
        path = run_dir_for(run_id) / "scene.json"
        # File not made (yet): 404.
        if not path.is_file():
            abort(404)
        return no_store(send_file(path, mimetype="application/json"))

    @app.get("/api/result/<run_id>/overlay.jpg")
    def result_overlay(run_id):
        path = run_dir_for(run_id) / "overlay.jpg"
        if not path.is_file():
            abort(404)
        return no_store(send_file(path, mimetype="image/jpeg"))

    # Return the finished application to main().
    return app


def make_callbacks(state: dict, out_root: Path, args):
    """Create the two functions ServerLink calls when a request finishes or fails."""

    def on_done(run_id, info):
        # Called when the model finishes a request successfully.
        try:
            # Build the overlay image and the 3D scene file from the result.
            summary = build_views(out_root / run_id)

            # Turn the best grasp into a move proposal for the user to approve (or None if there are no grasps).
            best = summary["best_grasp"]
            proposal = None
            if best is not None:
                proposal = {"run_id": run_id, **best, "frame": args.move_frame, "log": [],
                            # Shown in the approval panel: the robot first stops this far (m) back from the grasp.
                            "approach_distance": args.approach_distance,
                            # Deadline after which the proposal can no longer be approved.
                            "expires_at": time.time() + args.approval_timeout}
                if best["score"] >= args.min_score:
                    # Good enough: wait for the user's decision.
                    proposal["status"], proposal["message"] = "awaiting", ""
                else:
                    # Too weak to offer. A "skipped" proposal is still shown, so the user knows why there is no prompt.
                    proposal["status"] = "skipped"
                    proposal["message"] = (f"Best score {best['score']:.3f} is below --min-score "
                                           f"{args.min_score}; no move is proposed.")
            # Publish it (under its lock, because the web routes and the move thread also use it).
            with state["move_lock"]:
                state["proposal"] = proposal

            # The info saved when this capture started (prompt, time, ...).
            meta = state["meta"].get(run_id, {})
            # The summary the web page shows for the newest result.
            result = {"id": run_id, "prompt": meta.get("prompt", ""), "captured_at": meta.get("captured_at", ""),
                      "seconds": info.get("seconds"), **summary}
            # Publish it so /api/status reports it.
            state["latest"] = result
            # Update meta.json on disk with the result numbers as well (this now includes the best grasp's pose).
            (out_root / run_id / "meta.json").write_text(json.dumps({**meta, **result}, indent=2))
        except Exception as exc:  # noqa: BLE001
            # Building the views failed: show why.
            state["last_error"] = f"Inference finished but building the views failed: {type(exc).__name__}: {exc}"
        finally:
            # Whatever happened, this capture is no longer in progress.
            state["pending"] = None

    def on_error(run_id, message):
        # Called when a request fails: remember the reason and free the "busy" flag.
        state["last_error"] = f"Inference failed: {message}"
        state["pending"] = None

    # Hand both functions back to main().
    return on_done, on_error


def main():
    """Parse options, start the camera and server link, and run the web server until stopped."""
    # Define every command-line option.
    parser = argparse.ArgumentParser(description="Gungnir_Client: browser UI for the L515 + Contact-GraspNet pipeline")
    # The server address is optional and positional: `python Gungnir_Client.py 192.168.1.50`.
    parser.add_argument("server", nargs="?", default=os.environ.get("GUNGNIR_SERVER", "127.0.0.1"),
                        help="ContactGraspnet_Server address, e.g. 192.168.1.50 (default: $GUNGNIR_SERVER or 127.0.0.1)")
    parser.add_argument("--token", default=os.environ.get("GUNGNIR_TOKEN"), help="shared secret (default: $GUNGNIR_TOKEN)")
    # Network address and port of THIS program's web page: localhost only by default, so only an SSH tunnel can reach it.
    parser.add_argument("--host", default="127.0.0.1", help="web page bind address (default: localhost only; use an SSH tunnel)")
    parser.add_argument("--port", type=int, default=8000, help="web page port")
    # Depth range in metres: used for the depth colour map and sent to the model as its crop.
    parser.add_argument("--min-depth", type=float, default=0.1, help="metres; also the depth crop sent to the model")
    parser.add_argument("--max-depth", type=float, default=9)
    # Where captures and results are saved.
    parser.add_argument("--output-dir", default=str(ROOT / "output" / "runs"))
    # Frame rate of the live view.
    parser.add_argument("--stream-fps", type=float, default=15, help="live-view rate; lower it over a slow link")
    # Flag (no value): swap the red and blue channels.
    parser.add_argument("--swap-rb", action="store_true", help="flip red/blue if the live colours look wrong")
    # --- robot motion (all of these only matter after the user approves a proposed move) ---
    # The planner script that executes the move. Assumed to sit next to this file; its file name is a guess, so
    # change the default (or pass the option) if your script is called something else.
    parser.add_argument("--planner-script", default=str(ROOT / "Cartesian_Path_Planner.py"),
                        help="path of the Cartesian path planner script (default: Cartesian_Path_Planner.py next to this file)")
    # The frame the grasp pose is expressed in. Contact-GraspNet returns poses in the frame of the point cloud, which
    # is the camera frame. If l515_node aligns depth to the COLOUR image (the intrinsics K come from the colour
    # camera_info), the matching frame is camera_color_optical_frame instead.
    parser.add_argument("--move-frame", default="camera_depth_optical_frame",
                        help="TF frame of the grasp poses, passed to the planner as --frame")
    # Passed straight through to the planner: the end-effector link, the MoveIt group, and the speed scaling.
    parser.add_argument("--move-eef", default="tcp_frame", help="end-effector link, passed to the planner as --eef")
    parser.add_argument("--move-group", default="manipulator", help="MoveIt planning group, passed as --group")
    parser.add_argument("--move-speed", type=float, default=0.2, help="velocity/acceleration scaling, 0 < x <= 1")
    # The robot first moves to a point this far back from the grasp, along the approach (z) axis, stops, and then
    # moves in to the grasp. 0 = go straight to the grasp.
    parser.add_argument("--approach-distance", type=float, default=0.05,
                        help="metres to stop short of the grasp along the approach axis first (0 = straight in)")
    # A frame that does not move with the arm (the camera does), used by the planner to fix both points in place.
    parser.add_argument("--move-fixed-frame", default="base_link",
                        help="frame fixed to the robot base, passed to the planner as --fixed-frame")
    # Grasps scoring below this are not offered to the robot (0 = always offer the best one).
    parser.add_argument("--min-score", type=float, default=0.0, help="do not propose a move if the best score is lower")
    # How long a proposal stays approvable (a grasp is only valid for the arm position at capture time).
    parser.add_argument("--approval-timeout", type=float, default=120.0,
                        help="seconds a proposed move can be approved before it expires")
    args = parser.parse_args()
    # The planner expects a speed scaling factor in (0, 1]; reject anything else before the page starts.
    if not 0.0 < args.move_speed <= 1.0:
        parser.error("--move-speed must be greater than 0 and at most 1")
    # The planner accepts 0 to 0.5 m; reject anything else here so the mistake shows at start-up.
    if not 0.0 <= args.approach_distance <= 0.5:
        parser.error("--approach-distance must be between 0 and 0.5 metres")

    # Make sure the output folder exists (creating parents; fine if it already does).
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Shared state between the web routes and the callbacks:
    # pending = id of the running capture; last_error = message to show; latest = newest result summary;
    # meta = per-run information saved at capture time;
    # proposal = the best-grasp robot move (dict, see make_callbacks) or None; move_lock = the lock that guards it.
    state = {"pending": None, "last_error": None, "latest": None, "meta": {},
             "proposal": None, "move_lock": threading.Lock()}
    # The two functions ServerLink calls on success / failure.
    on_done, on_error = make_callbacks(state, out_root, args)
    # Initialise ROS 2 before constructing CameraFeed, because CameraFeed is now a ROS 2 Node.
    rclpy.init(args=None)
    # Keep nullable handles so the finally block also works if either constructor raises an exception.
    server = None
    feed = None

    # Only print warnings from the web server; the page polls /api/status twice a second and would flood the terminal.
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    def terminate(signum, frame):
        # Turn a termination signal into KeyboardInterrupt, which the try/finally below handles like Ctrl-C.
        raise KeyboardInterrupt  # run the same cleanup as Ctrl-C (close the camera)

    # kill / systemd stop (SIGTERM) now runs the cleanup too.
    signal.signal(signal.SIGTERM, terminate)
    if hasattr(signal, "SIGHUP"):  # an SSH session dropping sends SIGHUP; without this the camera is never closed
        signal.signal(signal.SIGHUP, terminate)
    try:
        # Create the HTTP server link and the ROS 2 camera subscriber node inside the cleanup-protected block.
        server = ServerLink(server_url(args.server), args.token, on_done, on_error)
        feed = CameraFeed(args.min_depth, args.max_depth, args.stream_fps, args.swap_rb)
        # Start the camera and the server connection first, so they come up while FastSAM loads in create_app().
        feed.start()
        server.start()
        print(f"Connecting to ContactGraspnet_Server at {server.url} ... loading the segmentation model", flush=True)
        # Build the web app (loads FastSAM: the slow part).
        app = create_app(feed, server, state, args)
        print(f"Serving on http://{args.host}:{args.port}  (results in {out_root})", flush=True)
        # Run the web server until interrupted. threaded=True serves several browsers at once (the streams keep
        # connections open); use_reloader=False stops Flask from starting a second copy that would also open the camera.
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    finally:
        # Stop the HTTP server link if its constructor completed successfully.
        if server is not None:
            server.stop()
        # Stop and destroy the ROS subscriber node if its constructor completed successfully.
        if feed is not None:
            feed.stop()
            feed.destroy_node()
        # Shut down the ROS 2 client library cleanly after all node resources are released.
        if rclpy.ok():
            rclpy.shutdown()


# The web page, sent to the browser as one HTML document. The Python r prefix keeps backslashes (such as the \n
# in the JavaScript) exactly as written. Comments inside it are HTML <!-- --> comments, CSS /* */ comments and
# JavaScript // comments; the browser ignores them.
INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<!-- Make the page scale properly on small screens -->
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gungnir Client</title>
<style>
  /* Colour palette (reused through the stylesheet) and box sizing */
  :root { --bg:#0f1318; --panel:#171d25; --line:#2a3340; --text:#e6edf3; --muted:#8b98a8;
          --accent:#3b82f6; --ok:#22c55e; --warn:#f59e0b; --bad:#ef4444; --best:#ff00ff; }
  * { box-sizing: border-box; }
  /* Page background, text colour and font */
  body { margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui, sans-serif; }
  /* Top bar holding the title, the status pills and the link to the output window */
  header { display:flex; flex-wrap:wrap; gap:10px; align-items:center; padding:10px 16px;
           border-bottom:1px solid var(--line); background:var(--panel); }
  header h1 { font-size:16px; margin:0 12px 0 0; }
  /* Small rounded status badges; the ok/warn/bad variants change the colour */
  .pill { padding:2px 10px; border-radius:999px; border:1px solid var(--line); font-size:12px; color:var(--muted); }
  .pill.ok { color:var(--ok); border-color:var(--ok); } .pill.warn { color:var(--warn); border-color:var(--warn); }
  .pill.bad { color:var(--bad); border-color:var(--bad); }
  /* Flexible gap that pushes the link to the right edge of the header */
  header .spacer { flex:1; }
  a { color:var(--accent); }
  /* Page body: sections stacked with a gap between them */
  main { padding:16px; display:grid; gap:16px; }
  section { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; }
  /* The robot-move panel gets a magenta border (the same colour as the best grasp in the images) so it is
     clearly the part of the page that can move hardware */
  section#move-panel { border-color:var(--best); }
  /* Small uppercase section headings */
  h2 { font-size:13px; margin:0 0 8px; color:var(--muted); font-weight:600; letter-spacing:.04em; text-transform:uppercase; }
  /* Two equal columns; collapses to one column on narrow screens (see the media query) */
  .two { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
  @media (max-width: 900px) { .two { grid-template-columns:1fr; } }
  figure { margin:0; } figcaption { color:var(--muted); font-size:12px; margin-top:4px; }
  /* Camera/overlay images: full width, 4:3 shape, black background behind them */
  img.view { width:100%; display:block; background:#000; border-radius:4px; aspect-ratio:4/3; object-fit:contain; }
  /* Box that Plotly draws the 3D scene into (also shows placeholder text before the first result) */
  #scene3d { width:100%; height:480px; background:#000; border-radius:4px; display:flex; align-items:center;
             justify-content:center; color:var(--muted); }
  /* Row with the prompt box and the buttons */
  .controls { display:flex; gap:10px; align-items:stretch; flex-wrap:wrap; }
  .controls input { flex:1; min-width:220px; padding:10px 12px; border-radius:6px; border:1px solid var(--line);
                    background:#0b0f14; color:var(--text); font:inherit; }
  /* Buttons: blue by default, greyed out when disabled, outlined for the secondary one */
  button { padding:10px 18px; border-radius:6px; border:0; background:var(--accent); color:#fff; font:inherit;
           font-weight:600; cursor:pointer; }
  button:disabled { background:#334155; color:#94a3b8; cursor:not-allowed; }
  button.secondary { background:transparent; border:1px solid var(--line); color:var(--muted); font-weight:400; padding:6px 12px; }
  /* The approve button is green: it is the one action that starts robot motion */
  button.go { background:#16a34a; }
  button.go:disabled { background:#334155; }
  /* Status/error line under the buttons (the .bad variant is red) */
  #message { margin-top:8px; min-height:1.4em; color:var(--muted); } #message.bad { color:var(--bad); }
  /* Same idea for the line under the approve/decline buttons (.ok = green, .bad = red) */
  #move-message { margin-top:8px; min-height:1.4em; color:var(--muted); }
  #move-message.ok { color:var(--ok); } #move-message.bad { color:var(--bad); }
  /* The pose numbers of the proposed move are shown in a fixed-width font so the columns line up */
  #move-summary { margin-bottom:10px; } #move-summary code { font-family:ui-monospace, monospace; color:var(--text); }
  /* One-line summary above the model output */
  #meta { color:var(--muted); margin-bottom:8px; }
  /* Collapsible log area with a scrollable box */
  details { margin-top:8px; } pre { max-height:220px; overflow:auto; background:#0b0f14; padding:8px; border-radius:4px;
           font-size:12px; color:#9fb0c3; }
  /* In the separate output window (?view=results) hide everything marked live-only */
  body.results-only .live-only { display:none; }
</style>
</head>
<body>
<!-- Header: title, two status pills (filled in by the JavaScript below), and the link to the output window -->
<header>
  <h1>Gungnir Client</h1>
  <span class="pill" id="pill-camera">camera ...</span>
  <span class="pill" id="pill-server">model ...</span>
  <span class="spacer"></span>
  <!-- target="grasp-output" reuses the same browser window/tab each time the link is clicked -->
  <span class="live-only"><a href="/?view=results" target="grasp-output">Open output in a separate window</a></span>
</header>
<main>
  <!-- Live camera section: the two img tags load the endless MJPEG streams from this program -->
  <section class="live-only" id="live">
    <h2>Live camera</h2>
    <div class="two">
      <figure><img class="view" src="/stream/rgb" alt="RGB stream"><figcaption>RGB</figcaption></figure>
      <figure><img class="view" src="/stream/depth" alt="Depth stream">
        <figcaption>Depth: <span id="depth-range"></span> (blue = near, red = far; black = outside the range or no return, ignored by the model)</figcaption></figure>
    </div>
  </section>
  <!-- Capture section: prompt text box, the capture button, the restart button, a message line and the log -->
  <section class="live-only" id="controls">
    <h2>Capture</h2>
    <div class="controls">
      <input id="prompt" type="text" maxlength="2000" placeholder="Prompt (saved with each capture; not used by the model yet)">
      <!-- Starts disabled; the JavaScript enables it when the camera and model are both ready -->
      <button id="capture" disabled>Capture &amp; run</button>
      <button id="restart" class="secondary" title="Restart the model on the GPU computer (use after a CUDA error)">Restart model</button>
    </div>
    <div id="message"></div>
    <details><summary>ContactGraspnet_Server log</summary><pre id="log"></pre></details>
  </section>
  <!-- Model output section: a summary line, the interactive 3D scene, and the 2D overlay image -->
  <section>
    <h2>Model output</h2>
    <div id="meta">No result yet. Press "Capture &amp; run".</div>
    <div class="two">
      <figure><div id="scene3d">3D scene appears here</div><figcaption>3D point cloud and grasps (drag to rotate, scroll to zoom). The best grasp is the thick magenta one.</figcaption></figure>
      <figure><img class="view" id="overlay" alt="Grasps overlaid on the captured RGB image"><figcaption>Grasps projected onto the captured RGB image (gold = top 10%, green = top 30%, magenta = best grasp, proposed for the robot)</figcaption></figure>
    </div>
  </section>
  <!-- Proposed robot move: hidden until a result has a best grasp. The buttons only send the user's decision to
       the server; nothing moves unless the user approves (and confirms the browser dialog). -->
  <section id="move-panel" hidden>
    <h2>Proposed robot move</h2>
    <div id="move-summary"></div>
    <div class="controls">
      <button id="move-approve" class="go" disabled>Approve &amp; move robot</button>
      <button id="move-decline" class="secondary" disabled>Decline</button>
    </div>
    <div id="move-message"></div>
    <details><summary>Cartesian planner output</summary><pre id="move-log"></pre></details>
  </section>
</main>
<!-- Plotly's JavaScript library, served by this same program (no internet needed) -->
<script src="/plotly.min.js"></script>
<script>
  // If the address ends in ?view=results this is the separate output window: hide the live parts
  // (via the results-only CSS class) and retitle the tab.
  if (new URLSearchParams(location.search).get('view') === 'results') {
    document.body.classList.add('results-only');
    document.title = 'Gungnir output';
  }
  // Shortcut: $('some-id') returns the page element with that id.
  const $ = id => document.getElementById(id);
  // shownResult: id of the result currently displayed; submitting: true while a capture request is being sent.
  let shownResult = null, submitting = false;
  // lastProposal: the move proposal most recently received from the server (the one the buttons act on).
  // moveError: an error from an approve/decline request, shown under the buttons while the proposal is awaiting.
  // movePosting: true while an approve/decline request is being sent (blocks double clicks).
  let lastProposal = null, moveError = null, movePosting = false;

  // Set a pill's text and colour class (ok / warn / bad, or none).
  function setPill(el, text, cls) { el.textContent = text; el.className = 'pill ' + (cls || ''); }
  // Show a message under the buttons, in red when `bad` is true.
  function say(text, bad) { $('message').textContent = text || ''; $('message').className = bad ? 'bad' : ''; }

  // Update the whole page from one status object returned by /api/status.
  function render(s) {
    const cam = s.camera;
    // Camera pill: fps when working, the error text when not, otherwise "starting".
    setPill($('pill-camera'),
            cam.ok ? `camera ${cam.fps} fps` : (cam.error ? 'camera: ' + cam.error.slice(0, 60) : 'camera starting...'),
            cam.ok ? 'ok' : (cam.error ? 'bad' : 'warn'));
    // Model pill: its state, plus the restart count if it has restarted; the colour follows the state.
    const w = s.server, cls = {ready:'ok', busy:'warn', starting:'warn', error:'bad', stopped:'bad'}[w.state];
    setPill($('pill-server'), 'model: ' + w.state + (w.restarts ? ` (restarts: ${w.restarts})` : ''), cls);
    // Show the depth range used by the colour map.
    $('depth-range').textContent = `${s.config.min_depth}-${s.config.max_depth} m`;
    // True while the robot is executing an approved move (a new capture is not allowed then).
    const moving = !!(s.proposal && s.proposal.status === 'running');
    // The Capture button is enabled only when the model is ready, the camera works, nothing is running,
    // no request is already being sent and the robot is not moving.
    $('capture').disabled = !(w.state === 'ready' && cam.ok && !s.busy && !submitting && !moving);
    // Message line: an error wins, then "running", then the model's own status text when it is not ready.
    if (s.last_error) say(s.last_error, true);
    else if (s.busy) say('Running inference on the captured frame...');
    else if (w.state !== 'ready') say(w.message, w.state === 'error');
    else if (!submitting) say('');
    // Fill the log box with the model server's recent output.
    $('log').textContent = (w.log || []).join('\n');
    // If a newer result exists than the one displayed, load it.
    if (s.result && s.result.id !== shownResult) loadResult(s.result);
    // Update the robot-move panel (shown only when there is a proposal).
    renderMove(s.proposal);
  }

  // Update the "Proposed robot move" panel from the proposal object in the status (null = no proposal).
  function renderMove(p) {
    const panel = $('move-panel');
    // A different proposal than before: forget any old request error.
    if (!p || !lastProposal || p.run_id !== lastProposal.run_id) moveError = null;
    lastProposal = p;
    // No proposal (no result yet, no grasps found, or a new capture is running): hide the whole panel.
    if (!p) { panel.hidden = true; return; }
    panel.hidden = false;
    // Format the target pose: position in metres, orientation as roll/pitch/yaw in radians (what the planner takes).
    const f = v => v.toFixed(3);
    $('move-summary').innerHTML =
      `Best grasp #${p.index} (score <b>${p.score.toFixed(3)}</b>), drawn in magenta above.<br>` +
      `Target in <code>${p.frame}</code>: ` +
      `x <code>${f(p.position[0])}</code> y <code>${f(p.position[1])}</code> z <code>${f(p.position[2])}</code> m, ` +
      `roll <code>${f(p.rpy[0])}</code> pitch <code>${f(p.rpy[1])}</code> yaw <code>${f(p.rpy[2])}</code> rad.` +
      // One extra sentence when the approach move is on: the robot stops short of the grasp first.
      (p.approach_distance > 0
        ? `<br>The robot first stops <b>${(p.approach_distance * 100).toFixed(1)} cm</b> back from this point ` +
          `along the approach axis, then moves in.`
        : '');
    // The buttons work only while the proposal is waiting for an answer and no request is being sent.
    const awaiting = p.status === 'awaiting';
    $('move-approve').disabled = !awaiting || movePosting;
    $('move-decline').disabled = !awaiting || movePosting;
    // The line under the buttons: what the user is being asked, or how the move turned out.
    let text = p.message, kind = '';
    if (awaiting) text = moveError || `Move the robot to this grasp? This proposal expires in ${p.seconds_left} s.`;
    if (awaiting && moveError) kind = 'bad';
    else if (p.status === 'succeeded') kind = 'ok';
    else if (p.status === 'failed') kind = 'bad';
    $('move-message').textContent = text;
    $('move-message').className = kind;
    // The planner's own output (live while it runs).
    $('move-log').textContent = (p.log || []).join('\n');
  }

  // Send the user's decision (approve or decline) for the proposal that is on screen.
  async function sendMoveDecision(kind) {
    if (!lastProposal) return;
    movePosting = true; moveError = null;
    $('move-approve').disabled = true; $('move-decline').disabled = true;
    try {
      // The proposal's run id travels with the request, so the server can reject it if a newer proposal exists.
      const r = await fetch('/api/move/' + kind, {method: 'POST', headers: {'Content-Type': 'application/json'},
                                                  body: JSON.stringify({id: lastProposal.run_id})});
      if (!r.ok) moveError = (await r.json()).error || 'The request failed';
    } catch (e) { moveError = 'Request failed: ' + e; }
    movePosting = false;
  }

  // Approve button: ask for one explicit confirmation (it moves real hardware), then tell the server to go ahead.
  $('move-approve').addEventListener('click', () => {
    if (!lastProposal) return;
    const ok = confirm('The robot will move to the magenta grasp.\n\n' +
                       'Make sure the workspace is clear and the emergency stop is within reach.\n\nMove the robot now?');
    if (ok) sendMoveDecision('approve');
  });
  // Decline button: tell the server not to move; nothing is sent to the robot.
  $('move-decline').addEventListener('click', () => sendMoveDecision('decline'));

  // Display one finished result: the summary line, the 2D overlay and the 3D scene.
  async function loadResult(r) {
    // Remember it first, so it is not loaded twice while the downloads are running.
    shownResult = r.id;
    const best = r.best_score == null ? 'n/a' : r.best_score.toFixed(3);
    // Summary line: time, grasp count, best score, inference time, and the prompt if there is one.
    $('meta').textContent = `${r.captured_at} | ${r.num_grasps} grasps | best score ${best} | inference ${r.seconds}s` +
                            (r.prompt ? ` | prompt: ${r.prompt}` : '');
    // Changing the image address makes the browser download the new overlay picture.
    $('overlay').src = `/api/result/${r.id}/overlay.jpg`;
    try {
      // Download the 3D scene description, clear the placeholder text, and let Plotly draw it
      // (react updates the existing plot if there is one).
      const fig = await (await fetch(`/api/result/${r.id}/scene.json`)).json();
      $('scene3d').textContent = '';
      Plotly.react('scene3d', fig.data, fig.layout, {responsive: true, displaylogo: false});
    } catch (e) { $('scene3d').textContent = '3D scene failed to load: ' + e; }
  }

  // Capture button: ask this program to capture the current frame and run the model.
  $('capture').addEventListener('click', async () => {
    // Block double clicks while the request is being sent.
    submitting = true; $('capture').disabled = true; say('Capturing...');
    try {
      // POST the prompt text as JSON.
      const r = await fetch('/api/capture', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                             body: JSON.stringify({prompt: $('prompt').value})});
      // On failure (not ready, no frame...) show the explanation.
      if (!r.ok) say((await r.json()).error || 'Capture failed', true);
    } catch (e) { say('Capture request failed: ' + e, true); }
    submitting = false;
  });
  // Restart button: ask this program to have the GPU computer restart the model.
  $('restart').addEventListener('click', () => fetch('/api/server/restart', {method: 'POST'}));

  // Poll for status twice a second. The next poll is scheduled only after the current one finishes, so slow
  // responses cannot pile up.
  async function poll() {
    try { render(await (await fetch('/api/status', {cache: 'no-store'})).json()); }
    catch (e) { setPill($('pill-camera'), 'server unreachable', 'bad'); }
    setTimeout(poll, 500);
  }
  // Start polling as soon as the page has loaded.
  poll();
</script>
</body>
</html>
"""


# Run main() only when this file is started directly (not when it is imported, e.g. by a test).
if __name__ == "__main__":
    main()