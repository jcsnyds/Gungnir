#!/usr/bin/env python3
"""Gungnir_Client: browser UI for the L515 + Contact-GraspNet pipeline. Runs on the camera computer.

    python Gungnir_Client.py <gpu-computer-ip>        # then open http://localhost:8000

Start ContactGraspnet_Server.py on the GPU computer first (it prints this exact command). To view the page
from your own computer: ssh -L 8000:localhost:8000 <user>@<camera-computer>, then browse to http://localhost:8000.

The page shows the live RGB and depth streams, a prompt box (saved with each capture, not used by the model
yet), and a Capture & Run button. The model output appears next to the stream and in a separate output window
(http://localhost:8000/?view=results) that follows the latest result.

Architecture: this process owns the camera and the SAM2 segmentation model and never imports TensorFlow.
Each capture is uploaded over HTTP to ContactGraspnet_Server (which keeps the model loaded on the GPU
computer); the result is downloaded and drawn here. The camera can only be opened by one process: stop Odin.py
before starting this. There is no authentication on the web page: the default bind address is localhost; use
--host 0.0.0.0 only on a network you trust.

Server address: the first argument (or $GUNGNIR_SERVER); "192.168.1.50", "192.168.1.50:8765" and
"http://192.168.1.50:8765" all work. If the server was started with a token, pass --token (or $GUNGNIR_TOKEN).

AI use: Written by Claude (Anthropic; Claude Sonnet 5.5 via claude.ai), October 2026, from the author's
requirements and existing code (l515.py, inference.py, data.py, visualization_utils.py). Originally named
grasp_ui.py. November 2026 revision: the model now runs on a separate computer, so the subprocess supervisor
was replaced by ServerLink (HTTP), startup was streamlined, and the comments were condensed. Claude tested
this only against a simulated camera, segmentation model and Contact-GraspNet server, not the real hardware.
Reviewed and tested by: Ryan Poliquin
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
from flask import Flask, Response, abort, jsonify, request, send_file

ROOT = Path(__file__).resolve().parent
DEFAULT_PORT = 8765  # ContactGraspnet_Server's default port
INFER_TIMEOUT = 300  # seconds to wait for one inference (upload + model + download)
JPEG_QUALITY = 70  # quality of the live-view JPEGs
# A valid run id looks like 20261017-143005 or 20261017-143005-2; anything else is rejected (path traversal).
RUN_ID_RE = re.compile(r"\d{8}-\d{6}(-\d+)?")
# Same tiers as visualization_utils.visualize_grasps: gold = top 10%, green = top 30%, the rest faint grey (BGR).
GOLD_BGR, GREEN_BGR, GREY_BGR = (0, 215, 255), (50, 205, 50), (170, 170, 170)


# ======================================================================================
# Camera
# ======================================================================================
class CameraFeed(threading.Thread):
    """Reads frames from the L515 in a loop and keeps the latest one (plus pre-encoded JPEGs for the streams)."""

    def __init__(self, min_depth, max_depth, stream_fps, swap_rb):
        super().__init__(daemon=True, name="camera")
        self.min_depth, self.max_depth = float(min_depth), float(max_depth)  # depth range (m) of the colour map
        self.min_interval = 1.0 / max(1.0, float(stream_fps))  # minimum seconds between encoded stream frames
        self.swap_rb = bool(swap_rb)
        self.cond = threading.Condition()  # lock + wake-up for stream readers
        self.seq = 0  # number of encoded frames so far; readers use it to spot new frames
        self.depth = None  # float32 HxW, metres (0 = no return)
        self.rgb = None  # uint8 HxWx3, RGB order
        self.K = None  # 3x3 float32 intrinsics of the colour stream (depth is aligned to it)
        self.rgb_jpeg = self.depth_jpeg = None
        self.error = None  # latest camera error (None = fine)
        self.fps = 0.0
        self.last_frame_time = 0.0
        self.size = None  # (width, height)
        self._halt = threading.Event()  # not "_stop": that name belongs to threading.Thread

    def colorize_depth(self, depth: np.ndarray) -> np.ndarray:
        """Turbo colormap over [min_depth, max_depth]; black where the model ignores the pixel."""
        scaled = np.clip((depth - self.min_depth) / (self.max_depth - self.min_depth), 0.0, 1.0)
        bgr = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        bgr[~((depth >= self.min_depth) & (depth <= self.max_depth))] = 0  # outside the range or no return
        return bgr

    def snapshot(self):
        """Latest (depth, rgb, K), or None before the first frame. The arrays are never mutated afterwards."""
        with self.cond:
            if self.depth is None or self.K is None:
                return None
            return self.depth, self.rgb, self.K

    @property
    def ok(self) -> bool:
        return self.seq > 0 and (time.time() - self.last_frame_time) < 3.0

    def stop(self):
        self._halt.set()
        self.join(timeout=5)

    def run(self):
        # Imported here so a missing driver is reported in the UI instead of crashing start-up.
        try:
            from l515 import L515  # installed by IntelRealsense_Driver/setup.py
        except Exception as exc:  # noqa: BLE001
            self.error = f"Cannot import l515: {exc}"
            print("[camera]", self.error, file=sys.stderr)
            return

        cam, failures = None, 0
        frames, window_start, last_encode = 0, time.time(), 0.0  # FPS window and last JPEG time

        while not self._halt.is_set():
            try:
                if cam is None:  # first iteration, or after a reopen
                    cam = L515()
                    self.K = None
                cam.capture_image()  # blocks until the next aligned colour+depth pair
                depth = cam.depth_image  # float32 metres
                raw = cam.color_image  # BGR as delivered by the native shim
                rgb = np.ascontiguousarray(raw if self.swap_rb else raw[..., ::-1])
                if self.K is None:  # the camera matrix does not change while streaming
                    self.K = np.asarray(cam.get_k(), dtype=np.float32)

                now = time.time()
                rgb_jpeg = depth_jpeg = None
                if now - last_encode >= self.min_interval:  # encode only as often as the stream FPS needs
                    params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                    rgb_jpeg = cv2.imencode(".jpg", rgb[..., ::-1], params)[1].tobytes()  # OpenCV wants BGR
                    depth_jpeg = cv2.imencode(".jpg", self.colorize_depth(depth), params)[1].tobytes()
                    last_encode = now

                with self.cond:  # readers never see a half-updated frame
                    self.depth, self.rgb = depth, rgb
                    self.size = (int(depth.shape[1]), int(depth.shape[0]))
                    if rgb_jpeg is not None:
                        self.rgb_jpeg, self.depth_jpeg = rgb_jpeg, depth_jpeg
                        self.seq += 1
                    self.last_frame_time = now
                    self.error = None
                    self.cond.notify_all()

                failures = 0
                frames += 1
                if now - window_start >= 2.0:
                    self.fps = frames / (now - window_start)
                    frames, window_start = 0, now
            except Exception as exc:  # noqa: BLE001
                failures += 1
                self.error = f"{type(exc).__name__}: {exc}"
                print(f"[camera] {self.error} (failure {failures})", file=sys.stderr)
                if failures >= 5 and cam is not None:  # reopen the device
                    try:
                        cam.close()
                    except Exception:  # noqa: BLE001
                        pass
                    cam = None
                time.sleep(min(2.0, 0.2 * failures))

        if cam is not None:  # stop() was called: release the camera so it can be used again
            try:
                cam.close()
            except Exception:  # noqa: BLE001
                pass

    def mjpeg(self, which: str):
        """Generator yielding an endless MJPEG stream ("rgb" or "depth") for one browser connection."""
        last = -1
        while True:
            with self.cond:
                self.cond.wait_for(lambda: self.seq != last, timeout=2.0)
                if self.seq == last:
                    frame = None
                else:
                    last = self.seq
                    frame = self.rgb_jpeg if which == "rgb" else self.depth_jpeg
            if frame is None:
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(frame) + frame + b"\r\n"


# ======================================================================================
# Link to ContactGraspnet_Server (on the GPU computer)
# ======================================================================================
def server_url(spec: str) -> str:
    """Accept "host", "host:port" or a full URL and return "http://host:port"."""
    spec = spec.strip()
    parsed = urlparse(spec if "://" in spec else "http://" + spec)
    if not parsed.hostname:
        raise SystemExit(f"Bad server address: {spec!r}")
    return f"{parsed.scheme}://{parsed.hostname}:{parsed.port or DEFAULT_PORT}"


class ServerLink:
    """Talks HTTP to ContactGraspnet_Server: polls /health for its status and uploads captures to /infer."""

    POLL_INTERVAL = 1.0  # seconds between status checks
    UNREACHABLE_AFTER = 5  # failed checks in a row before the state becomes "error" (restarts take a few seconds)

    def __init__(self, url, token, on_done, on_error):
        self.url, self.token = url, token
        self.on_done, self.on_error = on_done, on_error
        self.lock = threading.RLock()
        self.state = "starting"  # starting | ready | busy | error | stopped
        self.message = f"Connecting to ContactGraspnet_Server at {url} ..."
        self.pending = None  # run id in flight (None when idle)
        self.restarts = 0  # server restarts seen
        self.forward_passes = None  # model setting, reported by the server
        self._log, self._boot_id, self._misses = [], None, 0
        self._halt = threading.Event()

    # -- HTTP ----------------------------------------------------------------------
    def _open(self, path, data=None, headers=None, timeout=5.0):
        req = urllib.request.Request(self.url + path, data=data)  # GET if data is None, else POST
        if self.token:
            req.add_header("Authorization", "Bearer " + self.token)
        for key, value in (headers or {}).items():
            req.add_header(key, value)
        return urllib.request.urlopen(req, timeout=timeout)

    @staticmethod
    def _http_error(exc: urllib.error.HTTPError) -> str:
        if exc.code == 401:
            return "ContactGraspnet_Server rejected the token (HTTP 401): use the same --token on both computers."
        try:
            detail = json.loads(exc.read().decode()).get("error")
        except Exception:  # noqa: BLE001
            detail = None
        return detail or f"ContactGraspnet_Server returned HTTP {exc.code}."

    # -- lifecycle -------------------------------------------------------------------
    def start(self):
        threading.Thread(target=self._poll, daemon=True, name="server-health").start()

    def stop(self):  # the remote server keeps running; it belongs to the other computer
        self._halt.set()
        with self.lock:
            self.state = "stopped"

    def restart(self):
        """Ask the server to restart its model (the 'Restart model' button, e.g. after a CUDA error)."""
        try:
            with self._open("/restart", data=b""):
                pass
        except urllib.error.HTTPError as exc:
            note = self._http_error(exc)
        except Exception as exc:  # noqa: BLE001
            note = f"Restart request failed: {exc}"
        else:
            with self.lock:
                self.state, self.message = "starting", "Restart requested; waiting for ContactGraspnet_Server..."
            return
        with self.lock:
            self.message = note

    def log_tail(self, n=20):
        with self.lock:
            return self._log[-n:]

    # -- status polling ----------------------------------------------------------------
    def _poll(self):
        while not self._halt.is_set():
            try:
                with self._open("/health", timeout=3.0) as reply:
                    self._apply(json.loads(reply.read().decode()))
            except urllib.error.HTTPError as exc:  # answered, but with an error (usually a wrong token)
                self._miss(self._http_error(exc), immediately=True)
            except Exception as exc:  # noqa: BLE001
                self._miss(f"Cannot reach ContactGraspnet_Server at {self.url}: {exc}")
            self._halt.wait(self.POLL_INTERVAL)

    def _apply(self, info):
        with self.lock:
            self._misses = 0
            self._log = list(info.get("log", []))
            if info.get("forward_passes") is not None:
                self.forward_passes = info["forward_passes"]
            boot = info.get("boot_id")  # changes whenever the server process is replaced
            if self._boot_id is not None and boot != self._boot_id:
                self.restarts += 1
            self._boot_id = boot
            remote = info.get("state", "starting")
            if self.pending is not None and remote != "starting":
                self.state, self.message = "busy", "Running inference..."  # our own request is in flight
            else:
                self.state, self.message = remote, info.get("message", "")

    def _miss(self, message, immediately=False):
        with self.lock:
            self._misses += 1
            if immediately or self._misses >= self.UNREACHABLE_AFTER:
                self.state, self.message = "error", message

    # -- requests ----------------------------------------------------------------------
    def submit(self, run_id, input_path, output_path, z_range) -> bool:
        """Start one inference request. Returns False if the server is not ready."""
        with self.lock:
            if self.state != "ready":
                return False
            self.state, self.message, self.pending = "busy", "Running inference...", run_id
        threading.Thread(target=self._run, args=(run_id, input_path, output_path, z_range),
                         daemon=True, name="infer-request").start()
        return True

    def _run(self, run_id, input_path, output_path, z_range):
        """Upload the input file, wait for the result, save it. Runs in its own thread."""
        started = time.time()
        try:
            headers = {"Content-Type": "application/octet-stream", "X-Run-Id": run_id,
                       "X-Z-Min": str(z_range[0]), "X-Z-Max": str(z_range[1])}  # depth crop applied by the model
            with self._open("/infer", data=Path(input_path).read_bytes(), headers=headers,
                            timeout=INFER_TIMEOUT) as reply:
                payload = reply.read()
                info = {"num_grasps": int(reply.headers.get("X-Num-Grasps", 0)),
                        "seconds": float(reply.headers.get("X-Seconds", round(time.time() - started, 2)))}
            Path(output_path).write_bytes(payload)
            error = None
        except urllib.error.HTTPError as exc:
            error = self._http_error(exc)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        with self.lock:
            self.pending = None
            if error is None:
                self.state, self.message = "ready", "Model ready"
            else:  # the server usually restarts itself after an error; the next status check says when it is back
                self.state, self.message = "starting", "Inference failed; waiting for ContactGraspnet_Server..."
        if error is None:  # outside the lock: the callbacks do slow work
            self.on_done(run_id, info)
        else:
            self.on_error(run_id, error)


# ======================================================================================
# Segmentation model
# ======================================================================================
class SegmentationModel:
    """SAM2 automatic mask generator: turns an RGB image into an instance-id map."""

    def __init__(self):
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        checkpoint = "checkpoints/sam2.1_hiera_small.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam2 = build_sam2(model_cfg, checkpoint, device=device)
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2, points_per_side=32, pred_iou_thresh=0.7, stability_score_thresh=0.92,
            min_mask_region_area=100)

    def segment(self, image):
        masks = sorted(self.mask_generator.generate(image), key=lambda x: x["area"], reverse=True)  # largest first
        instance_map = np.zeros(image.shape[:2], dtype=np.uint16)
        for object_id, mask in enumerate(masks, start=1):
            # Only assign pixels that have not been assigned yet.
            instance_map[mask["segmentation"] & (instance_map == 0)] = object_id
        return instance_map


# ======================================================================================
# Output views (3D scene + 2D overlay)
# ======================================================================================
def grasp_points(grasps: np.ndarray, gripper_points: np.ndarray) -> np.ndarray:
    """Gripper wireframe points for every grasp in camera coordinates: (N,4,4),(K,3) -> (N,K,3)."""
    rot, trans = grasps[:, :3, :3], grasps[:, :3, 3]
    return np.einsum("nij,kj->nki", rot, gripper_points) + trans[:, None, :]  # rotation * point + translation


def score_tiers(scores: np.ndarray):
    """Boolean masks (gold, green, grey): top 10% / next 20% / rest."""
    if scores.size == 0:
        empty = np.zeros(0, dtype=bool)
        return empty, empty, empty
    p70, p90 = np.percentile(scores, [70, 90])
    gold = scores >= p90
    green = (scores >= p70) & ~gold
    return gold, green, ~(gold | green)


def draw_overlay(rgb, grasps, scores, K, gripper_points, segments, segmap=None) -> bytes:
    """Grasp wireframes projected onto the captured RGB image (JPEG bytes)."""
    img = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)  # OpenCV draws in BGR
    n = len(grasps)
    if segmap is not None:  # lightly overlay the colourised segmentation map
        num_objects = np.max(segmap)
        colors = np.random.default_rng(42).integers(0, 256, size=(num_objects + 1, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]  # background = black
        img = cv2.addWeighted(img, 0.9, colors[segmap], 0.1, 0)
    if n:
        pts = grasp_points(grasps, gripper_points)  # (N, K, 3) in camera coordinates
        z = pts[..., 2]
        ok = z > 1e-3  # points behind the camera cannot be projected
        safe_z = np.where(ok, z, 1.0)
        u = np.clip(K[0, 0] * pts[..., 0] / safe_z + K[0, 2], -1e4, 1e4)  # pinhole projection
        v = np.clip(K[1, 1] * pts[..., 1] / safe_z + K[1, 2], -1e4, 1e4)

        def draw(canvas, indices, color, thickness):
            for i in indices:
                for a, b in segments:
                    if ok[i, a] and ok[i, b]:
                        cv2.line(canvas, (int(round(u[i, a])), int(round(v[i, a]))),
                                 (int(round(u[i, b])), int(round(v[i, b]))), color, thickness, cv2.LINE_AA)

        gold, green, grey = score_tiers(scores)
        faint = img.copy()  # low-scoring grasps are drawn on a copy and blended in at 40%
        draw(faint, np.flatnonzero(grey), GREY_BGR, 1)
        img = cv2.addWeighted(faint, 0.4, img, 0.6, 0)
        draw(img, np.flatnonzero(green), GREEN_BGR, 2)
        draw(img, np.flatnonzero(gold), GOLD_BGR, 2)  # gold last, so it is never hidden

    label = f"{n} grasps" + (f" | best score {float(scores.max()):.2f}" if n else " | none found")
    cv2.putText(img, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)  # outline
    cv2.putText(img, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


def scene_figure_json(pc, colors, grasps, scores, gripper_points, segments, max_points=30000) -> str:
    """Interactive 3D scene as Plotly JSON. All grasps of one tier share ONE trace (not 6 per grasp)."""
    import plotly.graph_objects as go
    import plotly.io as pio

    traces = []
    if len(pc):
        keep = np.arange(len(pc))
        if len(pc) > max_points:  # too many points make the browser slow: fixed-seed random subset
            keep = np.random.default_rng(0).choice(len(pc), max_points, replace=False)
        pc = pc[keep]
        marker = dict(size=1.6, color="lightgray")
        if len(colors) > 0:  # use the model's per-point colours if they cover every kept point
            c = np.asarray(colors)
            if c.ndim == 2 and c.shape[1] == 3 and len(c) >= keep.max() + 1:
                c = c[keep]
                if c.dtype.kind == "f" and c.size and c.max() <= 1.0:
                    c = c * 255
                c = np.clip(c, 0, 255).astype(np.uint8)
                marker["color"] = [f"rgb({r},{g},{b})" for r, g, b in c]
        traces.append(go.Scatter3d(x=pc[:, 0], y=pc[:, 1], z=pc[:, 2], mode="markers",
                                   marker=marker, name="Point cloud", hoverinfo="skip"))

    if len(grasps):
        pts = grasp_points(grasps, gripper_points)

        def lines(indices):
            # Start/end of every segment of the selected grasps; a NaN row after each makes Plotly lift the pen,
            # so thousands of separate segments fit in one layer.
            a, b = pts[indices][:, segments[:, 0], :], pts[indices][:, segments[:, 1], :]
            return np.stack([a, b, np.full_like(a, np.nan)], axis=2).reshape(-1, 3)

        gold, green, grey = score_tiers(scores)
        for mask, name, color, width in ((grey, "other grasps", "rgba(130,130,130,0.35)", 3),
                                         (green, "top 30%", "limegreen", 4),
                                         (gold, "top 10%", "gold", 4)):
            idx = np.flatnonzero(mask)
            if idx.size:
                xyz = lines(idx)
                traces.append(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines",
                                           line=dict(color=color, width=width), name=name, hoverinfo="skip"))

    for axis, color in enumerate(("red", "green", "blue")):  # camera axes, 10 cm long
        end = np.zeros(3)
        end[axis] = 0.1
        traces.append(go.Scatter3d(x=[0, end[0]], y=[0, end[1]], z=[0, end[2]], mode="lines",
                                   line=dict(color=color, width=6), showlegend=False, hoverinfo="skip"))

    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_dark", margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=1.0),
        scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)", aspectmode="data",
                   # OpenCV camera frame has y pointing down: flip "up" and view from behind the camera.
                   camera=dict(up=dict(x=0, y=-1, z=0), eye=dict(x=0.0, y=-0.4, z=-1.6))))
    return pio.to_json(fig)


def build_views(run_dir: Path) -> dict:
    """Create overlay.jpg and scene.json for one finished run; returns summary numbers."""
    result, captured = np.load(run_dir / "result.npz"), np.load(run_dir / "input.npz")
    grasps, scores = result["grasps"], result["scores"]
    gp, seg = result["gripper_points"], result["gripper_segments"]
    (run_dir / "overlay.jpg").write_bytes(draw_overlay(captured["rgb"], grasps, scores, captured["K"], gp, seg))
    (run_dir / "scene.json").write_text(
        scene_figure_json(result["pc_full"], result["pc_colors"], grasps, scores, gp, seg))
    return {"num_grasps": int(len(grasps)), "best_score": float(scores.max()) if scores.size else None}


# ======================================================================================
# Web app
# ======================================================================================
def create_app(feed: CameraFeed, server: ServerLink, state: dict, args) -> Flask:
    """Build the Flask web application and register all of its URL routes."""
    app = Flask(__name__)
    out_root = Path(args.output_dir)
    lock = threading.Lock()  # keeps status reads and capture requests from interleaving
    plotly_js = {}  # caches Plotly's (large) JavaScript after the first request
    seg_model = SegmentationModel()  # SAM2

    def run_dir_for(run_id: str) -> Path:
        if not RUN_ID_RE.fullmatch(run_id):  # blocks "../" tricks
            abort(404)
        path = out_root / run_id
        if not path.is_dir():
            abort(404)
        return path

    def no_store(response):
        response.headers["Cache-Control"] = "no-store"  # always show the latest data
        return response

    @app.get("/")
    def index():
        return no_store(Response(INDEX_HTML, mimetype="text/html"))

    @app.get("/plotly.min.js")
    def plotly_bundle():  # served locally so the browser needs no internet access
        if "js" not in plotly_js:
            from plotly.offline import get_plotlyjs
            plotly_js["js"] = get_plotlyjs()
        response = Response(plotly_js["js"], mimetype="application/javascript")
        response.headers["Cache-Control"] = "public, max-age=86400"
        return response

    @app.get("/stream/<which>")
    def stream(which):
        if which not in ("rgb", "depth"):
            abort(404)
        return no_store(Response(feed.mjpeg(which), mimetype="multipart/x-mixed-replace; boundary=frame"))

    @app.get("/api/status")
    def status():
        with lock:
            payload = {
                "camera": {"ok": feed.ok, "fps": round(feed.fps, 1), "error": feed.error, "size": feed.size},
                "server": {"state": server.state, "message": server.message, "restarts": server.restarts,
                           "log": server.log_tail(20)},
                "busy": state["pending"],
                "last_error": state["last_error"],
                "result": state["latest"],
                "config": {"min_depth": args.min_depth, "max_depth": args.max_depth,
                           "forward_passes": server.forward_passes},
            }
        return no_store(jsonify(payload))

    @app.post("/api/capture")
    def capture():
        prompt = str((request.get_json(silent=True) or {}).get("prompt", ""))[:2000]
        with lock:
            if state["pending"] is not None or server.state != "ready":
                return jsonify(error="The model is not ready (or is already busy)."), 409
            frame = feed.snapshot()
            if frame is None:
                return jsonify(error="No camera frame available yet."), 503
            depth, rgb, K = frame
            # TODO: Add image cropping based on "prompt"

            # Segment the RGB image so the model can tell which regions belong to different objects.
            seg_map = seg_model.segment(rgb)
            run_id = time.strftime("%Y%m%d-%H%M%S")
            run_dir, n = out_root / run_id, 1
            while run_dir.exists():  # two captures in the same second: add a counter
                n += 1
                run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{n}"
                run_dir = out_root / run_id
            run_dir.mkdir(parents=True)
            # Same keys as Odin.py's input_pc.npz, so inference.py can be pointed at any saved run.
            # TODO: have ContactGraspnet_Server use seg_map (it is uploaded but not read yet)
            np.savez(run_dir / "input.npz", depth=depth.astype(np.float32), K=K.astype(np.float32), rgb=rgb, seg_map=seg_map)
            meta = {"id": run_id, "prompt": prompt, "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "z_range": [args.min_depth, args.max_depth], "forward_passes": server.forward_passes}
            (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
            state["meta"][run_id] = meta
            state["last_error"] = None
            state["pending"] = run_id
            if not server.submit(run_id, run_dir / "input.npz", run_dir / "result.npz",
                                 z_range=(args.min_depth, args.max_depth)):
                state["pending"] = None
                return jsonify(error="ContactGraspnet_Server is not ready."), 503
        return jsonify(id=run_id)

    @app.post("/api/server/restart")
    def restart_server():
        threading.Thread(target=server.restart, daemon=True).start()  # returns at once; a restart takes a while
        return jsonify(ok=True)

    @app.get("/api/result/<run_id>/scene.json")
    def result_scene(run_id):
        path = run_dir_for(run_id) / "scene.json"
        if not path.is_file():
            abort(404)
        return no_store(send_file(path, mimetype="application/json"))

    @app.get("/api/result/<run_id>/overlay.jpg")
    def result_overlay(run_id):
        path = run_dir_for(run_id) / "overlay.jpg"
        if not path.is_file():
            abort(404)
        return no_store(send_file(path, mimetype="image/jpeg"))

    return app


def make_callbacks(state: dict, out_root: Path):
    """Create the two functions ServerLink calls when a request finishes or fails."""

    def on_done(run_id, info):
        try:
            summary = build_views(out_root / run_id)
            meta = state["meta"].get(run_id, {})
            result = {"id": run_id, "prompt": meta.get("prompt", ""), "captured_at": meta.get("captured_at", ""),
                      "seconds": info.get("seconds"), **summary}
            state["latest"] = result
            (out_root / run_id / "meta.json").write_text(json.dumps({**meta, **result}, indent=2))
        except Exception as exc:  # noqa: BLE001
            state["last_error"] = f"Inference finished but building the views failed: {type(exc).__name__}: {exc}"
        finally:
            state["pending"] = None

    def on_error(run_id, message):
        state["last_error"] = f"Inference failed: {message}"
        state["pending"] = None

    return on_done, on_error


def main():
    parser = argparse.ArgumentParser(description="Gungnir_Client: browser UI for the L515 + Contact-GraspNet pipeline")
    parser.add_argument("server", nargs="?", default=os.environ.get("GUNGNIR_SERVER", "127.0.0.1"),
                        help="ContactGraspnet_Server address, e.g. 192.168.1.50 (default: $GUNGNIR_SERVER or 127.0.0.1)")
    parser.add_argument("--token", default=os.environ.get("GUNGNIR_TOKEN"), help="shared secret (default: $GUNGNIR_TOKEN)")
    parser.add_argument("--host", default="127.0.0.1", help="web page bind address (default: localhost only; use an SSH tunnel)")
    parser.add_argument("--port", type=int, default=8000, help="web page port")
    parser.add_argument("--min-depth", type=float, default=0.1, help="metres; also the depth crop sent to the model")
    parser.add_argument("--max-depth", type=float, default=9)
    parser.add_argument("--output-dir", default=str(ROOT / "output" / "runs"))
    parser.add_argument("--stream-fps", type=float, default=15, help="live-view rate; lower it over a slow link")
    parser.add_argument("--swap-rb", action="store_true", help="flip red/blue if the live colours look wrong")
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    state = {"pending": None, "last_error": None, "latest": None, "meta": {}}  # shared with the callbacks
    on_done, on_error = make_callbacks(state, out_root)
    server = ServerLink(server_url(args.server), args.token, on_done, on_error)
    feed = CameraFeed(args.min_depth, args.max_depth, args.stream_fps, args.swap_rb)

    logging.getLogger("werkzeug").setLevel(logging.WARNING)  # the page polls /api/status twice a second

    def terminate(signum, frame):
        raise KeyboardInterrupt  # run the same cleanup as Ctrl-C (close the camera)

    signal.signal(signal.SIGTERM, terminate)
    if hasattr(signal, "SIGHUP"):  # an SSH session dropping sends SIGHUP; without this the camera is never closed
        signal.signal(signal.SIGHUP, terminate)
    try:
        # Start the camera and the server connection first, so they come up while SAM2 loads in create_app().
        feed.start()
        server.start()
        print(f"Connecting to ContactGraspnet_Server at {server.url} ... loading the segmentation model", flush=True)
        app = create_app(feed, server, state, args)
        print(f"Serving on http://{args.host}:{args.port}  (results in {out_root})", flush=True)
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)  # reloader would open the camera twice
    finally:
        server.stop()
        feed.stop()


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gungnir Client</title>
<style>
  :root { --bg:#0f1318; --panel:#171d25; --line:#2a3340; --text:#e6edf3; --muted:#8b98a8;
          --accent:#3b82f6; --ok:#22c55e; --warn:#f59e0b; --bad:#ef4444; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui, sans-serif; }
  header { display:flex; flex-wrap:wrap; gap:10px; align-items:center; padding:10px 16px;
           border-bottom:1px solid var(--line); background:var(--panel); }
  header h1 { font-size:16px; margin:0 12px 0 0; }
  .pill { padding:2px 10px; border-radius:999px; border:1px solid var(--line); font-size:12px; color:var(--muted); }
  .pill.ok { color:var(--ok); border-color:var(--ok); } .pill.warn { color:var(--warn); border-color:var(--warn); }
  .pill.bad { color:var(--bad); border-color:var(--bad); }
  header .spacer { flex:1; }
  a { color:var(--accent); }
  main { padding:16px; display:grid; gap:16px; }
  section { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; }
  h2 { font-size:13px; margin:0 0 8px; color:var(--muted); font-weight:600; letter-spacing:.04em; text-transform:uppercase; }
  .two { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
  @media (max-width: 900px) { .two { grid-template-columns:1fr; } }
  figure { margin:0; } figcaption { color:var(--muted); font-size:12px; margin-top:4px; }
  img.view { width:100%; display:block; background:#000; border-radius:4px; aspect-ratio:4/3; object-fit:contain; }
  #scene3d { width:100%; height:480px; background:#000; border-radius:4px; display:flex; align-items:center;
             justify-content:center; color:var(--muted); }
  .controls { display:flex; gap:10px; align-items:stretch; flex-wrap:wrap; }
  .controls input { flex:1; min-width:220px; padding:10px 12px; border-radius:6px; border:1px solid var(--line);
                    background:#0b0f14; color:var(--text); font:inherit; }
  button { padding:10px 18px; border-radius:6px; border:0; background:var(--accent); color:#fff; font:inherit;
           font-weight:600; cursor:pointer; }
  button:disabled { background:#334155; color:#94a3b8; cursor:not-allowed; }
  button.secondary { background:transparent; border:1px solid var(--line); color:var(--muted); font-weight:400; padding:6px 12px; }
  #message { margin-top:8px; min-height:1.4em; color:var(--muted); } #message.bad { color:var(--bad); }
  #meta { color:var(--muted); margin-bottom:8px; }
  details { margin-top:8px; } pre { max-height:220px; overflow:auto; background:#0b0f14; padding:8px; border-radius:4px;
           font-size:12px; color:#9fb0c3; }
  body.results-only .live-only { display:none; }  /* the separate output window hides the live parts */
</style>
</head>
<body>
<header>
  <h1>Gungnir Client</h1>
  <span class="pill" id="pill-camera">camera ...</span>
  <span class="pill" id="pill-server">model ...</span>
  <span class="spacer"></span>
  <span class="live-only"><a href="/?view=results" target="grasp-output">Open output in a separate window</a></span>
</header>
<main>
  <section class="live-only" id="live">
    <h2>Live camera</h2>
    <div class="two">
      <figure><img class="view" src="/stream/rgb" alt="RGB stream"><figcaption>RGB</figcaption></figure>
      <figure><img class="view" src="/stream/depth" alt="Depth stream">
        <figcaption>Depth: <span id="depth-range"></span> (blue = near, red = far; black = outside the range or no return, ignored by the model)</figcaption></figure>
    </div>
  </section>
  <section class="live-only" id="controls">
    <h2>Capture</h2>
    <div class="controls">
      <input id="prompt" type="text" maxlength="2000" placeholder="Prompt (saved with each capture; not used by the model yet)">
      <button id="capture" disabled>Capture &amp; run</button>
      <button id="restart" class="secondary" title="Restart the model on the GPU computer (use after a CUDA error)">Restart model</button>
    </div>
    <div id="message"></div>
    <details><summary>ContactGraspnet_Server log</summary><pre id="log"></pre></details>
  </section>
  <section>
    <h2>Model output</h2>
    <div id="meta">No result yet. Press "Capture &amp; run".</div>
    <div class="two">
      <figure><div id="scene3d">3D scene appears here</div><figcaption>3D point cloud and grasps (drag to rotate, scroll to zoom)</figcaption></figure>
      <figure><img class="view" id="overlay" alt="Grasps overlaid on the captured RGB image"><figcaption>Grasps projected onto the captured RGB image (gold = top 10%, green = top 30%)</figcaption></figure>
    </div>
  </section>
</main>
<script src="/plotly.min.js"></script>
<script>
  // ?view=results is the separate output window: hide the live parts and retitle the tab.
  if (new URLSearchParams(location.search).get('view') === 'results') {
    document.body.classList.add('results-only');
    document.title = 'Gungnir output';
  }
  const $ = id => document.getElementById(id);
  let shownResult = null, submitting = false;  // id of the displayed result; true while a capture is being sent

  function setPill(el, text, cls) { el.textContent = text; el.className = 'pill ' + (cls || ''); }
  function say(text, bad) { $('message').textContent = text || ''; $('message').className = bad ? 'bad' : ''; }

  // Update the whole page from one /api/status object.
  function render(s) {
    const cam = s.camera;
    setPill($('pill-camera'),
            cam.ok ? `camera ${cam.fps} fps` : (cam.error ? 'camera: ' + cam.error.slice(0, 60) : 'camera starting...'),
            cam.ok ? 'ok' : (cam.error ? 'bad' : 'warn'));
    const w = s.server, cls = {ready:'ok', busy:'warn', starting:'warn', error:'bad', stopped:'bad'}[w.state];
    setPill($('pill-server'), 'model: ' + w.state + (w.restarts ? ` (restarts: ${w.restarts})` : ''), cls);
    $('depth-range').textContent = `${s.config.min_depth}-${s.config.max_depth} m`;
    // Capture is allowed only when the model is ready, the camera works, and nothing is running or being sent.
    $('capture').disabled = !(w.state === 'ready' && cam.ok && !s.busy && !submitting);
    if (s.last_error) say(s.last_error, true);
    else if (s.busy) say('Running inference on the captured frame...');
    else if (w.state !== 'ready') say(w.message, w.state === 'error');
    else if (!submitting) say('');
    $('log').textContent = (w.log || []).join('\n');
    if (s.result && s.result.id !== shownResult) loadResult(s.result);
  }

  // Display one finished result: summary line, 2D overlay and 3D scene.
  async function loadResult(r) {
    shownResult = r.id;  // set first so it is not loaded twice while downloading
    const best = r.best_score == null ? 'n/a' : r.best_score.toFixed(3);
    $('meta').textContent = `${r.captured_at} | ${r.num_grasps} grasps | best score ${best} | inference ${r.seconds}s` +
                            (r.prompt ? ` | prompt: ${r.prompt}` : '');
    $('overlay').src = `/api/result/${r.id}/overlay.jpg`;
    try {
      const fig = await (await fetch(`/api/result/${r.id}/scene.json`)).json();
      $('scene3d').textContent = '';
      Plotly.react('scene3d', fig.data, fig.layout, {responsive: true, displaylogo: false});
    } catch (e) { $('scene3d').textContent = '3D scene failed to load: ' + e; }
  }

  $('capture').addEventListener('click', async () => {
    submitting = true; $('capture').disabled = true; say('Capturing...');
    try {
      const r = await fetch('/api/capture', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                             body: JSON.stringify({prompt: $('prompt').value})});
      if (!r.ok) say((await r.json()).error || 'Capture failed', true);
    } catch (e) { say('Capture request failed: ' + e, true); }
    submitting = false;
  });
  $('restart').addEventListener('click', () => fetch('/api/server/restart', {method: 'POST'}));

  // Poll twice a second; the next poll is scheduled only after the current one finishes.
  async function poll() {
    try { render(await (await fetch('/api/status', {cache: 'no-store'})).json()); }
    catch (e) { setPill($('pill-camera'), 'server unreachable', 'bad'); }
    setTimeout(poll, 500);
  }
  poll();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()