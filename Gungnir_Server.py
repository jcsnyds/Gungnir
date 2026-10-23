#!/usr/bin/env python3
"""Gungnir_Server: browser UI for the L515 + Contact-GraspNet pipeline.

    python Gungnir_Server.py           # then open http://localhost:8000 (see SSH tunnel below)

Run it on the Jetson from the project folder with the venv active. From your own computer:

    ssh -L 8000:localhost:8000 edgeai5@<jetson-ip>      # then browse to http://localhost:8000

The page shows the live RGB and depth streams, a prompt box (saved with each capture, not used by the
model yet), and a Capture & Run button. The model output appears next to the stream and in a separate
output window (http://localhost:8000/?view=results) that follows the latest result.

Architecture: this process owns the camera and never imports TensorFlow. The model lives in a separate
long-running process (ContactGraspnet_Client.py), so repeat captures do not reload it.
ContactGraspnet_Client runs with CUDA_LAUNCH_BLOCKING=1, which your pipeline currently needs, and is
restarted automatically if it crashes.
The camera can only be opened by one process: stop Odin.py before starting this.

There is no authentication. The default bind address is localhost; use --host 0.0.0.0 only on a network
you trust.

AI use: Written by Claude (Anthropic; Claude Sonnet 5.5 via claude.ai), October 2026, from the
author's requirements and existing code (l515.py, inference.py, data.py, visualization_utils.py).
Originally named grasp_ui.py. Claude tested it only against a simulated camera and model, not the
real hardware.
Reviewed and tested by: Ryan Poliquin
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, request, send_file

# Define file paths
ROOT = Path(__file__).resolve().parent
RUN_ID_RE = re.compile(r"\d{8}-\d{6}(-\d+)?")

# Same tiers as visualization_utils.visualize_grasps: gold = top 10%, green = top 30%, the rest faint grey.
GOLD_BGR, GREEN_BGR, GREY_BGR = (0, 215, 255), (50, 205, 50), (170, 170, 170)


# ======================================================================================
# Camera
# ======================================================================================
class CameraFeed(threading.Thread):
    """Reads frames from the L515 in a loop and keeps the latest one (plus pre-encoded JPEGs for the streams)."""

    def __init__(self, min_depth, max_depth, stream_fps, jpeg_quality, swap_rb):
        super().__init__(daemon=True, name="camera")
        self.min_depth, self.max_depth = float(min_depth), float(max_depth)
        self.min_interval = 1.0 / max(1.0, float(stream_fps))
        self.jpeg_quality = int(jpeg_quality)
        self.swap_rb = bool(swap_rb)
        self.cond = threading.Condition()
        self.seq = 0
        self.depth = None  # float32 HxW, meters (0 = no return)
        self.rgb = None  # uint8 HxWx3, RGB order
        self.K = None  # 3x3 float32 intrinsics of the color stream (depth is aligned to it)
        self.rgb_jpeg = None
        self.depth_jpeg = None
        self.error = None
        self.fps = 0.0
        self.last_frame_time = 0.0
        self.size = None
        self._halt = threading.Event()  # not "_stop": that name belongs to threading.Thread

    # -- helpers ---------------------------------------------------------------------
    def colorize_depth(self, depth: np.ndarray) -> np.ndarray:
        """Turbo colormap over [min_depth, max_depth]; black where the model ignores the pixel."""
        span = self.max_depth - self.min_depth
        # scale so the current images max and min depths are between 0 and 1
        scaled = np.clip((depth - self.min_depth) / span, 0.0, 1.0)
        # Apply a colormap to colorize the depth
        bgr = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        # Remove any pixels out of the specified depth range
        bgr[~((depth >= self.min_depth) & (depth <= self.max_depth))] = 0
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

    # -- thread ----------------------------------------------------------------------
    def run(self):
        try:
            from l515 import L515, L515Error  # noqa: F401  (installed by IntelRealsense_Driver/setup.py)
        except Exception as exc:  # noqa: BLE001
            self.error = f"Cannot import l515: {exc}"
            print("[camera]", self.error, file=sys.stderr)
            return

        # Allocate memory for all variables
        cam, failures = None, 0
        frames, window_start, last_encode = 0, time.time(), 0.0

        # Main loop that continues until a halt condition occurs
        while not self._halt.is_set():
            try:
                # define camera object on first iteration
                if cam is None:
                    cam = L515()
                    self.K = None
                # capture a new image
                cam.capture_image()
                # get data from camera
                depth = cam.depth_image
                raw = cam.color_image  # BGR as delivered by the native shim
                rgb = np.ascontiguousarray(raw if self.swap_rb else raw[..., ::-1])
                if self.K is None:
                    self.K = np.asarray(cam.get_k(), dtype=np.float32)

                # record the time of the current frame
                now = time.time()
                rgb_jpeg = depth_jpeg = None

                # Only encode the images if a minimum amount of time has passes, used to get desired FPS
                if now - last_encode >= self.min_interval:
                    # Encode the rgb and depth images into .jpg
                    params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                    rgb_jpeg = cv2.imencode(".jpg", rgb[..., ::-1], params)[1].tobytes()
                    depth_jpeg = cv2.imencode(".jpg", self.colorize_depth(depth), params)[1].tobytes()
                    last_encode = now

                with self.cond:
                    # Update camera feed variables
                    self.depth, self.rgb = depth, rgb
                    self.size = (int(depth.shape[1]), int(depth.shape[0]))
                    if rgb_jpeg is not None:
                        self.rgb_jpeg, self.depth_jpeg = rgb_jpeg, depth_jpeg
                        self.seq += 1
                    self.last_frame_time = now
                    self.error = None
                    self.cond.notify_all()

                # Show a new frame was recorded
                failures = 0
                frames += 1
                # Compute currest FPS
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

        if cam is not None:
            try:
                cam.close()
            except Exception:  # noqa: BLE001
                pass

    # -- MJPEG -----------------------------------------------------------------------
    def mjpeg(self, which: str):
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
# ContactGraspnet_Client supervision
# ======================================================================================
class ContactGraspnetClient:
    """Runs ContactGraspnet_Client.py as a subprocess, speaks JSON lines to it, and restarts it when it dies."""

    MAX_STARTUP_FAILURES = 3

    def __init__(self, argv, env, cwd, on_done, on_error):
        # Define arguments to start up the model
        self.argv, self.env, self.cwd = argv, env, cwd

        self.on_done, self.on_error = on_done, on_error
        self.lock = threading.RLock()
        self.proc = None
        self.state = "stopped"  # starting | ready | busy | error | stopped
        self.message = ""
        self.pending = None
        self.restarts = 0
        self.log = collections.deque(maxlen=300)
        self._log_lock = threading.Lock()
        self._fatal = None
        self._startup_failures = 0
        self._was_ready = False
        self._closing = False

    # -- lifecycle ---------------------------------------------------------------------
    def start(self):
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                return
            self._closing = False
            self._was_ready = False
            self._fatal = None
            self.state, self.message = "starting", "Loading the model (this takes a while)..."
            self.proc = subprocess.Popen(
                self.argv, cwd=self.cwd, env=self.env, text=True, bufsize=1,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            proc = self.proc
        for target in (self._read_stdout, self._read_stderr, self._watch):
            threading.Thread(target=target, args=(proc,), daemon=True).start()

    def restart(self):
        """Manual restart (the 'Restart model' button). Also clears the give-up state."""
        with self.lock:
            self._startup_failures = 0
            proc = self.proc
        if proc is not None and proc.poll() is None:
            with self.lock:
                self._closing = True  # so _watch does not auto-restart a second copy
            proc.kill()
            proc.wait()
        self.start()

    def stop(self):
        with self.lock:
            self._closing = True
            proc = self.proc
            self.state = "stopped"
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            proc.stdin.flush()
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            proc.kill()

    # -- requests ----------------------------------------------------------------------
    def submit(self, run_id, input_path, output_path) -> bool:
        with self.lock:
            if self.state != "ready" or self.proc is None:
                return False
            message = {"cmd": "infer", "id": run_id, "input": str(input_path), "output": str(output_path)}
            try:
                self.proc.stdin.write(json.dumps(message) + "\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError):
                return False
            self.state, self.message, self.pending = "busy", "Running inference...", run_id
            return True

    def log_tail(self, n=20):
        with self._log_lock:
            return list(self.log)[-n:]

    def _append_log(self, line):
        with self._log_lock:
            self.log.append(line)

    # -- reader threads ----------------------------------------------------------------
    def _read_stdout(self, proc):
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                self._append_log(f"[stdout] {line}")
                continue
            self._handle(proc, message)

    def _read_stderr(self, proc):
        for line in proc.stderr:
            line = line.rstrip()
            self._append_log(line)
            print("[client]", line, file=sys.stderr)

    def _handle(self, proc, message):
        event = message.get("event")
        if event == "ready":
            with self.lock:
                if proc is not self.proc:
                    return
                self.state, self.message = "ready", "Model ready"
                self._was_ready, self._startup_failures = True, 0
        elif event == "done":
            with self.lock:
                run_id, self.pending = self.pending, None
                if proc is self.proc:
                    self.state, self.message = "ready", "Model ready"
            self.on_done(run_id, message)
        elif event == "error":
            with self.lock:
                run_id, self.pending = self.pending, None
                self.state, self.message = "starting", "Inference failed; restarting the model..."
            self.on_error(run_id, message.get("message", "unknown error"))
        elif event == "fatal":
            with self.lock:
                self._fatal = message.get("message")
                self.message = f"ContactGraspnet_Client could not start: {self._fatal}"

    def _watch(self, proc):
        code = proc.wait()
        with self.lock:
            if proc is not self.proc:
                return
            closing, was_ready, pending = self._closing, self._was_ready, self.pending
            self.pending = None
            if closing:
                return
            if not was_ready:
                self._startup_failures += 1
            give_up = self._startup_failures >= self.MAX_STARTUP_FAILURES
            if give_up:
                self.state = "error"
                self.message = (f"ContactGraspnet_Client failed to start {self._startup_failures} times (exit code {code})"
                                f"{': ' + self._fatal if self._fatal else ''}. "
                                "Check the log, fix the cause, then press Restart model.")
            else:
                self.state = "starting"
                self.message = f"ContactGraspnet_Client exited (code {code}){': ' + self._fatal if self._fatal else ''}; restarting..."
                self.restarts += 1
        if pending is not None:
            self.on_error(pending, f"ContactGraspnet_Client exited unexpectedly (code {code}).")
        if not give_up:
            time.sleep(1.0)
            self.start()


# ======================================================================================
# Output views (3D scene + 2D overlay)
# ======================================================================================
def grasp_points(grasps: np.ndarray, gripper_points: np.ndarray) -> np.ndarray:
    """Gripper wireframe points for every grasp in camera coordinates: (N,4,4),(K,3) -> (N,K,3)."""
    rot, trans = grasps[:, :3, :3], grasps[:, :3, 3]
    return np.einsum("nij,kj->nki", rot, gripper_points) + trans[:, None, :]


def score_tiers(scores: np.ndarray):
    """Boolean masks (gold, green, grey): top 10% / next 20% / rest."""
    if scores.size == 0:
        empty = np.zeros(0, dtype=bool)
        return empty, empty, empty
    p70, p90 = np.percentile(scores, [70, 90])
    gold = scores >= p90
    green = (scores >= p70) & ~gold
    return gold, green, ~(gold | green)


def draw_overlay(rgb, grasps, scores, K, gripper_points, segments) -> bytes:
    """Grasp wireframes projected onto the captured RGB image (JPEG bytes)."""
    img = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    n = len(grasps)
    if n:
        pts = grasp_points(grasps, gripper_points)
        z = pts[..., 2]
        ok = z > 1e-3
        safe_z = np.where(ok, z, 1.0)
        u = np.clip(K[0, 0] * pts[..., 0] / safe_z + K[0, 2], -1e4, 1e4)
        v = np.clip(K[1, 1] * pts[..., 1] / safe_z + K[1, 2], -1e4, 1e4)

        def draw(canvas, indices, color, thickness):
            for i in indices:
                for a, b in segments:
                    if ok[i, a] and ok[i, b]:
                        cv2.line(canvas, (int(round(u[i, a])), int(round(v[i, a]))),
                                 (int(round(u[i, b])), int(round(v[i, b]))), color, thickness, cv2.LINE_AA)

        gold, green, grey = score_tiers(scores)
        faint = img.copy()
        draw(faint, np.flatnonzero(grey), GREY_BGR, 1)
        img = cv2.addWeighted(faint, 0.4, img, 0.6, 0)
        draw(img, np.flatnonzero(green), GREEN_BGR, 2)
        draw(img, np.flatnonzero(gold), GOLD_BGR, 2)

    label = f"{n} grasps" + (f" | best score {float(scores.max()):.2f}" if n else " | none found")
    cv2.putText(img, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


def scene_figure_json(pc, colors, grasps, scores, gripper_points, segments, max_points=30000) -> str:
    """Interactive 3D scene as Plotly JSON. All grasps of one tier share ONE trace (not 6 per grasp)."""
    import plotly.graph_objects as go
    import plotly.io as pio

    traces = []
    if len(pc):
        keep = np.arange(len(pc))
        if len(pc) > max_points:
            keep = np.random.default_rng(0).choice(len(pc), max_points, replace=False)
        pc = pc[keep]
        marker = dict(size=1.6, color="lightgray")
        if len(colors) > 0:
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
            a, b = pts[indices][:, segments[:, 0], :], pts[indices][:, segments[:, 1], :]
            return np.stack([a, b, np.full_like(a, np.nan)], axis=2).reshape(-1, 3)  # NaN row breaks the line

        gold, green, grey = score_tiers(scores)
        for mask, name, color, width in ((grey, "other grasps", "rgba(130,130,130,0.35)", 3),
                                         (green, "top 30%", "limegreen", 4),
                                         (gold, "top 10%", "gold", 4)):
            idx = np.flatnonzero(mask)
            if idx.size:
                xyz = lines(idx)
                traces.append(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines",
                                           line=dict(color=color, width=width), name=name, hoverinfo="skip"))

    for axis, color in enumerate(("red", "green", "blue")):  # camera axes, as plot_opencv_cam=True drew them
        end = np.zeros(3)
        end[axis] = 0.1
        traces.append(go.Scatter3d(x=[0, end[0]], y=[0, end[1]], z=[0, end[2]], mode="lines",
                                   line=dict(color=color, width=6), showlegend=False, hoverinfo="skip"))

    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", y=1.0),
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)", aspectmode="data",
            # OpenCV camera frame has y pointing down: flip "up" and view from behind the camera.
            camera=dict(up=dict(x=0, y=-1, z=0), eye=dict(x=0.0, y=-0.4, z=-1.6)),
        ),
    )
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
def create_app(feed: CameraFeed, client: ContactGraspnetClient, state: dict, args) -> Flask:
    app = Flask(__name__)
    out_root = Path(args.output_dir)
    lock = threading.Lock()
    plotly_js = {}

    def run_dir_for(run_id: str) -> Path:
        if not RUN_ID_RE.fullmatch(run_id):
            abort(404)
        path = out_root / run_id
        if not path.is_dir():
            abort(404)
        return path

    def no_store(response):
        response.headers["Cache-Control"] = "no-store"
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
                "client": {"state": client.state, "message": client.message, "restarts": client.restarts,
                           "log": client.log_tail(20)},
                "busy": state["pending"],
                "last_error": state["last_error"],
                "result": state["latest"],
                "config": {"min_depth": args.min_depth, "max_depth": args.max_depth,
                           "forward_passes": args.forward_passes},
            }
        return no_store(jsonify(payload))

    @app.post("/api/capture")
    def capture():
        prompt = str((request.get_json(silent=True) or {}).get("prompt", ""))[:2000]
        with lock:
            if state["pending"] is not None or client.state != "ready":
                return jsonify(error="The model is not ready (or is already busy)."), 409
            frame = feed.snapshot()
            if frame is None:
                return jsonify(error="No camera frame available yet."), 503
            depth, rgb, K = frame
            run_id = time.strftime("%Y%m%d-%H%M%S")
            run_dir, n = out_root / run_id, 1
            while run_dir.exists():
                n += 1
                run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{n}"
                run_dir = out_root / run_id
            run_dir.mkdir(parents=True)
            # Same keys as Odin.py's input_pc.npz, so inference.py can be pointed at any saved run.
            np.savez(run_dir / "input.npz", depth=depth.astype(np.float32), K=K.astype(np.float32), rgb=rgb)
            meta = {"id": run_id, "prompt": prompt, "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "z_range": [args.min_depth, args.max_depth], "forward_passes": args.forward_passes}
            (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
            state["meta"][run_id] = meta
            state["last_error"] = None
            state["pending"] = run_id
            if not client.submit(run_id, run_dir / "input.npz", run_dir / "result.npz"):
                state["pending"] = None
                return jsonify(error="Could not hand the frame to ContactGraspnet_Client."), 503
        return jsonify(id=run_id)

    @app.post("/api/client/restart")
    def restart_client():
        threading.Thread(target=client.restart, daemon=True).start()
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


def make_callbacks(state: dict, lock_free_out_root: Path):
    def on_done(run_id, info):
        try:
            summary = build_views(lock_free_out_root / run_id)
            meta = state["meta"].get(run_id, {})
            result = {"id": run_id, "prompt": meta.get("prompt", ""), "captured_at": meta.get("captured_at", ""),
                      "seconds": info.get("seconds"), **summary}
            state["latest"] = result
            (lock_free_out_root / run_id / "meta.json").write_text(json.dumps({**meta, **result}, indent=2))
        except Exception as exc:  # noqa: BLE001
            state["last_error"] = f"Inference finished but building the views failed: {type(exc).__name__}: {exc}"
        finally:
            state["pending"] = None

    def on_error(run_id, message):
        state["last_error"] = f"Inference failed: {message}"
        state["pending"] = None

    return on_done, on_error


def package_dir(name: str):
    """Directory of an installed package without importing it (importing TensorFlow here would defeat the split)."""
    spec = importlib.util.find_spec(name)
    return list(spec.submodule_search_locations)[0] if spec and spec.submodule_search_locations else None


def main():
    parser = argparse.ArgumentParser(description="Gungnir_Server: browser UI for the L515 + Contact-GraspNet pipeline")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost only; use an SSH tunnel)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--min-depth", type=float, default=0.1, help="metres; also the depth crop passed to the model")
    parser.add_argument("--max-depth", type=float, default=2.5)
    parser.add_argument("--forward-passes", type=int, default=3, help="fixed when the model is built")
    parser.add_argument("--ckpt-dir", default=None, help="default: contact_graspnet/checkpoints/scene_test_2048_bs3_hor_sigma_0025")
    parser.add_argument("--output-dir", default=str(ROOT / "output" / "runs"))
    parser.add_argument("--stream-fps", type=float, default=15, help="live-view rate; lower it over a slow link")
    parser.add_argument("--jpeg-quality", type=int, default=70)
    parser.add_argument("--swap-rb", action="store_true", help="flip red/blue if the live colours look wrong")
    parser.add_argument("--no-launch-blocking", action="store_true",
                        help="do not set CUDA_LAUNCH_BLOCKING=1 in ContactGraspnet_Client (faster, but your pipeline needs it today)")
    parser.add_argument("--client-script", default=str(ROOT / "ContactGraspnet_Client.py"))
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    env = {**os.environ, "TF_USE_LEGACY_KERAS": "1", "PYTHONUNBUFFERED": "1"}
    if not args.no_launch_blocking:
        env["CUDA_LAUNCH_BLOCKING"] = "1"
    tf_dir = package_dir("tensorflow")
    if tf_dir:
        env["LD_LIBRARY_PATH"] = f"{tf_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"

    client_argv = [sys.executable, args.client_script, "--forward-passes", str(args.forward_passes),
                   "--z-min", str(args.min_depth), "--z-max", str(args.max_depth)]
    if args.ckpt_dir:
        client_argv += ["--ckpt-dir", str(Path(args.ckpt_dir).resolve())]

    state = {"pending": None, "last_error": None, "latest": None, "meta": {}}
    on_done, on_error = make_callbacks(state, out_root)
    client = ContactGraspnetClient(client_argv, env, str(ROOT), on_done, on_error)
    feed = CameraFeed(args.min_depth, args.max_depth, args.stream_fps, args.jpeg_quality, args.swap_rb)

    logging.getLogger("werkzeug").setLevel(logging.WARNING)  # the page polls /api/status twice a second
    app = create_app(feed, client, state, args)
    feed.start()
    client.start()
    print(f"Serving on http://{args.host}:{args.port}  (results in {out_root})", flush=True)

    def terminate(signum, frame):
        raise KeyboardInterrupt  # run the same cleanup as Ctrl-C (close the camera, stop ContactGraspnet_Client)

    signal.signal(signal.SIGTERM, terminate)
    if hasattr(signal, "SIGHUP"):  # an SSH session dropping sends SIGHUP; without this the camera is never closed
        signal.signal(signal.SIGHUP, terminate)
    try:
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    finally:
        client.stop()
        feed.stop()


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gungnir Server</title>
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
  body.results-only .live-only { display:none; }
</style>
</head>
<body>
<header>
  <h1>Gungnir Server</h1>
  <span class="pill" id="pill-camera">camera ...</span>
  <span class="pill" id="pill-client">model ...</span>
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
      <button id="restart" class="secondary" title="Restart the model process (use after a CUDA error)">Restart model</button>
    </div>
    <div id="message"></div>
    <details><summary>ContactGraspnet_Client log</summary><pre id="log"></pre></details>
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
  if (new URLSearchParams(location.search).get('view') === 'results') {
    document.body.classList.add('results-only');
    document.title = 'Gungnir output';
  }
  const $ = id => document.getElementById(id);
  let shownResult = null, submitting = false, lastStatus = null;

  function setPill(el, text, cls) { el.textContent = text; el.className = 'pill ' + (cls || ''); }
  function say(text, bad) { $('message').textContent = text || ''; $('message').className = bad ? 'bad' : ''; }

  function render(s) {
    lastStatus = s;
    const cam = s.camera;
    setPill($('pill-camera'),
            cam.ok ? `camera ${cam.fps} fps` : (cam.error ? 'camera: ' + cam.error.slice(0, 60) : 'camera starting...'),
            cam.ok ? 'ok' : (cam.error ? 'bad' : 'warn'));
    const w = s.client, cls = {ready:'ok', busy:'warn', starting:'warn', error:'bad', stopped:'bad'}[w.state];
    setPill($('pill-client'), 'model: ' + w.state + (w.restarts ? ` (restarts: ${w.restarts})` : ''), cls);
    $('depth-range').textContent = `${s.config.min_depth}-${s.config.max_depth} m`;
    $('capture').disabled = !(w.state === 'ready' && cam.ok && !s.busy && !submitting);
    if (s.last_error) say(s.last_error, true);
    else if (s.busy) say('Running inference on the captured frame...');
    else if (w.state !== 'ready') say(w.message, w.state === 'error');
    else if (!submitting) say('');
    $('log').textContent = (w.log || []).join('\n');
    if (s.result && s.result.id !== shownResult) loadResult(s.result);
  }

  async function loadResult(r) {
    shownResult = r.id;
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
  $('restart').addEventListener('click', () => fetch('/api/client/restart', {method: 'POST'}));

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
