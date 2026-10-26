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
real hardware. Line-by-line explanatory comments (Python, CSS, HTML and JavaScript) were added by
Claude in a later pass (October 2026); that pass changed comments only, and the executable code is
identical to the version the author uploaded.
Reviewed and tested by: Ryan Poliquin
"""

# Store type annotations as text instead of evaluating them when the file is loaded, so annotations
# such as `CameraFeed` used before the class is fully defined work on any Python 3 version.
from __future__ import annotations

import argparse  # reads the command-line options (--port, --host, --min-depth, ...)
import collections  # provides deque, a list with a fixed maximum length (used for the client log)
import importlib.util  # finds where a package is installed without importing it
import json  # encodes/decodes the JSON messages and the saved meta.json files
import logging  # controls how much the web server prints to the terminal
import os  # environment variables for the child process
import re  # regular expressions, used to validate run ids
import signal  # lets us react to SIGTERM / SIGHUP so cleanup runs on shutdown
import subprocess  # starts ContactGraspnet_Client.py as a separate process
import sys  # the Python interpreter path (sys.executable) and the error stream (sys.stderr)
import threading  # background threads, locks, and events
import time  # timestamps, sleeping, and run-id generation
from pathlib import Path  # path objects that behave the same on any operating system

import cv2  # OpenCV: JPEG encoding, color maps, and drawing lines/text on images
import numpy as np  # arrays: images, point clouds, and grasp matrices
from flask import Flask, Response, abort, jsonify, request, send_file  # the web framework and its helpers

# Define file paths
# Folder that contains this script: resolve() makes the path absolute (following any symlinks) and
# .parent removes the file name. Output folders and the client script are located relative to it.
ROOT = Path(__file__).resolve().parent
# A valid run id looks like 20261017-143005 or 20261017-143005-2 (date-time, optional counter). Used to
# reject anything else so a request can never point at a different folder (path traversal).
RUN_ID_RE = re.compile(r"\d{8}-\d{6}(-\d+)?")

# Same tiers as visualization_utils.visualize_grasps: gold = top 10%, green = top 30%, the rest faint grey.
# The colors are in OpenCV's BGR order (blue, green, red), each channel 0-255.
GOLD_BGR, GREEN_BGR, GREY_BGR = (0, 215, 255), (50, 205, 50), (170, 170, 170)


# ======================================================================================
# Camera
# ======================================================================================
class CameraFeed(threading.Thread):
    """Reads frames from the L515 in a loop and keeps the latest one (plus pre-encoded JPEGs for the streams)."""

    def __init__(self, min_depth, max_depth, stream_fps, jpeg_quality, swap_rb):
        # Initialize the Thread base class. daemon=True means this thread never keeps the program alive
        # on its own; name is only a label shown in debuggers.
        super().__init__(daemon=True, name="camera")
        # Depth range (meters) used for the depth color map; stored as floats.
        self.min_depth, self.max_depth = float(min_depth), float(max_depth)
        # Minimum seconds between two encoded stream frames (1 / fps); fps is never allowed below 1.
        self.min_interval = 1.0 / max(1.0, float(stream_fps))
        # JPEG quality, 0-100 (higher = bigger, sharper images).
        self.jpeg_quality = int(jpeg_quality)
        # True if red and blue should be swapped for this camera.
        self.swap_rb = bool(swap_rb)
        # A condition variable: a lock plus a way for other threads to wait until new data arrives.
        self.cond = threading.Condition()
        # Counts how many encoded frames have been produced; stream readers use it to spot new frames.
        self.seq = 0
        self.depth = None  # float32 HxW, meters (0 = no return)
        self.rgb = None  # uint8 HxWx3, RGB order
        self.K = None  # 3x3 float32 intrinsics of the color stream (depth is aligned to it)
        # The latest frames already compressed to JPEG bytes, ready to stream to browsers.
        self.rgb_jpeg = None
        self.depth_jpeg = None
        # Text of the most recent camera error (None when everything is fine).
        self.error = None
        # Measured frames per second, updated every ~2 seconds.
        self.fps = 0.0
        # Wall-clock time of the last successful frame, used to decide whether the camera is alive.
        self.last_frame_time = 0.0
        # (width, height) of the camera image once known.
        self.size = None
        self._halt = threading.Event()  # not "_stop": that name belongs to threading.Thread

    # -- helpers ---------------------------------------------------------------------
    def colorize_depth(self, depth: np.ndarray) -> np.ndarray:
        """Turbo colormap over [min_depth, max_depth]; black where the model ignores the pixel."""
        # Width of the depth range, used to rescale depth values to 0-1.
        span = self.max_depth - self.min_depth
        # scale so the current images max and min depths are between 0 and 1
        # (values outside the range are clipped to 0 or 1)
        scaled = np.clip((depth - self.min_depth) / span, 0.0, 1.0)
        # Apply a colormap to colorize the depth
        # (convert 0-1 to 0-255 integers first; the result is a BGR color image)
        bgr = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        # Remove any pixels out of the specified depth range
        # (the ~ inverts the mask, so pixels outside [min_depth, max_depth], or with no depth, become black)
        bgr[~((depth >= self.min_depth) & (depth <= self.max_depth))] = 0
        # Return the colorized image.
        return bgr

    def snapshot(self):
        """Latest (depth, rgb, K), or None before the first frame. The arrays are never mutated afterwards."""
        # Hold the lock so the three values are read from the same frame.
        with self.cond:
            # No frame (or no intrinsics) has arrived yet.
            if self.depth is None or self.K is None:
                return None
            # Hand back the stored arrays themselves; the camera thread replaces them instead of editing them.
            return self.depth, self.rgb, self.K

    @property
    def ok(self) -> bool:
        # True once at least one frame was encoded AND the last frame is less than 3 seconds old.
        return self.seq > 0 and (time.time() - self.last_frame_time) < 3.0

    def stop(self):
        # Ask the camera loop to finish...
        self._halt.set()
        # ...and wait up to 5 seconds for it to close the camera and exit.
        self.join(timeout=5)

    # -- thread ----------------------------------------------------------------------
    def run(self):
        """The camera thread's main function: runs from start() until stop() is called."""
        # Import the camera driver inside the thread so a missing driver is reported in the UI
        # instead of crashing the whole server at start-up.
        try:
            from l515 import L515, L515Error  # noqa: F401  (installed by IntelRealsense_Driver/setup.py)
        except Exception as exc:  # noqa: BLE001
            # Remember the error text so /api/status can show it, print it, and give up.
            self.error = f"Cannot import l515: {exc}"
            print("[camera]", self.error, file=sys.stderr)
            return

        # Allocate memory for all variables
        # cam: the camera object (None until opened); failures: consecutive errors in a row.
        cam, failures = None, 0
        # frames: frames counted in the current FPS window; window_start: when that window began;
        # last_encode: when the last JPEG pair was produced.
        frames, window_start, last_encode = 0, time.time(), 0.0

        # Main loop that continues until a halt condition occurs
        while not self._halt.is_set():
            try:
                # define camera object on first iteration (and again after a reopen)
                if cam is None:
                    # Open the L515 and start streaming.
                    cam = L515()
                    # Intrinsics are re-read from the freshly opened camera below.
                    self.K = None
                # capture a new image
                # (blocks until the next aligned color+depth frame pair arrives)
                cam.capture_image()
                # get data from camera
                # Depth image in meters (float32).
                depth = cam.depth_image
                raw = cam.color_image  # BGR as delivered by the native shim
                # Make an RGB copy: reverse the channel order (BGR -> RGB) unless swap_rb says the camera
                # already delivers RGB; ascontiguousarray makes the memory layout contiguous for OpenCV.
                rgb = np.ascontiguousarray(raw if self.swap_rb else raw[..., ::-1])
                # Read the camera matrix once per opened camera (it does not change while streaming).
                if self.K is None:
                    self.K = np.asarray(cam.get_k(), dtype=np.float32)

                # record the time of the current frame
                now = time.time()
                # These stay None unless we decide to encode this frame.
                rgb_jpeg = depth_jpeg = None

                # Only encode the images if a minimum amount of time has passed, used to get the desired FPS
                if now - last_encode >= self.min_interval:
                    # Encode the rgb and depth images into .jpg
                    # (the quality setting is passed to OpenCV as a flag/value pair)
                    params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                    # OpenCV expects BGR, so flip the RGB image back; [1] picks the encoded buffer from the
                    # (success, buffer) pair and .tobytes() turns it into plain bytes.
                    rgb_jpeg = cv2.imencode(".jpg", rgb[..., ::-1], params)[1].tobytes()
                    # Same for the colorized depth image (already BGR).
                    depth_jpeg = cv2.imencode(".jpg", self.colorize_depth(depth), params)[1].tobytes()
                    # Remember when we last encoded.
                    last_encode = now

                # Take the lock so readers never see a half-updated frame.
                with self.cond:
                    # Update camera feed variables
                    self.depth, self.rgb = depth, rgb
                    # Image size as (width, height); numpy shape is (rows, columns) = (height, width).
                    self.size = (int(depth.shape[1]), int(depth.shape[0]))
                    # Only publish new JPEGs (and bump the counter) if this frame was encoded.
                    if rgb_jpeg is not None:
                        self.rgb_jpeg, self.depth_jpeg = rgb_jpeg, depth_jpeg
                        self.seq += 1
                    # Record that a frame arrived just now and clear any earlier error.
                    self.last_frame_time = now
                    self.error = None
                    # Wake up any stream readers waiting for a new frame.
                    self.cond.notify_all()

                # Show a new frame was recorded
                # (a success resets the error streak; frames feeds the FPS estimate)
                failures = 0
                frames += 1
                # Compute current FPS
                # (once a window of 2+ seconds has passed: frames divided by elapsed time, then restart the window)
                if now - window_start >= 2.0:
                    self.fps = frames / (now - window_start)
                    frames, window_start = 0, now
            except Exception as exc:  # noqa: BLE001
                # Any camera problem: count it, remember the message, and log it.
                failures += 1
                self.error = f"{type(exc).__name__}: {exc}"
                print(f"[camera] {self.error} (failure {failures})", file=sys.stderr)
                # After 5 failures in a row, close the camera so the loop above reopens it from scratch.
                if failures >= 5 and cam is not None:  # reopen the device
                    try:
                        cam.close()
                    except Exception:  # noqa: BLE001
                        # Closing a broken camera can fail too; ignore that.
                        pass
                    cam = None
                # Wait before retrying: 0.2 s per failure so far, but never more than 2 s.
                time.sleep(min(2.0, 0.2 * failures))

        # The loop has ended (stop() was called): release the camera so it can be used again.
        if cam is not None:
            try:
                cam.close()
            except Exception:  # noqa: BLE001
                pass

    # -- MJPEG -----------------------------------------------------------------------
    def mjpeg(self, which: str):
        """Generator that yields an endless MJPEG stream ("rgb" or "depth") for one browser connection."""
        # The last frame number this viewer has been sent (-1 = none yet).
        last = -1
        # Keep going until the browser disconnects (Flask then stops iterating this generator).
        while True:
            # Take the lock to read the shared frame data safely.
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
            # One multipart chunk: the boundary marker, headers for this JPEG (type and length), a blank
            # line, the JPEG bytes, and a line break. The browser replaces the displayed image each time.
            yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(frame) + frame + b"\r\n"


# ======================================================================================
# ContactGraspnet_Client supervision
# ======================================================================================
class ContactGraspnetClient:
    """Runs ContactGraspnet_Client.py as a subprocess, speaks JSON lines to it, and restarts it when it dies."""

    # After this many failed start-ups in a row, stop retrying automatically.
    MAX_STARTUP_FAILURES = 3

    def __init__(self, argv, env, cwd, on_done, on_error):
        # Define arguments to start up the model
        # argv: the command line to run; env: its environment variables; cwd: its working folder.
        self.argv, self.env, self.cwd = argv, env, cwd

        # Functions to call when a request finishes / fails (provided by make_callbacks).
        self.on_done, self.on_error = on_done, on_error
        # A re-entrant lock protecting the fields below (the same thread may take it twice).
        self.lock = threading.RLock()
        # The subprocess object (None before the first start).
        self.proc = None
        self.state = "stopped"  # starting | ready | busy | error | stopped
        # Human-readable status text shown in the UI.
        self.message = ""
        # Run id of the request currently being processed (None when idle).
        self.pending = None
        # How many times the client has been restarted after exiting.
        self.restarts = 0
        # The last 300 lines the client wrote to stderr (shown in the log panel); old lines fall off.
        self.log = collections.deque(maxlen=300)
        # Separate lock for the log, so log reads/writes never wait on the main lock.
        self._log_lock = threading.Lock()
        # The client's own "could not start" message, if it sent one.
        self._fatal = None
        # Consecutive start-ups that failed before the client became ready.
        self._startup_failures = 0
        # True once the current process has reported "ready".
        self._was_ready = False
        # True while we are shutting the client down on purpose (so we do not restart it).
        self._closing = False

    # -- lifecycle ---------------------------------------------------------------------
    def start(self):
        """Launch the client process and the three threads that watch it."""
        with self.lock:
            # If it is already running, there is nothing to do.
            if self.proc is not None and self.proc.poll() is None:
                return
            # Reset the per-process flags for the new run.
            self._closing = False
            self._was_ready = False
            self._fatal = None
            self.state, self.message = "starting", "Loading the model (this takes a while)..."
            # Start the process. Text mode with line buffering; stdin/stdout/stderr are pipes so we can
            # send requests, read protocol replies, and capture log output.
            self.proc = subprocess.Popen(
                self.argv, cwd=self.cwd, env=self.env, text=True, bufsize=1,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            # Keep a local reference so the watcher threads follow THIS process even if self.proc changes later.
            proc = self.proc
        # One thread reads protocol messages, one reads log lines, one waits for the process to exit.
        for target in (self._read_stdout, self._read_stderr, self._watch):
            threading.Thread(target=target, args=(proc,), daemon=True).start()

    def restart(self):
        """Manual restart (the 'Restart model' button). Also clears the give-up state."""
        with self.lock:
            # Forget earlier start-up failures so automatic retries work again.
            self._startup_failures = 0
            proc = self.proc
        # If the process is still alive, kill it.
        if proc is not None and proc.poll() is None:
            with self.lock:
                self._closing = True  # so _watch does not auto-restart a second copy
            proc.kill()
            # Wait for it to disappear completely.
            proc.wait()
        # Start a fresh process.
        self.start()

    def stop(self):
        """Shut the client down for good (used when the server exits)."""
        with self.lock:
            # Mark this as intentional so the watcher does not restart it.
            self._closing = True
            proc = self.proc
            self.state = "stopped"
        # Nothing to stop if it never started or already exited.
        if proc is None or proc.poll() is not None:
            return
        try:
            # Politely ask it to quit and give it 5 seconds...
            proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            proc.stdin.flush()
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            # ...otherwise (error or timeout) kill it.
            proc.kill()

    # -- requests ----------------------------------------------------------------------
    def submit(self, run_id, input_path, output_path) -> bool:
        """Send one inference request. Returns True if it was handed to the client, False otherwise."""
        with self.lock:
            # Only accept work when the model is loaded and idle.
            if self.state != "ready" or self.proc is None:
                return False
            # The request in the pipe protocol: the command, an id, and the input/output file paths.
            message = {"cmd": "infer", "id": run_id, "input": str(input_path), "output": str(output_path)}
            try:
                # Write one JSON line to the client's stdin and flush it so it is sent right away.
                self.proc.stdin.write(json.dumps(message) + "\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError):
                # The pipe is closed (the client died); report failure.
                return False
            # Mark the client busy and remember which request it is working on.
            self.state, self.message, self.pending = "busy", "Running inference...", run_id
            return True

    def log_tail(self, n=20):
        """Return the last n log lines (a copy, safe to use while the log keeps growing)."""
        with self._log_lock:
            return list(self.log)[-n:]

    def _append_log(self, line):
        # Add one line to the log (the deque drops the oldest line when it is full).
        with self._log_lock:
            self.log.append(line)

    # -- reader threads ----------------------------------------------------------------
    def _read_stdout(self, proc):
        """Read protocol messages from the client's stdout until it closes."""
        # Iterating over the pipe yields one line at a time (blocking while there is none).
        for line in proc.stdout:
            line = line.strip()
            # Skip empty lines.
            if not line:
                continue
            try:
                # Each protocol line is a JSON object.
                message = json.loads(line)
            except json.JSONDecodeError:
                # Not JSON (unexpected): keep it in the log so it is not lost.
                self._append_log(f"[stdout] {line}")
                continue
            # Act on the decoded message.
            self._handle(proc, message)

    def _read_stderr(self, proc):
        """Copy the client's stderr (TensorFlow logs, tracebacks) into the log and the terminal."""
        for line in proc.stderr:
            # Remove only the trailing newline/whitespace, keeping indentation.
            line = line.rstrip()
            self._append_log(line)
            # Also echo it to this server's terminal with a [client] prefix.
            print("[client]", line, file=sys.stderr)

    def _handle(self, proc, message):
        """React to one protocol message from the client."""
        # Which kind of message is it? (ready / done / error / fatal)
        event = message.get("event")
        if event == "ready":
            with self.lock:
                # Ignore messages from an old process that has already been replaced.
                if proc is not self.proc:
                    return
                # The model is loaded: accept requests, and reset the failure counter.
                self.state, self.message = "ready", "Model ready"
                self._was_ready, self._startup_failures = True, 0
        elif event == "done":
            with self.lock:
                # Take the pending id (and clear it) so we know which request finished.
                run_id, self.pending = self.pending, None
                # Back to idle, unless this message came from a replaced process.
                if proc is self.proc:
                    self.state, self.message = "ready", "Model ready"
            # Tell the app (outside the lock, because the callback does slow work).
            self.on_done(run_id, message)
        elif event == "error":
            with self.lock:
                run_id, self.pending = self.pending, None
                # The client exits after an error, so show that it is about to restart.
                self.state, self.message = "starting", "Inference failed; restarting the model..."
            self.on_error(run_id, message.get("message", "unknown error"))
        elif event == "fatal":
            with self.lock:
                # The client could not start; keep its explanation for the status message.
                self._fatal = message.get("message")
                self.message = f"ContactGraspnet_Client could not start: {self._fatal}"

    def _watch(self, proc):
        """Wait for the client process to exit, then decide whether to restart it."""
        # Block until the process ends and get its exit code.
        code = proc.wait()
        with self.lock:
            # A newer process has replaced this one (e.g. after a manual restart): nothing to do.
            if proc is not self.proc:
                return
            # Snapshot the flags and clear the pending request.
            closing, was_ready, pending = self._closing, self._was_ready, self.pending
            self.pending = None
            # An intentional shutdown: do not restart.
            if closing:
                return
            # If it died before ever reporting ready, count that as a failed start-up.
            if not was_ready:
                self._startup_failures += 1
            # Too many failed start-ups in a row: stop trying.
            give_up = self._startup_failures >= self.MAX_STARTUP_FAILURES
            if give_up:
                self.state = "error"
                # Build the message from adjacent strings; includes the client's own explanation if any.
                self.message = (f"ContactGraspnet_Client failed to start {self._startup_failures} times (exit code {code})"
                                f"{': ' + self._fatal if self._fatal else ''}. "
                                "Check the log, fix the cause, then press Restart model.")
            else:
                # Otherwise it will be restarted below.
                self.state = "starting"
                self.message = f"ContactGraspnet_Client exited (code {code}){': ' + self._fatal if self._fatal else ''}; restarting..."
                self.restarts += 1
        # A request was in flight when the client died: report it as failed.
        if pending is not None:
            self.on_error(pending, f"ContactGraspnet_Client exited unexpectedly (code {code}).")
        # Restart after a short pause (outside the lock).
        if not give_up:
            time.sleep(1.0)
            self.start()


# ======================================================================================
# Output views (3D scene + 2D overlay)
# ======================================================================================
def grasp_points(grasps: np.ndarray, gripper_points: np.ndarray) -> np.ndarray:
    """Gripper wireframe points for every grasp in camera coordinates: (N,4,4),(K,3) -> (N,K,3)."""
    # Split each 4x4 pose into its 3x3 rotation block and its 3-element translation vector.
    rot, trans = grasps[:, :3, :3], grasps[:, :3, 3]
    # For every grasp n and wireframe point k, compute rotation * point + translation.
    # einsum "nij,kj->nki": sum over j of rot[n,i,j] * point[k,j], giving a (grasps, points, 3) array.
    return np.einsum("nij,kj->nki", rot, gripper_points) + trans[:, None, :]


def score_tiers(scores: np.ndarray):
    """Boolean masks (gold, green, grey): top 10% / next 20% / rest."""
    # No grasps: return three empty masks.
    if scores.size == 0:
        empty = np.zeros(0, dtype=bool)
        return empty, empty, empty
    # The 70th and 90th percentile score values.
    p70, p90 = np.percentile(scores, [70, 90])
    # Gold: score at or above the 90th percentile.
    gold = scores >= p90
    # Green: at or above the 70th percentile but not gold.
    green = (scores >= p70) & ~gold
    # Grey: everything that is neither gold nor green.
    return gold, green, ~(gold | green)


def draw_overlay(rgb, grasps, scores, K, gripper_points, segments) -> bytes:
    """Grasp wireframes projected onto the captured RGB image (JPEG bytes)."""
    # OpenCV draws in BGR, so convert the RGB image (copying it so the original is untouched).
    img = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    # Number of grasps to draw.
    n = len(grasps)
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
        # Draw the low-scoring grasps on a copy, then blend it at 40% so they look faint.
        faint = img.copy()
        draw(faint, np.flatnonzero(grey), GREY_BGR, 1)
        img = cv2.addWeighted(faint, 0.4, img, 0.6, 0)
        # Draw the better grasps on top at full strength (gold last, so it is never hidden).
        draw(img, np.flatnonzero(green), GREEN_BGR, 2)
        draw(img, np.flatnonzero(gold), GOLD_BGR, 2)

    # Caption text: the grasp count, plus the best score if there are any grasps.
    label = f"{n} grasps" + (f" | best score {float(scores.max()):.2f}" if n else " | none found")
    # Draw the caption twice: a thick black outline first, then thin white text on top, so it is readable
    # on any background.
    cv2.putText(img, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    # Encode the finished image as a high-quality JPEG and return its bytes.
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


def scene_figure_json(pc, colors, grasps, scores, gripper_points, segments, max_points=30000) -> str:
    """Interactive 3D scene as Plotly JSON. All grasps of one tier share ONE trace (not 6 per grasp)."""
    # Imported here so the server starts even before Plotly is needed.
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
        # Keep only the chosen points.
        pc = pc[keep]
        # Default marker style: small light-grey dots.
        marker = dict(size=1.6, color="lightgray")
        # If the model returned per-point colors, use them.
        if len(colors) > 0:
            c = np.asarray(colors)
            # Only use them if they are an (N, 3) array covering every point we kept.
            if c.ndim == 2 and c.shape[1] == 3 and len(c) >= keep.max() + 1:
                # Same subset as the points.
                c = c[keep]
                # Floating-point colors in 0-1 are scaled to 0-255.
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
            # separate segments can live in one layer. reshape flattens everything into a list of 3D points.
            return np.stack([a, b, np.full_like(a, np.nan)], axis=2).reshape(-1, 3)  # NaN row breaks the line

        # Split the grasps into score tiers.
        gold, green, grey = score_tiers(scores)
        # One layer per tier: (mask of grasps, legend name, line color, line width).
        for mask, name, color, width in ((grey, "other grasps", "rgba(130,130,130,0.35)", 3),
                                         (green, "top 30%", "limegreen", 4),
                                         (gold, "top 10%", "gold", 4)):
            # Indices of the grasps in this tier.
            idx = np.flatnonzero(mask)
            # Skip empty tiers.
            if idx.size:
                xyz = lines(idx)
                traces.append(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines",
                                           line=dict(color=color, width=width), name=name, hoverinfo="skip"))

    # Draw three short lines at the origin showing the camera's x (red), y (green), z (blue) axes.
    for axis, color in enumerate(("red", "green", "blue")):  # camera axes, as plot_opencv_cam=True drew them
        # A 10 cm line along the given axis, starting at the origin.
        end = np.zeros(3)
        end[axis] = 0.1
        traces.append(go.Scatter3d(x=[0, end[0]], y=[0, end[1]], z=[0, end[2]], mode="lines",
                                   line=dict(color=color, width=6), showlegend=False, hoverinfo="skip"))

    # Combine the layers into one figure and style it.
    fig = go.Figure(data=traces)
    fig.update_layout(
        # Dark theme to match the page.
        template="plotly_dark",
        # No margins, so the plot fills its box.
        margin=dict(l=0, r=0, t=0, b=0),
        # Horizontal legend along the top.
        legend=dict(orientation="h", y=1.0),
        scene=dict(
            # Axis labels in meters.
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            # Equal scale on all three axes, so shapes are not stretched.
            aspectmode="data",
            # OpenCV camera frame has y pointing down: flip "up" and view from behind the camera.
            camera=dict(up=dict(x=0, y=-1, z=0), eye=dict(x=0.0, y=-0.4, z=-1.6)),
        ),
    )
    # Serialize the figure to a JSON string the browser's Plotly library can draw.
    return pio.to_json(fig)


def build_views(run_dir: Path) -> dict:
    """Create overlay.jpg and scene.json for one finished run; returns summary numbers."""
    # Open the model's result file and the frame that was sent to it.
    result, captured = np.load(run_dir / "result.npz"), np.load(run_dir / "input.npz")
    # Grasp poses and scores.
    grasps, scores = result["grasps"], result["scores"]
    # Gripper wireframe points and the point pairs to connect.
    gp, seg = result["gripper_points"], result["gripper_segments"]
    # Make the 2D overlay (grasps drawn on the captured RGB image) and save it as a JPEG file.
    (run_dir / "overlay.jpg").write_bytes(draw_overlay(captured["rgb"], grasps, scores, captured["K"], gp, seg))
    # Make the 3D scene description and save it as JSON text.
    (run_dir / "scene.json").write_text(
        scene_figure_json(result["pc_full"], result["pc_colors"], grasps, scores, gp, seg))
    # Summary numbers for the status line (best_score is None when there are no grasps).
    return {"num_grasps": int(len(grasps)), "best_score": float(scores.max()) if scores.size else None}


# ======================================================================================
# Web app
# ======================================================================================
def create_app(feed: CameraFeed, client: ContactGraspnetClient, state: dict, args) -> Flask:
    """Build the Flask web application and register all of its URL routes."""
    # The Flask application object; __name__ tells Flask where this module lives.
    app = Flask(__name__)
    # Folder where every capture gets its own sub-folder.
    out_root = Path(args.output_dir)
    # Lock that keeps status reads and capture requests from interleaving.
    lock = threading.Lock()
    # Small dictionary used to cache the (large) Plotly JavaScript text after the first request.
    plotly_js = {}

    def run_dir_for(run_id: str) -> Path:
        # Reject any run id that does not look like a real one (blocks "../" tricks): HTTP 404.
        if not RUN_ID_RE.fullmatch(run_id):
            abort(404)
        # The folder for this run.
        path = out_root / run_id
        # If it does not exist, also 404.
        if not path.is_dir():
            abort(404)
        return path

    def no_store(response):
        # Tell the browser never to cache this response, so it always sees the latest data.
        response.headers["Cache-Control"] = "no-store"
        return response

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
                # Model process state, its status text, restart count, and recent log lines.
                "client": {"state": client.state, "message": client.message, "restarts": client.restarts,
                           "log": client.log_tail(20)},
                # Id of the capture being processed (None when idle).
                "busy": state["pending"],
                # Text of the last failure, shown to the user.
                "last_error": state["last_error"],
                # Summary of the newest finished result (None before the first one).
                "result": state["latest"],
                # Settings the page displays.
                "config": {"min_depth": args.min_depth, "max_depth": args.max_depth,
                           "forward_passes": args.forward_passes},
            }
        # Convert the dictionary to a JSON response.
        return no_store(jsonify(payload))

    @app.post("/api/capture")
    def capture():
        # Read the prompt text from the JSON body (empty if absent); limit it to 2000 characters.
        prompt = str((request.get_json(silent=True) or {}).get("prompt", ""))[:2000]
        with lock:
            # Refuse (HTTP 409 Conflict) if a capture is running or the model is not ready.
            if state["pending"] is not None or client.state != "ready":
                return jsonify(error="The model is not ready (or is already busy)."), 409
            # Grab the latest camera frame.
            frame = feed.snapshot()
            # No frame yet: HTTP 503 Service Unavailable.
            if frame is None:
                return jsonify(error="No camera frame available yet."), 503
            depth, rgb, K = frame
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
            np.savez(run_dir / "input.npz", depth=depth.astype(np.float32), K=K.astype(np.float32), rgb=rgb)
            # Information saved alongside the capture.
            meta = {"id": run_id, "prompt": prompt, "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "z_range": [args.min_depth, args.max_depth], "forward_passes": args.forward_passes}
            # Write it as indented JSON.
            (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
            # Keep it in memory too, so the result handler can reuse it.
            state["meta"][run_id] = meta
            # A new capture clears the previous error and marks this run as in progress.
            state["last_error"] = None
            state["pending"] = run_id
            # Hand the request to the model process; if that fails, undo the "in progress" mark.
            if not client.submit(run_id, run_dir / "input.npz", run_dir / "result.npz"):
                state["pending"] = None
                return jsonify(error="Could not hand the frame to ContactGraspnet_Client."), 503
        # Success: tell the page which run was started.
        return jsonify(id=run_id)

    @app.post("/api/client/restart")
    def restart_client():
        # Restart in a background thread so this request returns immediately (the restart can take a while).
        threading.Thread(target=client.restart, daemon=True).start()
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


def make_callbacks(state: dict, lock_free_out_root: Path):
    """Create the two functions the client supervisor calls when a request finishes or fails."""

    def on_done(run_id, info):
        # Called when the model finishes a request successfully.
        try:
            # Build the overlay image and the 3D scene file from the result.
            summary = build_views(lock_free_out_root / run_id)
            # The info saved when this capture started (prompt, time, ...).
            meta = state["meta"].get(run_id, {})
            # The summary the web page shows for the newest result.
            result = {"id": run_id, "prompt": meta.get("prompt", ""), "captured_at": meta.get("captured_at", ""),
                      "seconds": info.get("seconds"), **summary}
            # Publish it so /api/status reports it.
            state["latest"] = result
            # Update meta.json on disk with the result numbers as well.
            (lock_free_out_root / run_id / "meta.json").write_text(json.dumps({**meta, **result}, indent=2))
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


def package_dir(name: str):
    """Directory of an installed package without importing it (importing TensorFlow here would defeat the split)."""
    # find_spec locates the package on disk without running its code.
    spec = importlib.util.find_spec(name)
    # For a package, submodule_search_locations lists its folder(s); take the first. Otherwise return None.
    return list(spec.submodule_search_locations)[0] if spec and spec.submodule_search_locations else None


def main():
    """Parse options, start the camera and model threads, and run the web server until stopped."""
    # Define every command-line option.
    parser = argparse.ArgumentParser(description="Gungnir_Server: browser UI for the L515 + Contact-GraspNet pipeline")
    # Network address to listen on: localhost only by default, so only an SSH tunnel can reach it.
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost only; use an SSH tunnel)")
    # TCP port of the web server.
    parser.add_argument("--port", type=int, default=8000)
    # Depth range in meters: used for the depth color map and passed to the model as its crop.
    parser.add_argument("--min-depth", type=float, default=0.1, help="metres; also the depth crop passed to the model")
    parser.add_argument("--max-depth", type=float, default=2.5)
    # Parallel forward passes of the model (fixed when the model is built).
    parser.add_argument("--forward-passes", type=int, default=3, help="fixed when the model is built")
    # Which checkpoint folder to load (None = the client's default).
    parser.add_argument("--ckpt-dir", default=None, help="default: contact_graspnet/checkpoints/scene_test_2048_bs3_hor_sigma_0025")
    # Where captures and results are saved.
    parser.add_argument("--output-dir", default=str(ROOT / "output" / "runs"))
    # Frame rate of the live view, and the JPEG quality used for it.
    parser.add_argument("--stream-fps", type=float, default=15, help="live-view rate; lower it over a slow link")
    parser.add_argument("--jpeg-quality", type=int, default=70)
    # Flag (no value): swap the red and blue channels.
    parser.add_argument("--swap-rb", action="store_true", help="flip red/blue if the live colours look wrong")
    # Flag: do not force synchronous CUDA launches in the model process.
    parser.add_argument("--no-launch-blocking", action="store_true",
                        help="do not set CUDA_LAUNCH_BLOCKING=1 in ContactGraspnet_Client (faster, but your pipeline needs it today)")
    # Path of the model process script.
    parser.add_argument("--client-script", default=str(ROOT / "ContactGraspnet_Client.py"))
    # Read the options from the command line.
    args = parser.parse_args()

    # Make sure the output folder exists (creating parents; fine if it already does).
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Environment variables for the model process: a copy of ours plus legacy Keras (needed by the repo's
    # TF1-style layers) and unbuffered Python output (so log lines appear immediately).
    env = {**os.environ, "TF_USE_LEGACY_KERAS": "1", "PYTHONUNBUFFERED": "1"}
    # Make every CUDA call finish before the next starts; slower, but your pipeline currently needs it.
    if not args.no_launch_blocking:
        env["CUDA_LAUNCH_BLOCKING"] = "1"
    # Find TensorFlow's install folder (without importing it).
    tf_dir = package_dir("tensorflow")
    if tf_dir:
        # Put it first on the library search path so the custom ops find TensorFlow's shared libraries.
        env["LD_LIBRARY_PATH"] = f"{tf_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"

    # The command that starts the model process: this same Python, the client script, and its options.
    client_argv = [sys.executable, args.client_script, "--forward-passes", str(args.forward_passes),
                   "--z-min", str(args.min_depth), "--z-max", str(args.max_depth)]
    # If a checkpoint folder was given, pass it as an absolute path.
    if args.ckpt_dir:
        client_argv += ["--ckpt-dir", str(Path(args.ckpt_dir).resolve())]

    # Shared state between the web routes and the callbacks:
    # pending = id of the running capture; last_error = message to show; latest = newest result summary;
    # meta = per-run information saved at capture time.
    state = {"pending": None, "last_error": None, "latest": None, "meta": {}}
    # The two functions the supervisor calls on success / failure.
    on_done, on_error = make_callbacks(state, out_root)
    # The model process supervisor (not started yet) and the camera thread (not started yet).
    client = ContactGraspnetClient(client_argv, env, str(ROOT), on_done, on_error)
    feed = CameraFeed(args.min_depth, args.max_depth, args.stream_fps, args.jpeg_quality, args.swap_rb)

    # Only print warnings from the web server; the page polls /api/status twice a second and would flood the terminal.
    logging.getLogger("werkzeug").setLevel(logging.WARNING)  # the page polls /api/status twice a second
    # Build the web app.
    app = create_app(feed, client, state, args)
    # Start the camera thread and the model process (both run in the background).
    feed.start()
    client.start()
    # Tell the user where to point the browser (flush so it shows up immediately).
    print(f"Serving on http://{args.host}:{args.port}  (results in {out_root})", flush=True)

    def terminate(signum, frame):
        # Turn a termination signal into KeyboardInterrupt, which the try/finally below handles like Ctrl-C.
        raise KeyboardInterrupt  # run the same cleanup as Ctrl-C (close the camera, stop ContactGraspnet_Client)

    # kill / systemd stop (SIGTERM) now runs the cleanup too.
    signal.signal(signal.SIGTERM, terminate)
    if hasattr(signal, "SIGHUP"):  # an SSH session dropping sends SIGHUP; without this the camera is never closed
        signal.signal(signal.SIGHUP, terminate)
    try:
        # Run the web server until interrupted. threaded=True serves several browsers at once (the streams
        # keep connections open); use_reloader=False stops Flask from starting a second copy that would
        # also try to open the camera.
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    finally:
        # Always runs on exit: stop the model process first, then release the camera.
        client.stop()
        feed.stop()


# The web page, sent to the browser as one HTML document. The Python r prefix keeps backslashes (such as
# the \n in the JavaScript) exactly as written. Comments inside it are HTML <!-- --> comments, CSS /* */
# comments and JavaScript // comments; the browser ignores them.
INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<!-- Make the page scale properly on small screens -->
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gungnir Server</title>
<style>
  /* Color palette and box sizing: the variables below are reused through the stylesheet */
  :root { --bg:#0f1318; --panel:#171d25; --line:#2a3340; --text:#e6edf3; --muted:#8b98a8;
          --accent:#3b82f6; --ok:#22c55e; --warn:#f59e0b; --bad:#ef4444; }
  * { box-sizing: border-box; }
  /* Page background, text color and font */
  body { margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui, sans-serif; }
  /* Top bar holding the title, the status pills and the link to the output window */
  header { display:flex; flex-wrap:wrap; gap:10px; align-items:center; padding:10px 16px;
           border-bottom:1px solid var(--line); background:var(--panel); }
  header h1 { font-size:16px; margin:0 12px 0 0; }
  /* Small rounded status badges; the ok/warn/bad variants change the color */
  .pill { padding:2px 10px; border-radius:999px; border:1px solid var(--line); font-size:12px; color:var(--muted); }
  .pill.ok { color:var(--ok); border-color:var(--ok); } .pill.warn { color:var(--warn); border-color:var(--warn); }
  .pill.bad { color:var(--bad); border-color:var(--bad); }
  /* Flexible gap that pushes the link to the right edge of the header */
  header .spacer { flex:1; }
  a { color:var(--accent); }
  /* Page body: sections stacked with a gap between them */
  main { padding:16px; display:grid; gap:16px; }
  section { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; }
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
  /* Status/error line under the buttons (the .bad variant is red) */
  #message { margin-top:8px; min-height:1.4em; color:var(--muted); } #message.bad { color:var(--bad); }
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
  <h1>Gungnir Server</h1>
  <span class="pill" id="pill-camera">camera ...</span>
  <span class="pill" id="pill-client">model ...</span>
  <span class="spacer"></span>
  <!-- target="grasp-output" reuses the same browser window/tab each time the link is clicked -->
  <span class="live-only"><a href="/?view=results" target="grasp-output">Open output in a separate window</a></span>
</header>
<main>
  <!-- Live camera section: the two img tags load the endless MJPEG streams from the server -->
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
      <button id="restart" class="secondary" title="Restart the model process (use after a CUDA error)">Restart model</button>
    </div>
    <div id="message"></div>
    <details><summary>ContactGraspnet_Client log</summary><pre id="log"></pre></details>
  </section>
  <!-- Model output section: a summary line, the interactive 3D scene, and the 2D overlay image -->
  <section>
    <h2>Model output</h2>
    <div id="meta">No result yet. Press "Capture &amp; run".</div>
    <div class="two">
      <figure><div id="scene3d">3D scene appears here</div><figcaption>3D point cloud and grasps (drag to rotate, scroll to zoom)</figcaption></figure>
      <figure><img class="view" id="overlay" alt="Grasps overlaid on the captured RGB image"><figcaption>Grasps projected onto the captured RGB image (gold = top 10%, green = top 30%)</figcaption></figure>
    </div>
  </section>
</main>
<!-- Plotly's JavaScript library, served by this same server (no internet needed) -->
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
  // shownResult: id of the result currently displayed; submitting: true while a capture request is being sent;
  // lastStatus: the most recent status object received from the server.
  let shownResult = null, submitting = false, lastStatus = null;

  // Set a pill's text and color class (ok / warn / bad, or none).
  function setPill(el, text, cls) { el.textContent = text; el.className = 'pill ' + (cls || ''); }
  // Show a message under the buttons, in red when `bad` is true.
  function say(text, bad) { $('message').textContent = text || ''; $('message').className = bad ? 'bad' : ''; }

  // Update the whole page from one status object returned by /api/status.
  function render(s) {
    lastStatus = s;
    const cam = s.camera;
    // Camera pill: fps when working, the error text when not, otherwise "starting".
    setPill($('pill-camera'),
            cam.ok ? `camera ${cam.fps} fps` : (cam.error ? 'camera: ' + cam.error.slice(0, 60) : 'camera starting...'),
            cam.ok ? 'ok' : (cam.error ? 'bad' : 'warn'));
    // Model pill: its state, plus the restart count if it has restarted; the color follows the state.
    const w = s.client, cls = {ready:'ok', busy:'warn', starting:'warn', error:'bad', stopped:'bad'}[w.state];
    setPill($('pill-client'), 'model: ' + w.state + (w.restarts ? ` (restarts: ${w.restarts})` : ''), cls);
    // Show the depth range used by the color map.
    $('depth-range').textContent = `${s.config.min_depth}-${s.config.max_depth} m`;
    // The Capture button is enabled only when the model is ready, the camera works, nothing is running
    // and no request is already being sent.
    $('capture').disabled = !(w.state === 'ready' && cam.ok && !s.busy && !submitting);
    // Message line: an error wins, then "running", then the model's own status text when it is not ready.
    if (s.last_error) say(s.last_error, true);
    else if (s.busy) say('Running inference on the captured frame...');
    else if (w.state !== 'ready') say(w.message, w.state === 'error');
    else if (!submitting) say('');
    // Fill the log box with the model process's recent output.
    $('log').textContent = (w.log || []).join('\n');
    // If a newer result exists than the one displayed, load it.
    if (s.result && s.result.id !== shownResult) loadResult(s.result);
  }

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

  // Capture button: ask the server to capture the current frame and run the model.
  $('capture').addEventListener('click', async () => {
    // Block double clicks while the request is being sent.
    submitting = true; $('capture').disabled = true; say('Capturing...');
    try {
      // POST the prompt text as JSON.
      const r = await fetch('/api/capture', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                             body: JSON.stringify({prompt: $('prompt').value})});
      // On failure (not ready, no frame...) show the server's explanation.
      if (!r.ok) say((await r.json()).error || 'Capture failed', true);
    } catch (e) { say('Capture request failed: ' + e, true); }
    submitting = false;
  });
  // Restart button: ask the server to restart the model process.
  $('restart').addEventListener('click', () => fetch('/api/client/restart', {method: 'POST'}));

  // Poll the server for status twice a second. The next poll is scheduled only after the current one
  // finishes, so slow responses cannot pile up.
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