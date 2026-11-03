#!/usr/bin/env python3
"""ContactGraspnet_Server: persistent Contact-GraspNet inference server. Runs on the GPU computer.

    python ContactGraspnet_Server.py

That is all: it sets up its own environment (legacy Keras, CUDA_LAUNCH_BLOCKING=1, TensorFlow on
LD_LIBRARY_PATH), loads the model once, and prints the command to run on the camera computer.

HTTP interface used by Gungnir_Client.py:
    GET  /health   -> {"state": "starting|ready|busy", "message", "boot_id", "forward_passes", "log": [...]}
    POST /infer    body = input .npz (depth, K, rgb, ...); headers X-Z-Min / X-Z-Max = depth crop in metres
                   200 -> body = result .npz, headers X-Num-Grasps / X-Seconds;  else JSON {"error": ...}
    POST /restart  -> {"ok": true}, then the server restarts itself

Restarts: after an inference error (a CUDA "illegal address" error is sticky) or a /restart request, the
server replaces itself with a fresh copy of the same command (os.execv: same terminal, same PID, so it also
works under systemd). A failure while starting (no GPU, missing checkpoint, port in use) is printed and
the server exits instead of retrying, because retrying cannot fix those.

Security: the repo's data loader opens the upload with allow_pickle=True, which could run code from a crafted
.npz. Uploads containing pickled objects are therefore rejected before the loader sees them. Use --token (or
$GUNGNIR_TOKEN) as well if the network is shared; the camera computer needs the same token.

AI use: Written by Claude (Anthropic; Claude Sonnet 5.5 via claude.ai), October 2026 (originally
grasp_worker.py; model setup follows contact_graspnet/inference.py). November 2026 revision: replaced the
stdin/stdout pipe protocol with the HTTP interface so it can run on a different computer from
Gungnir_Client.py, and condensed the comments. Tested only against a fake model (HTTP, error and restart
paths), not with TensorFlow or the real hardware; the GraspModel TensorFlow code was not run by Claude.
Reviewed and tested by: Ryan Poliquin
"""

from __future__ import annotations

import argparse
import collections
import hmac
import importlib.util
import json
import os
import queue
import secrets
import socket
import sys
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parent
DEFAULT_PORT = 8765
DEFAULT_Z = (0.1, 9.0)  # depth crop (m) used if a client does not send X-Z-Min / X-Z-Max
MAX_UPLOAD = 256 * 1024 * 1024  # bytes; a normal capture is under 10 MB

# Pairs of indices into the 7 gripper wireframe points built in GraspModel.__init__. Matches
# visualization_utils.draw_grasps except the last pair, (5, 6) instead of (4, 6). Points 2 and 4 coincide,
# so the pair (2, 4) draws nothing.
GRIPPER_SEGMENTS = np.array([[0, 1], [1, 2], [2, 3], [2, 4], [4, 5], [5, 6]], dtype=np.int32)

LOG = collections.deque(maxlen=300)  # last output lines, served through /health
LOG_LOCK = threading.Lock()
_TERMINAL = None  # copy of the real stderr, set by capture_output()


# ======================================================================================
# Process plumbing: environment, log capture, exit and restart
# ======================================================================================
def command() -> list:
    """The command that started this process (used to start a fresh copy of it)."""
    return [sys.executable, str(SCRIPT), *sys.argv[1:]]


def bootstrap_env(no_launch_blocking: bool) -> None:
    """Set the environment variables the model needs. If any had to be added, restart once so they apply."""
    wanted = {"TF_USE_LEGACY_KERAS": "1",  # the repo's tf.compat.v1.layers need legacy Keras
              "PYTHONUNBUFFERED": "1"}  # log lines appear immediately
    if not no_launch_blocking:
        wanted["CUDA_LAUNCH_BLOCKING"] = "1"  # your pipeline currently needs this
    spec = importlib.util.find_spec("tensorflow")  # locates TensorFlow without importing it
    if spec and spec.submodule_search_locations:
        tf_dir = list(spec.submodule_search_locations)[0]
        current = os.environ.get("LD_LIBRARY_PATH", "")
        if tf_dir not in current.split(":"):  # the custom ops must find TensorFlow's shared libraries
            wanted["LD_LIBRARY_PATH"] = f"{tf_dir}:{current}"
    added = {k: v for k, v in wanted.items() if k == "LD_LIBRARY_PATH" or k not in os.environ}
    if added:
        os.environ.update(added)
        os.execv(sys.executable, command())  # the dynamic linker reads LD_LIBRARY_PATH only at process start


def capture_output() -> None:
    """Copy everything printed to stdout/stderr (even by C++ code) into LOG, and still show it on the terminal."""
    global _TERMINAL
    _TERMINAL = os.dup(2)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 1)
    os.dup2(write_fd, 2)
    os.close(write_fd)
    terminal = os.dup(_TERMINAL)

    def pump():
        with os.fdopen(read_fd, "r", errors="replace") as source, os.fdopen(terminal, "w", buffering=1) as out:
            for line in source:
                line = line.rstrip("\n")
                with LOG_LOCK:
                    LOG.append(line)
                out.write(line + "\n")

    threading.Thread(target=pump, daemon=True, name="log-pump").start()


def log_tail(n: int = 20) -> list:
    with LOG_LOCK:
        return list(LOG)[-n:]


def _flush_all() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001
            pass
    time.sleep(0.3)  # lets the log pump copy the last lines to the terminal


def die(code: int = 0) -> None:
    """Exit now. os._exit skips TensorFlow's slow, sometimes-hanging teardown."""
    _flush_all()
    os._exit(code)


def restart_self() -> None:
    """Replace this process with a fresh copy of itself (new CUDA context, model reloaded)."""
    _flush_all()
    if _TERMINAL is not None:  # give the new process the real terminal back, not our log pipe
        os.dup2(_TERMINAL, 1)
        os.dup2(_TERMINAL, 2)
    os.execv(sys.executable, command())


def lan_address() -> str:
    """Best guess of this computer's LAN address, for the hint printed at start-up."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no packet is sent; this just picks the outgoing interface
            return s.getsockname()[0]
    except OSError:
        return "<this-computer's-ip>"


# ======================================================================================
# Model
# ======================================================================================
class GraspModel:
    """Contact-GraspNet loaded once. Mirrors the setup in contact_graspnet/inference.py."""

    def __init__(self, repo_dir: str, ckpt_dir: str, forward_passes: int):
        repo = Path(repo_dir)
        sys.path.insert(0, str(repo / "contact_graspnet"))  # config_utils, data, contact_grasp_estimator, ...
        sys.path.append(str(repo))

        import tensorflow.compat.v1 as tf  # TF1-style API the repo was written for (after the env vars are set)

        tf.disable_eager_execution()
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if not gpus:  # the custom pointnet2 ops run on the GPU
            raise RuntimeError("TensorFlow sees no GPU; the custom pointnet2 ops need one.")
        tf.config.experimental.set_memory_growth(gpus[0], True)  # the Jetson shares RAM with the GPU

        # Imported only now because they load the compiled custom TensorFlow ops.
        import config_utils
        import mesh_utils
        from contact_grasp_estimator import GraspEstimator
        from data import load_available_input_data

        self.load_available_input_data = load_available_input_data
        self.forward_passes = int(forward_passes)  # fixes the network's batch size

        # load_weights() only prints "no checkpoint in ..." and carries on with random weights.
        # Fail loudly instead, so a wrong path can never produce plausible-looking nonsense.
        state = tf.train.get_checkpoint_state(ckpt_dir)
        if not (state and state.model_checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

        config = config_utils.load_config(ckpt_dir, batch_size=self.forward_passes, arg_configs=[])
        self.estimator = GraspEstimator(config)
        self.estimator.build_network()
        saver = tf.train.Saver(save_relative_paths=True)

        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True
        session_config.allow_soft_placement = True  # run ops that have no GPU version on the CPU
        self.sess = tf.Session(config=session_config)
        self.estimator.load_weights(self.sess, saver, ckpt_dir, mode="test")

        # Gripper wireframe in the grasp frame, built exactly as visualization_utils.draw_grasps does.
        control = mesh_utils.create_gripper("panda").get_control_point_tensor(1, False, convex_hull=False).squeeze()
        midpoint = 0.5 * (control[1] + control[2])
        base_lines = np.array([np.zeros(3), midpoint, control[1], control[3], control[1], control[2], control[4]])
        opening = float(config["DATA"]["gripper_width"])  # maximum gripper opening in metres (0.08 here)
        points = base_lines.copy()
        points[2:, 0] = np.sign(base_lines[2:, 0]) * opening / 2  # spread the fingers to the full opening
        self.gripper_points = points.astype(np.float32)

    def infer(self, input_path: str, output_path: str, z_range=None) -> dict:
        """Run the model on one input .npz and write the grasps to output_path (.npz)."""
        started = time.time()
        z_range = [float(z_range[0]), float(z_range[1])] if z_range else list(DEFAULT_Z)
        # Returns (segmap, rgb, depth, camera matrix, point cloud, point colours); missing entries are None.
        segmap, rgb, depth, cam_K, pc_full, pc_colors = self.load_available_input_data(input_path, K=None)
        pc_segments = {}  # no object segments: the whole scene is used
        if pc_full is None:  # our input has depth, not a point cloud: back-project it inside the depth range
            pc_full, pc_segments, pc_colors = self.estimator.extract_point_clouds(
                depth, cam_K, segmap=segmap, rgb=rgb, skip_border_objects=False, z_range=z_range)
        # Results are dicts keyed by segment id (-1 = whole scene): grasp poses, scores, contact points.
        pred_grasps_cam, scores, contact_pts, _ = self.estimator.predict_scene_grasps(
            self.sess, pc_full, pc_segments=pc_segments, local_regions=False, filter_grasps=False,
            forward_passes=self.forward_passes)
        save_result(
            output_path,
            grasps=np.asarray(pred_grasps_cam[-1], dtype=np.float32).reshape(-1, 4, 4),  # (N,4,4) poses, camera frame
            scores=np.asarray(scores[-1], dtype=np.float32).reshape(-1),  # (N,) confidence
            contact_pts=np.asarray(contact_pts[-1], dtype=np.float32).reshape(-1, 3),  # (N,3) surface contact points
            pc_full=np.asarray(pc_full, dtype=np.float32).reshape(-1, 3),  # (M,3) the cloud the model saw
            pc_colors=pc_colors,  # per-point colours (may be None)
            gripper_points=self.gripper_points)
        return {"num_grasps": int(np.asarray(scores[-1]).size), "seconds": round(time.time() - started, 2)}


def save_result(path, *, grasps, scores, contact_pts, pc_full, pc_colors, gripper_points) -> None:
    """Flat arrays only (no pickles), so the client can load results without allow_pickle."""
    if pc_colors is None:
        pc_colors = np.zeros((0, 3), dtype=np.uint8)  # keep the key present
    np.savez(path, grasps=grasps, scores=scores, contact_pts=contact_pts, pc_full=pc_full,
             pc_colors=np.asarray(pc_colors), gripper_points=np.asarray(gripper_points, dtype=np.float32),
             gripper_segments=GRIPPER_SEGMENTS)


# ======================================================================================
# HTTP interface
# ======================================================================================
def validate_npz(path: Path) -> None:
    """Raise unless the file is a plain .npz of numeric arrays (no pickled Python objects)."""
    with np.load(path, allow_pickle=False) as data:  # refuses anything that would have to be unpickled
        for key in data.files:
            data[key]  # noqa: B018 - reading each array forces the check for every entry


class Job:
    """One inference request, handed from an HTTP thread to the main thread (TensorFlow only runs there)."""

    def __init__(self, input_path, output_path, z_range):
        self.input_path, self.output_path, self.z_range = input_path, output_path, z_range
        self.info = self.error = None
        self.done = threading.Event()  # main thread: finished (either way)
        self.responded = threading.Event()  # HTTP thread: reply sent


class Shared:
    """Status shared between the HTTP threads and the main thread."""

    def __init__(self, token, forward_passes):
        self.lock = threading.Lock()
        self.state, self.message = "starting", "Loading the model (this takes a while)..."
        self.boot_id = secrets.token_hex(4)  # changes on every (re)start so clients can see restarts
        self.token, self.forward_passes = token, forward_passes
        self.jobs = queue.Queue()

    def set(self, state, message):
        with self.lock:
            self.state, self.message = state, message


def make_handler(shared: Shared):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # /health is polled every second; keep the terminal quiet
            pass

        def _send(self, code, body, content_type, extra=None):
            try:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                for key, value in (extra or {}).items():
                    self.send_header(key, str(value))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the client went away

        def _json(self, code, payload):
            self._send(code, json.dumps(payload).encode(), "application/json")

        def _authorized(self) -> bool:
            if not shared.token:
                return True
            return hmac.compare_digest(self.headers.get("Authorization", "").encode(),
                                       ("Bearer " + shared.token).encode())

        def do_GET(self):
            if self.path != "/health":
                return self._json(404, {"error": "Not found."})
            if not self._authorized():
                return self._json(401, {"error": "Missing or wrong token."})
            with shared.lock:
                payload = {"state": shared.state, "message": shared.message, "boot_id": shared.boot_id,
                           "forward_passes": shared.forward_passes}
            payload["log"] = log_tail(20)
            self._json(200, payload)

        def do_POST(self):
            if not self._authorized():
                return self._json(401, {"error": "Missing or wrong token."})
            if self.path == "/restart":
                shared.set("starting", "Restart requested; restarting the model...")
                self._json(200, {"ok": True})
                print("[server] restart requested by a client", flush=True)
                threading.Timer(0.5, restart_self).start()
            elif self.path == "/infer":
                self._infer()
            else:
                self._json(404, {"error": "Not found."})

        def _infer(self):
            try:
                length = int(self.headers.get("Content-Length") or 0)
                z_range = (float(self.headers.get("X-Z-Min", DEFAULT_Z[0])),
                           float(self.headers.get("X-Z-Max", DEFAULT_Z[1])))
            except ValueError:
                return self._json(400, {"error": "Bad Content-Length or X-Z-Min/X-Z-Max header."})
            if not 0 < length <= MAX_UPLOAD:
                return self._json(400, {"error": f"Body must be 1 byte to {MAX_UPLOAD // 2**20} MB."})
            with shared.lock:  # claim the model: one inference at a time
                if shared.state != "ready":
                    return self._json(409, {"error": f"ContactGraspnet_Server is {shared.state}, not ready."})
                shared.state, shared.message = "busy", "Running inference..."

            with tempfile.TemporaryDirectory(prefix="gungnir_") as tmp:
                in_path, out_path = Path(tmp) / "input.npz", Path(tmp) / "result.npz"
                try:
                    body = self.rfile.read(length)
                    if len(body) != length:
                        raise ValueError("Upload was cut short.")
                    in_path.write_bytes(body)
                    validate_npz(in_path)
                except Exception as exc:  # noqa: BLE001 - nothing reached the model: release it again
                    shared.set("ready", "Model ready")
                    return self._json(400, {"error": f"Rejected input: {type(exc).__name__}: {exc}"})

                job = Job(in_path, out_path, z_range)
                shared.jobs.put(job)
                job.done.wait()
                try:
                    if job.error is not None:
                        self._json(500, {"error": job.error})
                    else:
                        self._send(200, out_path.read_bytes(), "application/octet-stream",
                                   {"X-Num-Grasps": job.info["num_grasps"], "X-Seconds": job.info["seconds"]})
                finally:
                    job.responded.set()  # after an error the main thread waits for this before restarting

    return Handler


def serve_jobs(model: GraspModel, shared: Shared) -> None:
    """Main-thread loop: run queued inference jobs one at a time. Never returns."""
    while True:
        job = shared.jobs.get()
        try:
            job.info = model.infer(str(job.input_path), str(job.output_path), job.z_range)
        except BaseException as exc:  # noqa: BLE001
            traceback.print_exc()
            job.error = f"{type(exc).__name__}: {exc}"
        if job.error is None:
            shared.set("ready", "Model ready")
            job.done.set()
        else:
            shared.set("starting", "Inference failed; restarting the model...")
            job.done.set()
            job.responded.wait(timeout=10)  # let the error reply go out first
            restart_self()  # the CUDA context may be poisoned: start a fresh process


# ======================================================================================
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="ContactGraspnet_Server: Contact-GraspNet inference server (GPU computer)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="0.0.0.0", help="bind address (default: all interfaces)")
    parser.add_argument("--token", default=os.environ.get("GUNGNIR_TOKEN"), help="shared secret (default: $GUNGNIR_TOKEN)")
    parser.add_argument("--repo-dir", default=str(ROOT / "contact_graspnet"), help="Contact-GraspNet repository root")
    parser.add_argument("--ckpt-dir", default=None, help="default: <repo>/checkpoints/scene_test_2048_bs3_hor_sigma_0025")
    parser.add_argument("--forward-passes", type=int, default=3, help="more passes -> more grasp candidates; fixed at start-up")
    parser.add_argument("--no-launch-blocking", action="store_true",
                        help="do not set CUDA_LAUNCH_BLOCKING=1 (faster, but your pipeline needs it today)")
    args = parser.parse_args(argv)

    bootstrap_env(args.no_launch_blocking)  # may restart this process once
    capture_output()
    ckpt_dir = args.ckpt_dir or str(Path(args.repo_dir) / "checkpoints" / "scene_test_2048_bs3_hor_sigma_0025")
    import signal
    for name in ("SIGINT", "SIGTERM", "SIGHUP"):  # Ctrl-C, kill, closed SSH session: exit without TF's slow teardown
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), lambda signum, frame: die(0))

    shared = Shared(args.token, args.forward_passes)
    try:  # listen first, so clients see "starting" (not "unreachable") while the model loads
        httpd = ThreadingHTTPServer((args.host, args.port), make_handler(shared))
    except OSError as exc:
        print(f"[server] cannot listen on {args.host}:{args.port}: {exc}", flush=True)
        die(2)
    threading.Thread(target=httpd.serve_forever, daemon=True, name="http").start()
    print(f"[server] listening on port {args.port}. On the camera computer run:\n"
          f"         python Gungnir_Client.py {lan_address()}" + (" --token <your token>" if args.token else ""), flush=True)
    if not args.token:
        print("[server] no token set: anyone who can reach this port can use the GPU (fine on a private LAN).", flush=True)

    try:
        model = GraspModel(args.repo_dir, ckpt_dir, args.forward_passes)
    except BaseException:  # noqa: BLE001 - report anything, including SystemExit from imports
        traceback.print_exc()
        print("[server] could not start the model (see above); exiting.", flush=True)
        die(2)
    shared.set("ready", "Model ready")
    print("[server] model ready", flush=True)
    serve_jobs(model, shared)
    return 0


if __name__ == "__main__":
    sys.exit(main())