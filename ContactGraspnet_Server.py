#!/usr/bin/env python3
"""ContactGraspnet_Server: persistent Contact-GraspNet inference server. Runs on the GPU computer.

    python ContactGraspnet_Server.py

That is all: it sets up its own environment (legacy Keras, CUDA_LAUNCH_BLOCKING=1, TensorFlow on
LD_LIBRARY_PATH), loads the model once, and prints the command to run on the camera computer.

----------------------------------------------------------------------------------------------------
COMMUNICATION PROTOCOL (used by Gungnir_Client.py)
----------------------------------------------------------------------------------------------------
Transport: plain HTTP/1.0 over TCP, default port 8765. Every request opens its own connection, sends one
request, receives one reply, and closes. The CLIENT always starts the conversation; the server only answers.

    GET  /health   -> 200 {"state": "starting|ready|busy", "message": str, "boot_id": str,
                           "forward_passes": int, "log": [last 20 log lines]}
    POST /infer    request : body = an input .npz file (arrays: depth, K, rgb, seg_map)
                             headers = X-Z-Min / X-Z-Max (depth crop in metres), X-Run-Id (informational)
                   200     : body = the result .npz file (arrays: grasps, scores, contact_pts, pc_full,
                             pc_colors, gripper_points, gripper_segments);
                             headers = X-Num-Grasps, X-Seconds
                   400     : the upload was rejected (bad headers, cut short, not a plain .npz)
                   409     : the model is not "ready" (still loading, or busy with another request)
                   500     : the model raised an error; the server restarts itself right afterwards
    POST /restart  -> 200 {"ok": true}; the server then restarts itself (new process, model reloaded)

    If the server was started with a token, every request must carry "Authorization: Bearer <token>"
    (otherwise: 401).

Typical conversation (time runs downwards):

    camera computer (client)                          GPU computer (this file)
    ------------------------                          ------------------------
    GET /health  every 1 s  ------------------------>  {"state": "starting"}   (model still loading)
    GET /health  ------------------------------------>  {"state": "ready"}
    (user presses Capture)
    POST /infer  [input.npz, ~5-10 MB]  ------------->  state -> "busy", job queued, main thread runs it
                                                        (meanwhile GET /health -> {"state": "busy"})
                 <-------------------------------------  200 [result.npz]; state -> "ready"
    (client saves result.npz and draws the views)

Restart rule: after an inference error (a CUDA "illegal address" error is sticky) or a /restart request,
the server replaces itself with a fresh copy of the same command (os.execv: same terminal, same PID, so it
also works under systemd). The client sees a different "boot_id" in /health and counts it as a restart.
A failure while STARTING (no GPU, missing checkpoint, port in use) is printed and the server exits instead of
retrying, because retrying cannot fix those.

Security: the repo's data loader opens the upload with allow_pickle=True, which could run code from a crafted
.npz. Uploads containing pickled objects are therefore rejected before the loader sees them. Use --token (or
$GUNGNIR_TOKEN) as well if the network is shared; the camera computer needs the same token.

AI use: Written by Claude (Anthropic; Claude Sonnet 5.5 via claude.ai), October 2026 (originally
grasp_worker.py; model setup follows contact_graspnet/inference.py). November 2026 revision: replaced the
stdin/stdout pipe protocol with the HTTP interface so it can run on a different computer from
Gungnir_Client.py; line-by-line comments restored in a later pass (comments only: the executable code is
identical to the version the author uploaded in November 2026). Tested only against a fake model (HTTP, error and restart paths), not with
TensorFlow or the real hardware; the GraspModel TensorFlow code was not run by Claude.
Reviewed and tested by: Ryan Poliquin
"""

# Store type annotations as text instead of evaluating them when the file is loaded, so annotations such as
# `tuple[float, float]` work on any Python 3 version.
from __future__ import annotations

import argparse  # reads the command-line options (--port, --token, ...)
import collections  # deque: a list with a fixed maximum length (used for the log buffer)
import hmac  # compare_digest: compares two secrets in constant time (so timing cannot leak the token)
import importlib.util  # find_spec: finds where a package is installed WITHOUT importing it
import json  # encodes/decodes the JSON replies
import os  # environment variables, low-level file descriptors, execv, _exit
import queue  # thread-safe queue that hands inference jobs from the HTTP threads to the main thread
import secrets  # random "boot id" so clients can tell when the server restarted
import socket  # used only to guess this computer's LAN address for the start-up hint
import sys  # sys.executable (the Python interpreter), sys.argv (our own command line), sys.path, sys.exit
import tempfile  # a scratch folder per request for its input and result files
import threading  # background threads, locks and events
import time  # stopwatch for inference and short sleeps
import traceback  # prints the full error trace when something fails
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # the web server (standard library)
from pathlib import Path  # path objects that behave the same on any operating system

import numpy as np  # arrays: depth images, point clouds, grasp matrices

# Absolute path of this script (resolve() makes it absolute and follows symlinks).
SCRIPT = Path(__file__).resolve()
# The folder that contains this script; the Contact-GraspNet repo is expected next to it.
ROOT = SCRIPT.parent
# TCP port the server listens on; the client uses the same default.
DEFAULT_PORT = 8765
# Depth crop (min, max) in metres, used if a client does not send X-Z-Min / X-Z-Max.
DEFAULT_Z = (0.1, 9.0)
# Largest upload accepted, in bytes (256 MB). A normal capture is under 10 MB; this stops absurd requests.
MAX_UPLOAD = 256 * 1024 * 1024

# Each row is a pair of indices into the 7 gripper wireframe points built in GraspModel.__init__; a grasp is
# drawn by joining each pair with a line. Matches visualization_utils.draw_grasps except the last pair, which
# is (5, 6) here instead of (4, 6). Points 2 and 4 are the same point (both are control point 1), so the
# pair (2, 4) draws nothing.
GRIPPER_SEGMENTS = np.array([[0, 1], [1, 2], [2, 3], [2, 4], [4, 5], [5, 6]], dtype=np.int32)

# The last 300 lines the process printed (stdout and stderr). Served to the client through /health.
LOG = collections.deque(maxlen=300)
# Lock for LOG: reading a deque while another thread appends to it can raise an error.
LOG_LOCK = threading.Lock()
# A copy of the real terminal's file descriptor; set by capture_output() so restart_self() can restore it.
_TERMINAL = None


# ======================================================================================
# Process plumbing: environment, log capture, exit and restart
# ======================================================================================
def command() -> list:
    """The command that started this process (used to start a fresh copy of it)."""
    # [python interpreter, this script, then the original command-line options].
    return [sys.executable, str(SCRIPT), *sys.argv[1:]]


def bootstrap_env(no_launch_blocking: bool) -> None:
    """Set the environment variables the model needs. If any had to be added, restart once so they apply."""
    # Variables we want to exist, with the value to use if they are missing.
    wanted = {"TF_USE_LEGACY_KERAS": "1",  # the repo's tf.compat.v1.layers need legacy Keras
              "PYTHONUNBUFFERED": "1"}  # Python output appears immediately instead of in blocks
    # Make every CUDA call finish before the next starts: slower, but your pipeline currently needs it.
    if not no_launch_blocking:
        wanted["CUDA_LAUNCH_BLOCKING"] = "1"
    # Locate TensorFlow's install folder without importing it (importing would load TensorFlow too early).
    spec = importlib.util.find_spec("tensorflow")
    if spec and spec.submodule_search_locations:
        # For a package, submodule_search_locations lists its folder(s); take the first.
        tf_dir = list(spec.submodule_search_locations)[0]
        # What LD_LIBRARY_PATH (the shared-library search path) currently contains.
        current = os.environ.get("LD_LIBRARY_PATH", "")
        # If TensorFlow's folder is not on it yet, put it first: the custom ops need TensorFlow's libraries.
        if tf_dir not in current.split(":"):
            wanted["LD_LIBRARY_PATH"] = f"{tf_dir}:{current}"
    # Keep only what actually has to change: LD_LIBRARY_PATH (already checked above) and any variable that is
    # not set yet. Variables the user set themselves are left alone.
    added = {k: v for k, v in wanted.items() if k == "LD_LIBRARY_PATH" or k not in os.environ}
    if added:
        os.environ.update(added)
        # Start this program again in place of the current process. LD_LIBRARY_PATH is read by the dynamic
        # linker only when a process starts, so changing it inside a running process would have no effect.
        # The second run finds everything already set and carries on.
        os.execv(sys.executable, command())


def capture_output() -> None:
    """Copy everything printed to stdout/stderr (even by C++ code) into LOG, and still show it on the terminal."""
    # "global" lets this function assign the module-level _TERMINAL variable.
    global _TERMINAL
    # os.dup(2) makes a copy of file descriptor 2 (standard error): the real terminal.
    _TERMINAL = os.dup(2)
    # A pipe: whatever is written to write_fd can be read from read_fd.
    read_fd, write_fd = os.pipe()
    # Point descriptors 1 (stdout) and 2 (stderr) at the pipe. From now on every print() AND every message
    # written by C/C++ libraries (TensorFlow's own logging) goes into the pipe.
    os.dup2(write_fd, 1)
    os.dup2(write_fd, 2)
    # The original write_fd is no longer needed (1 and 2 are copies of it).
    os.close(write_fd)
    # A second copy of the terminal for the reader thread below to write to.
    terminal = os.dup(_TERMINAL)

    def pump():
        # Open the pipe's read end and the terminal copy as text streams (closed automatically at the end).
        with os.fdopen(read_fd, "r", errors="replace") as source, os.fdopen(terminal, "w", buffering=1) as out:
            # Reading the pipe yields one line at a time, blocking while there is none.
            for line in source:
                line = line.rstrip("\n")
                # Remember the line for /health...
                with LOG_LOCK:
                    LOG.append(line)
                # ...and echo it to the real terminal.
                out.write(line + "\n")

    # Run the reader in the background; daemon=True so it never keeps the program alive.
    threading.Thread(target=pump, daemon=True, name="log-pump").start()


def log_tail(n: int = 20) -> list:
    """Return the last n log lines (a copy, safe to use while the log keeps growing)."""
    with LOG_LOCK:
        return list(LOG)[-n:]


def _flush_all() -> None:
    """Push any buffered output out, and pause briefly so the log pump can copy it to the terminal."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001
            pass  # flushing can fail if the stream is already closed; ignore that
    time.sleep(0.3)  # the log pump copies the final lines during this pause


def die(code: int = 0) -> None:
    """Exit now. os._exit skips TensorFlow's slow, sometimes-hanging teardown."""
    _flush_all()
    os._exit(code)


def restart_self() -> None:
    """Replace this process with a fresh copy of itself (new CUDA context, model reloaded)."""
    _flush_all()
    # Give the new process the real terminal back (descriptors 1 and 2 currently point at our log pipe).
    if _TERMINAL is not None:
        os.dup2(_TERMINAL, 1)
        os.dup2(_TERMINAL, 2)
    # execv replaces the running program in place: same process id, same terminal, nothing left of the old state
    # (including a possibly corrupted CUDA context). The listening socket is closed automatically by the exec.
    os.execv(sys.executable, command())


def lan_address() -> str:
    """Best guess of this computer's LAN address, for the hint printed at start-up."""
    try:
        # A UDP socket is just a convenient way to ask the operating system which interface it would use.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no packet is actually sent for UDP connect()
            return s.getsockname()[0]  # the local address that would be used
    except OSError:
        return "<this-computer's-ip>"


# ======================================================================================
# Model
# ======================================================================================
class GraspModel:
    """Contact-GraspNet loaded once. Mirrors the setup in contact_graspnet/inference.py."""

    def __init__(self, repo_dir: str, ckpt_dir: str, forward_passes: int):
        # Turn the repository path text into a Path object.
        repo = Path(repo_dir)
        # Put the repo's inner folder first on Python's module search path so `import config_utils` etc. finds it.
        sys.path.insert(0, str(repo / "contact_graspnet"))  # config_utils, data, contact_grasp_estimator, ...
        # Also add the repo root (last), as inference.py does, for any imports that expect it.
        sys.path.append(str(repo))

        # Imported here (not at the top of the file) so the slow TensorFlow import happens only now, after the
        # environment variables are set. "compat.v1" is the TensorFlow 1 style API the repo was written for.
        import tensorflow.compat.v1 as tf

        # Use TensorFlow 1 "graph mode" (placeholders and sessions) instead of eager execution.
        tf.disable_eager_execution()
        # Ask TensorFlow which GPUs it can see.
        gpus = tf.config.experimental.list_physical_devices("GPU")
        # The custom pointnet2 operations run on the GPU, so with no GPU the model cannot work: stop early.
        if not gpus:
            raise RuntimeError("TensorFlow sees no GPU; the custom pointnet2 ops need one.")
        # Allocate GPU memory only as needed instead of grabbing it all up front (the Jetson shares RAM).
        tf.config.experimental.set_memory_growth(gpus[0], True)

        # These repo modules are imported only now, after TensorFlow is configured, because importing them
        # loads the compiled custom TensorFlow operations.
        import config_utils  # loads the model's config.yaml (and applies overrides such as the batch size)
        import mesh_utils  # builds the Panda gripper model used for the grasp wireframe
        from contact_grasp_estimator import GraspEstimator  # the Contact-GraspNet network and its inference code
        from data import load_available_input_data  # reads the .npz input (depth, K, rgb)

        # Keep the loader on the object so infer() can use it later.
        self.load_available_input_data = load_available_input_data
        # Number of parallel forward passes; it fixes the network's batch size when the graph is built below.
        self.forward_passes = int(forward_passes)

        # load_weights() only prints "no checkpoint in ..." and carries on with random weights.
        # Fail loudly instead, so a wrong path can never produce plausible-looking nonsense.
        # get_checkpoint_state reads the "checkpoint" file in the folder; it returns None if there is none.
        state = tf.train.get_checkpoint_state(ckpt_dir)
        # Stop with a clear error unless a checkpoint file exists and names a model checkpoint.
        if not (state and state.model_checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

        # Load the model configuration from the checkpoint folder; batch_size is set to the number of
        # forward passes, and no extra command-line style overrides are applied.
        config = config_utils.load_config(ckpt_dir, batch_size=self.forward_passes, arg_configs=[])
        # Create the Contact-GraspNet object from that configuration.
        self.estimator = GraspEstimator(config)
        # Build the TensorFlow graph (the network layers); the weights are still random at this point.
        self.estimator.build_network()
        # A Saver reads saved variable values from a checkpoint into the graph;
        # save_relative_paths matches how the checkpoint was written.
        saver = tf.train.Saver(save_relative_paths=True)

        # Settings for the TensorFlow session that will run the graph.
        session_config = tf.ConfigProto()
        # Grow GPU memory use on demand rather than reserving everything immediately.
        session_config.gpu_options.allow_growth = True
        # If an operation has no GPU version, let TensorFlow run it on the CPU instead of failing.
        session_config.allow_soft_placement = True
        # Create the session: the object that actually executes the graph.
        self.sess = tf.Session(config=session_config)
        # Copy the trained weights from the checkpoint into the graph's variables.
        self.estimator.load_weights(self.sess, saver, ckpt_dir, mode="test")

        # Gripper wireframe in the grasp frame, built exactly as visualization_utils.draw_grasps does.
        # Control points are key points on the Panda gripper's geometry; the result has shape (1, 6, 3)
        # (a batch of one), which squeeze() reduces to (6, 3).
        control = mesh_utils.create_gripper("panda").get_control_point_tensor(1, False, convex_hull=False).squeeze()
        # Point halfway between control points 1 and 2.
        midpoint = 0.5 * (control[1] + control[2])
        # The 7 wireframe points: gripper origin, that midpoint, then five points picked from the control points.
        base_lines = np.array([np.zeros(3), midpoint, control[1], control[3], control[1], control[2], control[4]])
        # Maximum opening of the gripper in metres, from the model's config (0.08 for this checkpoint).
        opening = float(config["DATA"]["gripper_width"])
        # Work on a copy so base_lines itself stays unchanged.
        points = base_lines.copy()
        # For points 2-6, set the x coordinate to +/- half the opening (keeping each point's original side),
        # which spreads the fingers to the full opening width.
        points[2:, 0] = np.sign(base_lines[2:, 0]) * opening / 2
        # Store as 32-bit floats, which is what the saved results and the client expect.
        self.gripper_points = points.astype(np.float32)

    def infer(self, input_path: str, output_path: str, z_range=None) -> dict:
        """Run the model on one input .npz and write the grasps to output_path (.npz)."""
        # Start the stopwatch for this request.
        started = time.time()
        # Depth crop for this request: the one the client sent, otherwise the default.
        z_range = [float(z_range[0]), float(z_range[1])] if z_range else list(DEFAULT_Z)
        # Read the input file; K=None means "use the intrinsics stored in the file". Returns
        # (segmentation map, rgb image, depth image, camera matrix, point cloud, point colours);
        # entries that are not in the file come back as None.
        segmap, rgb, depth, cam_K, pc_full, pc_colors = self.load_available_input_data(input_path, K=None)
        # No object segments: the whole scene is used.
        pc_segments = {}
        # Our input has depth rather than a ready-made point cloud, so convert depth to 3D points.
        if pc_full is None:
            # Back-project the depth image to a point cloud with the camera matrix, keeping only points inside
            # the depth range; also returns the matching colours from the rgb image.
            pc_full, pc_segments, pc_colors = self.estimator.extract_point_clouds(
                depth, cam_K, segmap=segmap, rgb=rgb, skip_border_objects=False, z_range=z_range)
        # Run the network. The results are dicts keyed by segment id (-1 is the whole scene): grasp poses
        # (4x4 matrices), scores, and contact points. The fourth result (gripper openings) is unused.
        pred_grasps_cam, scores, contact_pts, _ = self.estimator.predict_scene_grasps(
            self.sess,  # the TensorFlow session holding the loaded weights
            pc_full,  # the scene point cloud
            pc_segments=pc_segments,  # empty: no per-object segments
            local_regions=False,  # do not crop around segments
            filter_grasps=False,  # do not filter grasps by segment
            forward_passes=self.forward_passes)  # more passes -> more grasp candidates
        # Write everything the client needs into the output file.
        save_result(
            output_path,
            grasps=np.asarray(pred_grasps_cam[-1], dtype=np.float32).reshape(-1, 4, 4),  # (N,4,4) poses, camera frame
            scores=np.asarray(scores[-1], dtype=np.float32).reshape(-1),  # (N,) confidence of each grasp
            contact_pts=np.asarray(contact_pts[-1], dtype=np.float32).reshape(-1, 3),  # (N,3) surface contact points
            pc_full=np.asarray(pc_full, dtype=np.float32).reshape(-1, 3),  # (M,3) the point cloud the model saw
            pc_colors=pc_colors,  # per-point colours (may be None)
            gripper_points=self.gripper_points)  # the 7 wireframe points used to draw a gripper
        # Number of grasps found, and how many seconds this request took (rounded to 2 decimals).
        return {"num_grasps": int(np.asarray(scores[-1]).size), "seconds": round(time.time() - started, 2)}


def save_result(path, *, grasps, scores, contact_pts, pc_full, pc_colors, gripper_points) -> None:
    """Flat arrays only (no pickles), so the client can load results without allow_pickle."""
    # If there are no colours, store an empty (0, 3) array so the key always exists in the file.
    if pc_colors is None:
        pc_colors = np.zeros((0, 3), dtype=np.uint8)
    # Write all arrays into one uncompressed .npz archive; each keyword becomes a named array in the file.
    np.savez(path, grasps=grasps, scores=scores, contact_pts=contact_pts, pc_full=pc_full,
             pc_colors=np.asarray(pc_colors), gripper_points=np.asarray(gripper_points, dtype=np.float32),
             gripper_segments=GRIPPER_SEGMENTS)


# ======================================================================================
# HTTP interface
# ======================================================================================
def validate_npz(path: Path) -> None:
    """Raise unless the file is a plain .npz of numeric arrays (no pickled Python objects)."""
    # allow_pickle=False makes NumPy refuse any array that would have to be unpickled (which can run code).
    with np.load(path, allow_pickle=False) as data:
        # data.files lists the array names; reading each one forces the check for every entry.
        for key in data.files:
            data[key]  # noqa: B018 - the read itself is the check


class Job:
    """One inference request, handed from an HTTP thread to the main thread (TensorFlow only runs there)."""

    def __init__(self, input_path, output_path, z_range):
        # Where to read the input from, where to write the result, and the depth crop to use.
        self.input_path, self.output_path, self.z_range = input_path, output_path, z_range
        # Filled in by the main thread: the summary after success, or the error text after failure.
        self.info = self.error = None
        # Set by the main thread when the job has finished (either way); the HTTP thread waits on it.
        self.done = threading.Event()
        # Set by the HTTP thread once its reply has been sent; after an error the main thread waits for it
        # before restarting, so the client always receives the error message.
        self.responded = threading.Event()


class Shared:
    """Status shared between the HTTP threads and the main thread."""

    def __init__(self, token, forward_passes):
        # Protects state/message, which several threads read and write.
        self.lock = threading.Lock()
        # starting -> ready <-> busy; message is the text shown in the client's UI.
        self.state, self.message = "starting", "Loading the model (this takes a while)..."
        # Random id that changes on every (re)start, so clients can see that the server restarted.
        self.boot_id = secrets.token_hex(4)
        # The required token (None = no authentication) and the model's batch size (reported in /health).
        self.token, self.forward_passes = token, forward_passes
        # Jobs waiting for the main thread. Only one is ever queued, because state "busy" refuses the others.
        self.jobs = queue.Queue()

    def set(self, state, message):
        """Change state and message together, under the lock."""
        with self.lock:
            self.state, self.message = state, message


def make_handler(shared: Shared):
    """Build the request-handler class for the web server (it needs access to `shared`)."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            # The default handler prints one line per request; /health is polled every second, so stay quiet.
            pass

        def _send(self, code, body, content_type, extra=None):
            """Send one complete reply: status line, headers, then the body (bytes)."""
            try:
                self.send_response(code)  # e.g. 200 OK
                self.send_header("Content-Type", content_type)
                # Content-Length tells the client where the body ends (the connection is closed afterwards).
                self.send_header("Content-Length", str(len(body)))
                # Extra headers, e.g. X-Num-Grasps.
                for key, value in (extra or {}).items():
                    self.send_header(key, str(value))
                self.end_headers()  # blank line between the headers and the body
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the client went away; nothing to do

        def _json(self, code, payload):
            """Send a dict as a JSON reply."""
            self._send(code, json.dumps(payload).encode(), "application/json")

        def _authorized(self) -> bool:
            """True if no token is required, or the request carries the right one."""
            if not shared.token:
                return True
            # compare_digest compares in constant time; both sides are converted to bytes first.
            return hmac.compare_digest(self.headers.get("Authorization", "").encode(),
                                       ("Bearer " + shared.token).encode())

        def do_GET(self):
            """Handle GET requests: only /health exists."""
            if self.path != "/health":
                return self._json(404, {"error": "Not found."})
            if not self._authorized():
                return self._json(401, {"error": "Missing or wrong token."})
            # Read the shared status under the lock so state and message belong together.
            with shared.lock:
                payload = {"state": shared.state, "message": shared.message, "boot_id": shared.boot_id,
                           "forward_passes": shared.forward_passes}
            # The log has its own lock, so it is added outside the status lock.
            payload["log"] = log_tail(20)
            self._json(200, payload)

        def do_POST(self):
            """Handle POST requests: /restart and /infer."""
            if not self._authorized():
                return self._json(401, {"error": "Missing or wrong token."})
            if self.path == "/restart":
                # Show "starting" right away so clients stop sending work.
                shared.set("starting", "Restart requested; restarting the model...")
                # Reply first...
                self._json(200, {"ok": True})
                print("[server] restart requested by a client", flush=True)
                # ...then restart half a second later, giving the reply time to reach the client.
                threading.Timer(0.5, restart_self).start()
            elif self.path == "/infer":
                self._infer()
            else:
                self._json(404, {"error": "Not found."})

        def _infer(self):
            """Receive an input file, have the main thread run the model, and send back the result file."""
            # Read the request headers: upload size and the optional depth crop.
            try:
                length = int(self.headers.get("Content-Length") or 0)
                z_range = (float(self.headers.get("X-Z-Min", DEFAULT_Z[0])),
                           float(self.headers.get("X-Z-Max", DEFAULT_Z[1])))
            except ValueError:
                return self._json(400, {"error": "Bad Content-Length or X-Z-Min/X-Z-Max header."})
            # Refuse empty or oversized uploads.
            if not 0 < length <= MAX_UPLOAD:
                return self._json(400, {"error": f"Body must be 1 byte to {MAX_UPLOAD // 2**20} MB."})
            # Claim the model under the lock: only one inference at a time, and only when it is ready.
            with shared.lock:
                if shared.state != "ready":
                    return self._json(409, {"error": f"ContactGraspnet_Server is {shared.state}, not ready."})
                shared.state, shared.message = "busy", "Running inference..."

            # A scratch folder that exists until the reply has been sent (then it is deleted automatically).
            with tempfile.TemporaryDirectory(prefix="gungnir_") as tmp:
                in_path, out_path = Path(tmp) / "input.npz", Path(tmp) / "result.npz"
                try:
                    # Read exactly `length` bytes of body from the connection.
                    body = self.rfile.read(length)
                    if len(body) != length:
                        raise ValueError("Upload was cut short.")
                    # Save it, then check that it is a plain .npz without pickled objects.
                    in_path.write_bytes(body)
                    validate_npz(in_path)
                except Exception as exc:  # noqa: BLE001
                    # Nothing reached the model, so release it again and tell the client why.
                    shared.set("ready", "Model ready")
                    return self._json(400, {"error": f"Rejected input: {type(exc).__name__}: {exc}"})

                # Hand the job to the main thread (TensorFlow only ever runs there) and wait until it is done.
                job = Job(in_path, out_path, z_range)
                shared.jobs.put(job)
                job.done.wait()
                try:
                    if job.error is not None:
                        self._json(500, {"error": job.error})
                    else:
                        # Success: the body is the result file; the summary numbers travel in headers.
                        self._send(200, out_path.read_bytes(), "application/octet-stream",
                                   {"X-Num-Grasps": job.info["num_grasps"], "X-Seconds": job.info["seconds"]})
                finally:
                    # Tell the main thread the reply is out (it waits for this before restarting after an error).
                    job.responded.set()

    return Handler


def serve_jobs(model: GraspModel, shared: Shared) -> None:
    """Main-thread loop: run queued inference jobs one at a time. Never returns."""
    while True:
        # Block until an HTTP thread queues a job.
        job = shared.jobs.get()
        try:
            job.info = model.infer(str(job.input_path), str(job.output_path), job.z_range)
        except BaseException as exc:  # noqa: BLE001
            # Print the full trace (it also goes into the log that /health serves).
            traceback.print_exc()
            job.error = f"{type(exc).__name__}: {exc}"
        if job.error is None:
            # Success: ready for the next request, then wake the HTTP thread so it sends the result.
            shared.set("ready", "Model ready")
            job.done.set()
        else:
            shared.set("starting", "Inference failed; restarting the model...")
            job.done.set()
            # Give the HTTP thread time to send the error reply (up to 10 s), then restart: after a CUDA error
            # the session is unusable, so we start a fresh process.
            job.responded.wait(timeout=10)
            restart_self()


# ======================================================================================
# Start-up
# ======================================================================================
def main(argv=None) -> int:
    """Parse options, start listening, load the model, and serve requests."""
    # Define the command-line options.
    parser = argparse.ArgumentParser(description="ContactGraspnet_Server: Contact-GraspNet inference server (GPU computer)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="0.0.0.0", help="bind address (default: all interfaces)")
    parser.add_argument("--token", default=os.environ.get("GUNGNIR_TOKEN"), help="shared secret (default: $GUNGNIR_TOKEN)")
    parser.add_argument("--repo-dir", default=str(ROOT / "contact_graspnet"), help="Contact-GraspNet repository root")
    parser.add_argument("--ckpt-dir", default=None, help="default: <repo>/checkpoints/scene_test_2048_bs3_hor_sigma_0025")
    parser.add_argument("--forward-passes", type=int, default=5, help="more passes -> more grasp candidates; fixed at start-up")
    parser.add_argument("--no-launch-blocking", action="store_true",
                        help="do not set CUDA_LAUNCH_BLOCKING=1 (faster, but your pipeline needs it today)")
    args = parser.parse_args(argv)

    # Set the environment the model needs (this may restart the process once, before anything else happens).
    bootstrap_env(args.no_launch_blocking)
    # From here on, collect all output into the log buffer as well as showing it.
    capture_output()
    # The checkpoint folder: the one given, or the default model inside the repo.
    ckpt_dir = args.ckpt_dir or str(Path(args.repo_dir) / "checkpoints" / "scene_test_2048_bs3_hor_sigma_0025")
    import signal
    # Ctrl-C (SIGINT), kill (SIGTERM) and a closed SSH session (SIGHUP) all exit at once, because TensorFlow's
    # own shutdown can hang.
    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        if hasattr(signal, name):  # SIGHUP does not exist on Windows
            signal.signal(getattr(signal, name), lambda signum, frame: die(0))

    # Status object shared with the HTTP threads.
    shared = Shared(args.token, args.forward_passes)
    # Start listening BEFORE loading the model, so clients see "starting" (not "unreachable") while it loads.
    try:
        httpd = ThreadingHTTPServer((args.host, args.port), make_handler(shared))
    except OSError as exc:  # most likely: the port is already in use
        print(f"[server] cannot listen on {args.host}:{args.port}: {exc}", flush=True)
        die(2)
    # Serve requests in a background thread (each request gets its own thread); the main thread stays free for
    # TensorFlow.
    threading.Thread(target=httpd.serve_forever, daemon=True, name="http").start()
    # Tell the user what to type on the camera computer.
    print(f"[server] listening on port {args.port}. On the camera computer run:\n"
          f"         python Gungnir_Client.py {lan_address()}" + (" --token <your token>" if args.token else ""), flush=True)
    if not args.token:
        print("[server] no token set: anyone who can reach this port can use the GPU (fine on a private LAN).", flush=True)

    # Build the model: the slow part (importing TensorFlow, building the graph, loading the weights).
    try:
        model = GraspModel(args.repo_dir, ckpt_dir, args.forward_passes)
    except BaseException:  # noqa: BLE001 - report anything, including SystemExit from imports
        traceback.print_exc()
        print("[server] could not start the model (see above); exiting.", flush=True)
        die(2)
    # The model is loaded: clients may send work now.
    shared.set("ready", "Model ready")
    print("[server] model ready", flush=True)
    # Run inference jobs on this (main) thread until the process exits.
    serve_jobs(model, shared)
    return 0


# Run main() only when this file is started directly (not when it is imported, e.g. by a test), and pass its
# return value to sys.exit as the process exit code.
if __name__ == "__main__":
    sys.exit(main())