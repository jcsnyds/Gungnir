#!/usr/bin/env python3
"""ContactGraspnet_Server: persistent Contact-GraspNet inference process (the only one that imports TensorFlow).

Started by Gungnir_Client.py. Loads the model once, then answers requests so each capture skips the
model build and checkpoint restore that `inference.py` repeats on every run.

Protocol: one JSON object per line.
    Gungnir_Client -> ContactGraspnet_Server (stdin):
        {"cmd": "infer", "id": ..., "input": "<npz>", "output": "<npz>"}
        {"cmd": "quit"}
    ContactGraspnet_Server -> Gungnir_Client (stdout):
        {"event": "ready"}
        {"event": "done", "id": ..., "num_grasps": N, "seconds": S}
        {"event": "error", "id": ..., "message": ...}   (client exits afterwards)
        {"event": "fatal", "message": ...}              (failed to start)
Everything else the process prints goes to stderr, so stdout carries only protocol messages.

After any inference error this client exits on purpose: a CUDA "illegal address" error is sticky, so
the session is unusable and Gungnir_Client starts a fresh client.

AI use: Written by Claude (Anthropic; Claude Sonnet 5.5 via claude.ai), October 2026. Originally
named grasp_worker.py. Its model setup follows contact_graspnet/inference.py. Claude
tested only the message-passing loop, against a fake model; the TensorFlow code was not run by Claude.
Line-by-line explanatory comments were added by Claude in a later pass (October 2026); that pass
changed comments only, and the executable code is identical to the version the author uploaded.
Reviewed and tested by: Ryan Poliquin
"""

# Store type annotations as text instead of evaluating them when the file is loaded. This lets the
# annotation `tuple[float, float]` below work on any Python 3 version, not just recent ones.
from __future__ import annotations

import argparse  # reads the command-line options this process is started with
import json  # every message on the pipe is one JSON object written on a single line
import os  # environment variables, low-level file descriptors, and os._exit()
import sys  # module search path (sys.path), the request stream (sys.stdin), and sys.exit()
import time  # wall-clock timer for how long each inference takes
import traceback  # prints the full error trace to stderr when something fails
from pathlib import Path  # path objects that behave the same on any operating system

import numpy as np  # array library; the model results are saved as NumPy arrays

# Folder that contains this script: resolve() makes the path absolute (following any symlinks) and
# .parent removes the file name. It is used below to find the Contact-GraspNet repository next to this file.
ROOT = Path(__file__).resolve().parent

# Each row is a pair of indices into the 7 gripper wireframe points built in GraspModel.__init__; a grasp is
# drawn by joining each pair with a line. These match visualization_utils.draw_grasps except the last pair,
# which is (5, 6) here instead of (4, 6). Points 2 and 4 are the same point (both are control point 1),
# so the pair (2, 4) adds no visible line.
GRIPPER_SEGMENTS = np.array([[0, 1], [1, 2], [2, 3], [2, 4], [4, 5], [5, 6]], dtype=np.int32)

# The real standard-output stream, reserved for protocol messages. It is None until main() sets it.
_proto = None


def emit(message: dict) -> None:
    """Send one protocol message (a dict) to Gungnir_Client as a single line of JSON."""
    # json.dumps turns the dict into text; the "\n" marks the end of the message for the reader.
    _proto.write(json.dumps(message) + "\n")
    # Flush so the line leaves our buffer immediately instead of waiting until the buffer fills up.
    _proto.flush()


class GraspModel:
    """Contact-GraspNet loaded once. Mirrors the setup in contact_graspnet/inference.py."""

    def __init__(self, repo_dir: str, ckpt_dir: str, forward_passes: int, z_range: tuple[float, float]):
        # The repo's code uses tf.compat.v1.layers, which only exists in legacy Keras (the tf_keras package).
        # setdefault leaves the variable alone if it is already set. It must be set before TensorFlow loads.
        os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # must be set before TensorFlow is imported
        # Turn the repository path text into a Path object.
        repo = Path(repo_dir)
        # The folder inside the repo that holds config_utils.py, data.py, contact_grasp_estimator.py, etc.
        inner = repo / "contact_graspnet"
        # Put that folder first on Python's module search path so `import config_utils` etc. finds it.
        sys.path.insert(0, str(inner))  # config_utils, data, contact_grasp_estimator, mesh_utils, ...
        # Also add the repo root (last), as inference.py does, for any imports that expect it.
        sys.path.append(str(repo))

        # Imported here (not at the top of the file) so the slow TensorFlow import happens only once the
        # environment variable above is set. "compat.v1" is the TensorFlow 1 style API the repo was written for.
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
        import mesh_utils  # builds the Panda gripper model used to draw the grasp wireframe
        from contact_grasp_estimator import GraspEstimator  # the Contact-GraspNet network and its inference code
        from data import load_available_input_data  # reads the .npz input (depth, K, rgb)

        # Keep references on the object so infer() can use them later.
        self.tf = tf
        self.load_available_input_data = load_available_input_data
        # Number of parallel forward passes; it fixes the network's batch size when the graph is built below.
        self.forward_passes = int(forward_passes)
        # Depth crop (minimum, maximum) in meters, stored as a plain list of floats.
        self.z_range = [float(z_range[0]), float(z_range[1])]

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
        # A Saver knows how to read the saved variable values from a checkpoint into the graph;
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

        # Gripper wireframe in the grasp frame: built exactly as visualization_utils.draw_grasps does.
        # Control points are key points on the Panda gripper's geometry. The result has shape (1, 6, 3)
        # (a batch of one), which squeeze() reduces to (6, 3).
        control = mesh_utils.create_gripper("panda").get_control_point_tensor(1, False, convex_hull=False).squeeze()
        # Point halfway between control points 1 and 2.
        midpoint = 0.5 * (control[1] + control[2])
        # The 7 wireframe points: gripper origin, that midpoint, then five points picked from the control points.
        base_lines = np.array([np.zeros(3), midpoint, control[1], control[3], control[1], control[2], control[4]])
        # Maximum opening of the gripper in meters, from the model's config (0.08 for this checkpoint).
        opening = float(config["DATA"]["gripper_width"])
        # Work on a copy so base_lines itself stays unchanged.
        points = base_lines.copy()
        # For points 2-6, set the x coordinate to +/- half the opening (keeping each point's original side),
        # which spreads the fingers to the full opening width.
        points[2:, 0] = np.sign(base_lines[2:, 0]) * opening / 2
        # Store as 32-bit floats, which is what the saved results and the client expect.
        self.gripper_points = points.astype(np.float32)

    def infer(self, input_path: str, output_path: str) -> dict:
        """Run the model on one saved frame (input .npz) and write the grasps to output_path (.npz)."""
        # Start the stopwatch for this request.
        started = time.time()
        # Read the input file; K=None means "use the intrinsics stored in the file". The function returns
        # (segmentation map, rgb image, depth image, camera matrix, point cloud, point colors); the entries
        # that are not in the file come back as None.
        segmap, rgb, depth, cam_K, pc_full, pc_colors = self.load_available_input_data(input_path, K=None)
        # No object segments: we use the whole scene.
        pc_segments = {}
        # Our input has depth rather than a ready-made point cloud, so convert depth to 3D points.
        if pc_full is None:
            # Back-project the depth image to a point cloud using the camera matrix, keeping only points
            # inside the depth range; also returns the matching colors from the rgb image.
            pc_full, pc_segments, pc_colors = self.estimator.extract_point_clouds(
                depth, cam_K, segmap=segmap, rgb=rgb, skip_border_objects=False, z_range=self.z_range
            )
        # Run the network. The three results are dictionaries keyed by segment id (-1 is the whole scene):
        # grasp poses (4x4 matrices), scores, and contact points. The fourth result (gripper openings) is unused.
        pred_grasps_cam, scores, contact_pts, _ = self.estimator.predict_scene_grasps(
            self.sess,  # the TensorFlow session holding the loaded weights
            pc_full,  # the scene point cloud
            pc_segments=pc_segments,  # empty: no per-object segments
            local_regions=False,  # do not crop around segments
            filter_grasps=False,  # do not filter grasps by segment
            forward_passes=self.forward_passes,  # number of parallel passes (more passes -> more grasp candidates)
        )
        # Write everything the client needs into the output file.
        save_result(
            output_path,
            # (N, 4, 4) grasp poses in camera coordinates, as 32-bit floats
            grasps=np.asarray(pred_grasps_cam[-1], dtype=np.float32).reshape(-1, 4, 4),
            # (N,) confidence score for each grasp
            scores=np.asarray(scores[-1], dtype=np.float32).reshape(-1),
            # (N, 3) point on the object surface where each grasp touches
            contact_pts=np.asarray(contact_pts[-1], dtype=np.float32).reshape(-1, 3),
            # (M, 3) the point cloud the model saw
            pc_full=np.asarray(pc_full, dtype=np.float32).reshape(-1, 3),
            # per-point colors for the point cloud (may be None)
            pc_colors=pc_colors,
            # the 7 wireframe points used to draw a gripper
            gripper_points=self.gripper_points,
        )
        # Number of grasps found (the length of the scores array).
        num = int(np.asarray(scores[-1]).size)
        # Report the count and how many seconds this request took, rounded to 2 decimals.
        return {"num_grasps": num, "seconds": round(time.time() - started, 2)}


def save_result(path, *, grasps, scores, contact_pts, pc_full, pc_colors, gripper_points) -> None:
    """Flat arrays only (no pickles), so Gungnir_Client can load results without allow_pickle."""
    # If there are no colors, store an empty (0, 3) array so the key always exists in the file.
    if pc_colors is None:
        pc_colors = np.zeros((0, 3), dtype=np.uint8)
    # Write all arrays into one uncompressed .npz archive; each keyword becomes a named array in the file.
    np.savez(
        path,
        grasps=grasps,
        scores=scores,
        contact_pts=contact_pts,
        pc_full=pc_full,
        pc_colors=np.asarray(pc_colors),
        gripper_points=np.asarray(gripper_points, dtype=np.float32),
        gripper_segments=GRIPPER_SEGMENTS,
    )


def main(argv=None) -> int:
    """Start-up and request loop. Returns an exit code (the process normally ends through os._exit)."""
    # Define the command-line options this process accepts.
    parser = argparse.ArgumentParser(description="ContactGraspnet_Server: persistent Contact-GraspNet inference process")
    # Where the Contact-GraspNet repository lives (default: a "contact_graspnet" folder next to this script).
    parser.add_argument("--repo-dir", default=str(ROOT / "contact_graspnet"), help="Contact-GraspNet repository root")
    # Which trained model to load; None means "use the default folder below".
    parser.add_argument("--ckpt-dir", default=None, help="checkpoint directory (default: <repo>/checkpoints/scene_test_2048_bs3_hor_sigma_0025)")
    # How many parallel forward passes to run per request (also the network's batch size).
    parser.add_argument("--forward-passes", type=int, default=3, help="fixed when the model is built; restart ContactGraspnet_Server to change it")
    # Closest depth, in meters, that is kept.
    parser.add_argument("--z-min", type=float, default=0.1)
    # Farthest depth, in meters, that is kept.
    parser.add_argument("--z-max", type=float, default=2.5)
    # Read the options from the command line (or from argv when this function is called from a test).
    args = parser.parse_args(argv)
    # Use the checkpoint folder from the command line, or fall back to the default model inside the repo.
    ckpt_dir = args.ckpt_dir or str(Path(args.repo_dir) / "checkpoints" / "scene_test_2048_bs3_hor_sigma_0025")

    # Keep the real stdout for protocol messages; anything else that writes to fd 1 goes to stderr.
    # "global" lets this function assign the module-level _proto variable.
    global _proto
    # os.dup(1) makes a copy of file descriptor 1 (standard output); fdopen wraps the copy in a text stream
    # with line buffering. This copy keeps pointing at the pipe to Gungnir_Client.
    _proto = os.fdopen(os.dup(1), "w", buffering=1)
    # Point descriptor 1 at descriptor 2 (standard error). From now on any stray print() or C++ output that
    # would have gone to stdout lands in stderr (the log) instead, so it cannot corrupt the protocol stream.
    os.dup2(2, 1)

    # Build the model. This is the slow part (importing TensorFlow, building the graph, loading weights).
    try:
        model = GraspModel(args.repo_dir, ckpt_dir, args.forward_passes, (args.z_min, args.z_max))
    except BaseException as exc:  # noqa: BLE001 - report anything, including SystemExit from imports
        # Print the full trace to stderr (which the client shows in its log panel).
        traceback.print_exc()
        # Tell the client that start-up failed and why.
        emit({"event": "fatal", "message": f"{type(exc).__name__}: {exc}"})
        # Exit immediately with code 2; os._exit skips cleanup that could hang after a failed start.
        os._exit(2)

    # The model is loaded: tell the client it may send requests now.
    emit({"event": "ready"})

    # Read requests from standard input, one line at a time, until the client closes the pipe (end of file).
    for line in sys.stdin:  # EOF means Gungnir_Client went away
        # Remove the newline and surrounding spaces.
        line = line.strip()
        # Ignore blank lines.
        if not line:
            continue
        # Each line should be one JSON message.
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            # Not valid JSON: skip it and wait for the next line.
            continue
        # Which command is this? (None if the message has no "cmd" key)
        command = request.get("cmd")
        # "quit": leave the loop and shut down.
        if command == "quit":
            break
        # Anything other than "infer" is an unknown command: ignore it.
        if command != "infer":
            continue
        # Handle one inference request.
        try:
            # Run the model on the input file and write the result file.
            info = model.infer(request["input"], request["output"])
            # Report success, echoing the request id and adding the grasp count and the time taken.
            emit({"event": "done", "id": request.get("id"), **info})
        except BaseException as exc:  # noqa: BLE001
            # Print the full trace to stderr for the log.
            traceback.print_exc()
            # Tell the client what went wrong.
            emit({"event": "error", "id": request.get("id"), "message": f"{type(exc).__name__}: {exc}"})
            # Exit right away: after a CUDA error the session is unusable, so the client will start a new client.
            os._exit(3)  # the CUDA context may be poisoned; let Gungnir_Client start a fresh client

    # Normal shutdown (a "quit" command or the pipe closing). os._exit skips TensorFlow's slow, sometimes
    # hanging cleanup.
    os._exit(0)  # skip TensorFlow's slow, sometimes-hanging teardown


# Run main() only when this file is started directly (not when it is imported, e.g. by a test), and pass
# its return value to sys.exit as the process exit code.
if __name__ == "__main__":
    sys.exit(main())