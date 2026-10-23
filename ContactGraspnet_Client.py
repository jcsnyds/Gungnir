#!/usr/bin/env python3
"""ContactGraspnet_Client: persistent Contact-GraspNet inference process (the only one that imports TensorFlow).

Started by Gungnir_Server.py. Loads the model once, then answers requests so each capture skips the
model build and checkpoint restore that `inference.py` repeats on every run.

Protocol: one JSON object per line.
    Gungnir_Server -> ContactGraspnet_Client (stdin):
        {"cmd": "infer", "id": ..., "input": "<npz>", "output": "<npz>"}
        {"cmd": "quit"}
    ContactGraspnet_Client -> Gungnir_Server (stdout):
        {"event": "ready"}
        {"event": "done", "id": ..., "num_grasps": N, "seconds": S}
        {"event": "error", "id": ..., "message": ...}   (client exits afterwards)
        {"event": "fatal", "message": ...}              (failed to start)
Everything else the process prints goes to stderr, so stdout carries only protocol messages.

After any inference error this client exits on purpose: a CUDA "illegal address" error is sticky, so
the session is unusable and Gungnir_Server starts a fresh client.

AI use: Written by Claude (Anthropic; Claude Sonnet 5.5 via claude.ai), October 2026. Originally
named grasp_worker.py. Its model setup follows contact_graspnet/inference.py. Claude
tested only the message-passing loop, against a fake model; the TensorFlow code was not run by Claude.
Reviewed and tested by: Ryan Poliquin
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

# Pairs of indices into the gripper wireframe points; same connections as visualization_utils.draw_grasps.
GRIPPER_SEGMENTS = np.array([[0, 1], [1, 2], [2, 3], [2, 4], [4, 5], [5, 6]], dtype=np.int32)

_proto = None  # the real stdout, reserved for protocol messages


def emit(message: dict) -> None:
    _proto.write(json.dumps(message) + "\n")
    _proto.flush()


class GraspModel:
    """Contact-GraspNet loaded once. Mirrors the setup in contact_graspnet/inference.py."""

    def __init__(self, repo_dir: str, ckpt_dir: str, forward_passes: int, z_range: tuple[float, float]):
        os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # must be set before TensorFlow is imported
        repo = Path(repo_dir)
        inner = repo / "contact_graspnet"
        sys.path.insert(0, str(inner))  # config_utils, data, contact_grasp_estimator, mesh_utils, ...
        sys.path.append(str(repo))

        import tensorflow.compat.v1 as tf

        tf.disable_eager_execution()
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if not gpus:
            raise RuntimeError("TensorFlow sees no GPU; the custom pointnet2 ops need one.")
        tf.config.experimental.set_memory_growth(gpus[0], True)

        import config_utils
        import mesh_utils
        from contact_grasp_estimator import GraspEstimator
        from data import load_available_input_data

        self.tf = tf
        self.load_available_input_data = load_available_input_data
        self.forward_passes = int(forward_passes)
        self.z_range = [float(z_range[0]), float(z_range[1])]

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
        session_config.allow_soft_placement = True
        self.sess = tf.Session(config=session_config)
        self.estimator.load_weights(self.sess, saver, ckpt_dir, mode="test")

        # Gripper wireframe in the grasp frame: built exactly as visualization_utils.draw_grasps does.
        control = mesh_utils.create_gripper("panda").get_control_point_tensor(1, False, convex_hull=False).squeeze()
        midpoint = 0.5 * (control[1] + control[2])
        base_lines = np.array([np.zeros(3), midpoint, control[1], control[3], control[1], control[2], control[4]])
        opening = float(config["DATA"]["gripper_width"])
        points = base_lines.copy()
        points[2:, 0] = np.sign(base_lines[2:, 0]) * opening / 2
        self.gripper_points = points.astype(np.float32)

    def infer(self, input_path: str, output_path: str) -> dict:
        started = time.time()
        segmap, rgb, depth, cam_K, pc_full, pc_colors = self.load_available_input_data(input_path, K=None)
        pc_segments = {}
        if pc_full is None:
            pc_full, pc_segments, pc_colors = self.estimator.extract_point_clouds(
                depth, cam_K, segmap=segmap, rgb=rgb, skip_border_objects=False, z_range=self.z_range
            )
        pred_grasps_cam, scores, contact_pts, _ = self.estimator.predict_scene_grasps(
            self.sess,
            pc_full,
            pc_segments=pc_segments,
            local_regions=False,
            filter_grasps=False,
            forward_passes=self.forward_passes,
        )
        save_result(
            output_path,
            grasps=np.asarray(pred_grasps_cam[-1], dtype=np.float32).reshape(-1, 4, 4),
            scores=np.asarray(scores[-1], dtype=np.float32).reshape(-1),
            contact_pts=np.asarray(contact_pts[-1], dtype=np.float32).reshape(-1, 3),
            pc_full=np.asarray(pc_full, dtype=np.float32).reshape(-1, 3),
            pc_colors=pc_colors,
            gripper_points=self.gripper_points,
        )
        num = int(np.asarray(scores[-1]).size)
        return {"num_grasps": num, "seconds": round(time.time() - started, 2)}


def save_result(path, *, grasps, scores, contact_pts, pc_full, pc_colors, gripper_points) -> None:
    """Flat arrays only (no pickles), so Gungnir_Server can load results without allow_pickle."""
    if pc_colors is None:
        pc_colors = np.zeros((0, 3), dtype=np.uint8)
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
    parser = argparse.ArgumentParser(description="ContactGraspnet_Client: persistent Contact-GraspNet inference process")
    parser.add_argument("--repo-dir", default=str(ROOT / "contact_graspnet"), help="Contact-GraspNet repository root")
    parser.add_argument("--ckpt-dir", default=None, help="checkpoint directory (default: <repo>/checkpoints/scene_test_2048_bs3_hor_sigma_0025)")
    parser.add_argument("--forward-passes", type=int, default=3, help="fixed when the model is built; restart ContactGraspnet_Client to change it")
    parser.add_argument("--z-min", type=float, default=0.1)
    parser.add_argument("--z-max", type=float, default=2.5)
    args = parser.parse_args(argv)
    ckpt_dir = args.ckpt_dir or str(Path(args.repo_dir) / "checkpoints" / "scene_test_2048_bs3_hor_sigma_0025")

    # Keep the real stdout for protocol messages; anything else that writes to fd 1 goes to stderr.
    global _proto
    _proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)

    try:
        model = GraspModel(args.repo_dir, ckpt_dir, args.forward_passes, (args.z_min, args.z_max))
    except BaseException as exc:  # noqa: BLE001 - report anything, including SystemExit from imports
        traceback.print_exc()
        emit({"event": "fatal", "message": f"{type(exc).__name__}: {exc}"})
        os._exit(2)

    emit({"event": "ready"})

    for line in sys.stdin:  # EOF means Gungnir_Server went away
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        command = request.get("cmd")
        if command == "quit":
            break
        if command != "infer":
            continue
        try:
            info = model.infer(request["input"], request["output"])
            emit({"event": "done", "id": request.get("id"), **info})
        except BaseException as exc:  # noqa: BLE001
            traceback.print_exc()
            emit({"event": "error", "id": request.get("id"), "message": f"{type(exc).__name__}: {exc}"})
            os._exit(3)  # the CUDA context may be poisoned; let Gungnir_Server start a fresh client

    os._exit(0)  # skip TensorFlow's slow, sometimes-hanging teardown


if __name__ == "__main__":
    sys.exit(main())
