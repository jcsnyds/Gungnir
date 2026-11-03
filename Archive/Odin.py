# AI use: The Contact-GraspNet subprocess block (check=True, CalledProcessError handling, the
# TF_USE_LEGACY_KERAS and CUDA_LAUNCH_BLOCKING environment settings, and deleting stale results
# before inference) was written or suggested by Claude (Anthropic; Claude Sonnet 5.5 via
# claude.ai), October 2026. The rest was written by the author.

import subprocess
import os
import sys
import numpy as np
from pyevtk.hl import polyLinesToVTK, pointsToVTK
from l515 import L515
import tensorflow as tf

#=================================================================
# Constants
#=================================================================

USE_CAMERA = False

MIN_DEPTH = 0.1
MAX_DEPTH = 2.5

INFERENCE_PASSES = 3

gripper_depth = 0.06
gripper_width = 0.08
jaw_depth = 0.04

output_path = "input_pc.npz"
results_file = "output/model_predictions.npz"

#================================================================
# Preprocessing
# Capture image and save point cloud
#================================================================

#Capture image and get Depth Map
if USE_CAMERA:
    print('\nCapturing Image\n')
    cam = L515()
    cam.capture_image()
    depth = cam.get_depth()
    rgb = cam.get_color()
    cam_K = cam.get_k()
    point_cloud = cam.get_pointcloud(MIN_DEPTH, MAX_DEPTH)

    # Save as .npz with all keys expected by data.py (prevents UnboundLocalError)
    np.savez(
        output_path,
        depth = depth,        # Depth map from camera
        K=cam_K,              # Standard camera matrix key
        rgb=rgb,              # normal rgb image
    )

    print(f"Successfully saved to {output_path}")
else:
    print(f"Using last file saved to {output_path}")

#================================================================
# Model Inference
# Run Contact Graspnet
#================================================================
if os.path.exists(results_file):
    os.remove(results_file)   # never post-process a result left over from an earlier run

try:
    print("\nRunning Contact-GraspNet Inference...")

    tf_lib = os.path.dirname(tf.__file__)
    env = {
        **os.environ,
        "TF_USE_LEGACY_KERAS": "1",
        "CUDA_LAUNCH_BLOCKING": "1",
        "LD_LIBRARY_PATH": (
            f"{tf_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        ),
    }

    subprocess.run(
        [
            "python",
            "contact_graspnet/contact_graspnet/inference.py",
            f"--np_path={output_path}",
            "--ckpt_dir=contact_graspnet/checkpoints/scene_test_2048_bs3_hor_sigma_0025",
            f"--forward_passes={INFERENCE_PASSES}",
            f"--z_range=[{MIN_DEPTH},{MAX_DEPTH}]",
        ],
        check=True,
        env=env,
    )

    print("Successfully ran Contact-GraspNet Inference and saved output")
except subprocess.CalledProcessError as e:
    print(f"\nERROR: inference exited with code {e.returncode}")
    sys.exit(1)

#================================================================
# Post Processing
# Process Model Predictions and Save Outputs
#================================================================

# Load your compressed arrays
data = np.load(f"{results_file}", allow_pickle=True)
print("Your actual keys are:", data.files)

contact_pts = np.array(data['contact_pts'].item()[-1], dtype=np.float32)
scores = np.array(data['scores'].item()[-1], dtype=np.float32)
pred_grasps_cam = np.array(data['pred_grasps_cam'].item()[-1], dtype=np.float32)

num_grasps = len(scores)
print(f"Processing {num_grasps} grasps to match standard Contact-GraspNet layout...")

# Define standard parallel-jaw gripper dimensions (in meters)
# Local coordinates of the 5 points forming the classic parallel-jaw gripper frame shape
local_nodes = np.array([
    [0.0, 0.0, 0.0],  # 0: Base
    [-gripper_width / 2, 0.0, gripper_depth],  # 1: Left corner
    [-gripper_width / 2, 0.0, gripper_depth + jaw_depth],  # 2: Left jaw tip
    [gripper_width / 2, 0.0, gripper_depth],  # 3: Right corner
    [gripper_width / 2, 0.0, gripper_depth + jaw_depth]  # 4: Right jaw tip
], dtype=np.float32)

# Transform local gripper frames into the 3D Camera Scene
all_points = []
points_per_line = []
gripper_scores = []

for i in range(num_grasps):
    T = pred_grasps_cam[i]  # 4x4 Homogeneous transformation matrix
    R = T[:3, :3]
    t = T[:3, 3]

    # Global = R * Local + t
    global_nodes = (R @ local_nodes.T).T + t

    # Trace order: Base -> Left Corner -> Left Tip -> Left Corner -> Right Corner -> Right Tip
    line_sequence = [0, 1, 2, 1, 3, 4]
    for idx in line_sequence:
        all_points.append(global_nodes[idx])

    points_per_line.append(6)  # 6 coordinates trace out 1 gripper wireframe
    gripper_scores.append(scores[i])

all_points = np.array(all_points, dtype=np.float32)
points_per_line = np.array(points_per_line, dtype=np.int32)
gripper_scores = np.array(gripper_scores, dtype=np.float32)

# Separate X, Y, Z for pyevtk formatting
lx = np.ascontiguousarray(all_points[:, 0])
ly = np.ascontiguousarray(all_points[:, 1])
lz = np.ascontiguousarray(all_points[:, 2])

# Export the explicit gripper wireframes to VTK PolyData
# FIXED: Using cellData instead of lineData to attach variables to lines
polyLinesToVTK("./contact_graspnet_grippers", lx, ly, lz,
               pointsPerLine=points_per_line,
               cellData={"Grasp_Scores": gripper_scores})

# Export baseline contact point cloud for alignment checking
pointsToVTK("./grasp_points", np.ascontiguousarray(contact_pts[:, 0]),
            np.ascontiguousarray(contact_pts[:, 1]),
            np.ascontiguousarray(contact_pts[:, 2]),
            data={"Grasp_Scores": scores})


print("\nSuccess! Generated 'contact_graspnet_grippers.vtp'")
