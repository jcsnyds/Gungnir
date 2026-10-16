import subprocess
import numpy as np
from pyevtk.hl import polyLinesToVTK, pointsToVTK
from l515 import L515

#=================================================================
# Constants
#=================================================================

MIN_DEPTH = 0.1
MAX_DEPTH = 2.5

gripper_depth = 0.06
gripper_width = 0.08
jaw_depth = 0.04

#================================================================
# Capture image and save point cloud
#================================================================

#Capture image and get Depth Map
cam = L515()
cam.capture_image()
depth = cam.get_depth()
h, w = depth.shape

# Define Camera Intrinsics Matrix K
#     [fx,  0.0, cx],
#     [0.0, fy,  cy],
#     [0.0, 0.0, 1.0]
cam_K = cam.get_k()
fx = cam_K[0, 0]
fy = cam_K[1, 1]
cx = cam_K[0, 2]
cy = cam_K[1, 2]

# Project 2D Depth map to 3D Point Cloud (X, Y, Z) in meters
u, v = np.meshgrid(np.arange(w), np.arange(h))

# Filter valid depth points (in meters)
valid = (depth > MIN_DEPTH) & (depth < MAX_DEPTH)

z = depth[valid]
x = (u[valid] - cx) * z / fx
y = (v[valid] - cy) * z / fy

point_cloud = np.stack((x, y, z), axis=-1).astype(np.float32)
print(f"Generated 3D Point Cloud shape: {point_cloud.shape}")

# 4. Save as .npz with all keys expected by data.py (prevents UnboundLocalError)
output_path = "input_pc.npz"
np.savez(
    output_path,
    xyz=point_cloud,      # Core point cloud key
    pc_full=point_cloud,  # Fallback point cloud key
    K=cam_K,              # Standard camera matrix key
)

print(f"Successfully saved to {output_path}")

#================================================================
# Contact Graspnet Inference
#================================================================
subprocess.run(['python', 'contact_graspnet/contact_graspnet/inference.py', '--np_path=../input_pc.npz',
               '--ckpt_dir=checkpoints/scene_test_2048_bs3_hor_sigma_0025', '--forward_passes=5',
                f'--z_range=[{MIN_DEPTH},{MAX_DEPTH}]', '--no_visualize'])

#================================================================
# Process Model Predictions
#================================================================

# 1. Load your compressed arrays
data = np.load("contact_graspnet/results/predictions_input_pc.npz", allow_pickle=True)
print("Your actual keys are:", data.files)

contact_pts = np.array(data['contact_pts'].item()[-1], dtype=np.float32)
scores = np.array(data['scores'].item()[-1], dtype=np.float32)
pred_grasps_cam = np.array(data['pred_grasps_cam'].item()[-1], dtype=np.float32)

num_grasps = len(scores)
print(f"Processing {num_grasps} grasps to match standard Contact-GraspNet layout...")

# 2. Define standard parallel-jaw gripper dimensions (in meters)
# Local coordinates of the 5 points forming the classic parallel-jaw gripper frame shape
local_nodes = np.array([
    [0.0, 0.0, 0.0],  # 0: Base
    [-gripper_width / 2, 0.0, gripper_depth],  # 1: Left corner
    [-gripper_width / 2, 0.0, gripper_depth + jaw_depth],  # 2: Left jaw tip
    [gripper_width / 2, 0.0, gripper_depth],  # 3: Right corner
    [gripper_width / 2, 0.0, gripper_depth + jaw_depth]  # 4: Right jaw tip
], dtype=np.float32)

# 3. Transform local gripper frames into the 3D Camera Scene
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

# 4. Separate X, Y, Z for pyevtk formatting
lx = np.ascontiguousarray(all_points[:, 0])
ly = np.ascontiguousarray(all_points[:, 1])
lz = np.ascontiguousarray(all_points[:, 2])

# 5. Export the explicit gripper wireframes to VTK PolyData
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
