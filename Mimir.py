import numpy as np
import subprocess
from pyevtk.hl import polyLinesToVTK, pointsToVTK

# 1. Load your compressed arrays
data = np.load("contact_graspnet/results/predictions_input_pc.npz", allow_pickle=True)
print("Your actual keys are:", data.files)

contact_pts = np.array(data['contact_pts'].item()[-1], dtype=np.float32)
scores = np.array(data['scores'].item()[-1], dtype=np.float32)
pred_grasps_cam = np.array(data['pred_grasps_cam'].item()[-1], dtype=np.float32)

num_grasps = len(scores)
print(f"Processing {num_grasps} grasps to match standard Contact-GraspNet layout...")

# 2. Define standard parallel-jaw gripper dimensions (in meters)
gripper_depth = 0.06
gripper_width = 0.08
jaw_depth = 0.04

# Local coordinates of the 5 points forming the classic parallel-jaw gripper frame shape
local_nodes = np.array([
    [0.0, 0.0, 0.0],                                  # 0: Base
    [-gripper_width/2, 0.0, gripper_depth],            # 1: Left corner
    [-gripper_width/2, 0.0, gripper_depth + jaw_depth],  # 2: Left jaw tip
    [gripper_width/2, 0.0, gripper_depth],             # 3: Right corner
    [gripper_width/2, 0.0, gripper_depth + jaw_depth]   # 4: Right jaw tip
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

pointsToVTK("./grasp_points", np.ascontiguousarray(contact_pts[:, 0]),
                              np.ascontiguousarray(contact_pts[:, 1]),
                              np.ascontiguousarray(contact_pts[:, 2]),
            data={"Grasp_Scores": scores})

print("\nSuccess! Generated 'contact_graspnet_grippers.vtp'")