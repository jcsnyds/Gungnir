import numpy as np
from l515 import L515

# 1. Capture image and get Depth Map
cam = L515()
cam.capture_image()
depth = cam.get_depth()
h, w = depth.shape

# 2. Define Camera Intrinsics Matrix K
#     [fx,  0.0, cx],
#     [0.0, fy,  cy],
#     [0.0, 0.0, 1.0]
cam_K = cam.get_k()
fx = cam_K[0, 0]
fy = cam_K[1, 1]
cx = cam_K[0, 2]
cy = cam_K[1, 2]

# 3. Project 2D Depth map to 3D Point Cloud (X, Y, Z) in meters
u, v = np.meshgrid(np.arange(w), np.arange(h))

# Filter valid depth points (in meters)
valid = (depth > 0.1) & (depth < 2.5)

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