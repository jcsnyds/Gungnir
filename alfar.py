import numpy as np
from l515 import L515

# 1. Load the 4-channel RGB-D array (5712, 4284, 4)
rgbd_path = "output_rgbd.npy"
print(f"Loading {rgbd_path}...")
rgbd = np.load(rgbd_path)

# Extract Depth (Channel 3)
depth = rgbd[:, :, 3]/40

# 2. Downsample to avoid GPU memory overflow (stride=8 -> ~714x535)
stride = 8
depth_sub = depth[::stride, ::stride]
h, w = depth_sub.shape

# 3. Define Camera Intrinsics Matrix K
# Update fx, fy, cx, cy with your camera's actual parameters if available
fx = 3000.0 / stride
fy = 3000.0 / stride
cx = w / 2.0
cy = h / 2.0

cam_K = np.array([
    [fx,  0.0, cx],
    [0.0, fy,  cy],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

# 4. Project 2D Depth map to 3D Point Cloud (X, Y, Z) in meters
u, v = np.meshgrid(np.arange(w), np.arange(h))

# Filter valid depth points (in meters)
valid = (depth_sub > 0.1) & (depth_sub < 2.5)

z = depth_sub[valid]
x = (u[valid] - cx) * z / fx
y = (v[valid] - cy) * z / fy

point_cloud = np.stack((x, y, z), axis=-1).astype(np.float32)
print(f"Generated 3D Point Cloud shape: {point_cloud.shape}")

# 5. Save as .npz with all keys expected by data.py (prevents UnboundLocalError)
output_path = "input_pc.npz"
np.savez(
    output_path,
    xyz=point_cloud,      # Core point cloud key
    pc_full=point_cloud,  # Fallback point cloud key
    K=cam_K,              # Standard camera matrix key
)

print(f"Successfully saved to {output_path}")