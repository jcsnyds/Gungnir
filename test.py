import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Load model
checkpoint = "sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

device = "cuda" if torch.cuda.is_available() else "cpu"

sam2 = build_sam2(model_cfg, checkpoint, device=device)

# Automatic mask generator
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=32,
    pred_iou_thresh=0.7,
    stability_score_thresh=0.92,
    min_mask_region_area=100
)

# Load image
rgb = np.load("Sample Images/ColorMap1.npy")
image = cv2.imread("output/runs/20261023-114913/overlay.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Generate masks
masks = mask_generator.generate(rgb)

print(masks)

print(f"Generated {len(masks)} masks")

# Sort masks by area (largest first)
masks = sorted(masks, key=lambda x: x["area"], reverse=True)

# Create instance ID map
instance_map = np.zeros(rgb.shape[:2], dtype=np.uint16)

for object_id, mask in enumerate(masks, start=1):
    segmentation = mask["segmentation"]

    # Only assign pixels that haven't already been assigned
    instance_map[(segmentation) & (instance_map == 0)] = object_id

# Generate colors for each object ID
num_objects = len(masks)

colors = np.random.default_rng(42).integers(
    0, 256, size=(num_objects + 1, 3), dtype=np.uint8
)

colors[0] = [0, 0, 0]  # Background = black

# Convert instance map to colored image
colored_mask = colors[instance_map]

# Display
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(image_rgb)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(instance_map, cmap="nipy_spectral", interpolation="nearest")
axes[1].set_title("Instance ID Map")
axes[1].axis("off")

axes[2].imshow(colored_mask)
axes[2].set_title("Colored Instance Segmentation")
axes[2].axis("off")

plt.tight_layout()
plt.show()

# Print object information
print("Number of objects:", num_objects)
print("Instance map shape:", instance_map.shape)
print("Instance map data type:", instance_map.dtype)
print("Object IDs:", np.unique(instance_map))