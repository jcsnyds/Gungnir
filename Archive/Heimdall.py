import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

model_type = "DPT_Large" 
midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform if model_type == "DPT_Large" else midas_transforms.small_transform

img = cv2.imread("Sample Images/mug-1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_batch = transform(img).to(device)

print("Estimating depth...")
with torch.no_grad():
    prediction = midas(input_batch)
    
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

depth_map = prediction.cpu().numpy()

depth_min = depth_map.min()
depth_max = depth_map.max()
print(f"Max depth: {depth_max}, Min depth: {depth_min}")
depth_normalized = ((255 * (depth_map - depth_min) / (depth_max - depth_min))).astype("uint8")


rgbd_image = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
rgbd_image[:, :, 3] = depth_normalized
cv2.imwrite("output_rgbd.png", cv2.cvtColor(rgbd_image, cv2.COLOR_RGBA2BGRA))


rgbd_array = np.dstack((img, depth_map))
np.save("output_rgbd.npy", rgbd_array)

print("RGB-D conversion complete! Files saved.")

print("Opening plot window...")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(img)
axes[0].set_title("Original Image")
axes[0].axis("off")

depth_plot = axes[1].imshow(depth_map, cmap="inferno")
axes[1].set_title("MiDaS Depth Map")
axes[1].axis("off")

fig.colorbar(depth_plot, ax=axes[1], shrink=0.75, label="Relative Depth")

plt.tight_layout()
plt.show()