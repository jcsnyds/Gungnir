import cv2
import torch
import numpy as np

# 1. Load the MiDaS model
# Use "MiDaS_small" instead of "DPT_Large" if you need faster, CPU-only processing
model_type = "DPT_Large" 
midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# 2. Load the transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform if model_type == "DPT_Large" else midas_transforms.small_transform

# 3. Load and format your JPG image
# Replace "your_image.jpg" with the path to your file
img = cv2.imread("jcsnyds/Gungnir/mug-1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_batch = transform(img).to(device)

# 4. Predict the depth map
print("Estimating depth...")
with torch.no_grad():
    prediction = midas(input_batch)
    
    # Resize the predicted depth map to match the original image resolution
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

depth_map = prediction.cpu().numpy()

# 5. Normalize depth map to 0-255 for image formatting
depth_min = depth_map.min()
depth_max = depth_map.max()
depth_normalized = (255 * (depth_map - depth_min) / (depth_max - depth_min)).astype("uint8")

# 6. Create and save the RGB-D outputs

# Option A: Save as a 4-channel PNG (RGB + Depth mapped to the Alpha channel)
rgbd_image = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
rgbd_image[:, :, 3] = depth_normalized
cv2.imwrite("output_rgbd.png", cv2.cvtColor(rgbd_image, cv2.COLOR_RGBA2BGRA))

# Option B: Save as a raw NumPy array (keeps precise, uncompressed float values)
rgbd_array = np.dstack((img, depth_map))
np.save("output_rgbd.npy", rgbd_array)

print("RGB-D conversion complete! Files saved.")
