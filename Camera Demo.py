from L515 import L515

# Create a L515 object
camera = L515()

# Get the Intrinsic K matrix for the camera
K = camera.get_aligned_k()
print(K)

# Take an image and store it inside the L515 Object
camera.take_picture()

# There are 2 ways to get the RGB image and Depth map
# Method 1: Get Functions
color_image = camera.get_color()
print(color_image)
depth_map = camera.get_depth()
print(depth_map)

# Method 2: Object Attributes
color_image = camera.color_image
print(color_image)
depth_map = camera.depth_image
print(depth_map)

# Image size
print(f"Depth Map Shape:{depth_map.shape}")
print(f"RGB Image Shape:{color_image.shape}")

# Take a new picture and show it
camera.show_picture()

# Plot the depth map
camera.show_RGB_DepthMap()

# Show camera stream
for i in range(1000):
    camera.show_picture()

