from L515 import L515

cam = L515()
K = cam.get_k()
print(f"Camera Intrinsic Matrix K = {K}")

cam.take_picture()

print(f"Depth Map: {cam.get_depth_map()}")
print(f"RGB Image: {cam.get_color()}")

cam.show_RGB_DepthMap()

for i in range(500):
    cam.show_picture()
