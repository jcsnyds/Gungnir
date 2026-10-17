from l515 import L515
import numpy as np


cam = L515()
cam.capture_image()

np.save("Sample Images/DepthMap.npy", cam.get_depth())
np.save("Sample Images/DepthMap.png", cam.get_depth())
np.save("Sample Images/ColorMap.npy", cam.get_color())
np.save("Sample Images/ColorMap.png", cam.get_color())
np.save("Sample Images/Pointcloud.npy", cam.get_pointcloud(0.1, 2.5))
np.save("Sample Images/Pointcloud.png", cam.get_pointcloud(0.1, 2.5))
np.save("Sample Images/RGB.npy", cam.get_RGBD())
np.save("Sample Images/RGB.png", cam.get_RGBD())
