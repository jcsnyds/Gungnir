from l515 import L515
import numpy as np
import matplotlib.pyplot as plt

cam = L515()
cam.capture_image()

np.save("Sample Images/DepthMap.npy", cam.get_depth())
plt.imsave("Sample Images/DepthMap.png", cam.get_depth())

np.save("Sample Images/ColorMap.npy", cam.get_color())
plt.imsave("Sample Images/ColorMap.png", cam.get_color())

np.save("Sample Images/Pointcloud.npy", cam.get_pointcloud(0.1, 2.5))
plt.imsave("Sample Images/Pointcloud.png", cam.get_pointcloud(0.1, 2.5))

np.save("Sample Images/RGBD.npy", cam.get_RGBD())
