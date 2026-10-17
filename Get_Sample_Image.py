from l515 import L515
import numpy as np
import matplotlib.pyplot as plt
import cv2

cam = L515()
cam.capture_image()

np.save("Sample Images/DepthMap1.npy", cam.get_depth())
plt.imsave("Sample Images/DepthMap1.png", cam.get_depth())

np.save("Sample Images/ColorMap1.npy", cam.get_color())
plt.imsave("Sample Images/ColorMap1.png", cam.get_color())

np.save("Sample Images/Pointcloud1.npy", cam.get_pointcloud(0.1, 2.5))
plt.imsave("Sample Images/Pointcloud1.png", cam.get_pointcloud(0.1, 2.5))

np.save("Sample Images/RGBD1.npy", cam.get_RGBD())
