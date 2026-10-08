"""
Wrapper Class for Intel RealSense L515 Depth Camera

Requirements:
- Python Version 3.9 - 3.11
- pip install pyrealsense2==2.54.2.5684

NOTE: L515 support was removed from librealsense after v2.54.2, so install
a matching pyrealsense2 (e.g. pip install pyrealsense2==2.54.2.5684) and use
a Python version that release has wheels for.

Sources:
https://github.com/realsenseai/librealsense/blob/master/wrappers/python/

AI Statement:
Used Gemini to provide code to get Intrinsic Matrix K
Used Claude to resolve problem where old version of pyrealsense2 was only available on older versions of python.

Written by Ryan Poliquin
Debugged using Claude AI
10/5/2026
"""
from Gungnir.Archive import pyrealsense2 as rs
import matplotlib
matplotlib.use('QtAgg') # Or 'TkAgg' if you don't have PyQt installed
from matplotlib import pyplot as plt
import numpy as np
import cv2



class L515:
    def __init__(self, clipping_distance=9):
        # Set these first so close()/__del__ are safe even if init fails part-way
        self.pipeline = None
        self.running = False
        self.depth_image = None
        self.color_image = None
        self.clipping_distance = clipping_distance

        # Fail early, with a useful message, if no device is enumerated
        devices = rs.context().query_devices()
        if len(devices) == 0:
            raise RuntimeError(
                "No RealSense device found. Check the USB 3 connection and that "
                "your pyrealsense2 version still supports the L515 (<= 2.54.2)."
            )
        for dev in devices:
            print("Found device:", dev.get_info(rs.camera_info.name))

        # Create a context object. This object owns the handles to all connected realsense devices
        self.pipeline = rs.pipeline()

        # Create a config and configure the pipeline to stream different resolutions of color and depth streams
        self.config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        self.device = pipeline_profile.get_device()

        # Check to make sure an RGB Camera is Available
        found_rgb = False
        for s in self.device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("No RGB Camera found")
            exit(0)

        self.running = True

        # Configure the size of data received from the depth and color streams
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.profile = self.pipeline.start(self.config)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: ", self.depth_scale)

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        # Let auto-exposure settle
        warmup_frames = 5
        for _ in range(warmup_frames):
            frames = self.pipeline.wait_for_frames()

        # Define variables to store last image in
        self.depth_image = None
        self.color_image = None

    def take_picture(self):
        # Get frameset of color and depth
        frames = self.pipeline.wait_for_frames()

        # Align the depth frame to color frame
        frames = self.align.process(frames)

        # Extract Depth and Color Frames
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Validate that both frames are valid
        if not depth_frame or not color_frame:
            print("No depth and color frames found")

        # Save images and apply depth scale
        self.depth_image = np.asanyarray(depth_frame.get_data()) * self.depth_scale
        self.color_image = np.asanyarray(color_frame.get_data())
        return True

    def get_depth(self):
        return self.depth_image

    def get_color(self):
        return self.color_image

    def get_k(self):
        depth_profile = self.profile.get_stream(rs.stream.depth)
        depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()

        # K Matrix components
        fx = depth_intrinsics.fx
        fy = depth_intrinsics.fy
        cx = depth_intrinsics.ppx
        cy = depth_intrinsics.ppy

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        return K

    def show_RGB_DepthMap(self):
        # Flatten and format data for  plotting
        x_indices = np.arange(self.depth_image.shape[1])  # 480 columns
        y_indices = np.arange(self.depth_image.shape[0])  # 640 rows
        X, Y = np.meshgrid(x_indices, y_indices)
        x_plot = X.ravel()
        y_plot = Y.ravel()
        z_plot = self.depth_image.ravel()
        c_plot = self.color_image.reshape(-1, 3)/255.0

        downsample_rate = 25
        x_plot = x_plot[::downsample_rate]
        y_plot = y_plot[::downsample_rate]
        z_plot = z_plot[::downsample_rate]
        c_plot = c_plot[::downsample_rate]

        # initialize the plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(projection='3d')

        # Render the scatter plot
        sc = ax.scatter(
            x_plot,
            y_plot,
            z_plot,
            c=c_plot,  # Uses your separate color array mapped to color
            s=5,  # Size of individual scatter points
            alpha=0.8  # Opacity control for visual clarity
        )

        # Configure plot labels
        ax.set_xlabel('X Axis (Columns)')
        ax.set_ylabel('Y Axis (Rows)')
        ax.set_zlabel('Z Axis (Height)')
        ax.set_title('3D Scatter Plot from 2D NumPy Arrays')

        # Turns on mouse rotation and zoom trackpad support
        plt.ion()
        plt.show(block=True)  # Keeps the window open so you can interact with it

    def show_picture(self):
        self.take_picture()

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack(
            (self.depth_image, self.depth_image, self.depth_image))  # depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > self.clipping_distance) | (depth_image_3d <= 0), grey_color, self.color_image)

        # Render images: depth align to color on left depth on right
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=100), cv2.COLORMAP_JET)
        images = np.hstack((bg_removed, depth_colormap))

        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', images)
        cv2.waitKey(1)  # required for the window to draw; 0 = wait for keypress

    def __del__(self):
        if self.pipeline is not None and self.running:
            self.pipeline.stop()
            self.running = False
            cv2.destroyAllWindows()
