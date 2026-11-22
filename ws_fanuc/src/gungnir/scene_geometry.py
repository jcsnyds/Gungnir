#!/usr/bin/env python3
"""Camera-view geometry shared by scripts/make_world.py and view_pose.py (no ROS needed).

World frame: origin = robot base, table top at z = 0, +X away from the base.
Camera (optical) frame: x right, y down, z forward = the viewing direction."""
import math

import numpy as np

HFOV, IMG_W, IMG_H = 1.2217, 640, 480      # must match the sensor in config/gungnir_backends.xacro
FX = (IMG_W / 2) / math.tan(HFOV / 2)      # fx = fy (square pixels)
CX, CY = IMG_W / 2, IMG_H / 2


def look_at(position, target):
    """3x3 rotation whose columns (optical x right, y down, z forward) are expressed in the world frame, for a camera
    at `position` looking at `target`, with the image horizontal parallel to the table (no roll)."""
    f = np.asarray(target, float) - np.asarray(position, float)
    f = f / np.linalg.norm(f)
    right = np.cross(f, [0.0, 0.0, 1.0])
    if np.linalg.norm(right) < 1e-6:                  # looking straight down: image up = world +X
        right = np.cross(f, [1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    return np.column_stack((right, np.cross(f, right), f))


def view_from(center_xy, dist, pitch_deg, heading_deg, target_z=0.05):
    """Camera that looks at the table point (center_xy, target_z) from `dist` metres away, its optical axis pointing
    `pitch_deg` below the horizon (90 = straight down) towards world heading `heading_deg` (0 = +X).
    -> (position, rotation)"""
    p, h = math.radians(pitch_deg), math.radians(heading_deg)
    f = np.array([math.cos(p) * math.cos(h), math.cos(p) * math.sin(h), -math.sin(p)])
    target = np.array([center_xy[0], center_xy[1], target_z], float)
    position = target - dist * f
    return position, look_at(position, target)


def project(position, R, point):
    """(u, v, planar depth) of a world point in the camera image, or None if it is behind the camera.
    Gazebo's depth image holds planar depth (distance along the optical axis), which is what this returns."""
    p = R.T @ (np.asarray(point, float) - np.asarray(position, float))
    if p[2] <= 1e-6:
        return None
    return FX * p[0] / p[2] + CX, FX * p[1] / p[2] + CY, float(p[2])


def table_depth_at_centre(position, R):
    """Depth the table should have at the image centre (where the optical axis meets z = 0), or None."""
    fz = R[2, 2]
    return None if fz >= -1e-6 else float(position[2] / -fz)
