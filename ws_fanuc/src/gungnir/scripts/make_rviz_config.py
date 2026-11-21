#!/usr/bin/env python3
"""Create config/gungnir_camera.rviz from config/moveit.rviz by adding the camera and world displays.
Run again after regenerating moveit.rviz. The Grid is disabled (it would z-fight with the table top)."""
import sys
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parent.parent / "config"
SRC, DST = CONFIG / "moveit.rviz", CONFIG / "gungnir_camera.rviz"


def topic(value):
    return {"Depth": 5, "Durability Policy": "Volatile", "History Policy": "Keep Last",
            "Reliability Policy": "Reliable", "Value": value}


def image(name, value):
    return {"Class": "rviz_default_plugins/Image", "Enabled": True, "Name": name, "Normalize Range": True,
            "Max Value": 1, "Min Value": 0, "Median window": 5, "Topic": topic(value), "Value": True}


NEW = [
    image("Camera colour", "/camera/color/image_raw"),
    image("Camera depth (colour-coded)", "/camera/depth/image_colorized"),
    {"Class": "rviz_default_plugins/PointCloud2", "Enabled": True, "Name": "Camera cloud",
     "Alpha": 1, "Color Transformer": "RGB8", "Position Transformer": "XYZ", "Style": "Flat Squares",
     "Size (m)": 0.003, "Size (Pixels)": 2, "Use Fixed Frame": True, "Use rainbow": True,
     "Decay Time": 0, "Selectable": True, "Topic": topic("/camera/depth/color/points"), "Value": True},
    {"Class": "rviz_default_plugins/MarkerArray", "Enabled": True, "Name": "Sim world objects",
     "Namespaces": {}, "Topic": topic("/sim_world_markers"), "Value": True},
]

if not SRC.exists():
    sys.exit(f"{SRC} not found")
cfg = yaml.safe_load(SRC.read_text())
displays = cfg["Visualization Manager"]["Displays"]
for d in displays:
    if str(d.get("Class", "")).endswith("/Grid"):
        d["Enabled"] = False
for d in displays:
    if d.get("Class") == "moveit_rviz_plugin/MotionPlanning":
        d.setdefault("Planning Request", {})["Planning Group"] = "manipulator"
have = {d.get("Name") for d in displays}
displays.extend(d for d in NEW if d["Name"] not in have)
DST.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
print(f"wrote {DST} ({len(displays)} displays)")
