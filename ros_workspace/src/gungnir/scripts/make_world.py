#!/usr/bin/env python3
"""Generate the Gazebo world (table + random graspable objects) and a ground-truth file.

  make_world.py [--seed 1] [--count 5] [--center-x 1.0] [--center-y 0] [--pitch 45] [--view-dist 0.45] [--heading 0]

The robot's home pose points the camera at the middle of the object cluster (center-x, center-y, target-z) from
--view-dist metres, looking --pitch degrees below the horizon (90 = straight down) along world heading --heading
(0 = +X, away from the base). Objects (at most 0.12 m tall) are placed around the centre so that the top of every
object is visible from that camera: inside the image, 0.30-1.0 m away, and not hidden behind another object.
--spread-x / --spread-y: half-size of the placement area along / across the viewing direction.
World frame: origin = robot base, table top at z = 0 (the same plane as the base).
Writes worlds/gungnir_world.sdf and worlds/gungnir_world_truth.yaml (camera pose, expected pixels and depths)."""
import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scene_geometry import IMG_H, IMG_W, project, table_depth_at_centre, view_from  # noqa: E402

WORLD_DIR = Path(__file__).resolve().parent.parent / "worlds"
KINDS = ["box", "block", "can", "ball", "bottle"]
PALETTE = {"red": (0.80, 0.15, 0.15), "blue": (0.15, 0.30, 0.80), "green": (0.15, 0.65, 0.25),
           "yellow": (0.90, 0.80, 0.15), "orange": (0.90, 0.45, 0.10), "purple": (0.55, 0.25, 0.70),
           "white": (0.90, 0.90, 0.90), "cyan": (0.10, 0.70, 0.75)}
MARGIN_PX = 30                  # object tops must be this far inside the image border
MIN_DEPTH, MAX_DEPTH = 0.30, 1.00   # allowed camera-to-object distance (m); the sensor's near clip is 0.25

HEADER = """<?xml version="1.0"?>
<sdf version="1.8">
  <world name="gungnir_world">
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="ignition-gazebo-physics-system" name="ignition::gazebo::systems::Physics"/>
    <plugin filename="ignition-gazebo-user-commands-system" name="ignition::gazebo::systems::UserCommands"/>
    <plugin filename="ignition-gazebo-scene-broadcaster-system" name="ignition::gazebo::systems::SceneBroadcaster"/>
    <plugin filename="ignition-gazebo-sensors-system" name="ignition::gazebo::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <gravity>0 0 -9.81</gravity>
    <light type="directional" name="sun">
      <cast_shadows>false</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <!-- Table: top surface at z = 0 (the same plane as the robot base). Large, so a tilted camera never sees past it -->
    <model name="table"><static>true</static><pose>@TX@ 0 -0.05 0 0 0</pose>
      <link name="link">
        <collision name="c"><geometry><box><size>@TL@ @TW@ 0.1</size></box></geometry></collision>
        <visual name="v"><geometry><box><size>@TL@ @TW@ 0.1</size></box></geometry>
          <material><ambient>0.65 0.55 0.45 1</ambient><diffuse>0.65 0.55 0.45 1</diffuse></material></visual>
      </link></model>
"""
FOOTER = """
  </world>
</sdf>
"""


def cylinder(r, h):
    return f"<cylinder><radius>{r:.4f}</radius><length>{h:.4f}</length></cylinder>"


def box(w, d, h):
    return f"<box><size>{w:.4f} {d:.4f} {h:.4f}</size></box>"


def build(kind, rng):
    """Parts [(geometry_xml, centre_height)], footprint radius, top height, dimensions (and sphere radius for a ball)."""
    u = rng.uniform
    if kind in ("box", "block"):
        if kind == "box":
            w, d, h = u(0.05, 0.09), u(0.04, 0.07), u(0.03, 0.08)
        else:
            w, d, h = u(0.08, 0.12), u(0.05, 0.08), u(0.012, 0.02)
        return {"parts": [(box(w, d, h), h / 2)], "radius": 0.5 * math.hypot(w, d), "top": h,
                "dims": {"width": w, "depth": d, "height": h}}
    if kind == "can":
        r, h = u(0.025, 0.035), u(0.07, 0.12)
        return {"parts": [(cylinder(r, h), h / 2)], "radius": r, "top": h, "dims": {"radius": r, "height": h}}
    if kind == "ball":
        r = u(0.025, 0.04)
        return {"parts": [(f"<sphere><radius>{r:.4f}</radius></sphere>", r)], "radius": r, "top": 2 * r,
                "dims": {"radius": r}, "sphere_r": r}
    if kind == "bottle":
        r, hb, rn, hn = u(0.025, 0.032), 0.08, 0.012, 0.04
        return {"parts": [(cylinder(r, hb), hb / 2), (cylinder(rn, hn), hb + hn / 2)], "radius": r,
                "top": hb + hn, "dims": {"body_radius": r, "body_height": hb, "neck_radius": rn, "neck_height": hn}}
    raise ValueError(kind)


def model_xml(name, obj, x, y, yaw, rgb):
    color = f"{rgb[0]:.2f} {rgb[1]:.2f} {rgb[2]:.2f} 1"
    body = ""
    for i, (geom, zc) in enumerate(obj["parts"]):
        body += (f'\n        <collision name="c{i}"><pose>0 0 {zc:.4f} 0 0 0</pose>'
                 f'<geometry>{geom}</geometry></collision>'
                 f'\n        <visual name="v{i}"><pose>0 0 {zc:.4f} 0 0 0</pose><geometry>{geom}</geometry>'
                 f'<material><ambient>{color}</ambient><diffuse>{color}</diffuse></material></visual>')
    return (f'\n    <model name="{name}"><static>true</static>'
            f'<pose>{x:.4f} {y:.4f} 0 0 0 {yaw:.4f}</pose>\n      <link name="link">{body}\n      </link></model>')


def probe(obj, x, y, cam_pos, R):
    """(u, v, expected planar depth, 3D point) of the surface point used to check the object, or None.
    Boxes, cans and the bottle: the centre of the top face. Ball: where the ray through its centre hits it."""
    if obj.get("sphere_r"):
        r = obj["sphere_r"]
        c = np.array([x, y, r])
        p = project(cam_pos, R, c)
        if p is None:
            return None
        u, v, zc = p
        return u, v, zc * (1.0 - r / float(np.linalg.norm(c - cam_pos))), c
    q = np.array([x, y, obj["top"]])
    p = project(cam_pos, R, q)
    return None if p is None else (p[0], p[1], p[2], q)


def blocked(cam, q, others):
    """True if the straight line from the camera to point q passes through another object, each approximated by a
    vertical cylinder (x, y, radius, height) standing on the table."""
    dx, dy = q[0] - cam[0], q[1] - cam[1]
    a = dx * dx + dy * dy
    if a < 1e-12:
        return False
    for (x, y, r, h) in others:
        fx, fy = cam[0] - x, cam[1] - y
        b = 2 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - r * r
        disc = b * b - 4 * a * c
        if disc <= 0:
            continue
        root = math.sqrt(disc)
        t0, t1 = max((-b - root) / (2 * a), 0.0), min((-b + root) / (2 * a), 1.0)
        if t0 >= t1:
            continue
        z0, z1 = cam[2] + t0 * (q[2] - cam[2]), cam[2] + t1 * (q[2] - cam[2])
        if min(z0, z1) < h:
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--min-gap", type=float, default=0.02, help="clearance between objects (m)")
    ap.add_argument("--center-x", "--cam-x", dest="center_x", type=float, default=1.0)
    ap.add_argument("--center-y", "--cam-y", dest="center_y", type=float, default=0.0)
    ap.add_argument("--pitch", type=float, default=45.0, help="camera pitch below the horizon, deg (90 = straight down)")
    ap.add_argument("--view-dist", type=float, default=0.45, help="camera-to-cluster distance (m)")
    ap.add_argument("--heading", type=float, default=0.0, help="viewing direction, deg from world +X")
    ap.add_argument("--target-z", type=float, default=0.05, help="height of the point the camera looks at (m)")
    ap.add_argument("--spread-x", type=float, default=0.11, help="half-size of the placement area along the view direction")
    ap.add_argument("--spread-y", type=float, default=0.20, help="half-size of the placement area across the view direction")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    kinds = KINDS[:]
    rng.shuffle(kinds)
    objs = [build(kinds[i % len(kinds)], rng) for i in range(a.count)]
    for i, o in enumerate(objs):
        o["kind"] = kinds[i % len(kinds)]

    cam_pos, R = view_from((a.center_x, a.center_y), a.view_dist, a.pitch, a.heading, a.target_z)
    hd = math.radians(a.heading)
    ux, uy = math.cos(hd), math.sin(hd)

    def place():
        placed = []
        for o in objs:
            for _ in range(200):
                along, across = rng.uniform(-a.spread_x, a.spread_x), rng.uniform(-a.spread_y, a.spread_y)
                x = a.center_x + along * ux - across * uy
                y = a.center_y + along * uy + across * ux
                if all(math.hypot(x - px, y - py) >= o["radius"] + prad + a.min_gap for px, py, prad in placed):
                    placed.append((x, y, o["radius"]))
                    break
            else:
                return None
        return placed

    def why_not_visible(placed):
        for i, (o, (x, y, _)) in enumerate(zip(objs, placed)):
            p = probe(o, x, y, cam_pos, R)
            if p is None:
                return f"the {o['kind']} is behind the camera"
            u, v, depth, q = p
            if not (MARGIN_PX <= u <= IMG_W - MARGIN_PX and MARGIN_PX <= v <= IMG_H - MARGIN_PX):
                return f"the {o['kind']} falls outside the image"
            if not (MIN_DEPTH <= depth <= MAX_DEPTH):
                return f"the {o['kind']} is {depth:.2f} m from the camera"
            others = [(px, py, oo["radius"] + 0.005, oo["top"])
                      for j, (oo, (px, py, _)) in enumerate(zip(objs, placed)) if j != i]
            if blocked(cam_pos, q, others):
                return f"the {o['kind']} is hidden behind another object"
        return None

    placed, reason = None, "the objects do not fit in the placement area"
    for _ in range(500):
        candidate = place()
        if candidate is None:
            continue
        reason = why_not_visible(candidate)
        if reason is None:
            placed = candidate
            break
    if placed is None:
        sys.exit(f"Could not find a layout where every object is visible ({reason}). Try a smaller --count, "
                 "smaller --spread-x/--spread-y, a larger --view-dist, or a steeper --pitch.")

    names = list(PALETTE)
    rng.shuffle(names)
    x_min, x_max = -1.0, a.center_x + 3.0
    xml = (HEADER.replace("@TX@", f"{(x_min + x_max) / 2:.3f}")
           .replace("@TL@", f"{x_max - x_min:.3f}").replace("@TW@", "4.0"))
    truth_objs = []
    for i, (o, (x, y, _)) in enumerate(zip(objs, placed)):
        name, color, yaw = f"obj{i}_{o['kind']}", names[i % len(names)], rng.uniform(-math.pi, math.pi)
        xml += model_xml(name, o, x, y, yaw, PALETTE[color])
        u, v, depth, _ = probe(o, x, y, cam_pos, R)
        truth_objs.append({
            "name": name, "kind": o["kind"], "color": color,
            "dims_m": {k: round(val, 4) for k, val in o["dims"].items()},
            "world_xy_m": [round(x, 4), round(y, 4)], "yaw_rad": round(yaw, 4),
            "top_height_m": round(o["top"], 4),
            "expected_depth_m": round(float(depth), 4),
            "pixel_uv": [round(float(u), 1), round(float(v), 1)],
        })
    xml += FOOTER

    WORLD_DIR.mkdir(exist_ok=True)
    (WORLD_DIR / "gungnir_world.sdf").write_text(xml)
    td = table_depth_at_centre(cam_pos, R)
    truth = {"seed": a.seed,
             "camera": {"position": [round(float(c), 4) for c in cam_pos],
                        "rotation": [[round(float(c), 6) for c in row] for row in R],
                        "center_xy": [a.center_x, a.center_y], "target_z": a.target_z, "view_dist": a.view_dist,
                        "pitch_deg": a.pitch, "heading_deg": a.heading,
                        "table_depth_centre_m": round(td, 4) if td else None,
                        "note": "optical frame: x right, y down, z forward; rotation columns are these axes in world coordinates"},
             "objects": truth_objs}
    (WORLD_DIR / "gungnir_world_truth.yaml").write_text(yaml.safe_dump(truth, sort_keys=False))
    print(f"wrote {WORLD_DIR / 'gungnir_world.sdf'} and gungnir_world_truth.yaml")
    print(f"home camera: ({cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f}) m, looking at "
          f"({a.center_x:.2f}, {a.center_y:.2f}, {a.target_z:.2f}), pitch {a.pitch:g} deg, heading {a.heading:g} deg")
    for o in truth_objs:
        print(f"  {o['name']:14} {o['color']:7} at ({o['world_xy_m'][0]:.3f}, {o['world_xy_m'][1]:.3f})  "
              f"top {o['top_height_m'] * 100:4.1f} cm  pixel {o['pixel_uv']}  depth {o['expected_depth_m']:.3f} m")
    print("next: python3 view_pose.py --write     (needs a running launch; mock is enough)")


if __name__ == "__main__":
    main()
