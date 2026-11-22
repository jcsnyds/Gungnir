#!/usr/bin/env python3
"""Grab one frame from the simulated camera and save it.

  camera_snapshot.py [--out DIR] [--truth FILE]

Saves rgb.png, rgb_marked.png (magenta crosses where each object should be), depth_colour.png,
depth.npy (float32 m, 0 = no return), k.npy and points.npy. If the ground-truth file written by
make_world.py exists, prints expected vs measured depth at every object."""
import argparse
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import yaml

from l515_ros import L515, L515Error

DEFAULT_TRUTH = Path(__file__).resolve().parent.parent / "worlds" / "gungnir_world_truth.yaml"


def write_png(path, rgb):
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    Path(path).write_bytes(png)


def jet(t):
    r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def mark(img, u, v, size=8):
    h, w, _ = img.shape
    for d in range(-size, size + 1):
        for yy, xx in ((v, u + d), (v + d, u)):
            if 0 <= yy < h and 0 <= xx < w:
                img[yy, xx] = (255, 0, 255)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="~/sim_captures/latest")
    ap.add_argument("--truth", default=str(DEFAULT_TRUTH))
    ap.add_argument("--clean", action="store_true", help="read the noise-free depth (sim only)")
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    cam = L515(depth_topic="/camera/depth/image_clean" if args.clean else "/camera/depth/image_rect_raw")
    try:
        cam.capture_image()
        depth = cam.get_depth().copy()
        rgb = np.ascontiguousarray(cam.get_color())
        k = cam.get_k()
        points = cam.get_pointcloud(0.25, 9.0)
    except L515Error as e:
        sys.exit(f"ERROR: {e}")
    finally:
        cam.close()

    h, w = depth.shape
    valid = depth > 0
    print(f"depth {w}x{h}, valid pixels {100 * valid.mean():.1f}%")
    far = float("nan")
    if valid.any():
        far = float(np.percentile(depth[valid], 95))
        print(f"depth range {depth[valid].min():.3f} .. {depth[valid].max():.3f} m;  "
              f"far plane (table) ~ {far:.3f} m")
    print("K =\n", np.round(k, 1))

    dc = np.zeros((h, w, 3), np.uint8)
    if valid.any():
        lo, hi = np.percentile(depth[valid], [1, 99])
        dc = jet(np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1))
        dc[~valid] = 0

    marked = rgb.copy()
    truth = Path(args.truth).expanduser()
    if truth.exists():
        t = yaml.safe_load(truth.read_text())
        c = t["camera"]
        win = depth[h // 2 - 2:h // 2 + 3, w // 2 - 2:w // 2 + 3]
        win = win[win > 0]
        centre = float(np.median(win)) if win.size else float("nan")
        print(f"\nground truth: {truth.name} (seed {t['seed']}); table at the image centre: "
              f"expected {c['table_depth_centre_m']:.3f} m, measured {centre:.3f} m")
        print(f"{'object':16}{'pixel (u,v)':>14}{'expected m':>12}{'measured m':>12}{'error mm':>10}")
        for o in t["objects"]:
            u, v = (int(round(c)) for c in o["pixel_uv"])
            if not (2 <= u < w - 2 and 2 <= v < h - 2):
                print(f"{o['name']:16}{f'({u},{v})':>14}   outside the image")
                continue
            mark(marked, u, v)
            win = depth[v - 2:v + 3, u - 2:u + 3]
            win = win[win > 0]
            meas = float(np.median(win)) if win.size else float("nan")
            print(f"{o['name']:16}{f'({u},{v})':>14}{o['expected_depth_m']:>12.3f}{meas:>12.3f}"
                  f"{(meas - o['expected_depth_m']) * 1000:>10.1f}")
    else:
        print(f"\n(no ground-truth file at {truth})")

    write_png(out / "rgb.png", rgb)
    write_png(out / "rgb_marked.png", marked)
    write_png(out / "depth_colour.png", dc)
    np.save(out / "depth.npy", depth)
    np.save(out / "rgb.npy", rgb)
    np.save(out / "k.npy", k)
    np.save(out / "points.npy", points)
    print(f"\nsaved to {out}  ({len(points)} points)")


if __name__ == "__main__":
    main()
