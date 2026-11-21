#!/usr/bin/env python3
"""Measure how noisy a depth stream is. Works on the simulated camera and on the real one (same topics),
so the numbers can be compared directly and used to tune config/sim_camera_noise.yaml.

  depth_noise_report.py [--frames 20] [--topic /camera/depth/image_rect_raw]

Keep the scene still. Reports the fraction of valid pixels, pixels that flicker between valid and invalid, and the
per-pixel temporal standard deviation on stable surfaces (valid in every frame, away from depth edges), by depth."""
import argparse
import sys

import numpy as np

from l515_ros import L515, L515Error


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--topic", default="/camera/depth/image_rect_raw")
    args = ap.parse_args()

    cam, frames = L515(depth_topic=args.topic), []
    try:
        for _ in range(args.frames):
            cam.capture_image()
            frames.append(cam.get_depth().copy())
    except L515Error as e:
        sys.exit(f"ERROR: {e}")
    finally:
        cam.close()

    d = np.stack(frames)
    valid = d > 0
    always, ever = valid.all(axis=0), valid.any(axis=0)
    print(f"{len(frames)} frames of {d.shape[2]}x{d.shape[1]} from {args.topic}")
    print(f"valid pixels per frame     : {100 * valid.mean():.2f} %")
    print(f"never valid                : {100 * (~ever).mean():.2f} % of pixels")
    print(f"flicker (sometimes valid)  : {100 * (ever & ~always).mean():.2f} % of pixels")

    m = np.where(always, d.mean(axis=0), 0.0)
    gy = np.abs(np.diff(m, axis=0, prepend=m[:1]))
    gx = np.abs(np.diff(m, axis=1, prepend=m[:, :1]))
    flat = always & (gx < 0.01) & (gy < 0.01)
    if flat.sum() < 200:
        print("\nNot enough stable pixels for a noise estimate (is the scene still?)")
        return
    z, s = m[flat], d.std(axis=0)[flat] * 1000
    print(f"\ntemporal noise on stable surfaces ({int(flat.sum())} pixels): "
          f"median {np.median(s):.2f} mm, 95th percentile {np.percentile(s, 95):.2f} mm")
    print(f"{'depth (m)':>12}{'pixels':>9}{'median std (mm)':>17}")
    for a in np.arange(np.floor(z.min() / 0.05) * 0.05, z.max() + 0.05, 0.05):
        sel = (z >= a) & (z < a + 0.05)
        if sel.sum() >= 200:
            print(f"{a:6.2f}-{a + 0.05:.2f}{int(sel.sum()):9d}{np.median(s[sel]):17.2f}")


if __name__ == "__main__":
    main()
