from pathlib import Path
import sys

# Allow running this example directly from the project checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from l515 import L515


with L515() as camera:
    print("Depth scale:", camera.depth_scale)
    print("Depth K:\n", camera.get_k())
    print("Aligned/color K:\n", camera.get_aligned_k())

    camera.capture_image()
    print("Depth shape:", camera.get_depth().shape)
    print("Color shape:", camera.get_color().shape)
    print("Depth range (m):", camera.get_depth().min(), camera.get_depth().max())

    camera.capture_and_show_image()
