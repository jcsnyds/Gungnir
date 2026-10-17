#!/usr/bin/env python3
"""Project installer: RealSense L515 wrapper + Python dependencies, with GPU TensorFlow
and PyTorch on Jetson Orin (JetPack 7.2, CUDA 13.2, Python 3.12).

Run from the already-active virtual environment:

    python install.py                      # everything
    python install.py --skip-realsense     # Python dependencies only
    python install.py --skip-system-deps   # unknown flags are forwarded to IntelRealsense_Driver/setup.py
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable  # always the venv's interpreter, never whatever `python` resolves to
REALSENSE_SETUP = ROOT / "IntelRealsense_Driver" / "setup.py"
REQUIREMENTS = ROOT / "requirements.txt"
CONSTRAINTS = ROOT / "constraints.txt"

# CUDA 13.2 PyTorch builds. On JetPack 7.2 a plain `pip install torch` can give a build that
# does not run on Orin, so Jetson installs come from this index and are verified by running a kernel.
TORCH_INDEX = "https://download.pytorch.org/whl/cu132"

AARCH64_LINUX = platform.system() == "Linux" and platform.machine() == "aarch64"

TF_CHECK = """
import math, sys
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print('TensorFlow', tf.__version__, '| built with CUDA:', tf.test.is_built_with_cuda(), '| GPUs:', gpus)
if not gpus:
    sys.exit(1)
with tf.device('/GPU:0'):
    x = tf.random.normal([1024, 1024])
    y = float(tf.reduce_sum(x @ x))
print('GPU matmul sum =', y)
sys.exit(0 if math.isfinite(y) else 1)
"""

TORCH_CHECK = """
import math, sys
import torch
print('torch', torch.__version__, '| compiled arches:', torch.cuda.get_arch_list())
if not torch.cuda.is_available():
    sys.exit(1)
x = torch.randn(1024, 1024, device='cuda')
y = (x @ x).sum().item()
print('GPU matmul sum =', y)
sys.exit(0 if math.isfinite(y) else 1)
"""


def log(message: str) -> None:
    print(f"\n==> {message}", flush=True)


def run(command: list, **kwargs) -> subprocess.CompletedProcess:
    command = [str(part) for part in command]
    print("    $", " ".join(command), flush=True)
    return subprocess.run(command, check=True, **kwargs)


def pip(*args) -> None:
    run([PY, "-m", "pip", *args])


def is_installed(distribution: str) -> bool:
    try:
        metadata.version(distribution)
        return True
    except metadata.PackageNotFoundError:
        return False


def require_venv() -> None:
    if sys.prefix == sys.base_prefix and not os.environ.get("CONDA_PREFIX"):
        sys.exit(
            "No virtual environment is active. Activate yours first, e.g.\n"
            "    source /home/edgeai5/ME5990/edgeai/bin/activate\n"
            f"Current interpreter: {PY}"
        )
    if not CONSTRAINTS.is_file():
        sys.exit(f"Missing {CONSTRAINTS}")


def warn_if_system_site_packages() -> None:
    try:
        lines = (Path(sys.prefix) / "pyvenv.cfg").read_text().splitlines()
    except OSError:
        return
    if any(line.replace(" ", "").lower() == "include-system-site-packages=true" for line in lines):
        print(
            "\nNOTE: this venv has include-system-site-packages = true, so Ubuntu's apt builds of\n"
            "numpy/scipy/pandas/matplotlib are visible and `pip check` may report their metadata.\n"
            "requirements.txt uses minimum versions so the venv still gets its own copies. Setting\n"
            "the option to false is cleaner if you don't need JetPack's system Python modules."
        )


def torch_gpu_ok() -> bool:
    """True if PyTorch imports and runs a finite matmul on the GPU.

    Deliberately not an arch-list check: a wheel built for sm_80 runs on Orin (sm_87), because
    cubins are compatible within the same major architecture.
    """
    result = subprocess.run([PY, "-c", TORCH_CHECK], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def prepare_jetson() -> None:
    """Jetson Orin: replace CPU-only TensorFlow and make sure PyTorch runs on the GPU."""
    log("Preparing Jetson Orin (aarch64 Linux)")
    try:
        model = Path("/proc/device-tree/model").read_bytes().decode(errors="ignore")
    except OSError:
        model = ""
    if "orin" not in model.lower():
        print(
            f"WARNING: board model {model.strip(chr(0))!r} does not look like a Jetson Orin; "
            "the TensorFlow wheel in requirements.txt is built for Orin (sm_87) only."
        )
    if sys.version_info[:2] != (3, 12):
        sys.exit(
            "The Jetson TensorFlow wheel in requirements.txt is built for Python 3.12 (cp312); "
            f"this interpreter is {sys.version.split()[0]}."
        )

    # PyPI's aarch64 TensorFlow is CPU-only, and it shares the `tensorflow/` package directory
    # with the Jetson wheel, so it must be removed first.
    if is_installed("tensorflow"):
        log("Removing the CPU-only PyPI TensorFlow")
        pip("uninstall", "-y", "tensorflow")

    if not torch_gpu_ok():
        log("Installing CUDA 13.2 PyTorch")
        pip("install", "torch", "torchvision", "--extra-index-url", TORCH_INDEX)
        if not torch_gpu_ok():
            # PyPI can win over the extra index; force the CUDA build without touching its dependencies.
            log("PyTorch still does not run on the GPU; forcing the build from the CUDA 13.2 index")
            pip("install", "--force-reinstall", "--no-deps", "torch", "torchvision", "--index-url", TORCH_INDEX)


def check(label: str, code: str) -> bool:
    log(f"Checking {label}")
    ok = subprocess.run([PY, "-c", code]).returncode == 0
    print(f"    {label}: {'OK' if ok else 'FAILED'}")
    return ok


def verify() -> int:
    tf_ok = check("TensorFlow on GPU", TF_CHECK)
    torch_ok = check("PyTorch on GPU", TORCH_CHECK)
    if tf_ok and torch_ok:
        log("All GPU checks passed")
        return 0
    if AARCH64_LINUX:
        print(
            "\nA GPU check failed on this Jetson. Things to look at:\n"
            "  - TensorFlow: the community wheel is unofficial; see the repo's issues, or fall back to the\n"
            "    PyTorch port of Contact-GraspNet.\n"
            "  - PyTorch: the GPU matmul must run and print a finite sum (not nan). If it errors or prints nan,\n"
            "    try a prebuilt Orin wheel for CUDA 13.2 (see the Jetson-Orin-Wheels GitHub releases)."
        )
        return 1
    print("\nGPU checks failed. That is expected if this machine has no NVIDIA GPU.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-realsense", action="store_true", help="Skip the RealSense L515 setup.")
    parser.add_argument("--skip-verify", action="store_true", help="Skip the GPU checks at the end.")
    args, realsense_args = parser.parse_known_args()

    require_venv()
    warn_if_system_site_packages()
    # Inherited by every pip call below, including the ones inside the RealSense setup.py.
    os.environ["PIP_CONSTRAINT"] = str(CONSTRAINTS)

    if not args.skip_realsense:
        log("RealSense L515 wrapper")
        run([PY, REALSENSE_SETUP, *realsense_args])

    if AARCH64_LINUX:
        prepare_jetson()

    log("Installing requirements.txt")
    pip("install", "-r", REQUIREMENTS)

    return 0 if args.skip_verify else verify()


if __name__ == "__main__":
    sys.exit(main())