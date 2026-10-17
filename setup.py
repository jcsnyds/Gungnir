#!/usr/bin/env python3
"""Project installer: RealSense L515 wrapper + Python dependencies, with GPU TensorFlow
and PyTorch on Jetson Orin (JetPack 7.2, CUDA 13.2, Python 3.12).

Run from the already-active virtual environment:

    python setup.py                      # everything
    python setup.py --skip-realsense     # Python dependencies only
    python setup.py --skip-system-deps   # unknown flags are forwarded to IntelRealsense_Driver/setup.py
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import os
import platform
import subprocess
import sys
from pathlib import Path
import shlex
import shutil

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


# Contact-GraspNet's pointnet2 code uses tf.compat.v1.layers, which exists only on legacy Keras (tf_keras).
TF_LEGACY_LAYERS_CHECK = """
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
x = tf.compat.v1.placeholder(tf.float32, [None, 8, 4])
y = tf.compat.v1.layers.batch_normalization(x, training=True)
print('tf.compat.v1.layers OK:', y.name)
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



def build_contact_graspnet_tfops() -> None:
    """Build Contact-GraspNet TensorFlow custom ops natively for Jetson Orin."""
    if not AARCH64_LINUX:
        log("Skipping Contact-GraspNet CUDA ops: not running on ARM64 Linux")
        return

    import tensorflow as tf

    ops_root = ROOT / "contact_graspnet" / "pointnet2" / "tf_ops"
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")).resolve()
    nvcc = cuda_home / "bin" / "nvcc"
    cuda_lib = cuda_home / "targets" / "sbsa-linux" / "lib"
    tf_lib = Path(tf.__file__).resolve().parent

    if not nvcc.is_file():
        sys.exit(f"CUDA compiler not found: {nvcc}")

    if not cuda_lib.is_dir():
        sys.exit(f"CUDA library directory not found: {cuda_lib}")

    build_info = tf.sysconfig.get_build_info()
    tf_cuda = build_info.get("cuda_version", "unknown")

    log("Building Contact-GraspNet TensorFlow operators")
    print(f"    Architecture: {platform.machine()}, CUDA target: sm_87")
    print(f"    TensorFlow: {tf.__version__}, built against CUDA {tf_cuda}")
    print(f"    CUDA compiler: {nvcc}")
    print(f"    CUDA runtime directory: {cuda_lib}")

    if str(tf_cuda) != "13.2":
        print(
            "WARNING: TensorFlow's CUDA build version differs from the "
            "installed toolkit. Compilation may succeed while runtime "
            "compatibility remains unresolved."
        )

    backup_dir = ROOT.parent / "op_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)

    for name in (
        "sampling/tf_sampling_so.so",
        "grouping/tf_grouping_so.so",
        "3d_interpolation/tf_interpolate_so.so",
    ):
        original = ops_root / name
        if original.is_file():
            destination = backup_dir / original.name
            if not destination.exists():
                shutil.copy2(original, destination)
                print(f"    Backed up {original.name} to {destination}")

    env = os.environ.copy()
    env["CUDA_HOME"] = str(cuda_home)
    env["PATH"] = f"{cuda_home / 'bin'}:{env.get('PATH', '')}"
    env["TF_USE_LEGACY_KERAS"] = "1"
    env["LD_LIBRARY_PATH"] = ":".join(
        p for p in (
            str(tf_lib),
            str(cuda_lib),
            env.get("LD_LIBRARY_PATH", ""),
        ) if p
    )

    tf_cflags = tf.sysconfig.get_compile_flags()
    tf_lflags = tf.sysconfig.get_link_flags()
    common_flags = ["-std=c++17", *tf_cflags]
    cuda_link_flags = [
        f"-L{cuda_lib}",
        "-lcudart",
        f"-Wl,-rpath,{cuda_lib}",
        f"-Wl,-rpath,{tf_lib}",
    ]

    def compile_op(directory: Path, name: str) -> None:
        """Compile a CUDA kernel and its TensorFlow wrapper."""
        log(f"Compiling {name}")
        kernel = f"tf_{name}_g.cu"
        obj = f"tf_{name}_g.cu.o"
        wrapper = f"tf_{name}.cpp"
        output = f"tf_{name}_so.so"

        run([
            nvcc,
            "-std=c++17",
            "-arch=sm_87",
            "-c", kernel,
            "-o", obj,
            *tf_cflags,
            "-DGOOGLE_CUDA=1",
            "-Xcompiler=-fPIC",
        ], cwd=directory, env=env)

        run([
            "g++",
            "-shared",
            "-fPIC",
            *common_flags,
            wrapper,
            obj,
            "-o", output,
            *tf_lflags,
            *cuda_link_flags,
        ], cwd=directory, env=env)

    compile_op(ops_root / "sampling", "sampling")
    compile_op(ops_root / "grouping", "grouping")

    # This operator registers CPU kernels; it does not need nvcc.
    interpolation = ops_root / "3d_interpolation"
    log("Compiling interpolation operator")
    run([
        "g++",
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-O2",
        *tf_cflags,
        "tf_interpolate.cpp",
        "-o", "tf_interpolate_so.so",
        *tf_lflags,
        f"-Wl,-rpath,{tf_lib}",
    ], cwd=interpolation, env=env)

    # Confirm that all three rebuilt libraries are ARM64 ELF binaries.
    for relative in (
        "sampling/tf_sampling_so.so",
        "grouping/tf_grouping_so.so",
        "3d_interpolation/tf_interpolate_so.so",
    ):
        binary = ops_root / relative
        result = subprocess.run(
            ["file", str(binary)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"    {result.stdout.strip()}")
        if "ARM aarch64" not in result.stdout and "ARM64" not in result.stdout:
            sys.exit(f"Unexpected binary architecture: {binary}")

    # Smoke-test loading each library with the active TensorFlow.
    smoke_test = """
import os
import tensorflow as tf

root = os.environ["CONTACT_GRASPNET_TFOPS"]
for relative in (
    "sampling/tf_sampling_so.so",
    "grouping/tf_grouping_so.so",
    "3d_interpolation/tf_interpolate_so.so",
):
    path = os.path.join(root, relative)
    print("Loading:", path, flush=True)
    tf.load_op_library(path)
    print("Loaded successfully", flush=True)
"""
    env["CONTACT_GRASPNET_TFOPS"] = str(ops_root)
    run([PY, "-c", smoke_test], env=env)

    log("Contact-GraspNet custom operators compiled and loaded")



def check(label: str, code: str, env: dict | None = None) -> bool:
    log(f"Checking {label}")
    ok = subprocess.run([PY, "-c", code], env=env).returncode == 0
    print(f"    {label}: {'OK' if ok else 'FAILED'}")
    return ok


def verify() -> int:
    tf_ok = check("TensorFlow on GPU", TF_CHECK)
    torch_ok = check("PyTorch on GPU", TORCH_CHECK)
    if AARCH64_LINUX:
        legacy_env = {**os.environ, "TF_USE_LEGACY_KERAS": "1"}
        legacy_ok = check("tf.compat.v1.layers (needs TF_USE_LEGACY_KERAS=1)", TF_LEGACY_LAYERS_CHECK, env=legacy_env)
        tf_ok = tf_ok and legacy_ok
    if tf_ok and torch_ok:
        log("All GPU checks passed")
        if AARCH64_LINUX:
            print("    Reminder: export TF_USE_LEGACY_KERAS=1 (or set it at the top of inference.py) before running Contact-GraspNet.")
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
    parser.add_argument(
        "--skip-tf-ops",
        action="store_true",
        help="Skip building Contact-GraspNet TensorFlow custom operators.",
    )
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

    if AARCH64_LINUX:
        # tf-keras lists `tensorflow` as a dependency; without --no-deps pip would install PyPI's
        # CPU-only TensorFlow on top of the Jetson wheel. Pinned to the 2.21 line to match the wheel.
        log("Installing legacy Keras (tf-keras) for tf.compat.v1.layers")
        pip("install", "--no-deps", "tf-keras>=2.21,<2.22")

    if AARCH64_LINUX and not args.skip_tf_ops:
        build_contact_graspnet_tfops()

    return 0 if args.skip_verify else verify()


if __name__ == "__main__":
    sys.exit(main())