#!/usr/bin/env python3
"""Cross-platform project installer for Contact-GraspNet and the L515 wrapper.

Run from the project folder. The installer reuses an active environment, detects a
project-local .venv/ or venv/, or creates .venv/ automatically when needed:

    python setup.py                      # install dependencies, setup RealSense, then verify
    python setup.py --skip-realsense     # dependencies and verification only
    python setup.py --skip-verify        # install without final smoke tests
    python setup.py --skip-tf-ops        # skip Jetson-only TensorFlow CUDA-op compilation
    python setup.py --skip-sam2           # skip downloading the SAM 2.1 checkpoint
    python setup.py --skip-ros-setup      # skip automatic apt-based ROS setup (advanced)

The SAM 2.1 small checkpoint is stored in the project folder at
checkpoints/sam2.1_hiera_small.pt by default. Set SAM2_CHECKPOINT_DIR to override this location.

Optional platform-specific dependency files are supported:
    requirements-windows.txt / constraints-windows.txt
    requirements-linux.txt   / constraints-linux.txt
If absent, requirements.txt is used. On Ubuntu 22.04 the installer selects ROS 2 Humble; on
Ubuntu 24.04 it selects ROS 2 Jazzy, and installs the runtime modules used by Gungnir.
Jetson-specific TensorFlow/CUDA wheels should not be
listed in a Windows requirements file. Native L515 compilation also requires the driver's
IntelRealsense_Driver/setup.py to support the current operating system and compiler.

AI use: Originally written by Claude (Anthropic; Claude Sonnet 5.5 via claude.ai), October 2026,
as install.py: venv-aware installer, Jetson GPU TensorFlow/PyTorch handling, tf-keras install and
GPU checks. Claude checked only its syntax. Later edited by the author; parts such as
build_contact_graspnet_tfops() were not written by Claude in this conversation, so disclose
their source separately.
Virtual-environment auto-detection, creation, and interpreter re-execution were added with
ChatGPT (OpenAI), November 2026. ROS 2 apt installation detection, environment sourcing,
ROS Python verification, and ROS-compatible virtual-environment configuration were also
added with ChatGPT (OpenAI), November 2026. Existing installer functionality was retained.
Reviewed and tested by: Ryan Poliquin
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
import shlex
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parent
PY = sys.executable  # refreshed automatically after re-executing inside the selected environment
REALSENSE_SETUP = ROOT / "IntelRealsense_Driver" / "setup.py"
REQUIREMENTS = ROOT / "requirements.txt"
CONSTRAINTS = ROOT / "constraints.txt"
SYSTEM = platform.system()
IS_WINDOWS = SYSTEM == "Windows"
IS_LINUX = SYSTEM == "Linux"

# ROS 2 binary packages are mapped to their supported Ubuntu LTS releases.
ROS_DISTRO_BY_UBUNTU = {"jammy": "humble", "noble": "jazzy"}
# Required ROS packages: core Python API/CLI, launch support, messages, and cv_bridge.
ROS_PACKAGE_SUFFIXES = ("ros-base", "ros2launch", "sensor-msgs", "cv-bridge")
# Required to create the project venv on fresh Ubuntu installations.
ROS_SYSTEM_PACKAGES = ("python3-venv",)
# Official ROS package repository and signing key used by the Ubuntu deb installation instructions.
ROS_APT_REPOSITORY = "http://packages.ros.org/ros2/ubuntu"
ROS_APT_KEY_URL = "https://raw.githubusercontent.com/ros/rosdistro/master/ros.key"
ROS_APT_KEYRING = Path("/usr/share/keyrings/ros-archive-keyring.gpg")

# CUDA 13.2 PyTorch builds. On JetPack 7.2 a plain `pip install torch` can give a build that
# does not run on Orin, so Jetson installs come from this index and are verified by running a kernel.
TORCH_INDEX = "https://download.pytorch.org/whl/cu132"

AARCH64_LINUX = IS_LINUX and platform.machine().lower() in ("aarch64", "arm64")

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


def ubuntu_release() -> tuple[str, str]:
    """Return the Ubuntu ID and codename from /etc/os-release."""
    release_file = Path("/etc/os-release")
    if not release_file.is_file():
        sys.exit("ROS 2 auto-installation requires Ubuntu with /etc/os-release.")
    values = {}
    for line in release_file.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            values[key] = value.strip().strip('"').strip("'")
    return values.get("ID", "").lower(), values.get("UBUNTU_CODENAME", values.get("VERSION_CODENAME", "")).lower()


def selected_ros_distro() -> tuple[str, str]:
    """Choose the ROS 2 distribution matching the Ubuntu LTS release and architecture."""
    ubuntu_id, codename = ubuntu_release()
    if ubuntu_id != "ubuntu":
        sys.exit(f"ROS 2 auto-installation currently supports Ubuntu only; detected ID={ubuntu_id!r}.")
    distro = ROS_DISTRO_BY_UBUNTU.get(codename)
    if distro is None:
        supported = ", ".join(f"{name}: ROS 2 {ros}" for name, ros in ROS_DISTRO_BY_UBUNTU.items())
        sys.exit(f"Unsupported Ubuntu release {codename!r} for automatic ROS 2 installation. Supported: {supported}.")
    current = os.environ.get("ROS_DISTRO")
    if current and current != distro:
        sys.exit(
            f"The sourced ROS_DISTRO is {current!r}, but Ubuntu {codename} maps to ROS 2 {distro!r}. "
            "Use a matching Ubuntu/ROS combination or start a clean terminal."
        )
    return codename, distro


def privileged_prefix() -> list[str]:
    """Return the command prefix needed for package-manager and system-file changes."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return []
    sudo = shutil.which("sudo")
    if sudo is None:
        sys.exit("ROS 2 installation needs administrator privileges; install sudo or rerun as root.")
    return [sudo]


def run_privileged(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run one command with administrator privileges, prompting through sudo when needed."""
    return run([*privileged_prefix(), *command], **kwargs)


def deb_package_installed(package: str) -> bool:
    """Check whether an apt package is fully installed."""
    result = subprocess.run(
        ["dpkg-query", "-W", "-f=${Status}", package],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "install ok installed"


def ros_apt_repository_configured() -> bool:
    """Check the standard apt source files for an existing ROS 2 repository entry."""
    source_files = [Path("/etc/apt/sources.list")]
    source_dir = Path("/etc/apt/sources.list.d")
    if source_dir.is_dir():
        source_files.extend(source_dir.glob("*"))
    for source_file in source_files:
        try:
            if ROS_APT_REPOSITORY in source_file.read_text(errors="ignore"):
                return True
        except (OSError, IsADirectoryError):
            continue
    return False


def configure_ros_apt_repository(codename: str) -> None:
    """Add the official ROS 2 apt source if the machine does not already have it."""
    if ros_apt_repository_configured():
        log("ROS 2 apt repository already configured")
        return

    log("Configuring the official ROS 2 apt repository")
    run_privileged(["apt-get", "update"])
    run_privileged([
        "apt-get", "install", "-y",
        "curl", "ca-certificates", "gnupg", "software-properties-common",
    ])
    run_privileged(["add-apt-repository", "-y", "universe"])

    architecture = subprocess.run(
        ["dpkg", "--print-architecture"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="gungnir-ros-apt-") as temp_dir:
        temp_path = Path(temp_dir)
        key_path = temp_path / "ros-archive-keyring.gpg"
        source_path = temp_path / "ros2.list"
        urllib.request.urlretrieve(ROS_APT_KEY_URL, key_path)
        run_privileged(["install", "-D", "-m", "0644", key_path, ROS_APT_KEYRING])
        source_path.write_text(
            f"deb [arch={architecture} signed-by={ROS_APT_KEYRING}] {ROS_APT_REPOSITORY} {codename} main\n"
        )
        run_privileged([
            "install", "-D", "-m", "0644", source_path,
            "/etc/apt/sources.list.d/ros2.list",
        ])
    run_privileged(["apt-get", "update"])


def source_environment_script(script_path: Path) -> None:
    """Source a bash setup file and import its exported variables into this Python process."""
    if not script_path.is_file():
        sys.exit(f"ROS environment setup file not found: {script_path}")
    bash = shutil.which("bash") or "/bin/bash"
    command = f"source {shlex.quote(str(script_path))} >/dev/null && env -0"
    result = subprocess.run(
        [bash, "-c", command],
        env=os.environ.copy(),
        check=True,
        capture_output=True,
    )
    for entry in result.stdout.split(b"\0"):
        if b"=" not in entry:
            continue
        key, value = entry.split(b"=", 1)
        os.environ[os.fsdecode(key)] = os.fsdecode(value)


def ensure_ros2_installation() -> str:
    """Install the ROS runtime packages required by Gungnir, then source the ROS environment."""
    if not IS_LINUX:
        log("Skipping apt-based ROS 2 setup: this installer is not running on Linux")
        return ""

    codename, distro = selected_ros_distro()
    setup_script = Path("/opt/ros") / distro / "setup.bash"
    packages = [f"ros-{distro}-{suffix}" for suffix in ROS_PACKAGE_SUFFIXES]
    packages.extend(ROS_SYSTEM_PACKAGES)
    missing = [package for package in packages if not deb_package_installed(package)]

    if not setup_script.is_file() or missing:
        if missing:
            log(f"ROS 2 {distro} is missing required packages: {', '.join(missing)}")
        else:
            log(f"ROS 2 {distro} setup file is missing; installing the required runtime")
        configure_ros_apt_repository(codename)
        run_privileged(["apt-get", "install", "-y", *packages])

    if not setup_script.is_file():
        sys.exit(f"ROS 2 installation did not create the expected setup file: {setup_script}")

    current = os.environ.get("ROS_DISTRO")
    if current and current != distro:
        sys.exit(f"ROS_DISTRO={current!r} does not match the installed ROS 2 distribution {distro!r}.")
    if not current:
        log(f"Sourcing ROS 2 {distro} environment for setup.py and its child processes")
        source_environment_script(setup_script)

    overlay = os.environ.get("GUNGNIR_ROS_OVERLAY")
    if overlay:
        overlay_path = Path(overlay).expanduser().resolve()
        log(f"Sourcing user ROS workspace overlay: {overlay_path}")
        source_environment_script(overlay_path)

    return distro


def verify_ros2_python() -> None:
    """Verify the selected project interpreter can import the ROS modules used by the client."""
    code = """
import rclpy
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
print('rclpy:', rclpy.__file__)
print('sensor_msgs:', Image.__module__)
print('cv_bridge:', CvBridge.__module__)
"""
    log("Verifying ROS 2 imports in the project Python environment")
    result = subprocess.run([PY, "-c", code], env=os.environ.copy(), text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, file=sys.stderr, flush=True)
        sys.exit(
            "ROS 2 Python imports failed inside the selected virtual environment. "
            "The venv must use the same Python minor version as the apt-installed ROS distribution, "
            "and the /opt/ros/<distro>/setup.bash environment must be sourced."
        )


def is_installed(distribution: str) -> bool:
    try:
        metadata.version(distribution)
        return True
    except metadata.PackageNotFoundError:
        return False


def venv_python(environment_dir: Path) -> Path:
    """Return the Python executable inside a virtual environment."""
    if IS_WINDOWS:
        return environment_dir / "Scripts" / "python.exe"
    return environment_dir / "bin" / "python"


def is_usable_venv(environment_dir: Path) -> bool:
    """Check for the interpreter and configuration file that identify a venv."""
    return (environment_dir / "pyvenv.cfg").is_file() and venv_python(environment_dir).is_file()


def enable_system_site_packages(environment_dir: Path) -> bool:
    """Expose apt-installed Python modules to the venv, as required by ROS 2 binary packages."""
    if not IS_LINUX:
        return False
    config_path = environment_dir / "pyvenv.cfg"
    if not config_path.is_file():
        return False
    lines = config_path.read_text().splitlines()
    updated = []
    changed = False
    found_setting = False
    for line in lines:
        if line.strip().lower().startswith("include-system-site-packages"):
            updated.append("include-system-site-packages = true")
            found_setting = True
            changed = changed or line.split("=", 1)[-1].strip().lower() != "true"
        else:
            updated.append(line)
    if not found_setting:
        updated.append("include-system-site-packages = true")
        changed = True
    if changed:
        config_path.write_text("\n".join(updated) + "\n")
        log(f"Enabled Ubuntu system Python packages for ROS 2 in {environment_dir}")
    return changed


def activate_and_reexec(environment_dir: Path, interpreter: Path) -> None:
    """Re-launch this installer using the selected environment's interpreter.

    A child Python process cannot activate the parent terminal session. Re-executing
    this script with the environment's Python provides the same effect for this
    installer and all of the subprocesses it starts.
    """
    environment_dir = environment_dir.resolve()
    interpreter = Path(os.path.abspath(interpreter))  # Preserve venv Python symlinks.
    scripts_dir = environment_dir / ("Scripts" if IS_WINDOWS else "bin")

    child_env = os.environ.copy()
    child_env["VIRTUAL_ENV"] = str(environment_dir)
    child_env.pop("PYTHONHOME", None)
    child_env["PATH"] = str(scripts_dir) + os.pathsep + child_env.get("PATH", "")

    log(f"Starting setup.py with virtual environment: {environment_dir}")
    os.execve(
        str(interpreter),
        [str(interpreter), str(Path(__file__).resolve()), *sys.argv[1:]],
        child_env,
    )


def ensure_venv_and_reexec() -> None:
    """Use an active/project virtual environment, or create .venv/ if none exists."""
    # A currently running PEP 405 virtual environment always takes precedence.
    if sys.prefix != sys.base_prefix:
        log(f"Using active virtual environment: {sys.prefix}")
        if enable_system_site_packages(Path(sys.prefix)):
            child_env = os.environ.copy()
            os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], child_env)
        return

    # Respect an active Conda environment when this interpreter belongs to it.
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and Path(conda_prefix).resolve() == Path(sys.prefix).resolve():
        log(f"Using active Conda environment: {conda_prefix}")
        return

    # An activated environment may be present in the shell even when the user
    # explicitly invoked /usr/bin/python3 instead of the environment's Python.
    requested_environment = os.environ.get("VIRTUAL_ENV")
    if requested_environment:
        environment_dir = Path(requested_environment).expanduser()
        interpreter = venv_python(environment_dir)
        if is_usable_venv(environment_dir):
            enable_system_site_packages(environment_dir)
            activate_and_reexec(environment_dir, interpreter)
        print(
            f"WARNING: VIRTUAL_ENV points to an environment without Python: {environment_dir}",
            flush=True,
        )

    # Reuse a conventional environment already present in the project directory.
    for environment_dir in (ROOT / ".venv", ROOT / "venv"):
        interpreter = venv_python(environment_dir)
        if is_usable_venv(environment_dir):
            enable_system_site_packages(environment_dir)
            activate_and_reexec(environment_dir, interpreter)

    # No usable environment was found, so create the project's default .venv/.
    environment_dir = ROOT / ".venv"
    if environment_dir.exists():
        sys.exit(
            f"Cannot create a virtual environment because {environment_dir} already exists "
            "but does not contain a usable Python interpreter. Remove or rename that directory, "
            "then run setup.py again."
        )

    log(f"No virtual environment found; creating {environment_dir}")
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", "--system-site-packages", str(environment_dir)] if IS_LINUX
            else [sys.executable, "-m", "venv", str(environment_dir)],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        sys.exit(
            "Virtual environment creation failed. On Ubuntu/Debian, install the venv support "
            "package (for example, `sudo apt install python3-venv`) and run setup.py again. "
            f"Underlying exit code: {exc.returncode}"
        )

    interpreter = venv_python(environment_dir)
    if not is_usable_venv(environment_dir):
        sys.exit(f"Virtual environment creation did not produce a usable environment: {environment_dir}")

    enable_system_site_packages(environment_dir)
    activate_and_reexec(environment_dir, interpreter)


def platform_dependency_files() -> tuple[Path, Path | None]:
    """Choose platform-specific requirements/constraints when supplied."""
    suffix = "windows" if IS_WINDOWS else "linux" if IS_LINUX else platform.system().lower()
    platform_requirements = ROOT / f"requirements-{suffix}.txt"
    platform_constraints = ROOT / f"constraints-{suffix}.txt"
    requirements = platform_requirements if platform_requirements.is_file() else REQUIREMENTS
    constraints = platform_constraints if platform_constraints.is_file() else CONSTRAINTS

    if not requirements.is_file():
        sys.exit(f"Missing requirements file: {requirements}")
    if IS_WINDOWS and requirements == REQUIREMENTS:
        print(
            "\nWARNING: requirements-windows.txt was not found, so requirements.txt will be used.\n"
            "If requirements.txt contains Jetson/Linux-only wheels or packages, create a separate\n"
            "requirements-windows.txt for Windows (recommended)."
        )
    # Do not force Linux/Jetson constraints onto Windows.
    if IS_WINDOWS and constraints == CONSTRAINTS and not (ROOT / "constraints-windows.txt").is_file():
        constraints = None
    elif not constraints.is_file():
        constraints = None
    return requirements, constraints


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
    env["PATH"] = f"{cuda_home / 'bin'}{os.pathsep}{env.get('PATH', '')}"
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
            f"-I{cuda_home / 'targets' / 'sbsa-linux' / 'include'}",
            f"-I{cuda_home / 'include'}",
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
        "-shared",
        "-fPIC",
        *common_flags,
        "tf_interpolate.cpp",
        "-o", "tf_interpolate_so.so",
        *tf_lflags,
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
    if AARCH64_LINUX:
        tf_ok = check("TensorFlow on GPU", TF_CHECK)
        torch_ok = check("PyTorch on GPU", TORCH_CHECK)
        legacy_env = {**os.environ, "TF_USE_LEGACY_KERAS": "1"}
        legacy_ok = check("tf.compat.v1.layers (needs TF_USE_LEGACY_KERAS=1)", TF_LEGACY_LAYERS_CHECK, env=legacy_env)
        if tf_ok and torch_ok and legacy_ok:
            log("All Jetson GPU checks passed")
            print("    Reminder: set TF_USE_LEGACY_KERAS=1 before running Contact-GraspNet inference.")
            return 0
        print(
            "\nA Jetson GPU check failed. Confirm that the installed TensorFlow and PyTorch builds "
            "match the JetPack/CUDA environment."
        )
        return 1

    # Windows and general-purpose PCs may be CPU-only. Test imports and CPU execution first;
    # if a CUDA GPU is available, test it as an additional check without requiring one.
    tf_cpu_check = """
import math, tensorflow as tf
x = tf.random.normal([128, 128])
y = float(tf.reduce_sum(x @ x))
print('TensorFlow', tf.__version__, '| CPU matmul sum =', y)
assert math.isfinite(y)
"""
    torch_cpu_check = """
import math, torch
x = torch.randn(128, 128)
y = (x @ x).sum().item()
print('PyTorch', torch.__version__, '| CPU matmul sum =', y, '| CUDA available:', torch.cuda.is_available())
assert math.isfinite(y)
if torch.cuda.is_available():
    z = (torch.randn(128, 128, device='cuda') @ torch.randn(128, 128, device='cuda')).sum().item()
    print('PyTorch GPU matmul sum =', z)
    assert math.isfinite(z)
"""
    tf_ok = check("TensorFlow import/CPU operation", tf_cpu_check)
    torch_ok = check("PyTorch import/CPU operation (GPU tested if available)", torch_cpu_check)
    if tf_ok and torch_ok:
        log("Python framework checks passed")
        if IS_WINDOWS:
            print(
                "    Note: standard native Windows TensorFlow installs are generally CPU-only. "
                "For NVIDIA GPU TensorFlow, use a supported WSL2/Linux setup."
            )
        return 0
    print("\nOne or more Python framework checks failed. Review the errors above.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--skip-realsense", action="store_true", help="Skip the RealSense L515 native-wrapper setup.")
    parser.add_argument("--skip-verify", action="store_true", help="Skip final smoke tests.")
    parser.add_argument("--skip-tf-ops", action="store_true", help="Skip building Contact-GraspNet TensorFlow custom operators (Jetson only).")
    parser.add_argument("--skip-sam2", action="store_true", help="Skip downloading the SAM 2.1 checkpoint.")
    parser.add_argument("--skip-ros-setup", action="store_true", help="Skip ROS 2 apt installation and ROS Python checks.")
    args, realsense_args = parser.parse_known_args()

    if IS_LINUX and not args.skip_ros_setup:
        ensure_ros2_installation()
    elif args.skip_ros_setup:
        log("Skipping automatic ROS 2 installation and verification (--skip-ros-setup)")

    ensure_venv_and_reexec()
    warn_if_system_site_packages()
    requirements, constraints = platform_dependency_files()

    # PIP_CONSTRAINT is inherited by child setup scripts. Use platform-matched constraints only.
    if constraints is not None:
        os.environ["PIP_CONSTRAINT"] = str(constraints)
    else:
        os.environ.pop("PIP_CONSTRAINT", None)

    if not args.skip_realsense:
        if not REALSENSE_SETUP.is_file():
            sys.exit(f"RealSense setup script not found: {REALSENSE_SETUP}")
        log(f"RealSense L515 wrapper setup ({SYSTEM})")
        try:
            run([PY, REALSENSE_SETUP, *realsense_args])
        except subprocess.CalledProcessError as exc:
            if IS_WINDOWS:
                sys.exit(
                    "The RealSense wrapper setup failed on Windows. Its native backend must be "
                    "built as a Windows DLL with a supported Windows compiler. If you only need "
                    "Contact-GraspNet on Windows, rerun with --skip-realsense.\n"
                    f"Underlying exit code: {exc.returncode}"
                )
            raise

    if AARCH64_LINUX:
        prepare_jetson()

    log(f"Installing Python dependencies from {requirements.name}")
    pip("install", "-r", requirements)


    if AARCH64_LINUX:
        # tf-keras otherwise attempts to install PyPI TensorFlow over the Jetson-specific wheel.
        log("Installing legacy Keras (tf-keras) for tf.compat.v1.layers")
        pip("install", "--no-deps", "tf-keras>=2.21,<2.22")
        if not args.skip_tf_ops:
            build_contact_graspnet_tfops()
    elif IS_WINDOWS:
        log("Skipping Jetson-only TensorFlow CUDA-op compilation on Windows")
        print(
            "Windows note: this installer does not compile the Contact-GraspNet PointNet2 .so ops. "
            "Those are Linux shared objects; Windows would need a separate native Windows/CUDA "
            "implementation or a WSL2/Linux environment."
        )

    if IS_LINUX and not args.skip_ros_setup:
        verify_ros2_python()

    return 0 if args.skip_verify else verify()


if __name__ == "__main__":
    sys.exit(main())

