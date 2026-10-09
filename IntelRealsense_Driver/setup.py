#!/usr/bin/env python3
"""
Setup for the Intel RealSense L515 Python wrapper.

Run from an already-active virtual environment. This script never creates,
deletes, or activates a virtual environment.

Windows uses Intel's prebuilt librealsense 2.54.2 SDK DLL through ctypes, so
Visual Studio, CMake, and a C++ compiler are not required. Linux continues to
build the project's C-ABI shim against librealsense 2.54.2 with the system C++
toolchain. Both paths install the Python package dependencies into the current
environment and add python/ to its import path.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path


LIBREALSENSE_VERSION = "2.54.2"
LIBREALSENSE_URL = (
    "https://github.com/realsenseai/librealsense/"
    f"archive/refs/tags/v{LIBREALSENSE_VERSION}.tar.gz"
)
PROJECT_MIN_PYTHON = (3, 9)
NATIVE_LIBRARY_NAMES = {
    # Windows uses the prebuilt SDK C API directly through ctypes.
    "Windows": ["realsense2.dll"],
    # Linux retains the existing C-ABI shim built against librealsense 2.54.2.
    "Linux": ["libl515_native.so"],
}

WINDOWS_SDK_URL = (
    "https://github.com/IntelRealSense/librealsense/releases/download/v2.54.2/"
    "Intel.RealSense.SDK-WIN10-2.54.2.5684.exe"
)
WINDOWS_SDK_SHA256 = "b6bded035e2aef5263aa983faefbdd4c350e84f238e6bbbaaab5ac4de58bf0b7"


class SetupError(RuntimeError):
    pass


def log(message: str) -> None:
    print(f"\n==> {message}")


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    printable = " ".join(f'"{x}"' if " " in x else x for x in command)
    print(f"    $ {printable}")
    subprocess.run(command, cwd=cwd, env=env, check=True)


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def current_os() -> str:
    system = platform.system()
    if system not in NATIVE_LIBRARY_NAMES:
        raise SetupError(f"Unsupported operating system: {system}")
    return system


def is_virtual_environment() -> bool:
    """Return True for standard venv/virtualenv and Conda environments."""
    return (
        getattr(sys, "real_prefix", None) is not None
        or sys.prefix != sys.base_prefix
        or os.environ.get("VIRTUAL_ENV") is not None
        or os.environ.get("CONDA_PREFIX") is not None
    )


def require_existing_environment() -> None:
    """Refuse to modify the system Python; use the interpreter that launched setup.py."""
    if is_virtual_environment():
        return

    raise SetupError(
        "No active Python virtual/isolated environment was detected.\n\n"
        "Activate the existing environment you want to use, then run setup again.\n"
        "Examples:\n"
        "  Windows:    .venv\\Scripts\\activate   (or activate your existing env)\n"
        "  Linux:      source .venv/bin/activate   (or activate your existing env)\n\n"
        f"Current interpreter: {sys.executable}\n"
        f"Python: {sys.version.split()[0]}\n\n"
        "This setup script intentionally does not create or modify a new virtual environment."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up the native RealSense L515 wrapper.")
    parser.add_argument(
        "--skip-system-deps",
        action="store_true",
        help="Do not attempt OS package-manager installs for compiler/USB prerequisites.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Linux only: use an existing librealsense source tree instead of downloading v2.54.2.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refresh/re-copy the Windows SDK runtime, or re-download and rebuild the Linux native backend.",
    )
    parser.add_argument(
        "--test-camera",
        action="store_true",
        help="Open an L515 after setup and capture one frame.",
    )
    return parser.parse_args()


def python_version_ok() -> None:
    if sys.version_info < PROJECT_MIN_PYTHON:
        raise SetupError(
            f"Python {PROJECT_MIN_PYTHON[0]}.{PROJECT_MIN_PYTHON[1]}+ is required. "
            f"You are running Python {sys.version.split()[0]}."
        )

    if sys.version_info < (3, 12):
        print(
            "WARNING: Python 3.9-3.11 is supported by this project, but the primary goal "
            "of this architecture is Python 3.12+ compatibility."
        )


def sudo_command(command: list[str]) -> None:
    if os.name == "nt":
        run(command)
        return
    if os.geteuid() == 0:
        run(command)
        return
    if not command_exists("sudo"):
        raise SetupError(
            "This step requires administrator privileges, but sudo is not available.\n"
            f"Run manually as root: {' '.join(command)}"
        )
    run(["sudo", *command])


def install_python_dependencies(system: str) -> None:
    """Install packages into the exact interpreter running setup.py."""
    python = Path(sys.executable).resolve()
    log("Installing Python dependencies into the current environment")
    print(f"    Python: {python}")
    run([str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    packages = ["numpy", "opencv-python", "matplotlib"]
    if system == "Linux":
        packages.extend(["cmake", "ninja"])
    run([str(python), "-m", "pip", "install", "--upgrade", *packages])


def linux_package_manager() -> tuple[str, list[str]] | None:
    if command_exists("apt-get"):
        return "apt", ["build-essential", "libusb-1.0-0-dev", "libssl-dev", "pkg-config"]
    if command_exists("dnf"):
        return "dnf", ["gcc-c++", "make", "libusb1-devel", "openssl-devel", "pkgconf-pkg-config"]
    if command_exists("yum"):
        return "yum", ["gcc-c++", "make", "libusb1-devel", "openssl-devel", "pkgconfig"]
    if command_exists("pacman"):
        return "pacman", ["base-devel", "libusb", "openssl", "pkgconf"]
    if command_exists("zypper"):
        return "zypper", ["gcc-c++", "make", "libusb-1_0-devel", "libopenssl-devel", "pkg-config"]
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def windows_sdk_runtime_dirs() -> list[Path]:
    """Likely install locations for the official Intel RealSense SDK x64 DLLs."""
    roots = [
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")),
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
    ]
    dirs: list[Path] = []
    for root in roots:
        dirs.extend([
            root / "Intel RealSense SDK 2.0" / "bin" / "x64",
            root / "Intel RealSense SDK 2.0" / "bin",
        ])
    return dirs


def windows_sdk_api_version(directory: Path) -> int | None:
    """Return the native DLL's API version, or None if it cannot be loaded."""
    dll_path = directory / "realsense2.dll"
    if not dll_path.is_file():
        return None
    dll_dir_handle = None
    try:
        if hasattr(os, "add_dll_directory"):
            dll_dir_handle = os.add_dll_directory(str(directory))
        dll = ctypes.CDLL(str(dll_path))
        get_version = dll.rs2_get_api_version
        get_version.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        get_version.restype = ctypes.c_int
        error = ctypes.c_void_p()
        version = int(get_version(ctypes.byref(error)))
        if error.value:
            dll.rs2_free_error.argtypes = [ctypes.c_void_p]
            dll.rs2_free_error.restype = None
            dll.rs2_free_error(error)
            return None
        return version
    except (OSError, AttributeError, ValueError):
        return None
    finally:
        if dll_dir_handle is not None:
            dll_dir_handle.close()


def is_pinned_windows_sdk(directory: Path) -> bool:
    return windows_sdk_api_version(directory) == 25402


def find_windows_sdk_runtime() -> Path | None:
    for directory in windows_sdk_runtime_dirs():
        if is_pinned_windows_sdk(directory):
            return directory
    # Some installer builds use a slightly different install root; scan only
    # the expected Intel SDK directory, not the entire drive.
    for root in [
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Intel RealSense SDK 2.0",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Intel RealSense SDK 2.0",
    ]:
        if root.is_dir():
            for match in root.glob("**/realsense2.dll"):
                if match.is_file() and is_pinned_windows_sdk(match.parent):
                    return match.parent
    return None


def run_windows_installer_elevated(installer: Path, install_log: Path) -> int:
    """Run the SDK installer visibly with UAC elevation, and wait for it to finish.

    The previous silent launch returned Inno Setup exit code 2 on some systems.
    Using ShellExecuteEx with the ``runas`` verb lets Windows display the UAC
    prompt, while the visible installer exposes any version/downgrade prompts
    instead of silently cancelling before installation starts.
    """
    if os.name != "nt":
        raise SetupError("Elevated Windows installer can only run on Windows.")

    class SHELLEXECUTEINFOW(ctypes.Structure):
        _fields_ = [
            ("cbSize", ctypes.c_ulong),
            ("fMask", ctypes.c_ulong),
            ("hwnd", ctypes.c_void_p),
            ("lpVerb", ctypes.c_wchar_p),
            ("lpFile", ctypes.c_wchar_p),
            ("lpParameters", ctypes.c_wchar_p),
            ("lpDirectory", ctypes.c_wchar_p),
            ("nShow", ctypes.c_int),
            ("hInstApp", ctypes.c_void_p),
            ("lpIDList", ctypes.c_void_p),
            ("lpClass", ctypes.c_wchar_p),
            ("hkeyClass", ctypes.c_void_p),
            ("dwHotKey", ctypes.c_ulong),
            ("hIconOrMonitor", ctypes.c_void_p),
            ("hProcess", ctypes.c_void_p),
        ]

    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    info = SHELLEXECUTEINFOW()
    info.cbSize = ctypes.sizeof(info)
    info.fMask = 0x00000040  # SEE_MASK_NOCLOSEPROCESS
    info.lpVerb = "runas"
    info.lpFile = str(installer)
    # Inno Setup expects quotes around the log filename, not around /LOG= itself.
    info.lpParameters = f'/LOG="{install_log}"'
    info.lpDirectory = str(installer.parent)
    info.nShow = 1  # SW_SHOWNORMAL

    shell32.ShellExecuteExW.argtypes = [ctypes.POINTER(SHELLEXECUTEINFOW)]
    shell32.ShellExecuteExW.restype = ctypes.c_int
    if not shell32.ShellExecuteExW(ctypes.byref(info)):
        error = ctypes.get_last_error()
        if error == 1223:  # ERROR_CANCELLED (the user declined the UAC prompt)
            raise SetupError(
                "Windows elevation was cancelled. Rerun setup.py and approve the UAC prompt "
                "to install the RealSense SDK runtime."
            )
        raise OSError(error, "Could not start the elevated RealSense SDK installer")

    process = info.hProcess
    if not process:
        raise SetupError("Windows did not return a handle for the RealSense SDK installer.")
    try:
        kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        kernel32.WaitForSingleObject.restype = ctypes.c_ulong
        wait_result = kernel32.WaitForSingleObject(process, 0xFFFFFFFF)  # INFINITE
        if wait_result != 0:  # WAIT_OBJECT_0
            raise SetupError("Could not wait for the RealSense SDK installer to finish.")
        exit_code = ctypes.c_ulong()
        kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel32.GetExitCodeProcess.restype = ctypes.c_int
        if not kernel32.GetExitCodeProcess(process, ctypes.byref(exit_code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(exit_code.value)
    finally:
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int
        kernel32.CloseHandle(process)


def install_windows_sdk(project_root: Path, force: bool) -> Path:
    """Install/copy the official prebuilt SDK; no Visual Studio compiler required."""
    destination = project_root / "python" / "native_runtime"
    dll = destination / "realsense2.dll"
    if dll.is_file() and not force:
        if is_pinned_windows_sdk(destination):
            log("Using the existing prebuilt Windows RealSense 2.54.2 runtime")
            return destination
        log("Ignoring a project-local RealSense DLL that is not API version 2.54.2")
        shutil.rmtree(destination, ignore_errors=True)

    runtime_source = find_windows_sdk_runtime()
    if runtime_source is None:
        cache_dir = project_root / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        installer = cache_dir / "Intel.RealSense.SDK-WIN10-2.54.2.5684.exe"
        if not installer.is_file() or _sha256(installer).lower() != WINDOWS_SDK_SHA256:
            if installer.exists():
                installer.unlink()
            log("Downloading the official Intel RealSense SDK 2.54.2 Windows installer")
            print(f"    URL: {WINDOWS_SDK_URL}")
            try:
                urllib.request.urlretrieve(WINDOWS_SDK_URL, installer)
            except Exception as exc:
                raise SetupError(
                    "Could not download the prebuilt RealSense SDK. Check internet access, "
                    "or download the installer manually from the official v2.54.2 release."
                ) from exc
        actual = _sha256(installer).lower()
        if actual != WINDOWS_SDK_SHA256:
            raise SetupError(
                "The downloaded RealSense SDK installer failed SHA-256 verification. "
                f"Expected {WINDOWS_SDK_SHA256}, got {actual}."
            )

        install_log = cache_dir / "realsense_sdk_install.log"
        log("Starting the official RealSense SDK installer with Windows elevation")
        print("    The visible installer will open. Approve the UAC prompt and complete the wizard.")
        try:
            result_code = run_windows_installer_elevated(installer, install_log)
        except OSError as exc:
            raise SetupError(
                "Could not launch the RealSense SDK installer. Run the cached installer manually "
                f"as administrator: {installer}\nDetails: {exc}"
            ) from exc
        if result_code not in (0, 3010, 1641):
            if result_code == 2:
                reason = "The installer was cancelled before installation started."
            elif result_code == 5:
                reason = "The installer was cancelled during installation."
            else:
                reason = "The installer did not complete successfully."
            raise SetupError(
                f"{reason} Exit code: {result_code}. Installer log: {install_log}. "
                "Run setup.py again and complete the installer, or run the cached installer "
                "manually as administrator."
            )
        runtime_source = find_windows_sdk_runtime()

    if runtime_source is None or not is_pinned_windows_sdk(runtime_source):
        raise SetupError(
            "The installer completed, but the x64 DLL does not report librealsense API 2.54.2. "
            "Uninstall conflicting SDK versions if needed, then install Intel RealSense SDK v2.54.2."
        )

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    # Copy all sibling DLLs because realsense2.dll may rely on SDK-shipped native
    # dependencies. Only the x64 directory is considered.
    copied = []
    for dependency in runtime_source.glob("*.dll"):
        shutil.copy2(dependency, destination / dependency.name)
        copied.append(dependency.name)
    if not (destination / "realsense2.dll").is_file():
        raise SetupError("Failed to copy realsense2.dll into python/native_runtime.")
    log("Installed the prebuilt x64 RealSense runtime")
    print(f"    Source: {runtime_source}")
    print(f"    Copied DLLs: {', '.join(sorted(copied))}")
    return destination


def smoke_test_windows_runtime(project_root: Path, runtime_dir: Path) -> None:
    log("Checking the prebuilt Windows RealSense C API")
    dll_path = runtime_dir / "realsense2.dll"
    if not dll_path.is_file():
        raise SetupError(f"RealSense runtime missing: {dll_path}")
    try:
        directory_handle = os.add_dll_directory(str(runtime_dir)) if hasattr(os, "add_dll_directory") else None
        try:
            dll = ctypes.CDLL(str(dll_path))
            required = [
                # Validate the full C API surface used by l515_windows.py so a
                # mismatched SDK fails during setup rather than at camera startup.
                "rs2_get_api_version",
                "rs2_get_error_message",
                "rs2_free_error",
                "rs2_create_context",
                "rs2_delete_context",
                "rs2_create_pipeline",
                "rs2_delete_pipeline",
                "rs2_pipeline_stop",
                "rs2_create_config",
                "rs2_delete_config",
                "rs2_config_enable_stream",
                "rs2_pipeline_start_with_config",
                "rs2_pipeline_wait_for_frames",
                "rs2_create_align",
                "rs2_create_frame_queue",
                "rs2_start_processing_queue",
                "rs2_process_frame",
                "rs2_wait_for_frame",
                "rs2_embedded_frames_count",
                "rs2_extract_frame",
                "rs2_get_frame_stream_profile",
                "rs2_get_stream_profile_data",
                "rs2_get_video_stream_intrinsics",
                "rs2_depth_frame_get_units",
                "rs2_get_frame_data",
                "rs2_get_frame_width",
                "rs2_get_frame_height",
                "rs2_get_frame_stride_in_bytes",
                "rs2_release_frame",
                "rs2_delete_processing_block",
                "rs2_delete_frame_queue",
                "rs2_delete_pipeline_profile",
            ]
            missing = [name for name in required if not hasattr(dll, name)]
            if missing:
                raise SetupError("The SDK DLL lacks required C API exports: " + ", ".join(missing))
            get_version = dll.rs2_get_api_version
            get_version.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
            get_version.restype = ctypes.c_int
            error = ctypes.c_void_p()
            api_version = int(get_version(ctypes.byref(error)))
            if error.value:
                try:
                    message = dll.rs2_get_error_message(error)
                    details = message.decode("utf-8", errors="replace") if message else "unknown SDK error"
                finally:
                    dll.rs2_free_error(error)
                raise SetupError("Could not query the SDK API version: " + details)
            if api_version != 25402:
                raise SetupError(
                    f"Expected librealsense API 2.54.2 (25402), but DLL reports {api_version}."
                )
        finally:
            # The actual class retains its own DLL directory handle while loaded.
            if directory_handle is not None:
                directory_handle.close()
    except OSError as exc:
        raise SetupError(f"Could not load the prebuilt SDK DLL: {exc}") from exc

    import_probe = subprocess.run(
        [str(Path(sys.executable).resolve()), "-c", "from l515 import L515; print('Python import OK:', L515)"],
        capture_output=True,
        text=True,
        cwd=project_root / "python",
        check=False,
    )
    if import_probe.returncode != 0:
        raise SetupError("The Python wrapper could not be imported.\n" + import_probe.stderr.strip())
    print(import_probe.stdout.strip())


def ensure_system_prerequisites(system: str, skip: bool) -> Path | None:
    if skip:
        log("Skipping OS-level prerequisite installation by request")
        return None

    if system == "Linux":
        compiler_ready = command_exists("g++") and command_exists("make")
        pkg_config_ready = command_exists("pkg-config")
        usb_ready = False
        openssl_ready = False
        if pkg_config_ready:
            usb_ready = subprocess.run(
                ["pkg-config", "--exists", "libusb-1.0"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode == 0
            openssl_ready = subprocess.run(
                ["pkg-config", "--exists", "openssl"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode == 0
        if compiler_ready and pkg_config_ready and usb_ready and openssl_ready:
            return None

        package_info = linux_package_manager()
        if package_info is None:
            raise SetupError(
                "Could not identify a supported Linux package manager. Install a C++17-capable "
                "compiler, make, pkg-config, libusb-1.0 development headers, and OpenSSL headers, "
                "then rerun setup.py."
            )

        manager, packages = package_info
        log(f"Installing Linux build/USB prerequisites using {manager}")
        if manager == "apt":
            sudo_command(["apt-get", "update"])
            sudo_command(["apt-get", "install", "-y", *packages])
        elif manager == "dnf":
            sudo_command(["dnf", "install", "-y", *packages])
        elif manager == "yum":
            sudo_command(["yum", "install", "-y", *packages])
        elif manager == "pacman":
            sudo_command(["pacman", "-Sy", "--needed", "--noconfirm", *packages])
        elif manager == "zypper":
            sudo_command(["zypper", "--non-interactive", "install", *packages])
        return None

    raise SetupError(f"Unsupported operating system: {system}")


def source_has_expected_version(source_dir: Path) -> bool:
    """Verify the source tree is actually librealsense v2.54.2.

    librealsense v2.54.2 does not contain an rs_version.h file. Its version is
    defined in include/librealsense2/rs.h as separate API major/minor/patch
    macros. We parse those values directly instead of searching for the literal
    string "2.54.2".
    """
    rs_header = source_dir / "include" / "librealsense2" / "rs.h"
    if not rs_header.is_file():
        return False

    text = rs_header.read_text(encoding="utf-8", errors="ignore")

    import re

    components = {}
    for component in ("MAJOR", "MINOR", "PATCH"):
        match = re.search(
            rf"^\s*#define\s+RS2_API_{component}_VERSION\s+(\d+)\s*$",
            text,
            flags=re.MULTILINE,
        )
        if not match:
            return False
        components[component] = int(match.group(1))

    actual_version = (
        f"{components['MAJOR']}.{components['MINOR']}.{components['PATCH']}"
    )
    return actual_version == LIBREALSENSE_VERSION


def obtain_librealsense(project_root: Path, supplied_source: Path | None, force: bool) -> Path:
    target = supplied_source.resolve() if supplied_source else project_root / "third_party" / "librealsense"

    if supplied_source is not None:
        if not target.is_dir() or not (target / "CMakeLists.txt").is_file():
            raise SetupError(f"Invalid librealsense source directory: {target}")
        if not source_has_expected_version(target):
            raise SetupError(
                f"The supplied librealsense source does not appear to be v{LIBREALSENSE_VERSION}. "
                "This project intentionally requires 2.54.2 for L515 support."
            )
        return target

    if target.is_dir() and source_has_expected_version(target) and not force:
        log(f"Using existing librealsense v{LIBREALSENSE_VERSION} source tree")
        return target

    cache_dir = project_root / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / f"librealsense-{LIBREALSENSE_VERSION}.tar.gz"

    if force and archive.exists():
        archive.unlink()

    if not archive.exists():
        log(f"Downloading librealsense v{LIBREALSENSE_VERSION}")
        print(f"    URL: {LIBREALSENSE_URL}")
        try:
            urllib.request.urlretrieve(LIBREALSENSE_URL, archive)
        except Exception as exc:
            raise SetupError(
                "Could not download librealsense. Check your internet connection or provide "
                "an existing v2.54.2 source tree with --source-dir."
            ) from exc

    if target.exists():
        log("Replacing the existing librealsense source tree")
        shutil.rmtree(target)

    with tempfile.TemporaryDirectory(prefix="l515_extract_") as tmp:
        tmp_path = Path(tmp)
        with tarfile.open(archive, "r:gz") as tar:
            root = tmp_path.resolve()
            for member in tar.getmembers():
                member_path = (root / member.name).resolve()
                if os.path.commonpath([str(root), str(member_path)]) != str(root):
                    raise SetupError("The downloaded librealsense archive contains an unsafe path.")
            tar.extractall(tmp_path)
        extracted = tmp_path / f"librealsense-{LIBREALSENSE_VERSION}"
        if not extracted.is_dir():
            candidates = [p for p in tmp_path.iterdir() if p.is_dir()]
            if len(candidates) != 1:
                raise SetupError("Downloaded librealsense archive has an unexpected layout.")
            extracted = candidates[0]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(extracted), str(target))

    if not source_has_expected_version(target):
        raise SetupError(
            f"Downloaded librealsense source did not verify as v{LIBREALSENSE_VERSION}."
        )

    (target / ".l515_required_version").write_text(LIBREALSENSE_VERSION, encoding="utf-8")
    return target


def cmake_executable() -> str:
    """Find CMake installed in the current environment, then fall back to PATH."""
    python_dir = Path(sys.executable).resolve().parent
    candidates = [python_dir / "cmake.exe", python_dir / "cmake"]

    scripts_dir = os.environ.get("PATH", "").split(os.pathsep)
    for script_dir in scripts_dir:
        path = Path(script_dir)
        candidates.extend([path / "cmake.exe", path / "cmake"])

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    found = shutil.which("cmake")
    if found:
        return found
    raise SetupError("CMake was not found after installing the Python cmake package.")


def configure_and_build(
    project_root: Path,
    source_dir: Path,
    force: bool,
) -> Path:
    build_dir = project_root / "build"
    if force and build_dir.exists():
        log("Removing previous native build directory")
        shutil.rmtree(build_dir)

    cmake = cmake_executable()
    system = current_os()
    source_dir = source_dir.resolve()
    build_env = os.environ.copy()

    configure = [
        cmake,
        "-S",
        str(project_root),
        "-B",
        str(build_dir),
        f"-DLIBREALSENSE_SOURCE_DIR={source_dir}",
        "-DL515_BUILD_WITH_BUNDLED_LIBREALSENSE=ON",
        "-DBUILD_PYTHON_BINDINGS=OFF",
        "-DCMAKE_BUILD_TYPE=Release",
    ]

    log("Configuring the native L515 backend")
    run(configure, cwd=project_root, env=build_env)

    build = [cmake, "--build", str(build_dir), "--parallel"]

    log("Building the native L515 backend")
    run(build, cwd=project_root, env=build_env)

    names = NATIVE_LIBRARY_NAMES[system]
    for name in names:
        direct = build_dir / "lib" / name
        if direct.is_file():
            return direct

    for name in names:
        matches = list(build_dir.rglob(name))
        if matches:
            return matches[0]

    raise SetupError(
        "Native build completed, but l515_native was not found under the build directory.\n"
        f"Expected one of: {', '.join(names)}"
    )


def copy_native_runtime(project_root: Path, native_library: Path) -> Path:
    destination_dir = project_root / "python"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / native_library.name

    log(f"Installing native runtime next to python/l515.py: {destination.name}")
    shutil.copy2(native_library, destination)

    build_lib = native_library.parent
    runtime_prefixes = ("realsense2", "librealsense2")
    copied_runtime = []
    for item in build_lib.iterdir():
        lower = item.name.lower()
        if any(lower.startswith(prefix.lower()) for prefix in runtime_prefixes):
            target = destination_dir / item.name
            if item.is_file() or item.is_symlink():
                try:
                    if target.exists() or target.is_symlink():
                        target.unlink()
                    shutil.copy2(item, target, follow_symlinks=False)
                    copied_runtime.append(item.name)
                except OSError:
                    if item.is_file():
                        shutil.copy2(item.resolve(), target)
                        copied_runtime.append(item.name)

    if copied_runtime:
        print(f"    Copied RealSense runtime: {', '.join(copied_runtime)}")
    else:
        print("    WARNING: Could not find a colocated librealsense runtime library.")

    return destination


def find_linux_udev_rules(project_root: Path) -> Path | None:
    candidates = [
        project_root / "third_party" / "librealsense" / "config" / "99-realsense-libusb.rules",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def install_linux_udev_rules(project_root: Path) -> None:
    if current_os() != "Linux":
        return

    rules = find_linux_udev_rules(project_root)
    if rules is None:
        print("WARNING: librealsense udev rules file was not found; USB permissions were not changed.")
        return

    destination = Path("/etc/udev/rules.d/99-realsense-libusb.rules")
    try:
        log("Installing Linux RealSense udev rules")
        sudo_command(["cp", str(rules), str(destination)])
        if command_exists("udevadm"):
            sudo_command(["udevadm", "control", "--reload-rules"])
            sudo_command(["udevadm", "trigger"])
    except SetupError as exc:
        print(f"WARNING: Could not install udev rules automatically: {exc}")


def write_project_path_file(project_root: Path) -> Path:
    python = Path(sys.executable).resolve()
    probe = subprocess.run(
        [str(python), "-c", "import site; print(site.getsitepackages()[0])"],
        capture_output=True,
        text=True,
        check=True,
    )
    site_packages = Path(probe.stdout.strip().splitlines()[-1]).resolve()
    site_packages.mkdir(parents=True, exist_ok=True)
    pth = site_packages / "l515_project.pth"
    pth.write_text(str((project_root / "python").resolve()) + os.linesep, encoding="utf-8")
    return pth


def smoke_test_native_library(native_library: Path) -> None:
    log("Running Linux native-library smoke test")
    dll = ctypes.CDLL(str(native_library))

    required_symbols = ["l515_create", "l515_start", "l515_capture", "l515_destroy"]
    missing = [name for name in required_symbols if not hasattr(dll, name)]
    if missing:
        raise SetupError(
            "The native library loaded, but required C-ABI symbols are missing: "
            + ", ".join(missing)
        )

    import_probe = subprocess.run(
        [str(Path(sys.executable).resolve()), "-c", "from l515 import L515; print('Python import OK:', L515)"],
        capture_output=True,
        text=True,
        cwd=native_library.parent.parent,
        check=False,
    )
    if import_probe.returncode != 0:
        raise SetupError(
            "The Python wrapper could not be imported.\n" + import_probe.stderr.strip()
        )
    print(import_probe.stdout.strip())


def camera_test(project_root: Path) -> None:
    log("Testing the physical L515")
    run(
        [
            str(Path(sys.executable).resolve()),
            "-c",
            (
                "from l515 import L515; "
                "cam=L515(); "
                "print('Depth K:', cam.get_k()); "
                "cam.take_picture(); "
                "print('Frame capture OK:', cam.get_depth().shape, cam.get_color().shape); "
                "cam.close()"
            ),
        ],
        cwd=project_root,
    )


def print_success(project_root: Path) -> None:
    system = current_os()
    python = Path(sys.executable).resolve()
    print("\n" + "=" * 72)
    print("L515 setup completed successfully.")
    print("=" * 72)
    print(f"Project:       {project_root}")
    print(f"Python:        {python}")
    print(f"OS:            {system}")
    print(f"Python module: {project_root / 'python' / 'l515.py'}")
    if system == "Windows":
        print(f"SDK runtime:   {project_root / 'python' / 'native_runtime' / 'realsense2.dll'}")
        print("Build tools:   no Visual Studio/C++ compiler required")
    else:
        print(f"Native lib:    {project_root / 'python' / NATIVE_LIBRARY_NAMES[system][0]}")
    print("Environment:   existing environment; no virtual environment was created")

    print("\nYour application can use the class as before:")
    print("    from l515 import L515")
    print("    camera = L515()")
    print("    camera.take_picture()")
    print("    depth = camera.get_depth()")
    print("    color = camera.get_color()")
    print("    K = camera.get_k()")
    print("    camera.close()")
    print("\nNo pyrealsense2 package is installed or required.")


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    python_version_ok()
    require_existing_environment()
    system = current_os()

    log(f"Preparing {system} setup using the existing Python environment")
    print(f"    Python: {Path(sys.executable).resolve()}")
    print(f"    Prefix: {Path(sys.prefix).resolve()}")

    if system == "Windows":
        install_python_dependencies(system)
        runtime_dir = install_windows_sdk(project_root, args.force)
        write_project_path_file(project_root)
        smoke_test_windows_runtime(project_root, runtime_dir)
        if args.test_camera:
            camera_test(project_root)
        print_success(project_root)
        return 0

    ensure_system_prerequisites(system, args.skip_system_deps)
    install_python_dependencies(system)

    source_dir = obtain_librealsense(project_root, args.source_dir, args.force)
    native_library = configure_and_build(project_root, source_dir, args.force)
    installed_native = copy_native_runtime(project_root, native_library)

    write_project_path_file(project_root)
    install_linux_udev_rules(project_root)
    smoke_test_native_library(installed_native)

    if args.test_camera:
        camera_test(project_root)

    print_success(project_root)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SetupError as exc:
        print(f"\nSETUP ERROR:\n{exc}", file=sys.stderr)
        raise SystemExit(1)
    except subprocess.CalledProcessError as exc:
        print(
            "\nSETUP ERROR: a required command failed with exit code "
            f"{exc.returncode}: {exc.cmd}",
            file=sys.stderr,
        )
        raise SystemExit(exc.returncode or 1)
