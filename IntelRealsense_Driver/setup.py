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
    """Return True only when this interpreter is inside an isolated environment.

    Do not trust VIRTUAL_ENV alone: it can be left set in a shell that is running
    the system interpreter. Compare it with sys.prefix instead.
    """
    if getattr(sys, "real_prefix", None) is not None or sys.prefix != sys.base_prefix:
        return True

    current_prefix = Path(sys.prefix).resolve()
    for variable in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        value = os.environ.get(variable)
        if value and Path(value).resolve() == current_prefix:
            return True
    return False


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
    python = sys.executable
    log("Installing Python dependencies into the current environment")
    print(f"    Python executable: {python}")
    print(f"    Environment prefix: {sys.prefix}")
    run([python, "-m", "pip", "--version"])
    packages = ["numpy", "opencv-python", "matplotlib"]
    if system == "Linux":
        packages.extend(["cmake", "ninja"])
    run([python, "-m", "pip", "install", "--upgrade", *packages])


def linux_package_manager() -> tuple[str, list[str]] | None:
    if command_exists("apt-get"):
        return "apt", ["build-essential", "libusb-1.0-0-dev", "libudev-dev", "libssl-dev", "pkg-config"]
    if command_exists("dnf"):
        return "dnf", ["gcc-c++", "make", "libusb1-devel", "systemd-devel", "openssl-devel", "pkgconf-pkg-config"]
    if command_exists("yum"):
        return "yum", ["gcc-c++", "make", "libusb1-devel", "systemd-devel", "openssl-devel", "pkgconfig"]
    if command_exists("pacman"):
        return "pacman", ["base-devel", "libusb", "systemd", "openssl", "pkgconf"]
    if command_exists("zypper"):
        return "zypper", ["gcc-c++", "make", "libusb-1_0-devel", "libudev-devel", "libopenssl-devel", "pkg-config"]
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def windows_sdk_runtime_dirs() -> list[Path]:
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
    info.fMask = 0x00000040
    info.lpVerb = "runas"
    info.lpFile = str(installer)
    info.lpParameters = f'/LOG="{install_log}"'
    info.lpDirectory = str(installer.parent)
    info.nShow = 1

    shell32.ShellExecuteExW.argtypes = [ctypes.POINTER(SHELLEXECUTEINFOW)]
    shell32.ShellExecuteExW.restype = ctypes.c_int
    if not shell32.ShellExecuteExW(ctypes.byref(info)):
        error = ctypes.get_last_error()
        if error == 1223:
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
        wait_result = kernel32.WaitForSingleObject(process, 0xFFFFFFFF)
        if wait_result != 0:
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
            reason = "The installer did not complete successfully."
            if result_code == 2:
                reason = "The installer was cancelled before installation started."
            elif result_code == 5:
                reason = "The installer was cancelled during installation."
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
            if directory_handle is not None:
                directory_handle.close()
    except OSError as exc:
        raise SetupError(f"Could not load the prebuilt SDK DLL: {exc}") from exc

    import_probe = subprocess.run(
        [sys.executable, "-c", "from l515 import L515; print('Python import OK:', L515)"],
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
        udev_ready = False
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
            udev_ready = subprocess.run(
                ["pkg-config", "--exists", "libudev"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode == 0
        if compiler_ready and pkg_config_ready and usb_ready and openssl_ready and udev_ready:
            return None

        package_info = linux_package_manager()
        if package_info is None:
            raise SetupError(
                "Could not identify a supported Linux package manager. Install a C++17-capable "
                "compiler, make, pkg-config, libusb-1.0/libudev development headers, and OpenSSL headers, "
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


def patch_rsutils_sources(source_dir: Path) -> None:
    """Safely patch rsutils headers/sources to include <cstdint> for modern GCC toolchains."""
    version_h = source_dir / "third-party" / "rsutils" / "include" / "rsutils" / "version.h"
    version_cpp = source_dir / "third-party" / "rsutils" / "src" / "version.cpp"

    if version_h.is_file():
        content = version_h.read_text(encoding="utf-8", errors="ignore")
        if "#include <cstdint>" not in content:
            if "#include <string>" in content:
                content = content.replace("#include <string>", "#include <string>\n#include <cstdint>", 1)
                version_h.write_text(content, encoding="utf-8")
                log("Patched version.h to include <cstdint>")

    if version_cpp.is_file():
        content = version_cpp.read_text(encoding="utf-8", errors="ignore")
        if "#include <cstdint>" not in content:
            if '#include "version.h"' in content:
                content = content.replace('#include "version.h"', '#include "version.h"\n#include <cstdint>', 1)
                version_cpp.write_text(content, encoding="utf-8")
                log("Patched version.cpp to include <cstdint>")


def cmake_executable() -> str:
    python_dir = Path(sys.executable).parent
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


def build_parallel_jobs() -> int:
    override = os.environ.get("L515_BUILD_JOBS")
    if override:
        try:
            jobs = int(override)
        except ValueError as exc:
            raise SetupError("L515_BUILD_JOBS must be a positive integer.") from exc
        if jobs < 1:
            raise SetupError("L515_BUILD_JOBS must be a positive integer.")
        return jobs

    machine = platform.machine().lower()
    if machine in {"aarch64", "arm64", "armv8l"}:
        return 1
    return max(1, min(os.cpu_count() or 2, 4))


def run_build_with_log(
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(f'"{x}"' if " " in x else x for x in command)
    print(f"    $ {printable}")
    print(f"    Build log: {log_path}")

    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            raise SetupError(f"Could not start the native build command: {exc}") from exc

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()

    if return_code != 0:
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = "\n".join(lines[-70:])
        except OSError:
            tail = "(Could not read the build log.)"
        raise SetupError(
            f"Native CMake build failed with exit code {return_code}.\n"
            f"Full build log: {log_path}\n\n"
            f"Last build-output lines:\n{tail}"
        )


def configure_and_build(
        project_root: Path,
        source_dir: Path,
        force: bool,
) -> Path:
    build_dir = project_root / "build"
    if force and build_dir.exists():
        log("Removing previous native build directory")
        shutil.rmtree(build_dir)

    patch_rsutils_sources(source_dir)

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
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
    ]
    if system == "Linux":
        configure.append("-DFORCE_RSUSB_BACKEND=ON")

    log("Configuring the native L515 backend")
    run(configure, cwd=project_root, env=build_env)

    jobs = build_parallel_jobs()
    build = [cmake, "--build", str(build_dir), "--parallel", str(jobs)]
    build_log = project_root / ".cache" / "native_build.log"

    log(f"Building the native L515 backend with {jobs} parallel job(s)")
    run_build_with_log(
        build,
        cwd=project_root,
        env=build_env,
        log_path=build_log,
    )

    candidates = [
        build_dir / "libl515_native.so",
        build_dir / "Release" / "libl515_native.so",
        build_dir / "native" / "libl515_native.so",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    for found in build_dir.glob("**/libl515_native.so"):
        if found.is_file():
            return found

    raise SetupError(
        f"Native build completed, but could not find libl515_native.so in {build_dir}."
    )


def install_package_editable(project_root: Path) -> None:
    """Install the l515 package in editable mode so it can be imported anywhere in the environment."""
    log("Installing l515 package into the virtual environment in editable mode")


python_dir = project_root / "python"

# Create a minimal setup.py inside python/ if it doesn't exist so pip can install it
py_setup = python_dir / "setup.py"
if not py_setup.is_file():
    py_setup.write_text(
        'from setuptools import setup, find_packages\n\n'
        'setup(\n'
        '    name="l515",\n'
        '    version="1.0.0",\n'
        '    packages=find_packages(),\n'
        '    include_package_data=True,\n'
        ')\n',
        encoding="utf-8"
    )

run([sys.executable, "-m", "pip", "install", "-e", str(python_dir)])


def main() -> int:
    python_version_ok()
    args = parse_args()
    system = current_os()

    project_root = Path(__file__).resolve().parent

    require_existing_environment()
    install_python_dependencies(system)
    ensure_system_prerequisites(system, args.skip_system_deps)

    if system == "Windows":
        runtime_dir = install_windows_sdk(project_root, args.force)
        smoke_test_windows_runtime(project_root, runtime_dir)
    elif system == "Linux":
        source_dir = obtain_librealsense(project_root, args.source_dir, args.force)
        native_lib = configure_and_build(project_root, source_dir, args.force)
        destination = project_root / "python" / "native_runtime"
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(native_lib, destination / "libl515_native.so")
        log(f"Installed native backend to {destination / 'libl515_native.so'}")

        if args.test_camera:
            log("Running camera smoke test")
            run([sys.executable, "examples/test_l515.py"], cwd=project_root / "python")

    # Automatically install l515 globally inside the active virtual environment
    install_package_editable(project_root)

    log("Setup completed successfully!")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SetupError as exc:
        print(f"\nSETUP ERROR: {exc}", file=sys.stderr)
        sys.exit(1)