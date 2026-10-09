# Modern-Python, Cross-Platform RealSense L515 Wrapper

## What this project does

This project provides the Intel RealSense **L515** through modern Python without `pyrealsense2`.

- **Windows:** the setup downloads and verifies Intel's prebuilt RealSense SDK v2.54.2 installer, installs/copies its x64 `realsense2.dll` runtime, and calls the RealSense C API directly through Python's built-in `ctypes`. No Visual Studio, C++ compiler, CMake build, or .NET runtime is required.
- **Linux:** the existing C++ C-ABI wrapper is compiled against librealsense v2.54.2 and called through `ctypes`. This Linux path remains source-built and requires a C++ toolchain and USB development dependencies.

The project pins SDK **v2.54.2**, the legacy release containing L515 support. The Windows SDK installer is SHA-256 checked before use. Neither platform requires `pyrealsense2`, so the Python layer does not depend on the `pyrealsense2` wheel's CPython version limit.

## One-command setup

Copy the entire project directory to the target computer and run the setup script from the project directory **after activating the Python virtual environment you already intend to use**.

The setup script does **not** create, delete, activate, or replace a virtual environment. All Python packages and the import path file are installed into the environment associated with the Python interpreter that launches `setup.py`.

### Windows

Activate your existing environment, then run:

```powershell
python setup.py
```

Or run `setup.bat`. Setup installs Python dependencies into that active environment, downloads the official `Intel.RealSense.SDK-WIN10-2.54.2.5684.exe` if the SDK runtime is not already installed, verifies its SHA-256, and launches the SDK installer visibly with Windows UAC elevation. Complete the installer wizard; setup then copies the x64 runtime DLLs to `python/native_runtime/`.

**No Visual Studio or C++ compiler is used on Windows.** The installer is deliberately visible because silent installation can return exit code 2 when it is cancelled before installation starts or when a confirmation is declined. If installation is cancelled, rerun `python setup.py` and complete the wizard, or run `.cache/Intel.RealSense.SDK-WIN10-2.54.2.5684.exe` manually as administrator.

### Linux

Activate your existing environment, then run:

```bash
chmod +x setup.sh
./setup.sh
```

The script attempts to install the compiler, `libusb`, OpenSSL development headers, and `pkg-config` using the detected Linux package manager (`apt`, `dnf`, `yum`, `pacman`, or `zypper`). It also installs the RealSense udev rules when the v2.54.2 source tree contains them.

### Use an existing librealsense checkout (Linux only)

The setup script normally downloads the pinned source automatically, so **Git is not required**. To use your own checkout, point the setup script at a verified v2.54.2 source tree:

```bash
python setup.py --source-dir /path/to/librealsense-2.54.2
```

## What setup installs

The **active Python environment** receives:

```text
Windows: numpy, opencv-python, matplotlib
Linux:   numpy, opencv-python, matplotlib, cmake, ninja
```

It does **not** install `pyrealsense2`.

The setup process also creates a `.pth` file inside the **active environment** so `python/l515.py` is importable as a normal module.

Because the environment is expected to already be active, there is no additional activation step after setup. Your application can use the class in the same style as before:

```python
from L515 import L515

camera = L515()
camera.take_picture()

depth = camera.get_depth()
color = camera.get_color()
K = camera.get_k()

camera.close()
```

A context manager is also supported:

```python
from L515 import L515

with L515() as camera:
    camera.take_picture()
    depth = camera.get_depth()
    color = camera.get_color()
```

The example can be run with:

```bash
python examples/test_l515.py
```

To have the setup script test the physical camera as the final step:

```bash
python setup.py --test-camera
```

## Existing API

The class is intentionally compatible with the original public API:

```text
take_picture()
get_depth()
get_color()
get_k()
show_picture()
show_RGB_DepthMap()
close()
```

The original `get_k()` behavior is preserved: it returns the raw **depth-stream** intrinsic matrix.

The class also provides:

```python
get_aligned_k()
```

`take_picture()` aligns depth to the color stream. Therefore `get_aligned_k()` returns the color-camera intrinsics that correspond to the pixels in the aligned depth image.

## Depth and color data

`get_depth()` returns a NumPy array of depth values in **meters**.

`get_color()` returns the latest color image as a NumPy array in **BGR** format, matching the original OpenCV-oriented code.

The Windows backend copies frame data from the SDK-owned buffers into NumPy arrays before releasing frame handles. The Linux C-ABI backend likewise copies data into stable buffers before returning it to Python.

## Setup options

The setup script is idempotent. On Windows, rerunning it reuses the project-local runtime unless `--force` is passed. On Linux, it reuses the cached librealsense source and existing native build when possible. It always operates on the environment associated with the current `python` interpreter.

The setup script requires an already-active virtual/isolated environment. If no environment is active, setup stops rather than modifying the system Python.

Useful options:

```text
--skip-system-deps
    Skip OS package-manager installation. Useful on managed computers where
    system packages are installed separately by IT.

--source-dir PATH
    Linux only: use an existing librealsense 2.54.2 source tree.

--force
    Windows: recopy the installed SDK DLLs. Linux: refresh the source and rebuild.

--test-camera
    Instantiate the L515 and capture one frame after setup.
```

Examples:

```bash
python setup.py --force
python setup.py --test-camera
```

## Files

```text
l515_cross_platform/
├── CMakeLists.txt
├── README.md
├── setup.py
├── setup.bat
├── setup.sh
├── native/
│   └── l515_native.cpp
├── python/
│   ├── l515.py
│   └── l515_windows.py
└── examples/
    └── test_l515.py
```

Working directories created as needed:

```text
Windows: .cache/Intel.RealSense.SDK-WIN10-2.54.2.5684.exe
         python/native_runtime/*.dll
Linux:   .cache/                    Download cache for librealsense 2.54.2
         third_party/librealsense/ Pinned native SDK source
         build/                    CMake build tree
         python/libl515_native.so  Native C-ABI library
         python/librealsense2.so*  SDK runtime libraries
```

## Platform notes

### Windows

Windows uses the prebuilt SDK C API directly with `ctypes`. The project does not compile native code or invoke Visual Studio on this platform. DLLs are kept together in `python/native_runtime/`, and `os.add_dll_directory()` makes their dependencies visible to the Windows loader.

### Linux

Linux still compiles the C-ABI wrapper against the pinned librealsense source. Setup attempts to install compiler/USB prerequisites and RealSense udev rules when possible. If USB permissions are denied, reconnect the camera after installing udev rules instead of running the entire Python application as root.

macOS is not supported by this project version.

## Why two native paths are used

The Python class has one public API, but the native integration differs by platform:

- Windows calls the prebuilt SDK's stable C API through `ctypes`, avoiding a local C++ build and the need for Visual Studio.
- Linux builds a small C-ABI adapter against the pinned librealsense C++ API.

Both paths avoid the Python-specific `pyrealsense2` extension and preserve the same Python-facing `L515` class.

## AI Use Statement

AI tools, including OpenAI ChatGPT and other assistants, were used to investigate the `pyrealsense2`/Python-version compatibility issue; propose the platform-specific native architectures; generate and revise the Python wrapper, Windows `ctypes` C-API backend, Linux C++ C-ABI backend, setup scripts, error handling, and documentation; and diagnose setup/build errors. The Windows setup logic was generated with assistance to download and checksum-verify the official SDK installer and stage its runtime DLLs.

The code and documentation were iteratively reviewed and modified for the intended application. The project author, **Ryan Poliquin**, is responsible for integration, final review, and hardware validation. The project has been syntax-checked, but real camera capture still needs to be tested on the target Windows and Linux machines. AI assistance does not imply that every SDK, OS, driver, or hardware combination has been validated.

## Original project / attribution

Written by Ryan Poliquin.

The native implementation is based on the Intel RealSense **librealsense v2.54.2** SDK and is deliberately built without the official Python bindings.

Relevant sources:

- https://github.com/realsenseai/librealsense
- https://github.com/realsenseai/librealsense/releases/tag/v2.54.2
- https://pypi.org/project/pyrealsense2/2.54.2.5684/

## Important engineering note

This project solves the **Python ABI/version problem**. It does not create a new L515 SDK. The native backend still depends on the legacy librealsense code needed by the L515, so firmware, USB hardware, operating-system drivers, and native SDK compatibility can still affect operation.


