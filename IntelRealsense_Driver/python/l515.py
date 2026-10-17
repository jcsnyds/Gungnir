"""
Cross-platform Intel RealSense L515 wrapper for modern Python.

Architecture
------------
This class intentionally does NOT import pyrealsense2.

Windows loads the prebuilt librealsense 2.54.2 C API directly through ctypes,
so no Visual Studio compiler or custom DLL compilation is required. Linux uses
the existing small C-ABI wrapper built against librealsense 2.54.2.

The public API is intentionally close to the original L515 class:
    take_picture()
    get_depth()
    get_color()
    get_k()
    show_picture()
    show_RGB_DepthMap()
    close()

Python requirements:
    Python 3.9+
    numpy
    opencv-python      # only required by show_picture/close window handling
    matplotlib         # only required by show_RGB_DepthMap

Native requirements:
    librealsense 2.54.2 with L515 support, built as part of the native backend.
"""

from __future__ import annotations

import ctypes
import os
import platform
from pathlib import Path
from typing import Optional, Sequence
import numpy as np
import cv2


_ERROR_BUFFER_SIZE = 4096


class L515Error(RuntimeError):
    """Raised when the native L515 backend reports an error."""


class L515:
    """Intel RealSense L515 camera interface without a Python RealSense binding."""

    def __init__(
        self,
        clipping_distance: float = 9.0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        library_path: Optional[os.PathLike[str] | str] = None,
    ) -> None:
        self.pipeline = None  # compatibility with the old class
        self.running = False
        self.depth_image: Optional[np.ndarray] = None
        self.color_image: Optional[np.ndarray] = None
        self.clipping_distance = float(clipping_distance)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.depth_scale = 0.0
        self._native = None
        self._dll = None
        self._error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER_SIZE)
        self._dll_directory_handle = None
        self._windows_backend = None

        # Windows uses the prebuilt librealsense C API directly, bypassing the
        # custom native shim that must be compiled on Linux.
        if platform.system() == "Windows":
            try:
                from l515_windows import WindowsRealSenseBackend

                self._windows_backend = WindowsRealSenseBackend(
                    width=self.width,
                    height=self.height,
                    fps=self.fps,
                    library_path=library_path,
                )
                self.depth_scale = float(self._windows_backend.depth_scale)
                self.running = True
                self.pipeline = self  # preserve the legacy attribute
                return
            except Exception as exc:
                if self._windows_backend is not None:
                    self._windows_backend.close()
                self._windows_backend = None
                raise L515Error(f"Could not initialize the Windows RealSense backend: {exc}") from exc

        self._library_path = self._resolve_library_path(library_path)
        # Python 3.8+ uses the Windows DLL directory API for ctypes loading.
        # Keep the handle alive so librealsense2.dll, which sits beside
        # l515_native.dll, can be resolved by the Windows loader.
        if platform.system() == "Windows" and hasattr(os, "add_dll_directory"):
            self._dll_directory_handle = os.add_dll_directory(
                str(self._library_path.parent)
            )
        self._dll = ctypes.CDLL(str(self._library_path))
        self._configure_abi()

        handle = self._dll.l515_create(
            self.width,
            self.height,
            self.fps,
            self._error_buffer,
            _ERROR_BUFFER_SIZE,
        )
        if not handle:
            raise L515Error(self._native_error("Unable to create native L515 backend."))

        self._native = handle
        try:
            self._check(
                self._dll.l515_start(
                    self._native,
                    self._error_buffer,
                    _ERROR_BUFFER_SIZE,
                )
            )
            self.depth_scale = float(self._dll.l515_depth_scale(self._native))
            self.running = True
            self.pipeline = self  # compatibility placeholder; see note above
        except Exception:
            self.close()
            raise

    # ------------------------------------------------------------------
    # Native ABI setup / loading
    # ------------------------------------------------------------------
    def _configure_abi(self) -> None:
        dll = self._dll
        pvoid = ctypes.c_void_p
        c_char_p = ctypes.POINTER(ctypes.c_char)
        error_args = [pvoid, ctypes.POINTER(ctypes.c_char), ctypes.c_size_t]

        dll.l515_create.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        dll.l515_create.restype = pvoid

        dll.l515_start.argtypes = error_args
        dll.l515_start.restype = ctypes.c_int
        dll.l515_stop.argtypes = error_args
        dll.l515_stop.restype = ctypes.c_int
        dll.l515_capture.argtypes = error_args
        dll.l515_capture.restype = ctypes.c_int

        dll.l515_destroy.argtypes = [pvoid]
        dll.l515_destroy.restype = None

        dll.l515_depth_data.argtypes = [pvoid]
        dll.l515_depth_data.restype = ctypes.POINTER(ctypes.c_uint16)
        dll.l515_color_data.argtypes = [pvoid]
        dll.l515_color_data.restype = ctypes.POINTER(ctypes.c_uint8)

        dll.l515_depth_scale.argtypes = [pvoid]
        dll.l515_depth_scale.restype = ctypes.c_float
        dll.l515_width.argtypes = [pvoid]
        dll.l515_width.restype = ctypes.c_int
        dll.l515_height.argtypes = [pvoid]
        dll.l515_height.restype = ctypes.c_int
        dll.l515_is_running.argtypes = [pvoid]
        dll.l515_is_running.restype = ctypes.c_int

        dll.l515_get_depth_k.argtypes = [
            pvoid,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        dll.l515_get_depth_k.restype = ctypes.c_int

        dll.l515_get_color_k.argtypes = [
            pvoid,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        dll.l515_get_color_k.restype = ctypes.c_int

    @staticmethod
    def _resolve_library_path(library_path: Optional[os.PathLike[str] | str]) -> Path:
        """Find the platform-native l515_native library."""
        system = platform.system()
        if system == "Windows":
            filenames = ["l515_native.dll"]
        elif system == "Darwin":
            filenames = ["libl515_native.dylib", "l515_native.dylib"]
        elif system == "Linux":
            filenames = ["libl515_native.so"]
        else:
            raise L515Error(f"Unsupported operating system: {system}")

        if library_path is not None:
            path = Path(library_path).expanduser().resolve()
            if not path.is_file():
                raise L515Error(f"Native L515 library does not exist: {path}")
            return path

        env_path = os.environ.get("L515_NATIVE_PATH")
        candidates: list[Path] = []
        if env_path:
            candidates.append(Path(env_path).expanduser())

        here = Path(__file__).resolve().parent
        repo_root = here.parent
        cwd = Path.cwd()

        # Typical developer-build locations.
        bases = [
            here,
            repo_root,
            repo_root / "build" / "lib",
            cwd,
            cwd / "build" / "lib",
            cwd / "build" / "Release",
            cwd / "build" / "Debug",
        ]
        for base in bases:
            for filename in filenames:
                candidates.append(base / filename)

        # Handle a CMake multi-config build with the library one level deeper.
        for build_dir in (repo_root / "build", cwd / "build"):
            if build_dir.is_dir():
                for filename in filenames:
                    candidates.extend(build_dir.glob(f"*/{filename}"))

        for candidate in candidates:
            candidate = candidate.expanduser().resolve()
            if candidate.is_file():
                return candidate

        searched = "\n".join(f"  {p}" for p in candidates)
        raise L515Error(
            "Could not find the native L515 library. Build the CMake project "
            "first, copy the resulting library beside l515.py, or set "
            "L515_NATIVE_PATH. Searched:\n" + searched
        )

    def _native_error(self, fallback: str) -> str:
        message = self._error_buffer.value.decode("utf-8", errors="replace").strip()
        return message or fallback

    def _check(self, return_code: int) -> None:
        if return_code != 0:
            raise L515Error(self._native_error("Native L515 call failed."))

    # ------------------------------------------------------------------
    # Camera API
    # ------------------------------------------------------------------
    def capture_image(self) -> bool:
        """Capture one aligned color/depth frame pair."""
        if self._windows_backend is not None:
            if not self.running:
                raise L515Error("L515 is not running.")
            try:
                self.depth_image, self.color_image = self._windows_backend.take_picture()
                self.depth_scale = float(self._windows_backend.depth_scale)
                return True
            except Exception as exc:
                raise L515Error(f"Windows L515 capture failed: {exc}") from exc

        if not self.running or self._native is None:
            raise L515Error("L515 is not running.")

        self._check(
            self._dll.l515_capture(
                self._native,
                self._error_buffer,
                _ERROR_BUFFER_SIZE,
            )
        )

        pixel_count = self.width * self.height
        depth_ptr = self._dll.l515_depth_data(self._native)
        color_ptr = self._dll.l515_color_data(self._native)
        if not depth_ptr or not color_ptr:
            raise L515Error("Native backend returned an empty frame buffer.")

        # Copy the native buffers immediately. The pointers become invalid as
        # soon as another capture replaces the backend's frame buffers.
        depth_raw = np.ctypeslib.as_array(depth_ptr, shape=(pixel_count,)).copy()
        color = np.ctypeslib.as_array(
            color_ptr,
            shape=(pixel_count * 3,),
        ).copy()

        self.depth_image = (depth_raw.astype(np.float32) * self.depth_scale).reshape(
            self.height, self.width
        )
        self.color_image = color.reshape(self.height, self.width, 3)
        return True

    def get_depth(self) -> np.ndarray:
        """Return the most recently captured depth image in meters."""
        if self.depth_image is None:
            raise L515Error("No depth image has been captured yet.")
        return self.depth_image

    def get_color(self) -> np.ndarray:
        """Return the most recently captured BGR color image."""
        if self.color_image is None:
            raise L515Error("No color image has been captured yet.")
        return self.color_image

    def get_pointcloud(self) -> np.ndarray:
        # Define Camera Intrinsics Matrix K
        #     [fx,  0.0, cx],
        #     [0.0, fy,  cy],
        #     [0.0, 0.0, 1.0]
        cam_K = self.get_k()
        fx = cam_K[0, 0]
        fy = cam_K[1, 1]
        cx = cam_K[0, 2]
        cy = cam_K[1, 2]

        # Project 2D Depth map to 3D Point Cloud (X, Y, Z) in meters
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        # Filter valid depth points (in meters)
        valid = (self.depth > MIN_DEPTH) & (self.depth < MAX_DEPTH)

        z = self.depth[valid]
        x = (u[valid] - cx) * z / fx
        y = (v[valid] - cy) * z / fy

        point_cloud = np.stack((x, y, z), axis=-1).astype(np.float32)
        return point_cloud

    def get_RGBD(self) -> np.ndarray:
        """Return the most recently captured RGBD image in meters."""
        rgbd = np.dstack((self.color_image, self.depth_image))
        return rgbd

    def get_k(self) -> np.ndarray:
        """Return the color intrinsic matrix K used by depth aligned to color."""
        if self._windows_backend is not None:
            return self._windows_backend.get_aligned_k()
        return self._get_k(self._dll.l515_get_color_k)

    def _get_k(self, native_function) -> np.ndarray:
        if self._native is None:
            raise L515Error("L515 is not initialized.")
        values = (ctypes.c_float * 9)()
        self._check(
            native_function(
                self._native,
                values,
                self._error_buffer,
                _ERROR_BUFFER_SIZE,
            )
        )
        return np.asarray(values, dtype=np.float32).reshape(3, 3).copy()

    def show_image(self) -> None:
        assert self.depth_image is not None
        assert self.color_image is not None

        grey_color = 153
        depth_image_3d = np.dstack((self.depth_image,) * 3)
        bg_removed = np.where(
            (depth_image_3d > self.clipping_distance)
            | (depth_image_3d <= 0),
            grey_color,
            self.color_image,
        ).astype(np.uint8)

        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(self.depth_image, alpha=100),
            cv2.COLORMAP_JET,
        )
        images = np.hstack((bg_removed, depth_colormap))

        cv2.namedWindow("Align Example", cv2.WINDOW_NORMAL)
        cv2.imshow("Align Example", images)
        cv2.waitKey(1)

    def capture_and_show_image(self) -> None:
        """Capture and display color/depth side-by-side using OpenCV."""

        self.capture_image()
        self.show_image()

    def save_RGB(self):
        np.save("rgb.npy", self.color_image)

    def save_depth(self):
        np.save("depth.npy", self.depth_image)

    def save_RGBD(self):
        rgbd = self.get_RGBD()
        np.save("rgbd.npy", rgbd)

    def save_pointcloud(self):
        pointcloud = self.get_pointcloud()
        np.save("pointcloud.npy", point_cloud)

    def close(self) -> None:
        """Stop streaming and release the native camera handle."""
        if self._windows_backend is not None:
            try:
                self._windows_backend.close()
            finally:
                self._windows_backend = None
                self.pipeline = None
                self.running = False
                try:
                    import cv2
                    cv2.destroyAllWindows()
                except Exception:
                    pass
            return

        try:
            if self._native is not None:
                if self.running:
                    self._dll.l515_stop(
                        self._native,
                        self._error_buffer,
                        _ERROR_BUFFER_SIZE,
                    )
        finally:
            if self._native is not None:
                try:
                    self._dll.l515_destroy(self._native)
                finally:
                    self._native = None

            self.pipeline = None
            self.running = False

            if self._dll_directory_handle is not None:
                try:
                    self._dll_directory_handle.close()
                except Exception:
                    pass
                self._dll_directory_handle = None

            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:
                pass

    def __enter__(self) -> "L515":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
