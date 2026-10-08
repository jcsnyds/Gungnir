"""Windows backend using the prebuilt librealsense C API through ctypes.

No Visual Studio, CMake, C++ compilation, pyrealsense2, or .NET runtime is used.
The official Intel RealSense SDK installer provides realsense2.dll.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional

import numpy as np


# Values from librealsense v2.54.2 rs_types.h.
RS2_API_VERSION = 25402
RS2_STREAM_DEPTH = 1
RS2_STREAM_COLOR = 2
RS2_FORMAT_Z16 = 1
RS2_FORMAT_BGR8 = 6


class _Rs2Intrinsics(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("ppx", ctypes.c_float),
        ("ppy", ctypes.c_float),
        ("fx", ctypes.c_float),
        ("fy", ctypes.c_float),
        ("model", ctypes.c_int),
        ("coeffs", ctypes.c_float * 5),
    ]


class WindowsRealSenseBackend:
    """Direct ctypes bindings for the RealSense C API bundled with SDK 2.54.2."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        library_path: Optional[os.PathLike[str] | str] = None,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self._dll_directory_handle = None
        self._context = None
        self._pipeline = None
        self._config = None
        self._profile = None
        self._align = None
        self._queue = None
        self.depth_scale = 0.001
        self.depth_k = None
        self.color_k = None
        self._closed = False

        dll_path = self._resolve_dll_path(library_path)
        if hasattr(os, "add_dll_directory"):
            self._dll_directory_handle = os.add_dll_directory(str(dll_path.parent))
        try:
            self._dll = ctypes.CDLL(str(dll_path))
            self._configure_functions()
            self._context = self._call("rs2_create_context", RS2_API_VERSION)
            self._pipeline = self._call("rs2_create_pipeline", self._context)
            self._config = self._call("rs2_create_config")
            self._call(
                "rs2_config_enable_stream",
                self._config,
                RS2_STREAM_DEPTH,
                0,
                self.width,
                self.height,
                RS2_FORMAT_Z16,
                self.fps,
            )
            self._call(
                "rs2_config_enable_stream",
                self._config,
                RS2_STREAM_COLOR,
                0,
                self.width,
                self.height,
                RS2_FORMAT_BGR8,
                self.fps,
            )
            self._profile = self._call(
                "rs2_pipeline_start_with_config", self._pipeline, self._config
            )
            self._align = self._call("rs2_create_align", RS2_STREAM_COLOR)
            self._queue = self._call("rs2_create_frame_queue", 1)
            self._call("rs2_start_processing_queue", self._align, self._queue)

            # Warm up auto-exposure and cache both unaligned stream intrinsics.
            for _ in range(5):
                self._capture()
        except Exception:
            self.close()
            raise
        finally:
            if self._profile:
                try:
                    self._fn["rs2_delete_pipeline_profile"](self._profile)
                except Exception:
                    pass
                self._profile = None

    @staticmethod
    def _resolve_dll_path(library_path: Optional[os.PathLike[str] | str]) -> Path:
        here = Path(__file__).resolve().parent
        candidates: list[Path] = []
        if library_path is not None:
            requested = Path(library_path).expanduser().resolve()
            candidates.append(requested if requested.is_dir() else requested.parent)
        env_value = os.environ.get("L515_REALSENSE_DLL")
        if env_value:
            env_path = Path(env_value).expanduser().resolve()
            candidates.append(env_path if env_path.is_dir() else env_path.parent)
        candidates.append(here / "native_runtime")
        for directory in candidates:
            dll_path = directory / "realsense2.dll"
            if dll_path.is_file():
                return dll_path
        searched = "\n".join(f"  {d / 'realsense2.dll'}" for d in candidates)
        raise RuntimeError(
            "Could not find realsense2.dll. Run `python setup.py` to install the "
            "prebuilt Intel RealSense SDK runtime. Searched:\n" + searched
        )

    def _configure_functions(self) -> None:
        p = ctypes.c_void_p
        i = ctypes.c_int
        u = ctypes.c_uint
        f = ctypes.c_float
        errp = ctypes.POINTER(ctypes.c_void_p)
        intrp = ctypes.POINTER(_Rs2Intrinsics)
        self._fn = {}

        def bind(name, argtypes, restype=p):
            func = getattr(self._dll, name)
            func.argtypes = argtypes
            func.restype = restype
            self._fn[name] = func

        bind("rs2_get_error_message", [p], ctypes.c_char_p)
        bind("rs2_free_error", [p], None)
        bind("rs2_create_context", [i, errp])
        bind("rs2_delete_context", [p], None)
        bind("rs2_create_pipeline", [p, errp])
        bind("rs2_delete_pipeline", [p], None)
        bind("rs2_pipeline_stop", [p, errp], None)
        bind("rs2_create_config", [errp])
        bind("rs2_delete_config", [p], None)
        bind("rs2_config_enable_stream", [p, i, i, i, i, i, i, errp], None)
        bind("rs2_pipeline_start_with_config", [p, p, errp])
        bind("rs2_delete_pipeline_profile", [p], None)
        bind("rs2_pipeline_wait_for_frames", [p, u, errp])
        bind("rs2_create_align", [i, errp])
        bind("rs2_create_frame_queue", [i, errp])
        bind("rs2_start_processing_queue", [p, p, errp], None)
        bind("rs2_process_frame", [p, p, errp], None)
        bind("rs2_wait_for_frame", [p, u, errp])
        bind("rs2_embedded_frames_count", [p, errp], i)
        bind("rs2_extract_frame", [p, i, errp])
        bind("rs2_get_frame_stream_profile", [p, errp])
        bind("rs2_get_stream_profile_data", [p, ctypes.POINTER(i), ctypes.POINTER(i), ctypes.POINTER(i), ctypes.POINTER(i), ctypes.POINTER(i), errp], None)
        bind("rs2_get_video_stream_intrinsics", [p, intrp, errp], None)
        bind("rs2_depth_frame_get_units", [p, errp], f)
        bind("rs2_get_frame_data", [p, errp], p)
        bind("rs2_get_frame_width", [p, errp], i)
        bind("rs2_get_frame_height", [p, errp], i)
        bind("rs2_get_frame_stride_in_bytes", [p, errp], i)
        bind("rs2_release_frame", [p], None)
        bind("rs2_delete_processing_block", [p], None)
        bind("rs2_delete_frame_queue", [p], None)

    def _call(self, name: str, *args):
        error = ctypes.c_void_p()
        result = self._fn[name](*args, ctypes.byref(error))
        if error.value:
            try:
                message = self._fn["rs2_get_error_message"](error)
                detail = message.decode("utf-8", errors="replace") if message else "Unknown RealSense error"
            finally:
                self._fn["rs2_free_error"](error)
            raise RuntimeError(f"RealSense call {name} failed: {detail}")
        return result

    def _extract_stream_frames(self, composite) -> dict[int, int]:
        """Extract each child once and identify it from its stream profile."""
        output: dict[int, int] = {}
        frame_count = int(self._call("rs2_embedded_frames_count", composite))
        try:
            for index in range(frame_count):
                frame = self._call("rs2_extract_frame", composite, index)
                stream_profile = self._call("rs2_get_frame_stream_profile", frame)
                stream = ctypes.c_int()
                fmt = ctypes.c_int()
                stream_index = ctypes.c_int()
                unique_id = ctypes.c_int()
                fps = ctypes.c_int()
                self._call(
                    "rs2_get_stream_profile_data",
                    stream_profile,
                    ctypes.byref(stream),
                    ctypes.byref(fmt),
                    ctypes.byref(stream_index),
                    ctypes.byref(unique_id),
                    ctypes.byref(fps),
                )
                if stream.value in (RS2_STREAM_DEPTH, RS2_STREAM_COLOR):
                    if stream.value in output:
                        self._fn["rs2_release_frame"](frame)
                        raise RuntimeError(f"Duplicate RealSense stream {stream.value} in frameset.")
                    output[stream.value] = frame
                else:
                    self._fn["rs2_release_frame"](frame)
            return output
        except Exception:
            for frame in output.values():
                try:
                    self._fn["rs2_release_frame"](frame)
                except Exception:
                    pass
            raise

    def _intrinsics(self, frame) -> np.ndarray:
        stream_profile = self._call("rs2_get_frame_stream_profile", frame)
        intr = _Rs2Intrinsics()
        self._call("rs2_get_video_stream_intrinsics", stream_profile, ctypes.byref(intr))
        return np.array(
            [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    def _copy_frame(self, frame, channels: int, bytes_per_channel: int) -> np.ndarray:
        width = int(self._call("rs2_get_frame_width", frame))
        height = int(self._call("rs2_get_frame_height", frame))
        stride = int(self._call("rs2_get_frame_stride_in_bytes", frame))
        ptr = self._call("rs2_get_frame_data", frame)
        bytes_per_pixel = channels * bytes_per_channel
        row_bytes = width * bytes_per_pixel
        if not ptr or width <= 0 or height <= 0 or stride < row_bytes:
            raise RuntimeError(
                f"Invalid RealSense frame buffer: {width}x{height}, stride={stride}"
            )
        payload = ctypes.string_at(ptr, stride * height)
        rows = np.frombuffer(payload, dtype=np.uint8).reshape(height, stride)
        packed = rows[:, :row_bytes].copy()
        if bytes_per_channel == 2:
            return packed.view(np.uint16).reshape(height, width).astype(np.float32) * self.depth_scale
        return packed.reshape(height, width, channels)

    def _capture(self) -> tuple[np.ndarray, np.ndarray]:
        frames = None
        raw_depth = None
        raw_color = None
        aligned = None
        aligned_depth = None
        aligned_color = None
        try:
            frames = self._call("rs2_pipeline_wait_for_frames", self._pipeline, 5000)
            raw = self._extract_stream_frames(frames)
            raw_depth = raw.get(RS2_STREAM_DEPTH)
            raw_color = raw.get(RS2_STREAM_COLOR)
            if not raw_depth or not raw_color:
                raise RuntimeError("RealSense did not provide both depth and color frames.")

            # Keep the original depth intrinsics, plus the color intrinsics used
            # as the coordinate system for depth aligned to color.
            self.depth_k = self._intrinsics(raw_depth)
            self.color_k = self._intrinsics(raw_color)
            self.depth_scale = float(self._call("rs2_depth_frame_get_units", raw_depth))
            self._fn["rs2_release_frame"](raw_depth)
            raw_depth = None
            self._fn["rs2_release_frame"](raw_color)
            raw_color = None

            # The processing block takes ownership of the frameset. Retrieve its
            # output from the queue after applying depth-to-color alignment.
            self._call("rs2_process_frame", self._align, frames)
            frames = None
            aligned = self._call("rs2_wait_for_frame", self._queue, 5000)
            aligned_frames = self._extract_stream_frames(aligned)
            aligned_depth = aligned_frames.get(RS2_STREAM_DEPTH)
            aligned_color = aligned_frames.get(RS2_STREAM_COLOR)
            if not aligned_depth or not aligned_color:
                raise RuntimeError("RealSense alignment did not produce both output frames.")
            depth = self._copy_frame(aligned_depth, channels=1, bytes_per_channel=2)
            color = self._copy_frame(aligned_color, channels=3, bytes_per_channel=1)
            return depth, color
        finally:
            for frame in (raw_depth, raw_color, aligned_depth, aligned_color, aligned, frames):
                if frame:
                    try:
                        self._fn["rs2_release_frame"](frame)
                    except Exception:
                        pass

    def take_picture(self) -> tuple[np.ndarray, np.ndarray]:
        if self._closed:
            raise RuntimeError("The L515 camera has been closed.")
        return self._capture()

    def get_k(self) -> np.ndarray:
        if self.depth_k is None:
            raise RuntimeError("Depth intrinsics are not available until a frame is captured.")
        return self.depth_k.copy()

    def get_aligned_k(self) -> np.ndarray:
        if self.color_k is None:
            raise RuntimeError("Color intrinsics are not available until a frame is captured.")
        return self.color_k.copy()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        fn = getattr(self, "_fn", {})
        # Stop only if the pipeline was successfully started. Destructors may
        # call close() after partial initialization, so cleanup is best-effort.
        if self._pipeline:
            try:
                self._call("rs2_pipeline_stop", self._pipeline)
            except Exception:
                pass
        if self._profile:
            try:
                fn["rs2_delete_pipeline_profile"](self._profile)
            except Exception:
                pass
            self._profile = None
        if self._pipeline:
            try:
                fn["rs2_delete_pipeline"](self._pipeline)
            except Exception:
                pass
            self._pipeline = None
        if self._config:
            try:
                fn["rs2_delete_config"](self._config)
            except Exception:
                pass
            self._config = None
        if self._align:
            try:
                fn["rs2_delete_processing_block"](self._align)
            except Exception:
                pass
            self._align = None
        if self._queue:
            try:
                fn["rs2_delete_frame_queue"](self._queue)
            except Exception:
                pass
            self._queue = None
        if self._context:
            try:
                fn["rs2_delete_context"](self._context)
            except Exception:
                pass
            self._context = None
        if self._dll_directory_handle is not None:
            try:
                self._dll_directory_handle.close()
            except Exception:
                pass
            self._dll_directory_handle = None
