"""
frame_source.py — unified frame access for video proxy and still sequences.
"""

from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np

from intermediate import _FRAME_PATTERN, load_manifest


class FrameSource(ABC):
    @abstractmethod
    def frame_count(self) -> int:
        pass

    @abstractmethod
    def read_bgr(self, index: int) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class VideoFrameSource(FrameSource):
    """Frame-accurate access via MJPEG proxy (or any OpenCV-readable video)."""

    def __init__(self, video_path: str):
        self._path = video_path
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        self._count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def frame_count(self) -> int:
        return self._count

    def read_bgr(self, index: int) -> Optional[np.ndarray]:
        if index < 0 or index >= self._count:
            return None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class SequenceFrameSource(FrameSource):
    """Indexed access to cached intermediate TIFF sequence."""

    def __init__(self, intermediate_dir: str, preview_8bit: bool = True):
        self._dir = intermediate_dir
        self._preview_8bit = preview_8bit
        manifest = load_manifest(intermediate_dir)
        self._count = manifest["frame_count"]
        self._manifest = manifest

    @property
    def manifest(self) -> dict:
        return self._manifest

    def frame_count(self) -> int:
        return self._count

    def _path_for(self, index: int) -> str:
        import os
        return os.path.join(self._dir, _FRAME_PATTERN.format(index + 1))

    def read_bgr(self, index: int) -> Optional[np.ndarray]:
        if index < 0 or index >= self._count:
            return None
        frame = cv2.imread(self._path_for(index), cv2.IMREAD_UNCHANGED)
        if frame is None:
            return None
        if self._preview_8bit and frame.dtype == np.uint16:
            frame = (frame / 256).astype(np.uint8)
        return frame

    def read_bgr_full(self, index: int) -> Optional[np.ndarray]:
        """16-bit BGR without downconversion."""
        if index < 0 or index >= self._count:
            return None
        return cv2.imread(self._path_for(index), cv2.IMREAD_UNCHANGED)

    def close(self) -> None:
        pass
