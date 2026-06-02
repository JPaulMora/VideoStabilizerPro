"""
image_sequence.py — discover and decode sequential RAW/still folders.
"""

import os
import re
from typing import List, Optional

import cv2
import numpy as np

_RAW_EXTENSIONS = (".dng", ".tif", ".tiff")
_SEQUENCE_PATTERN = re.compile(r"(\d+)\s*$")


def _basename_lower(path: str) -> str:
    return os.path.basename(path).lower()


def is_raw_sequence_file(path: str) -> bool:
    """True if *path* looks like a member of a RAW still sequence."""
    name = _basename_lower(path)
    if name.endswith(".raw.dng"):
        return True
    return any(name.endswith(ext) for ext in _RAW_EXTENSIONS)


def frame_sort_key(path: str) -> tuple:
    """Sort by trailing numeric token, then full path for stability."""
    base = os.path.basename(path)
    m = _SEQUENCE_PATTERN.search(os.path.splitext(base)[0])
    if m is None:
        m = _SEQUENCE_PATTERN.search(base.replace(".raw", ""))
    num = int(m.group(1)) if m else 0
    return (num, path)


def list_frames(sequence_dir: str) -> List[str]:
    """
    Return sorted absolute paths to sequence members in *sequence_dir*.
    Raises ValueError if the directory has no matching files.
    """
    if not os.path.isdir(sequence_dir):
        raise ValueError(f"Not a directory: {sequence_dir}")

    paths = []
    for name in os.listdir(sequence_dir):
        full = os.path.join(sequence_dir, name)
        if os.path.isfile(full) and is_raw_sequence_file(full):
            paths.append(os.path.abspath(full))

    if not paths:
        raise ValueError(f"No RAW sequence files in: {sequence_dir}")

    return sorted(paths, key=frame_sort_key)


def is_image_sequence_dir(path: str) -> bool:
    try:
        return len(list_frames(path)) > 0
    except ValueError:
        return False


def source_mtime(sequence_dir: str) -> float:
    """Latest mtime among sequence members (0 if empty)."""
    try:
        frames = list_frames(sequence_dir)
    except ValueError:
        return 0.0
    return max(os.path.getmtime(p) for p in frames)


def read_raw_frame(path: str) -> np.ndarray:
    """
    Demosaic *path* to 16-bit BGR via rawpy.
    Falls back to cv2.imread for ordinary TIFF/PNG if rawpy fails.
    """
    try:
        import rawpy
    except ImportError as e:
        raise RuntimeError("rawpy is required for DNG decode; pip install rawpy") from e

    try:
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                output_bps=16,
            )
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except rawpy.LibRawFileUnsupportedError:
        pass

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def read_raw_preview(path: str, max_side: Optional[int] = None) -> np.ndarray:
    """8-bit BGR preview for tracking/UI."""
    frame = read_raw_frame(path)
    if frame.dtype == np.uint16:
        frame = (frame / 256).astype(np.uint8)
    if max_side and max(frame.shape[:2]) > max_side:
        h, w = frame.shape[:2]
        scale = max_side / max(h, w)
        frame = cv2.resize(
            frame,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return frame


def export_basename(path: str, suffix: str = ".tif") -> str:
    """Output filename preserving stem, normalised extension."""
    stem, _ = os.path.splitext(os.path.basename(path))
    if stem.lower().endswith(".raw"):
        stem = stem[:-4]
    return stem + suffix
