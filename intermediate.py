"""
intermediate.py — cached 16-bit RGB TIFF sequence from RAW folders.
"""

import json
import os
from typing import Callable, Dict, List, Optional

import cv2

from image_sequence import list_frames, read_raw_frame, source_mtime

_MANIFEST = "manifest.json"
_FRAME_PATTERN = "{:05d}.tif"


def intermediate_dir_for(sequence_dir: str) -> str:
    """Return cache path: ``<sequence_dir>/cache/<stem>_decoded/``."""
    sequence_dir = os.path.abspath(sequence_dir)
    stem = os.path.basename(sequence_dir.rstrip(os.sep))
    return os.path.join(sequence_dir, "cache", stem + "_decoded")


def _frame_path(intermediate_dir: str, index: int) -> str:
    return os.path.join(intermediate_dir, _FRAME_PATTERN.format(index + 1))


def _intermediate_is_fresh(sequence_dir: str, intermediate_dir: str) -> bool:
    if not os.path.isdir(intermediate_dir):
        return False
    manifest_path = os.path.join(intermediate_dir, _MANIFEST)
    if not os.path.isfile(manifest_path):
        return False
    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    expected = len(list_frames(sequence_dir))
    if manifest.get("frame_count") != expected:
        return False
    src_mtime = source_mtime(sequence_dir)
    cache_mtime = os.path.getmtime(intermediate_dir)
    return cache_mtime >= src_mtime


def load_manifest(intermediate_dir: str) -> Dict:
    path = os.path.join(intermediate_dir, _MANIFEST)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_intermediate(
    sequence_dir: str,
    intermediate_dir: str,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Decode all RAW members to numbered 16-bit TIFF files + manifest."""
    frames = list_frames(sequence_dir)
    os.makedirs(intermediate_dir, exist_ok=True)

    manifest_frames: List[Dict[str, str]] = []
    total = len(frames)

    for i, src_path in enumerate(frames):
        bgr = read_raw_frame(src_path)
        out_path = _frame_path(intermediate_dir, i)
        cv2.imwrite(out_path, bgr)
        manifest_frames.append({
            "index": i,
            "source": src_path,
            "basename": os.path.basename(src_path),
        })
        if progress_cb and i % 5 == 0:
            progress_cb(i, total)

    manifest = {
        "frame_count": total,
        "sequence_dir": os.path.abspath(sequence_dir),
        "frames": manifest_frames,
    }
    with open(os.path.join(intermediate_dir, _MANIFEST), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    os.utime(intermediate_dir, None)

    if progress_cb:
        progress_cb(total, total)


def ensure_intermediate(
    sequence_dir: str,
    intermediate_dir: Optional[str] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> str:
    """Return path to fresh intermediate cache for *sequence_dir*."""
    if intermediate_dir is None:
        intermediate_dir = intermediate_dir_for(sequence_dir)

    if not _intermediate_is_fresh(sequence_dir, intermediate_dir):
        build_intermediate(sequence_dir, intermediate_dir, progress_cb=progress_cb)

    return intermediate_dir
