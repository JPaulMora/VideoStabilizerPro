"""
raw_transform.py — integer Bayer transforms replayed from tracking geometry.
"""

import os
from typing import Dict, Tuple, Union

import cv2
import numpy as np

Transform = Union[
    Tuple[str, int, int],
    Tuple[str, int, int, int, int],
]


def read_bayer(path: str) -> np.ndarray:
    """Return a copy of the visible Bayer mosaic from a RAW/DNG file."""
    import rawpy

    with rawpy.imread(path) as raw:
        return raw.raw_image_visible.copy()


def align_even(x: int, y: int) -> Tuple[int, int]:
    return x & ~1, y & ~1


def translate_bayer(bayer: np.ndarray, tx: int, ty: int) -> np.ndarray:
    """Integer translation; exposed borders filled with 0."""
    h, w = bayer.shape[:2]
    out = np.zeros_like(bayer)
    src_x0 = max(0, -tx)
    src_y0 = max(0, -ty)
    dst_x0 = max(0, tx)
    dst_y0 = max(0, ty)
    copy_w = min(w - src_x0, w - dst_x0)
    copy_h = min(h - src_y0, h - dst_y0)
    if copy_w > 0 and copy_h > 0:
        out[dst_y0:dst_y0 + copy_h, dst_x0:dst_x0 + copy_w] = \
            bayer[src_y0:src_y0 + copy_h, src_x0:src_x0 + copy_w]
    return out


def crop_bayer(bayer: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    x, y = align_even(x, y)
    fh, fw = bayer.shape[:2]
    x = max(0, min(x, fw - 1))
    y = max(0, min(y, fh - 1))
    x2 = min(fw, x + w)
    y2 = min(fh, y + h)
    return bayer[y:y2, x:x2].copy()


def write_bayer_tiff(path: str, bayer: np.ndarray) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if not cv2.imwrite(path, bayer):
        raise RuntimeError(f"Cannot write Bayer TIFF: {path}")


def stabilization_translation(
    dx: float,
    dy: float,
    reference_center: Tuple[float, float],
    frame_size: Tuple[int, int],
    integer: bool = False,
) -> Tuple[float, float]:
    """
    Same translation as stabilizer.stabilize_frame (tx, ty) in warpAffine matrix.
    If *integer* is True, components are rounded for Bayer replay.
    """
    fw, fh = frame_size
    frame_cx, frame_cy = fw / 2, fh / 2
    ref_cx, ref_cy = reference_center
    tx = -dx + (frame_cx - ref_cx)
    ty = -dy + (frame_cy - ref_cy)
    if integer:
        return round(tx), round(ty)
    return tx, ty


def crop_rect(
    px: float,
    py: float,
    crop_w: int,
    crop_h: int,
    use_center: bool,
    frame_size: Tuple[int, int],
    even_align: bool = False,
) -> Tuple[int, int, int, int]:
    """Return (src_x1, src_y1, crop_w, crop_h) matching extract_crop logic."""
    fw, fh = frame_size
    src_x1 = int(px) - crop_w // 2 if use_center else int(px)
    src_y1 = int(py) - crop_h // 2 if use_center else int(py)
    if even_align:
        src_x1, src_y1 = align_even(src_x1, src_y1)
    return src_x1, src_y1, crop_w, crop_h


def frame_transform_from_points(
    frame_idx: int,
    points: Dict[int, Tuple[float, float]],
    reference_center: Tuple[float, float],
    frame_size: Tuple[int, int],
    *,
    crop_w: int = 0,
    crop_h: int = 0,
    use_center: bool = True,
    stabilize: bool = True,
    integer_raw: bool = False,
) -> Transform:
    """
    Compute per-frame geometry shared by RGB and Bayer exporters.

    stabilize=True  -> ("translate", tx, ty)
    stabilize=False -> ("crop", x, y, w, h)
    """
    fw, fh = frame_size
    fallback = reference_center
    cx, cy = points.get(frame_idx, fallback)
    dx, dy = cx - reference_center[0], cy - reference_center[1]

    if stabilize:
        tx, ty = stabilization_translation(
            dx, dy, reference_center, (fw, fh), integer=integer_raw,
        )
        return ("translate", int(tx), int(ty))

    x, y, w, h = crop_rect(
        cx, cy, crop_w, crop_h, use_center, (fw, fh), even_align=integer_raw,
    )
    return ("crop", x, y, w, h)
