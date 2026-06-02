import cv2
import numpy as np
from typing import Tuple

from raw_transform import stabilization_translation


def stabilize_frame(
    frame_bgr: np.ndarray,
    dx: float,
    dy: float,
    reference_center: Tuple[float, float],
) -> np.ndarray:
    """
    Translate frame so the tracked feature stays at the frame center.
    dx/dy = displacement from reference_center this frame.
    Revealed borders are filled black.
    """
    fh, fw = frame_bgr.shape[:2]
    tx, ty = stabilization_translation(dx, dy, reference_center, (fw, fh))
    border = 0
    M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
    return cv2.warpAffine(frame_bgr, M, (fw, fh),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def stabilize_frame_int(
    frame_bgr: np.ndarray,
    tx: int,
    ty: int,
) -> np.ndarray:
    """Integer-pixel translation (used for Bayer replay parity checks)."""
    from raw_transform import translate_bayer
    if frame_bgr.ndim == 2:
        return translate_bayer(frame_bgr, tx, ty)
    if frame_bgr.dtype == np.uint16:
        channels = [translate_bayer(frame_bgr[:, :, c], tx, ty) for c in range(3)]
        return np.stack(channels, axis=2)
    return translate_bayer(frame_bgr, tx, ty)
