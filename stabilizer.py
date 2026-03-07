import cv2
import numpy as np
from typing import Tuple


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
    frame_cx, frame_cy = fw / 2, fh / 2
    ref_cx, ref_cy = reference_center

    tx = -dx + (frame_cx - ref_cx)
    ty = -dy + (frame_cy - ref_cy)

    M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
    return cv2.warpAffine(frame_bgr, M, (fw, fh),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
