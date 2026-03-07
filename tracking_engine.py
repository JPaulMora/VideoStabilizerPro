from dataclasses import dataclass
from typing import Tuple, Optional
import cv2
import numpy as np


class TrackingLostError(Exception):
    pass


@dataclass
class TrackResult:
    center: Tuple[float, float]          # (cx, cy) video pixel coords
    score: float                          # TM_CCOEFF_NORMED score 0–1
    search_rect: Tuple[int, int, int, int]  # (x, y, w, h) video pixel coords
    match_rect: Tuple[int, int, int, int]   # (x, y, w, h) video pixel coords


class TrackingEngine:
    def __init__(self, search_padding=50, confidence_threshold=0.7,
                 template_update_interval=10):
        self.search_padding = search_padding
        self.confidence_threshold = confidence_threshold
        self.template_update_interval = template_update_interval
        self.template: Optional[np.ndarray] = None
        self.reference_center: Optional[Tuple] = None
        self.previous_center: Optional[Tuple] = None
        self._frame_count = 0

    @property
    def has_template(self) -> bool:
        return self.template is not None

    def set_template(self, frame_bgr: np.ndarray, roi: Tuple[int, int, int, int]):
        fh, fw = frame_bgr.shape[:2]
        x, y, w, h = roi
        # Clamp to frame bounds — negative coords from letterbox drags cause
        # numpy wrap-around slicing that produces empty or wrong-sized templates
        x = max(0, min(x, fw - 1))
        y = max(0, min(y, fh - 1))
        w = min(w, fw - x)
        h = min(h, fh - y)
        if w <= 0 or h <= 0:
            raise ValueError("ROI is outside frame bounds")
        self.template = frame_bgr[y:y + h, x:x + w].copy()
        cx, cy = x + w / 2, y + h / 2
        self.reference_center = (cx, cy)
        self.previous_center = (cx, cy)
        self._frame_count = 0

    def reset_position(self):
        """Call before re-running tracking from the start."""
        self.previous_center = self.reference_center
        self._frame_count = 0

    def track_frame(self, frame_bgr: np.ndarray) -> TrackResult:
        if self.template is None:
            raise TrackingLostError("No template set")

        fh, fw = frame_bgr.shape[:2]
        th, tw = self.template.shape[:2]
        pcx, pcy = self.previous_center
        pad = self.search_padding

        sx1 = max(0, int(pcx - tw // 2 - pad))
        sy1 = max(0, int(pcy - th // 2 - pad))
        sx2 = min(fw, int(pcx + tw // 2 + pad))
        sy2 = min(fh, int(pcy + th // 2 + pad))
        search_region = frame_bgr[sy1:sy2, sx1:sx2]

        # template or search region must be non-empty and search >= template
        if th == 0 or tw == 0:
            raise TrackingLostError("Template has zero dimension")
        if search_region.shape[0] < th or search_region.shape[1] < tw:
            raise TrackingLostError("Search region smaller than template")

        result = cv2.matchTemplate(search_region, self.template, cv2.TM_CCOEFF_NORMED)
        _, score, _, max_loc = cv2.minMaxLoc(result)

        if score < self.confidence_threshold:
            raise TrackingLostError(f"Score {score:.2f} below threshold")

        match_x = sx1 + max_loc[0]
        match_y = sy1 + max_loc[1]
        center = (match_x + tw / 2, match_y + th / 2)

        self._frame_count += 1
        if self._frame_count % self.template_update_interval == 0 and score > 0.9:
            self.template = frame_bgr[match_y:match_y + th, match_x:match_x + tw].copy()

        self.previous_center = center

        search_rect = (sx1, sy1, sx2 - sx1, sy2 - sy1)
        match_rect = (match_x, match_y, tw, th)
        return TrackResult(center=center, score=score,
                           search_rect=search_rect, match_rect=match_rect)
