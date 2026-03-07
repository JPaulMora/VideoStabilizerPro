import cv2
import numpy as np
from typing import Dict, Tuple, Callable, Optional
from stabilizer import stabilize_frame


def extract_crop(frame_rgb, px, py, crop_w, crop_h, use_center):
    """Extract a crop_w × crop_h region from frame_rgb around (px, py)."""
    fh, fw = frame_rgb.shape[:2]
    src_x1 = int(px) - crop_w // 2 if use_center else int(px)
    src_y1 = int(py) - crop_h // 2 if use_center else int(py)
    canvas = np.full((crop_h, crop_w, 3), 0x11, dtype=np.uint8)
    vid_x1 = max(0, src_x1); vid_x2 = min(fw, src_x1 + crop_w)
    vid_y1 = max(0, src_y1); vid_y2 = min(fh, src_y1 + crop_h)
    if vid_x2 > vid_x1 and vid_y2 > vid_y1:
        dx1 = vid_x1 - src_x1; dy1 = vid_y1 - src_y1
        canvas[dy1:dy1+(vid_y2-vid_y1), dx1:dx1+(vid_x2-vid_x1)] = \
            frame_rgb[vid_y1:vid_y2, vid_x1:vid_x2]
    return canvas


def export_video(
    src_path: str,
    out_path: str,
    points: Dict[int, Tuple[float, float]],
    crop_w: int,
    crop_h: int,
    use_center: bool,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> None:
    """
    Read src_path frame-by-frame, extract the tracked crop, write to out_path.
    progress_cb(current_frame, total_frames) called every 30 frames.
    """
    cap = cv2.VideoCapture(src_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fallback = (fw / 2, fh_vid / 2)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (crop_w, crop_h))

    for i in range(total):
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        px, py = points.get(i, fallback)
        crop = extract_crop(frame_rgb, px, py, crop_w, crop_h, use_center)
        writer.write(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        if progress_cb and i % 30 == 0:
            progress_cb(i, total)

    writer.release()
    cap.release()


def export_stabilized(
    src_path: str,
    out_path: str,
    points: Dict[int, Tuple[float, float]],
    reference_center: Tuple[float, float],
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Re-reads src_path, stabilizes each frame using stored points, writes out_path."""
    cap = cv2.VideoCapture(src_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ref_cx, ref_cy = reference_center
    fallback = reference_center

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (fw, fh))

    for i in range(total):
        ret, frame_bgr = cap.read()
        if not ret:
            break
        cx, cy = points.get(i, fallback)
        dx, dy = cx - ref_cx, cy - ref_cy
        writer.write(stabilize_frame(frame_bgr, dx, dy, reference_center))
        if progress_cb and i % 30 == 0:
            progress_cb(i, total)

    writer.release()
    cap.release()
