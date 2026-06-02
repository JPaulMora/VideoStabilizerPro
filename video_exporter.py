import os
import cv2
import numpy as np
from typing import Dict, Tuple, Callable, Optional, List

from stabilizer import stabilize_frame
from intermediate import load_manifest, _FRAME_PATTERN
from image_sequence import export_basename
from raw_transform import (
    frame_transform_from_points,
    read_bayer,
    translate_bayer,
    crop_bayer,
    write_bayer_tiff,
)


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


def _read_intermediate_bgr(intermediate_dir: str, index: int) -> np.ndarray:
    path = os.path.join(intermediate_dir, _FRAME_PATTERN.format(index + 1))
    frame = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise RuntimeError(f"Cannot read intermediate frame: {path}")
    return frame


def extract_crop_bgr(frame_bgr, px, py, crop_w, crop_h, use_center):
    """BGR crop; preserves uint8/uint16 dtype."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    fill = 0x11 if frame_bgr.dtype == np.uint8 else 0x1111
    fh, fw = frame_rgb.shape[:2]
    src_x1 = int(px) - crop_w // 2 if use_center else int(px)
    src_y1 = int(py) - crop_h // 2 if use_center else int(py)
    canvas = np.full((crop_h, crop_w, 3), fill, dtype=frame_rgb.dtype)
    vid_x1 = max(0, src_x1)
    vid_x2 = min(fw, src_x1 + crop_w)
    vid_y1 = max(0, src_y1)
    vid_y2 = min(fh, src_y1 + crop_h)
    if vid_x2 > vid_x1 and vid_y2 > vid_y1:
        dx1 = vid_x1 - src_x1
        dy1 = vid_y1 - src_y1
        canvas[dy1:dy1 + (vid_y2 - vid_y1), dx1:dx1 + (vid_x2 - vid_x1)] = \
            frame_rgb[vid_y1:vid_y2, vid_x1:vid_x2]
    return cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)


def _write_rgb_tiff(out_path: str, frame_bgr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    if not cv2.imwrite(out_path, frame_bgr):
        raise RuntimeError(f"Cannot write TIFF: {out_path}")


def export_stabilized_sequence(
    intermediate_dir: str,
    out_dir: str,
    points: Dict[int, Tuple[float, float]],
    reference_center: Tuple[float, float],
    *,
    export_rgb: bool = True,
    export_raw: bool = False,
    rgb_subdir: str = "rgb",
    raw_subdir: str = "raw",
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Export full-frame stabilized still sequence."""
    manifest = load_manifest(intermediate_dir)
    total = manifest["frame_count"]
    frames: List[dict] = manifest["frames"]

    sample = _read_intermediate_bgr(intermediate_dir, 0)
    fh, fw = sample.shape[:2]

    rgb_dir = os.path.join(out_dir, rgb_subdir) if export_rgb else None
    raw_dir = os.path.join(out_dir, raw_subdir) if export_raw else None
    if rgb_dir:
        os.makedirs(rgb_dir, exist_ok=True)
    if raw_dir:
        os.makedirs(raw_dir, exist_ok=True)

    ref_cx, ref_cy = reference_center
    fallback = reference_center

    for i in range(total):
        meta = frames[i]
        cx, cy = points.get(i, fallback)
        dx, dy = cx - ref_cx, cy - ref_cy
        stem = export_basename(meta["basename"], ".tif")

        if export_rgb:
            frame = _read_intermediate_bgr(intermediate_dir, i)
            out = stabilize_frame(frame, dx, dy, reference_center)
            _write_rgb_tiff(os.path.join(rgb_dir, stem), out)

        if export_raw:
            tform = frame_transform_from_points(
                i, points, reference_center, (fw, fh),
                stabilize=True, integer_raw=True,
            )
            _, tx, ty = tform
            bayer = read_bayer(meta["source"])
            out_bayer = translate_bayer(bayer, tx, ty)
            write_bayer_tiff(os.path.join(raw_dir, stem), out_bayer)

        if progress_cb and i % 5 == 0:
            progress_cb(i, total)

    if progress_cb:
        progress_cb(total, total)


def export_crop_sequence(
    intermediate_dir: str,
    out_dir: str,
    points: Dict[int, Tuple[float, float]],
    reference_center: Tuple[float, float],
    crop_w: int,
    crop_h: int,
    use_center: bool,
    *,
    export_rgb: bool = True,
    export_raw: bool = False,
    rgb_subdir: str = "rgb",
    raw_subdir: str = "raw",
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Export cropped stabilized still sequence."""
    manifest = load_manifest(intermediate_dir)
    total = manifest["frame_count"]
    frames: List[dict] = manifest["frames"]

    sample = _read_intermediate_bgr(intermediate_dir, 0)
    fh, fw = sample.shape[:2]

    rgb_dir = os.path.join(out_dir, rgb_subdir) if export_rgb else None
    raw_dir = os.path.join(out_dir, raw_subdir) if export_raw else None
    if rgb_dir:
        os.makedirs(rgb_dir, exist_ok=True)
    if raw_dir:
        os.makedirs(raw_dir, exist_ok=True)

    fallback = reference_center

    for i in range(total):
        meta = frames[i]
        cx, cy = points.get(i, fallback)
        stem = export_basename(meta["basename"], ".tif")

        if export_rgb:
            frame = _read_intermediate_bgr(intermediate_dir, i)
            crop = extract_crop_bgr(frame, cx, cy, crop_w, crop_h, use_center)
            _write_rgb_tiff(os.path.join(rgb_dir, stem), crop)

        if export_raw:
            tform = frame_transform_from_points(
                i, points, reference_center, (fw, fh),
                crop_w=crop_w, crop_h=crop_h, use_center=use_center,
                stabilize=False, integer_raw=True,
            )
            _, x, y, w, h = tform
            bayer = read_bayer(meta["source"])
            out_bayer = crop_bayer(bayer, x, y, w, h)
            write_bayer_tiff(os.path.join(raw_dir, stem), out_bayer)

        if progress_cb and i % 5 == 0:
            progress_cb(i, total)

    if progress_cb:
        progress_cb(total, total)
