"""
proxy.py — MJPEG proxy builder for frame-accurate seeking.

H.264/HEVC videos can only seek to keyframes, so cv2.cap.set(N) silently
snaps to a nearby keyframe and returns the wrong frame.  MJPEG encodes every
frame as an independent JPEG (every frame is a keyframe), making cap.set(N)
exact and deterministic.

Usage
-----
    from proxy import ensure_proxy

    proxy_path = ensure_proxy(src_path, progress_cb=my_callback)
    cap = cv2.VideoCapture(proxy_path)

The proxy is written next to the source file with a ``_proxy.avi`` suffix and
is reused on subsequent opens as long as it is newer than the source.
"""

import os
import cv2
from typing import Callable, Optional

_PROXY_SUFFIX = "_proxy.avi"
_MJPEG_QUALITY = 80   # 0–100; set via VIDEOWRITER_PROP_QUALITY where supported


def proxy_path_for(src_path: str) -> str:
    """Return the proxy path inside a ``cache/`` folder beside the source."""
    src_dir  = os.path.dirname(os.path.abspath(src_path))
    stem, _  = os.path.splitext(os.path.basename(src_path))
    cache_dir = os.path.join(src_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, stem + _PROXY_SUFFIX)


def _proxy_is_fresh(src_path: str, proxy_path: str) -> bool:
    """True if the proxy file exists and is at least as new as the source."""
    return (os.path.exists(proxy_path)
            and os.path.getmtime(proxy_path) >= os.path.getmtime(src_path))


def build_proxy(
    src_path: str,
    proxy_path: str,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> None:
    """
    Transcode *src_path* to an MJPEG AVI at *proxy_path*.
    *progress_cb(current, total)* is called roughly every 30 frames.
    Raises RuntimeError if the source cannot be opened or the writer fails.
    """
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source video: {src_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(proxy_path, fourcc, fps, (fw, fh))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create proxy writer at: {proxy_path}")

    # Best-effort quality hint — not all backends honour this
    writer.set(cv2.VIDEOWRITER_PROP_QUALITY, _MJPEG_QUALITY)

    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        if progress_cb and i % 30 == 0:
            progress_cb(i, total)

    writer.release()
    cap.release()

    if progress_cb:
        progress_cb(total, total)


def ensure_proxy(
    src_path: str,
    proxy_path: Optional[str] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> str:
    """
    Return a path to a fresh MJPEG proxy for *src_path*, building it if
    necessary.  Pass *proxy_path* to override the default location.
    """
    if proxy_path is None:
        proxy_path = proxy_path_for(src_path)

    if not _proxy_is_fresh(src_path, proxy_path):
        build_proxy(src_path, proxy_path, progress_cb=progress_cb)

    return proxy_path
