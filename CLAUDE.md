# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the app
python main.py

# Run all tests (exclude venv and test_vidstab.py — manual script)
pytest test_image_sequence.py test_intermediate.py test_raw_transform.py test_stabilizer.py test_tracking_engine.py test_video_exporter.py

# Run a single test file
pytest test_tracking_engine.py

# Run a single test
pytest test_tracking_engine.py::TestTrackingEngine::test_basic_track
```

No build step — pure Python. Activate the venv first: `source venv/bin/activate`.

## Architecture

The app is a single-window PyQt6 desktop tool. `main.py` holds `MainWindow` and all UI/event logic. The other modules are pure computation with no Qt dependency.

### Data flow

1. **Video:** user opens a file → `proxy.py` builds MJPEG (`cache/<name>_proxy.avi`). `self.frame_source` is a `VideoFrameSource` on the proxy. Export reads the original file.
2. **RAW sequence:** user opens a folder → `intermediate.py` demosaices each DNG to 16-bit TIFF in `cache/<folder>_decoded/` (see `manifest.json`). `SequenceFrameSource` drives playback/tracking. Export writes still folders (RGB TIFF and/or Bayer TIFF via integer transform replay on source DNGs).
3. User draws an ROI → `TrackingEngine.set_template()` stores the BGR crop and reference center.
4. Auto-tracking or single-frame tracking reads frames via `self.frame_source`, writing `self.points` and `self.tracking_overlays`.
5. Export uses `self.points` — MP4 for video; `export_*_sequence` for RAW folders (`video_exporter.py` + `raw_transform.py`).

### Key state in MainWindow

| Attribute | Purpose |
|---|---|
| `self.frame_source` | `VideoFrameSource` or `SequenceFrameSource` — all playback and seeking |
| `self.source_kind` | `"video"` or `"sequence"` |
| `self.source_path` | Original video file or RAW folder (export) |
| `self.intermediate_dir` | Decoded TIFF cache path (sequence only) |
| `self.proxy_path` | MJPEG proxy path (video only) |
| `self.points` | `{frame_idx: (px, py)}` — the tracking dataset; never cleared automatically |
| `self.tracking_overlays` | Per-frame visual overlays (search rect, match rect, lost flag); not used for export |
| `self._current_frame_bgr` | Cached last decoded frame; used by single-frame tracking and manual point to avoid re-seeking |
| `self._tracking_resume_frame` | Frame index from which the next Run Tracking pass should start; set on manual overrides and tracking loss |
| `self._manual_frames` | Set of frame indices the user set by hand; shown as orange ticks on the range bar |

### Seeking correctness

For **video**, H.264 seeking is non-deterministic without the MJPEG proxy. `VideoFrameSource` uses the proxy so `read_bgr(index)` is exact. For **sequences**, indices are always exact (numbered TIFF files).

### Color space convention

OpenCV reads BGR. Qt and the crop/export pipeline use RGB. Conversion happens at the boundary — frames stored in `_current_frame_bgr` are BGR; `_render_frame` converts before passing to widgets. Never pass BGR to `video_player.show_frame` or `extract_crop`.

### Tracking engine

`TrackingEngine` (`tracking_engine.py`) uses `cv2.TM_CCOEFF_NORMED` template matching within a search window padded by `search_padding=100px` around the previous center. No template auto-update — the template set at `set_template()` is fixed for the entire run. `reset_position()` resets `previous_center` to `reference_center` for a fresh pass.

### Module responsibilities

- `proxy.py` — MJPEG transcoding; `ensure_proxy()` is the only public entry point
- `image_sequence.py` — discover/sort/decode RAW folders (`rawpy`)
- `intermediate.py` — cached 16-bit RGB TIFF sequence + `manifest.json`
- `frame_source.py` — `VideoFrameSource` / `SequenceFrameSource`
- `raw_transform.py` — integer Bayer translate/crop; shared geometry with RGB export
- `tracking_engine.py` — template matcher; holds template and previous center
- `video_exporter.py` — video MP4 export + `export_*_sequence` for stills
- `stabilizer.py` — translation warp; uses `stabilization_translation()` from `raw_transform`
- `video_player.py` — `QLabel` subclass that paints the scaled video frame plus overlays and emits `roi_selected` / `point_selected` signals
- `crop_canvas.py` — `QLabel` subclass showing the crop preview; receives either a raw frame region or a pre-blended canvas
