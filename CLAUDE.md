# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the app
python main.py

# Run all tests (exclude venv)
pytest test_*.py

# Run a single test file
pytest test_tracking_engine.py

# Run a single test
pytest test_tracking_engine.py::TestTrackingEngine::test_basic_track
```

No build step — pure Python. Activate the venv first: `source venv/bin/activate`.

## Architecture

The app is a single-window PyQt6 desktop tool. `main.py` holds `MainWindow` and all UI/event logic. The other modules are pure computation with no Qt dependency.

### Data flow

1. User opens a video → `proxy.py` transcodes it to MJPEG (`cache/<name>_proxy.avi`) for frame-accurate seeking. `self.video_path` = original (export only). `self.proxy_path` = what the cap and tracker read from.
2. User draws an ROI → `TrackingEngine.set_template()` stores the BGR crop and reference center.
3. Auto-tracking (`_run_tracking`) or single-frame tracking (`_on_track_this_frame`) calls `TrackingEngine.track_frame()` on frames read from the proxy cap, writing results into `self.points: Dict[int, (float, float)]` and `self.tracking_overlays: Dict[int, (search_rect, match_rect, lost)]`.
4. Export reads from `self.video_path` (original quality) and uses `self.points` to drive either `video_exporter.export_video` (crop follows point) or `video_exporter.export_stabilized` (full frame translated to lock point).

### Key state in MainWindow

| Attribute | Purpose |
|---|---|
| `self.cap` | OpenCV cap on the proxy — used for all playback and seeking |
| `self.points` | `{frame_idx: (px, py)}` — the tracking dataset; never cleared automatically |
| `self.tracking_overlays` | Per-frame visual overlays (search rect, match rect, lost flag); not used for export |
| `self._current_frame_bgr` | Cached last decoded frame; used by single-frame tracking and manual point to avoid re-seeking |
| `self._tracking_resume_frame` | Frame index from which the next Run Tracking pass should start; set on manual overrides and tracking loss |
| `self._manual_frames` | Set of frame indices the user set by hand; shown as orange ticks on the range bar |

### Seeking correctness

`cap.set(N) + cap.read()` on H.264 is non-deterministic (snaps to keyframes). The MJPEG proxy makes every seek exact. After any `cap.read()`, always derive the true frame index as `int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1` — do not trust the requested index. This pattern is used in both `_seek_frame` and `_advance_frame`.

### Color space convention

OpenCV reads BGR. Qt and the crop/export pipeline use RGB. Conversion happens at the boundary — frames stored in `_current_frame_bgr` are BGR; `_render_frame` converts before passing to widgets. Never pass BGR to `video_player.show_frame` or `extract_crop`.

### Tracking engine

`TrackingEngine` (`tracking_engine.py`) uses `cv2.TM_CCOEFF_NORMED` template matching within a search window padded by `search_padding=100px` around the previous center. No template auto-update — the template set at `set_template()` is fixed for the entire run. `reset_position()` resets `previous_center` to `reference_center` for a fresh pass.

### Module responsibilities

- `proxy.py` — MJPEG transcoding; `ensure_proxy()` is the only public entry point
- `tracking_engine.py` — stateless-ish tracker; holds template and previous center
- `video_exporter.py` — `export_video` (crop) and `export_stabilized` (full frame warp); both read from `video_path`
- `stabilizer.py` — single-function translation warp used by `export_stabilized`
- `video_player.py` — `QLabel` subclass that paints the scaled video frame plus overlays and emits `roi_selected` / `point_selected` signals
- `crop_canvas.py` — `QLabel` subclass showing the crop preview; receives either a raw frame region or a pre-blended canvas
