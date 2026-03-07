import numpy as np
import pytest
from tracking_engine import TrackingEngine, TrackingLostError, TrackResult


def _solid_frame(h=100, w=100, color=(128, 128, 128)):
    frame = np.full((h, w, 3), color, dtype=np.uint8)
    return frame


def _frame_with_patch(h=200, w=200, patch_xy=(80, 80), patch_size=20):
    """Frame with a checkerboard patch for sufficient texture (non-zero std dev)."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    px, py = patch_xy
    ps = patch_size
    for dy in range(ps):
        for dx in range(ps):
            val = 255 if (dx + dy) % 2 == 0 else 80
            frame[py + dy, px + dx] = (val, val // 2, val // 3)
    return frame


def test_set_template_stores_reference():
    engine = TrackingEngine()
    frame = _frame_with_patch()
    engine.set_template(frame, (80, 80, 20, 20))
    assert engine.has_template
    assert engine.reference_center == (90.0, 90.0)
    assert engine.previous_center == (90.0, 90.0)


def test_track_frame_finds_patch():
    engine = TrackingEngine(search_padding=40, confidence_threshold=0.5)
    frame = _frame_with_patch(patch_xy=(80, 80))
    engine.set_template(frame, (80, 80, 20, 20))

    # Same frame — should find at same location with perfect score
    result = engine.track_frame(frame)
    assert isinstance(result, TrackResult)
    assert abs(result.center[0] - 90.0) < 2
    assert abs(result.center[1] - 90.0) < 2
    assert result.score > 0.9


def test_track_frame_finds_moved_patch():
    engine = TrackingEngine(search_padding=50, confidence_threshold=0.5)
    frame_ref = _frame_with_patch(patch_xy=(80, 80))
    engine.set_template(frame_ref, (80, 80, 20, 20))

    frame_moved = _frame_with_patch(patch_xy=(90, 85))
    result = engine.track_frame(frame_moved)
    assert abs(result.center[0] - 100.0) < 3
    assert abs(result.center[1] - 95.0) < 3


def test_track_frame_raises_when_no_template():
    engine = TrackingEngine()
    frame = _solid_frame()
    with pytest.raises(TrackingLostError, match="No template"):
        engine.track_frame(frame)


def test_track_frame_raises_on_low_confidence():
    engine = TrackingEngine(search_padding=30, confidence_threshold=0.99)
    frame_ref = _frame_with_patch(patch_xy=(80, 80))
    engine.set_template(frame_ref, (80, 80, 20, 20))

    # Completely different frame (random noise) — score won't reach 0.99
    rng = np.random.default_rng(42)
    frame_noise = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    with pytest.raises(TrackingLostError):
        engine.track_frame(frame_noise)


def test_reset_position_resets_to_reference():
    engine = TrackingEngine(search_padding=50, confidence_threshold=0.5)
    frame = _frame_with_patch(patch_xy=(80, 80))
    engine.set_template(frame, (80, 80, 20, 20))

    frame_moved = _frame_with_patch(patch_xy=(100, 100))
    engine.track_frame(frame_moved)
    assert engine.previous_center != engine.reference_center

    engine.reset_position()
    assert engine.previous_center == engine.reference_center
    assert engine._frame_count == 0


def test_search_rect_is_within_frame():
    engine = TrackingEngine(search_padding=50, confidence_threshold=0.5)
    frame = _frame_with_patch(h=200, w=200, patch_xy=(10, 10))
    engine.set_template(frame, (10, 10, 20, 20))
    result = engine.track_frame(frame)
    x, y, w, h = result.search_rect
    assert x >= 0 and y >= 0
    assert x + w <= 200
    assert y + h <= 200
