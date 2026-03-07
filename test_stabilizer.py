import numpy as np
import pytest
from stabilizer import stabilize_frame


def _frame(h=100, w=100, color=(50, 100, 150)):
    return np.full((h, w, 3), color, dtype=np.uint8)


def test_no_displacement_returns_identical():
    frame = _frame()
    ref = (50.0, 50.0)
    result = stabilize_frame(frame, 0.0, 0.0, ref)
    # When reference_center == frame center and dx=dy=0, translation is zero
    np.testing.assert_array_equal(result, frame)


def test_output_shape_preserved():
    frame = _frame(h=120, w=160)
    result = stabilize_frame(frame, 10.0, 5.0, (80.0, 60.0))
    assert result.shape == frame.shape


def test_translation_shifts_content():
    # Place a white dot at (50, 50) in a black frame
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[50, 50] = (255, 255, 255)

    # reference_center is frame center (50,50), dx=10 means feature moved right 10px
    # stabilizer should shift left by 10 to compensate
    result = stabilize_frame(frame, 10.0, 0.0, (50.0, 50.0))
    # The white pixel should now be at col 40
    assert result[50, 40, 0] == 255
    assert result[50, 50, 0] == 0


def test_black_border_fill():
    frame = np.full((100, 100, 3), 200, dtype=np.uint8)
    # Large displacement — border pixels must be black (0)
    result = stabilize_frame(frame, 40.0, 0.0, (50.0, 50.0))
    # Right edge columns should be black due to border fill
    assert result[50, 99, 0] == 0


def test_reference_offset_correction():
    # When reference_center differs from frame center, stabilizer corrects for it
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[30, 20] = (255, 255, 255)
    ref = (20.0, 30.0)  # reference center is at the white dot

    # No displacement (feature is still at ref): should move dot to frame center (50,50)
    result = stabilize_frame(frame, 0.0, 0.0, ref)
    assert result[50, 50, 0] == 255
