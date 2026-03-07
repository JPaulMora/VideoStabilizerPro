import numpy as np
import pytest
from video_exporter import extract_crop


def make_frame(w=640, h=480):
    """Create a test frame with a gradient so we can verify crop coordinates."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # Fill each pixel with its (x % 256) value in the red channel for easy checks
    for x in range(w):
        frame[:, x, 0] = x % 256
    for y in range(h):
        frame[y, :, 1] = y % 256
    return frame


class TestExtractCropCenter:
    def test_basic_center_crop(self):
        frame = make_frame(640, 480)
        crop = extract_crop(frame, px=320, py=240, crop_w=100, crop_h=100, use_center=True)
        assert crop.shape == (100, 100, 3)

    def test_center_crop_pixel_values(self):
        # When centered at (320, 240) with 100x100, x range is [270, 370), y range is [190, 290)
        frame = make_frame(640, 480)
        crop = extract_crop(frame, px=320, py=240, crop_w=100, crop_h=100, use_center=True)
        # Top-left of crop = pixel (270, 190) in frame
        assert crop[0, 0, 0] == 270 % 256  # red = x
        assert crop[0, 0, 1] == 190 % 256  # green = y

    def test_center_crop_shape(self):
        frame = make_frame(640, 480)
        crop = extract_crop(frame, px=320, py=240, crop_w=200, crop_h=150, use_center=True)
        assert crop.shape == (150, 200, 3)

    def test_center_crop_non_square(self):
        frame = make_frame(640, 480)
        crop = extract_crop(frame, px=100, py=100, crop_w=400, crop_h=225, use_center=True)
        assert crop.shape == (225, 400, 3)


class TestExtractCropTopLeft:
    def test_basic_topleft_crop(self):
        frame = make_frame(640, 480)
        crop = extract_crop(frame, px=50, py=50, crop_w=100, crop_h=100, use_center=False)
        assert crop.shape == (100, 100, 3)

    def test_topleft_crop_pixel_values(self):
        frame = make_frame(640, 480)
        # Top-left origin at (50, 50) — first pixel of crop = frame pixel (50, 50)
        crop = extract_crop(frame, px=50, py=50, crop_w=100, crop_h=100, use_center=False)
        assert crop[0, 0, 0] == 50 % 256
        assert crop[0, 0, 1] == 50 % 256


class TestExtractCropBoundaryClipping:
    def test_clipped_at_left_edge(self):
        # Point near left edge; crop should be partially filled, rest dark
        frame = make_frame(640, 480)
        crop = extract_crop(frame, px=10, py=240, crop_w=100, crop_h=100, use_center=True)
        assert crop.shape == (100, 100, 3)
        # First 40 columns (0..39) are out of frame → should be bg color 0x11
        assert np.all(crop[:, 0, 0] == 0x11)

    def test_clipped_at_top_edge(self):
        frame = make_frame(640, 480)
        crop = extract_crop(frame, px=320, py=10, crop_w=100, crop_h=100, use_center=True)
        assert crop.shape == (100, 100, 3)
        # First 40 rows are out of frame → bg color
        assert np.all(crop[0, :, 0] == 0x11)

    def test_clipped_at_right_edge(self):
        frame = make_frame(640, 480)
        crop = extract_crop(frame, px=630, py=240, crop_w=100, crop_h=100, use_center=True)
        assert crop.shape == (100, 100, 3)
        # Last columns past frame width should be bg
        assert np.all(crop[:, -1, 0] == 0x11)

    def test_clipped_at_bottom_edge(self):
        frame = make_frame(640, 480)
        crop = extract_crop(frame, px=320, py=470, crop_w=100, crop_h=100, use_center=True)
        assert crop.shape == (100, 100, 3)
        assert np.all(crop[-1, :, 0] == 0x11)

    def test_fully_outside_frame(self):
        # Point completely outside frame; entire crop should be bg color
        frame = make_frame(640, 480)
        crop = extract_crop(frame, px=-200, py=-200, crop_w=100, crop_h=100, use_center=True)
        assert crop.shape == (100, 100, 3)
        assert np.all(crop == 0x11)

    def test_point_at_origin_topleft(self):
        frame = make_frame(640, 480)
        crop = extract_crop(frame, px=0, py=0, crop_w=100, crop_h=100, use_center=False)
        assert crop.shape == (100, 100, 3)
        # Should be entirely within frame
        assert crop[0, 0, 0] == 0  # x=0 → red=0
        assert crop[0, 0, 1] == 0  # y=0 → green=0


class TestExtractCropOutputType:
    def test_dtype_preserved(self):
        frame = make_frame(640, 480)
        crop = extract_crop(frame, px=320, py=240, crop_w=100, crop_h=100, use_center=True)
        assert crop.dtype == np.uint8

    def test_channels(self):
        frame = make_frame(640, 480)
        crop = extract_crop(frame, px=320, py=240, crop_w=100, crop_h=100, use_center=True)
        assert crop.ndim == 3
        assert crop.shape[2] == 3
