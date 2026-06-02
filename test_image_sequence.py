import os
import tempfile

import pytest

from image_sequence import (
    export_basename,
    frame_sort_key,
    is_image_sequence_dir,
    is_raw_sequence_file,
    list_frames,
)


class TestFrameSortKey:
    def test_numeric_suffix(self):
        a = "/seq/STARWARS - 00001.raw.dng"
        b = "/seq/STARWARS - 00080.raw.dng"
        assert frame_sort_key(a) < frame_sort_key(b)

    def test_ordering(self):
        paths = [
            "/x/clip - 00010.raw.dng",
            "/x/clip - 00002.raw.dng",
            "/x/clip - 00001.raw.dng",
        ]
        assert [os.path.basename(p) for p in sorted(paths, key=frame_sort_key)] == [
            "clip - 00001.raw.dng",
            "clip - 00002.raw.dng",
            "clip - 00010.raw.dng",
        ]


class TestListFrames:
    def test_discovers_dng_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in (1, 2, 3):
                open(os.path.join(tmp, f"test - {i:05d}.raw.dng"), "w").close()
            open(os.path.join(tmp, "readme.txt"), "w").close()
            frames = list_frames(tmp)
            assert len(frames) == 3
            assert is_image_sequence_dir(tmp)

    def test_empty_dir_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError):
                list_frames(tmp)


class TestHelpers:
    def test_is_raw_sequence_file(self):
        assert is_raw_sequence_file("a.raw.dng")
        assert is_raw_sequence_file("a.DNG")
        assert not is_raw_sequence_file("a.mp4")

    def test_export_basename(self):
        assert export_basename("STARWARS - 00001.raw.dng") == "STARWARS - 00001.tif"
