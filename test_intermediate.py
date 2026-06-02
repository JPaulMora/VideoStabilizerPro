import json
import os
import time

import intermediate
from image_sequence import list_frames


def _touch(path: str, when: float) -> None:
    os.utime(path, (when, when))


class TestIntermediateFreshness:
    def test_rebuild_when_source_newer(self, tmp_path, monkeypatch):
        seq = tmp_path / "seq"
        seq.mkdir()
        for i in (1, 2):
            p = seq / f"f - {i:05d}.raw.dng"
            p.write_bytes(b"\x00")

        decoded = []

        def fake_read(path):
            decoded.append(path)
            import numpy as np
            return np.zeros((4, 4, 3), dtype=np.uint16)

        monkeypatch.setattr(intermediate, "read_raw_frame", fake_read)

        out = intermediate.ensure_intermediate(str(seq))
        assert len(decoded) == 2
        assert intermediate._intermediate_is_fresh(str(seq), out)

        # Touch source newer than cache
        time.sleep(0.05)
        _touch(str(seq / "f - 00001.raw.dng"), time.time() + 10)
        assert not intermediate._intermediate_is_fresh(str(seq), out)

        decoded.clear()
        intermediate.ensure_intermediate(str(seq))
        assert len(decoded) == 2

    def test_manifest_frame_count(self, tmp_path, monkeypatch):
        seq = tmp_path / "seq"
        seq.mkdir()
        (seq / "a - 00001.raw.dng").write_bytes(b"\x00")

        import numpy as np
        monkeypatch.setattr(
            intermediate, "read_raw_frame",
            lambda _p: np.zeros((2, 2, 3), dtype=np.uint16),
        )

        out = intermediate.ensure_intermediate(str(seq))
        manifest = intermediate.load_manifest(out)
        assert manifest["frame_count"] == 1
        assert len(list_frames(str(seq))) == 1
