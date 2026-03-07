import numpy as np
from PyQt6.QtWidgets import QLabel
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt


class CropCanvasWidget(QLabel):
    CROP_SIZE = 400
    BG_COLOR = (0x11, 0x11, 0x11)

    def __init__(self):
        super().__init__()
        self.setFixedSize(self.CROP_SIZE, self.CROP_SIZE)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._show_blank()

    def _show_blank(self):
        canvas = np.full((self.CROP_SIZE, self.CROP_SIZE, 3), 0x11, dtype=np.uint8)
        self._set_canvas(canvas)

    def _set_canvas(self, canvas: np.ndarray):
        h, w, ch = canvas.shape
        img = QImage(canvas.data, w, h, w * ch, QImage.Format.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(img))

    def show_frame(self, frame: np.ndarray, px: float, py: float, use_center: bool):
        frame_h, frame_w = frame.shape[:2]
        size = self.CROP_SIZE

        if use_center:
            half = size // 2
            src_x1 = int(px) - half
            src_y1 = int(py) - half
        else:
            src_x1 = int(px)
            src_y1 = int(py)

        src_x2 = src_x1 + size
        src_y2 = src_y1 + size

        canvas = np.full((size, size, 3), 0x11, dtype=np.uint8)

        vid_x1 = max(0, src_x1)
        vid_x2 = min(frame_w, src_x2)
        vid_y1 = max(0, src_y1)
        vid_y2 = min(frame_h, src_y2)

        if vid_x2 > vid_x1 and vid_y2 > vid_y1:
            dst_x1 = vid_x1 - src_x1
            dst_y1 = vid_y1 - src_y1
            dst_x2 = dst_x1 + (vid_x2 - vid_x1)
            dst_y2 = dst_y1 + (vid_y2 - vid_y1)
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = frame[vid_y1:vid_y2, vid_x1:vid_x2]

        self._set_canvas(canvas)
