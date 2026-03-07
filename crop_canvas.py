from PyQt6.QtWidgets import QLabel
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt
from video_exporter import extract_crop


class CropCanvasWidget(QLabel):
    BG_COLOR = (0x11, 0x11, 0x11)

    def __init__(self):
        super().__init__()
        self.crop_w = 400
        self.crop_h = 400
        self.setFixedSize(self.crop_w, self.crop_h)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._show_blank()

    def resize_canvas(self, w: int, h: int):
        self.crop_w = w
        self.crop_h = h
        self.setFixedSize(w, h)
        self._show_blank()

    def _show_blank(self):
        import numpy as np
        canvas = np.full((self.crop_h, self.crop_w, 3), 0x11, dtype=np.uint8)
        self._set_canvas(canvas)

    def _set_canvas(self, canvas):
        h, w, ch = canvas.shape
        img = QImage(canvas.data, w, h, w * ch, QImage.Format.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(img))

    def show_frame(self, frame, px: float, py: float, use_center: bool):
        canvas = extract_crop(frame, px, py, self.crop_w, self.crop_h, use_center)
        self._set_canvas(canvas)
