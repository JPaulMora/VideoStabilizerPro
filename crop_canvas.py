import numpy as np
from PyQt6.QtWidgets import QLabel, QSizePolicy
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QSize
from video_exporter import extract_crop


class CropCanvasWidget(QLabel):

    def __init__(self):
        super().__init__()
        self.crop_w = 400
        self.crop_h = 400
        self._current_canvas = None
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.setMaximumSize(self.crop_w, self.crop_h)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._show_blank()

    def sizeHint(self):
        return QSize(self.crop_w, self.crop_h)

    def resize_canvas(self, w: int, h: int):
        self.crop_w = w
        self.crop_h = h
        self.setMaximumSize(w, h)
        self.updateGeometry()
        self._show_blank()

    def _show_blank(self):
        canvas = np.full((self.crop_h, self.crop_w, 3), 0x11, dtype=np.uint8)
        self._current_canvas = canvas
        self._render_canvas()

    def _render_canvas(self):
        if self._current_canvas is None:
            return
        canvas = self._current_canvas
        h, w, ch = canvas.shape
        img = QImage(canvas.data, w, h, w * ch, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(img).scaled(
            self.width(), self.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_canvas()

    def show_frame(self, frame, px: float, py: float, use_center: bool):
        canvas = extract_crop(frame, px, py, self.crop_w, self.crop_h, use_center)
        self._current_canvas = canvas
        self._render_canvas()
