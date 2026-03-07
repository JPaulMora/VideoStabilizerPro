import numpy as np
from PyQt6.QtWidgets import QLabel
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt6.QtCore import Qt, QRect, QPoint


class VideoPlayerWidget(QLabel):

    def __init__(self):
        super().__init__()
        self.setScaledContents(False)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("QLabel { background-color: black; }")
        self.setMinimumSize(640, 480)

    def show_frame(self, frame: np.ndarray, px: float, py: float,
                   use_center: bool, crop_w: int = 400, crop_h: int = 400):
        frame_h, frame_w = frame.shape[:2]

        # Scale to fit widget while preserving aspect ratio
        widget_w = self.width()
        widget_h = self.height()

        scale = min(widget_w / frame_w, widget_h / frame_h)
        disp_w = int(frame_w * scale)
        disp_h = int(frame_h * scale)

        # Letterbox offsets
        offset_x = (widget_w - disp_w) // 2
        offset_y = (widget_h - disp_h) // 2

        # Convert frame to QPixmap
        h, w, ch = frame.shape
        img = QImage(frame.data, w, h, w * ch, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(img).scaled(
            disp_w, disp_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        # Create full-widget canvas
        full_pixmap = QPixmap(widget_w, widget_h)
        full_pixmap.fill(QColor(0, 0, 0))

        painter = QPainter(full_pixmap)
        painter.drawPixmap(offset_x, offset_y, pixmap)

        # Compute crop box source coords
        if use_center:
            src_x1 = px - crop_w // 2
            src_y1 = py - crop_h // 2
        else:
            src_x1 = px
            src_y1 = py

        # Map to display coords
        box_x = int(src_x1 * scale) + offset_x
        box_y = int(src_y1 * scale) + offset_y
        box_w = int(crop_w * scale)
        box_h = int(crop_h * scale)

        # Draw yellow crop box
        pen = QPen(QColor(255, 220, 0))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRect(QRect(box_x, box_y, box_w, box_h))

        # Draw red dot at base point
        dot_x = int(px * scale) + offset_x
        dot_y = int(py * scale) + offset_y
        pen2 = QPen(QColor(255, 60, 60))
        pen2.setWidth(2)
        painter.setPen(pen2)
        painter.setBrush(QColor(255, 60, 60))
        painter.drawEllipse(QPoint(dot_x, dot_y), 6, 6)

        painter.end()
        self.setPixmap(full_pixmap)
