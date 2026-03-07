from typing import Tuple, Optional
import numpy as np
from PyQt6.QtWidgets import QLabel
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush
from PyQt6.QtCore import Qt, QRect, QPoint, pyqtSignal


class VideoPlayerWidget(QLabel):

    roi_selected = pyqtSignal(int, int, int, int)   # x, y, w, h video coords
    point_selected = pyqtSignal(int, int)            # x, y video coords

    def __init__(self):
        super().__init__()
        self.setScaledContents(False)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("QLabel { background-color: black; }")
        self.setMinimumSize(640, 480)

        self._tracking_mode = False
        self._playback_paused = True
        self._drag_start: Optional[QPoint] = None
        self._drag_current: Optional[QPoint] = None
        self._search_rect: Optional[Tuple[int, int, int, int]] = None
        self._match_rect: Optional[Tuple[int, int, int, int]] = None
        self._tracking_lost = False
        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0
        self._last_render_args = None

    def show_frame(self, frame: np.ndarray, px: float, py: float,
                   use_center: bool, crop_w: int = 400, crop_h: int = 400):
        self._last_render_args = (frame, px, py, use_center, crop_w, crop_h)

        frame_h, frame_w = frame.shape[:2]

        widget_w = self.width()
        widget_h = self.height()

        scale = min(widget_w / frame_w, widget_h / frame_h)
        disp_w = int(frame_w * scale)
        disp_h = int(frame_h * scale)

        offset_x = (widget_w - disp_w) // 2
        offset_y = (widget_h - disp_h) // 2

        # Store for coordinate translation and overlay drawing
        self._scale = scale
        self._offset_x = offset_x
        self._offset_y = offset_y

        h, w, ch = frame.shape
        img = QImage(frame.data, w, h, w * ch, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(img).scaled(
            disp_w, disp_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        full_pixmap = QPixmap(widget_w, widget_h)
        full_pixmap.fill(QColor(0, 0, 0))

        painter = QPainter(full_pixmap)
        painter.drawPixmap(offset_x, offset_y, pixmap)

        # Crop box
        if use_center:
            src_x1 = px - crop_w // 2
            src_y1 = py - crop_h // 2
        else:
            src_x1 = px
            src_y1 = py

        box_x = int(src_x1 * scale) + offset_x
        box_y = int(src_y1 * scale) + offset_y
        box_w = int(crop_w * scale)
        box_h = int(crop_h * scale)

        pen = QPen(QColor(255, 220, 0))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRect(QRect(box_x, box_y, box_w, box_h))

        # Red dot at base point
        dot_x = int(px * scale) + offset_x
        dot_y = int(py * scale) + offset_y
        pen2 = QPen(QColor(255, 60, 60))
        pen2.setWidth(2)
        painter.setPen(pen2)
        painter.setBrush(QColor(255, 60, 60))
        painter.drawEllipse(QPoint(dot_x, dot_y), 6, 6)

        # Tracking overlays
        if self._tracking_mode:
            if self._search_rect is not None:
                self._draw_video_rect(painter, self._search_rect,
                                      QColor(0, 220, 220), fill=True, alpha=40)
            if self._match_rect is not None:
                color = QColor(220, 60, 60) if self._tracking_lost else QColor(60, 220, 60)
                self._draw_video_rect(painter, self._match_rect, color, line_width=2)

            # Drag selection rect
            if self._drag_start is not None and self._drag_current is not None:
                x1 = min(self._drag_start.x(), self._drag_current.x())
                y1 = min(self._drag_start.y(), self._drag_current.y())
                x2 = max(self._drag_start.x(), self._drag_current.x())
                y2 = max(self._drag_start.y(), self._drag_current.y())
                pen_drag = QPen(QColor(255, 255, 255))
                pen_drag.setWidth(1)
                painter.setPen(pen_drag)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(QRect(x1, y1, x2 - x1, y2 - y1))

        painter.end()
        self.setPixmap(full_pixmap)

    def _draw_video_rect(self, painter: QPainter,
                         rect: Tuple[int, int, int, int],
                         color: QColor,
                         fill: bool = False,
                         alpha: int = 255,
                         line_width: int = 1):
        x, y, w, h = rect
        wx = int(x * self._scale) + self._offset_x
        wy = int(y * self._scale) + self._offset_y
        ww = int(w * self._scale)
        wh = int(h * self._scale)

        pen_color = QColor(color.red(), color.green(), color.blue(), alpha if fill else 255)
        pen = QPen(pen_color)
        pen.setWidth(line_width)
        painter.setPen(pen)

        if fill:
            fill_color = QColor(color.red(), color.green(), color.blue(), alpha)
            painter.setBrush(QBrush(fill_color))
        else:
            painter.setBrush(Qt.BrushStyle.NoBrush)

        painter.drawRect(QRect(wx, wy, ww, wh))
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _widget_to_video(self, wx: float, wy: float) -> Tuple[float, float]:
        return ((wx - self._offset_x) / self._scale,
                (wy - self._offset_y) / self._scale)

    def set_paused(self, paused: bool):
        self._playback_paused = paused

    def set_tracking_mode(self, enabled: bool):
        self._tracking_mode = enabled
        self._drag_start = None
        self._drag_current = None
        self._search_rect = None
        self._match_rect = None
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)

    def set_tracking_overlays(self, search_rect, match_rect, lost: bool = False):
        self._search_rect = search_rect
        self._match_rect = match_rect
        self._tracking_lost = lost

    def mousePressEvent(self, e):
        if self._tracking_mode:
            self._drag_start = e.position().toPoint()
            self._drag_current = self._drag_start

    def mouseMoveEvent(self, e):
        if self._tracking_mode and self._drag_start is not None:
            self._drag_current = e.position().toPoint()
            if self._last_render_args is not None:
                self.show_frame(*self._last_render_args)

    def mouseReleaseEvent(self, e):
        if self._tracking_mode and self._drag_start is not None:
            vx1, vy1 = self._widget_to_video(self._drag_start.x(), self._drag_start.y())
            vx2, vy2 = self._widget_to_video(e.position().x(), e.position().y())
            x = int(min(vx1, vx2))
            y = int(min(vy1, vy2))
            w = int(abs(vx2 - vx1))
            h = int(abs(vy2 - vy1))
            self._drag_start = None
            self._drag_current = None
            if w > 8 and h > 8:
                self.roi_selected.emit(x, y, w, h)
            elif self._playback_paused:
                # Single click → manual center override, only when paused
                cx, cy = self._widget_to_video(e.position().x(), e.position().y())
                self.point_selected.emit(int(cx), int(cy))
