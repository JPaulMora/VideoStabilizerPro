import sys
import csv
import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout,
    QVBoxLayout, QPushButton, QLabel, QFrame,
    QFileDialog, QSlider, QRadioButton, QButtonGroup,
    QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer
from video_player import VideoPlayerWidget
from crop_canvas import CropCanvasWidget

BUTTON_STYLE = """
    QPushButton {
        background-color: #313244;
        color: #cdd6f4;
        border: none;
        border-radius: 6px;
        padding: 10px 12px;
        text-align: left;
        font-size: 13px;
    }
    QPushButton:hover { background-color: #45475a; }
    QPushButton:pressed { background-color: #585b70; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VideoStPro")
        self.setMinimumSize(1100, 650)

        # Playback state
        self.cap = None
        self.points = {}
        self.total_frames = 0
        self.current_frame_idx = 0
        self.is_playing = False
        self._slider_updating = False

        self.timer = QTimer()
        self.timer.timeout.connect(self._advance_frame)

        self._build_ui()

    def _build_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        root_layout = QHBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # --- Sidebar ---
        sidebar = QFrame()
        sidebar.setFixedWidth(200)
        sidebar.setObjectName("sidebar")
        sidebar.setStyleSheet("""
            #sidebar {
                background-color: #1e1e2e;
                border-right: 1px solid #313244;
            }
        """)

        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(12, 16, 12, 16)
        sidebar_layout.setSpacing(8)

        title = QLabel("VideoStPro")
        title.setStyleSheet("color: #cdd6f4; font-size: 16px; font-weight: bold; padding: 4px 0 12px 0;")
        sidebar_layout.addWidget(title)

        load_video_btn = QPushButton("Load Video")
        load_video_btn.setStyleSheet(BUTTON_STYLE)
        load_video_btn.clicked.connect(self._on_load_video)
        sidebar_layout.addWidget(load_video_btn)

        load_csv_btn = QPushButton("Load CSV")
        load_csv_btn.setStyleSheet(BUTTON_STYLE)
        load_csv_btn.clicked.connect(self._on_load_csv)
        sidebar_layout.addWidget(load_csv_btn)

        sep = QLabel("Settings")
        sep.setStyleSheet("color: #6c7086; font-size: 11px; padding: 8px 0 4px 0;")
        sidebar_layout.addWidget(sep)

        self.center_radio = QRadioButton("Center point")
        self.center_radio.setStyleSheet("color: #cdd6f4; font-size: 13px;")
        self.center_radio.setChecked(True)

        topleft_radio = QRadioButton("Top-left point")
        topleft_radio.setStyleSheet("color: #cdd6f4; font-size: 13px;")

        self._point_group = QButtonGroup()
        self._point_group.addButton(self.center_radio)
        self._point_group.addButton(topleft_radio)
        self._point_group.buttonClicked.connect(self._on_mode_changed)

        sidebar_layout.addWidget(self.center_radio)
        sidebar_layout.addWidget(topleft_radio)
        sidebar_layout.addStretch()

        # --- Content area ---
        content = QFrame()
        content.setObjectName("content")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Viewer row
        viewer_row = QHBoxLayout()
        viewer_row.setContentsMargins(0, 0, 0, 0)
        viewer_row.setSpacing(0)

        self.video_player = VideoPlayerWidget()
        self.video_player.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        viewer_row.addWidget(self.video_player, stretch=1)

        self.crop_canvas = CropCanvasWidget()
        viewer_row.addWidget(self.crop_canvas)

        content_layout.addLayout(viewer_row, stretch=1)

        # Controls row
        controls_row = QHBoxLayout()
        controls_row.setContentsMargins(8, 6, 8, 6)
        controls_row.setSpacing(8)

        self.play_pause_btn = QPushButton("▶")
        self.play_pause_btn.setFixedWidth(40)
        self.play_pause_btn.setStyleSheet(BUTTON_STYLE)
        self.play_pause_btn.clicked.connect(self._on_play_pause)
        controls_row.addWidget(self.play_pause_btn)

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self._on_slider_changed)
        controls_row.addWidget(self.frame_slider, stretch=1)

        self.frame_label = QLabel("0 / 0")
        self.frame_label.setStyleSheet("color: #cdd6f4; font-size: 13px; min-width: 80px;")
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        controls_row.addWidget(self.frame_label)

        controls_widget = QWidget()
        controls_widget.setStyleSheet("background-color: #1e1e2e; border-top: 1px solid #313244;")
        controls_widget.setLayout(controls_row)
        content_layout.addWidget(controls_widget)

        root_layout.addWidget(sidebar)
        root_layout.addWidget(content, stretch=1)

    # --- File loading ---

    def _on_load_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv);;All Files (*)"
        )
        if path:
            self._load_video(path)

    def _on_load_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open CSV", "/Users/nimble./Projects/VideoStPro",
            "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            self._load_csv(path)

    def _load_video(self, path: str):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.timer.setInterval(int(1000 / fps))
        self._slider_updating = True
        self.frame_slider.setRange(0, max(0, self.total_frames - 1))
        self.frame_slider.setValue(0)
        self._slider_updating = False
        self.current_frame_idx = 0
        self._seek_frame(0)

    def _load_csv(self, path: str):
        self.points = {}
        with open(path, newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
        # Row 0: metadata, Row 1: headers, Row 2+: data
        for row in rows[2:]:
            if len(row) < 4:
                continue
            try:
                frame = int(float(row[1]))
                px = float(row[2])
                py = float(row[3])
                self.points[frame] = (px, py)
            except (ValueError, IndexError):
                continue

    # --- Playback ---

    def _on_play_pause(self):
        if self.is_playing:
            self._pause()
        else:
            self._play()

    def _play(self):
        if not self.cap:
            return
        self.is_playing = True
        self.play_pause_btn.setText("⏸")
        self.timer.start()

    def _pause(self):
        self.is_playing = False
        self.play_pause_btn.setText("▶")
        self.timer.stop()

    def _on_slider_changed(self, value: int):
        if self._slider_updating:
            return
        self._pause()
        self._seek_frame(value)

    def _seek_frame(self, idx: int):
        if not self.cap:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame_idx = idx
            self._render_frame(frame)

    def _advance_frame(self):
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            self._pause()
            return
        self.current_frame_idx += 1
        self._slider_updating = True
        self.frame_slider.setValue(self.current_frame_idx)
        self._slider_updating = False
        self._render_frame(frame)

    def _render_frame(self, frame_bgr: np.ndarray):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_h, frame_w = frame_rgb.shape[:2]

        if self.points:
            px, py = self.points.get(
                self.current_frame_idx,
                (frame_w / 2, frame_h / 2)
            )
        else:
            px, py = frame_w / 2, frame_h / 2

        use_center = self.center_radio.isChecked()
        self.video_player.show_frame(frame_rgb, px, py, use_center)
        self.crop_canvas.show_frame(frame_rgb, px, py, use_center)
        self.frame_label.setText(f"{self.current_frame_idx} / {self.total_frames}")

    def _on_mode_changed(self):
        # Re-render current frame with new mode
        if not self.cap:
            return
        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        seek_to = max(0, self.current_frame_idx)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, seek_to)
        ret, frame = self.cap.read()
        if ret:
            self._render_frame(frame)
        # Restore position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
