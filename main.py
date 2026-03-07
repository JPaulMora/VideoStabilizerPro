import sys
import csv
import cv2
import numpy as np
from typing import Dict, Tuple, Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout,
    QVBoxLayout, QPushButton, QLabel, QFrame,
    QFileDialog, QSlider, QRadioButton, QButtonGroup,
    QSizePolicy, QLineEdit, QCheckBox
)
from PyQt6.QtCore import Qt, QTimer
from video_player import VideoPlayerWidget
from crop_canvas import CropCanvasWidget
import video_exporter
from tracking_engine import TrackingEngine, TrackingLostError

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

INPUT_STYLE = """
    QLineEdit {
        background-color: #313244;
        color: #cdd6f4;
        border: 1px solid #45475a;
        border-radius: 4px;
        padding: 4px 6px;
        font-size: 13px;
    }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VideoStPro")
        self.setMinimumSize(1100, 650)

        # Playback state
        self.cap = None
        self.video_path = None
        self.points = {}
        self.total_frames = 0
        self.current_frame_idx = 0
        self.is_playing = False
        self._slider_updating = False

        # Canvas size state
        self.crop_w = 400
        self.crop_h = 400

        # Tracking state
        self.tracking_engine = TrackingEngine()
        self.tracking_mode = False
        self.tracking_overlays: Dict[int, Tuple] = {}
        self._template_frame_idx: Optional[int] = None
        self._tracking_resume_frame: Optional[int] = None  # frame to resume from after manual fix

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
        sidebar.setFixedWidth(210)
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

        # --- Canvas Size section ---
        canvas_sep = QLabel("Canvas Size")
        canvas_sep.setStyleSheet("color: #6c7086; font-size: 11px; padding: 8px 0 4px 0;")
        sidebar_layout.addWidget(canvas_sep)

        wh_row = QHBoxLayout()
        wh_row.setSpacing(4)
        wh_row.addWidget(QLabel("W:"))
        self.canvas_w_input = QLineEdit("400")
        self.canvas_w_input.setFixedWidth(50)
        self.canvas_w_input.setStyleSheet(INPUT_STYLE)
        self.canvas_w_input.returnPressed.connect(self._apply_canvas_size)
        wh_row.addWidget(self.canvas_w_input)
        wh_row.addWidget(QLabel("H:"))
        self.canvas_h_input = QLineEdit("400")
        self.canvas_h_input.setFixedWidth(50)
        self.canvas_h_input.setStyleSheet(INPUT_STYLE)
        self.canvas_h_input.returnPressed.connect(self._apply_canvas_size)
        wh_row.addWidget(self.canvas_h_input)
        swap_btn = QPushButton("⇄")
        swap_btn.setFixedWidth(30)
        swap_btn.setStyleSheet(BUTTON_STYLE)
        swap_btn.clicked.connect(self._swap_canvas_size)
        wh_row.addWidget(swap_btn)

        for lbl in sidebar.findChildren(QLabel):
            if lbl.text() in ("W:", "H:"):
                lbl.setStyleSheet("color: #cdd6f4; font-size: 13px;")

        sidebar_layout.addLayout(wh_row)

        templates_row = QHBoxLayout()
        templates_row.setSpacing(4)
        for label, w, h in [("1:1", 400, 400), ("4:3", 400, 300), ("16:9", 400, 225), ("9:16", 225, 400)]:
            btn = QPushButton(label)
            btn.setStyleSheet(BUTTON_STYLE)
            btn.setFixedHeight(28)
            btn.clicked.connect(lambda *_, _w=w, _h=h: self._set_canvas_preset(_w, _h))
            templates_row.addWidget(btn)
        sidebar_layout.addLayout(templates_row)

        # Fix W:/H: label colors after adding to sidebar
        for lbl in sidebar.findChildren(QLabel):
            if lbl.text() in ("W:", "H:"):
                lbl.setStyleSheet("color: #cdd6f4; font-size: 13px;")

        # --- Tracking section ---
        tracking_sep = QLabel("Tracking")
        tracking_sep.setStyleSheet("color: #6c7086; font-size: 11px; padding: 8px 0 4px 0;")
        sidebar_layout.addWidget(tracking_sep)

        self.tracking_check = QCheckBox("Tracking mode")
        self.tracking_check.setStyleSheet("color: #cdd6f4; font-size: 13px;")
        self.tracking_check.toggled.connect(self._on_tracking_toggled)
        sidebar_layout.addWidget(self.tracking_check)

        self.tracking_status = QLabel("No template")
        self.tracking_status.setStyleSheet("color: #6c7086; font-size: 11px; padding: 2px 0;")
        self.tracking_status.setWordWrap(True)
        sidebar_layout.addWidget(self.tracking_status)

        self.run_tracking_btn = QPushButton("Run Tracking")
        self.run_tracking_btn.setStyleSheet(BUTTON_STYLE)
        self.run_tracking_btn.clicked.connect(self._run_tracking)
        sidebar_layout.addWidget(self.run_tracking_btn)

        self.export_stabilized_btn = QPushButton("Export Stabilized")
        self.export_stabilized_btn.setStyleSheet(BUTTON_STYLE)
        self.export_stabilized_btn.clicked.connect(self._on_export_stabilized)
        sidebar_layout.addWidget(self.export_stabilized_btn)

        sidebar_layout.addStretch()

        # Export button
        self.export_btn = QPushButton("Export Video")
        self.export_btn.setStyleSheet(BUTTON_STYLE)
        self.export_btn.clicked.connect(self._on_export_video)
        sidebar_layout.addWidget(self.export_btn)

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
        self.video_player.roi_selected.connect(self._on_roi_selected)
        self.video_player.point_selected.connect(self._on_manual_center)
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

        self.frame_input = QLineEdit("0")
        self.frame_input.setFixedWidth(60)
        self.frame_input.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.frame_input.setStyleSheet(INPUT_STYLE)
        self.frame_input.returnPressed.connect(self._on_frame_input)
        controls_row.addWidget(self.frame_input)

        self.frame_total_label = QLabel("/ 0")
        self.frame_total_label.setStyleSheet("color: #cdd6f4; font-size: 13px; min-width: 60px;")
        controls_row.addWidget(self.frame_total_label)

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
        self.video_path = path
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
        self.video_player.set_paused(False)
        self.timer.start()

    def _pause(self):
        self.is_playing = False
        self.play_pause_btn.setText("▶")
        self.video_player.set_paused(True)
        self.timer.stop()

    def _on_slider_changed(self, value: int):
        if self._slider_updating:
            return
        self._pause()
        self._seek_frame(value)

    def _on_frame_input(self):
        try:
            idx = int(self.frame_input.text())
        except ValueError:
            return
        idx = max(0, min(idx, self.total_frames - 1))
        self._pause()
        self._seek_frame(idx)

    def _seek_frame(self, idx: int):
        if not self.cap:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame_idx = idx
            self._render_frame(frame)

    def _advance_frame(self):
        if not self.cap or not self.is_playing:
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
        # Auto-pause on lost tracking frames — runs after render so the
        # lost frame is visible when playback stops
        if self.current_frame_idx in self.tracking_overlays:
            _, _, lost = self.tracking_overlays[self.current_frame_idx]
            if lost:
                self._pause()

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

        if self.tracking_mode and self.current_frame_idx in self.tracking_overlays:
            sr, mr, lost = self.tracking_overlays[self.current_frame_idx]
            self.video_player.set_tracking_overlays(sr, mr, lost=lost)
        else:
            self.video_player.set_tracking_overlays(None, None)

        self.video_player.show_frame(frame_rgb, px, py, use_center,
                                     crop_w=self.crop_w, crop_h=self.crop_h)
        self.crop_canvas.show_frame(frame_rgb, px, py, use_center)

        if not self.frame_input.hasFocus():
            self.frame_input.setText(str(self.current_frame_idx))
        self.frame_total_label.setText(f"/ {self.total_frames}")

    def _on_mode_changed(self):
        if not self.cap:
            return
        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        if ret:
            self._render_frame(frame)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

    # --- Canvas size ---

    def _apply_canvas_size(self):
        try:
            w = max(10, int(self.canvas_w_input.text()))
            h = max(10, int(self.canvas_h_input.text()))
        except ValueError:
            return
        self.crop_w = w
        self.crop_h = h
        self.crop_canvas.resize_canvas(w, h)
        self._on_mode_changed()

    def _swap_canvas_size(self):
        w = self.canvas_w_input.text()
        h = self.canvas_h_input.text()
        self.canvas_w_input.setText(h)
        self.canvas_h_input.setText(w)
        self._apply_canvas_size()

    def _set_canvas_preset(self, w: int, h: int):
        self.canvas_w_input.setText(str(w))
        self.canvas_h_input.setText(str(h))
        self._apply_canvas_size()

    # --- Export ---

    def _on_export_video(self):
        if not self.cap or not self.video_path:
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Export Video", "export.mp4", "MP4 Files (*.mp4)"
        )
        if not out_path:
            return

        self.play_pause_btn.setEnabled(False)
        self.export_btn.setEnabled(False)

        use_center = self.center_radio.isChecked()
        saved_idx = self.current_frame_idx

        def progress_cb(i, total):
            self.frame_input.setText(f"Exporting {i}/{total}")
            QApplication.processEvents()

        video_exporter.export_video(
            self.video_path,
            out_path,
            self.points,
            self.crop_w,
            self.crop_h,
            use_center,
            progress_cb,
        )

        self.play_pause_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self._seek_frame(saved_idx)
        self.frame_input.setText(str(saved_idx))

    # --- Tracking ---

    def _on_tracking_toggled(self, checked: bool):
        self.tracking_mode = checked
        self.video_player.set_tracking_mode(checked)
        if self.cap:
            self._seek_frame(self.current_frame_idx)

    def _on_roi_selected(self, x: int, y: int, w: int, h: int):
        if not self.cap:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame_bgr = self.cap.read()
        if not ret:
            return
        self.tracking_engine.set_template(frame_bgr, (x, y, w, h))
        self._template_frame_idx = self.current_frame_idx
        self.tracking_status.setText(f"Template @ frame {self.current_frame_idx}")
        self.tracking_status.setStyleSheet("color: #a6e3a1; font-size: 11px; padding: 2px 0;")

    def _run_tracking(self):
        if not self.video_path or not self.tracking_engine.has_template:
            return

        # Determine start frame: either fresh run or resuming after a manual fix
        resume_from = self._tracking_resume_frame
        if resume_from is not None:
            start_frame = resume_from + 1
            # Use the manually corrected point as the new search origin
            corrected = self.points.get(resume_from, self.tracking_engine.reference_center)
            self.tracking_engine.previous_center = corrected
            self._tracking_resume_frame = None
        else:
            start_frame = 0
            self.tracking_engine.reset_position()
            self.points = {}
            self.tracking_overlays = {}

        self.run_tracking_btn.setEnabled(False)
        cap = cv2.VideoCapture(self.video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        stopped_at = None
        for i in range(start_frame, total):
            ret, frame_bgr = cap.read()
            if not ret:
                break
            try:
                r = self.tracking_engine.track_frame(frame_bgr)
                self.points[i] = r.center
                self.tracking_overlays[i] = (r.search_rect, r.match_rect, False)
            except TrackingLostError:
                # Stop immediately — show the frame, let user fix and resume
                self.points[i] = self.tracking_engine.previous_center
                self.tracking_overlays[i] = (None, None, True)
                self._tracking_resume_frame = i
                stopped_at = i

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                px, py = self.points[i]
                self.video_player.set_tracking_overlays(None, None, lost=True)
                self.video_player.show_frame(frame_rgb, px, py,
                                             self.center_radio.isChecked(),
                                             crop_w=self.crop_w, crop_h=self.crop_h)
                self.current_frame_idx = i
                self._slider_updating = True
                self.frame_slider.setValue(i)
                self._slider_updating = False
                self.frame_input.setText(str(i))
                self.tracking_status.setText(
                    f"Lost at frame {i} — click to fix, then Run Tracking to continue")
                self.tracking_status.setStyleSheet(
                    "color: #fab387; font-size: 11px; padding: 2px 0;")
                break

            if i % 5 == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                px, py = self.points[i]
                sr, mr, lost = self.tracking_overlays[i]
                self.video_player.set_tracking_overlays(sr, mr, lost=lost)
                self.video_player.show_frame(frame_rgb, px, py,
                                             self.center_radio.isChecked(),
                                             crop_w=self.crop_w, crop_h=self.crop_h)
                self.current_frame_idx = i
                self._slider_updating = True
                self.frame_slider.setValue(i)
                self._slider_updating = False
                self.frame_input.setText(f"Tracking {i}/{total}")
                QApplication.processEvents()

        cap.release()
        self.run_tracking_btn.setEnabled(True)

        if stopped_at is None:
            self.tracking_status.setText(f"Done — {total} frames tracked")
            self.tracking_status.setStyleSheet(
                "color: #a6e3a1; font-size: 11px; padding: 2px 0;")

    def _on_export_stabilized(self):
        if not self.video_path or not self.tracking_engine.has_template or not self.points:
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Export Stabilized Video", "stabilized.mp4", "MP4 Files (*.mp4)"
        )
        if not out_path:
            return

        self.export_stabilized_btn.setEnabled(False)
        saved_idx = self.current_frame_idx

        def progress_cb(i, total):
            self.frame_input.setText(f"Stabilizing {i}/{total}")
            QApplication.processEvents()

        video_exporter.export_stabilized(
            self.video_path,
            out_path,
            self.points,
            self.tracking_engine.reference_center,
            progress_cb,
        )

        self.export_stabilized_btn.setEnabled(True)
        self._seek_frame(saved_idx)
        self.frame_input.setText(str(saved_idx))

    def _on_manual_center(self, x: int, y: int):
        self.points[self.current_frame_idx] = (float(x), float(y))
        # If this is the stopped frame, prime the engine so Resume starts from here
        if self.current_frame_idx == self._tracking_resume_frame:
            self.tracking_engine.previous_center = (float(x), float(y))
        self.tracking_status.setText(
            f"Manual point @ frame {self.current_frame_idx} — Run Tracking to continue")
        self.tracking_status.setStyleSheet("color: #89dceb; font-size: 11px; padding: 2px 0;")
        self._seek_frame(self.current_frame_idx)

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
