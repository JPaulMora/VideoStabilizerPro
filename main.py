import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout,
    QVBoxLayout, QPushButton, QLabel, QFrame
)
from PyQt6.QtCore import Qt
from video_player import VideoPlayerWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VideoStPro")
        self.setMinimumSize(900, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        root_layout = QHBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Sidebar
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

        sidebar_title = QLabel("VideoStPro")
        sidebar_title.setStyleSheet("color: #cdd6f4; font-size: 16px; font-weight: bold; padding: 4px 0 12px 0;")
        sidebar_layout.addWidget(sidebar_title)

        load_video_btn = QPushButton("Load Video")
        load_video_btn.setStyleSheet("""
            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border: none;
                border-radius: 6px;
                padding: 10px 12px;
                text-align: left;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #45475a;
            }
            QPushButton:pressed {
                background-color: #585b70;
            }
        """)
        load_video_btn.clicked.connect(self.load_video)
        sidebar_layout.addWidget(load_video_btn)

        sample_btn = QPushButton("Sample Action")
        sample_btn.setStyleSheet("""
            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border: none;
                border-radius: 6px;
                padding: 10px 12px;
                text-align: left;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #45475a;
            }
            QPushButton:pressed {
                background-color: #585b70;
            }
        """)
        sample_btn.clicked.connect(self.sample_function)
        sidebar_layout.addWidget(sample_btn)

        sidebar_layout.addStretch()

        # Main content area
        content = QFrame()
        content.setObjectName("content")

        content_layout = QVBoxLayout(content)
        content_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Sample label
        self.status_label = QLabel("Click 'Sample Action' in the sidebar.")
        self.status_label.setStyleSheet("color: #6c7086; font-size: 14px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Add video player widget
        self.video_player = VideoPlayerWidget()
        content_layout.addWidget(self.video_player, stretch=1)

        root_layout.addWidget(sidebar)
        root_layout.addWidget(content, stretch=1)

    def load_video(self):
        self.status_label.setText("Load video triggered!")
        self.status_label.setStyleSheet("color: #89b4fa; font-size: 14px;")

    def sample_function(self):
        self.status_label.setText("Sample function triggered!")
        self.status_label.setStyleSheet("color: #a6e3a1; font-size: 14px;")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()