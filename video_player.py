from PyQt6.QtWidgets import QLabel
from PyQt6.QtGui import QPixmap
from typing import Optional
import numpy as np

class VideoPlayerWidget(QLabel):

    def __int__(self):
        super().__init__()

        # Configure widget
        
        self.setStyleSheet("QLabel { background-color: black; }")
        self.setMinimumSize(640, 480)

        # Current frame and display state
        self.current_pixmap: Optional[QPixmap] = None
        self.current_frame: Optional[np.ndarray] = None
        self.display_scale = 1.0