from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QLabel, QPushButton, QStatusBar, QWidget, QProgressBar


class StatusBarMixin:
    def __init__(self):
        super().__init__()

        self.label_process = QProgressBar(self)
        self.label_process.setMinimum(0)
        self.label_process.setFormat("%v/%m")
        self.label_process.setFixedWidth(150)

        self.label_status = QLabel(self)

        self.zoom_label = QLabel()

        icon = QIcon.fromTheme(QIcon.ThemeIcon.ZoomFitBest)
        self.zoom_fit_button = QPushButton()
        self.zoom_fit_button.setIcon(icon)
        self.zoom_fit_button.setToolTip("Fit image to viewport")

        self.zoom_orig_button = QPushButton()
        icon = QIcon.fromTheme("zoom-original")
        self.zoom_orig_button.setIcon(icon)
        self.zoom_orig_button.setToolTip("Set original size of image")

        self.label_viewer_image_size = QLabel()

        self.status_bar = self._create_status_bar()
        self.setStatusBar(self.status_bar)

    def _create_status_bar(self):
        status_bar = QStatusBar()
        status_bar.addWidget(self.label_process)
        status_bar.addWidget(self.label_status)

        # Add a spacer widget to push the next status bar widgets to the right.
        spacer = QWidget()
        status_bar.addWidget(spacer, 1)

        status_bar.addWidget(self.label_viewer_image_size)
        status_bar.addWidget(self.zoom_label)
        status_bar.addWidget(self.zoom_fit_button)
        status_bar.addWidget(self.zoom_orig_button)

        return status_bar
