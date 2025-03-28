from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QLabel, QPushButton, QStatusBar, QWidget, QProgressBar, QFileDialog
from collections.abc import Callable


class StatusBarMixin:
    _load_image: Callable[[str], None]
    
    def __init__(self):
        super().__init__()

        self.label_process = QProgressBar(self)
        self.label_process.setMinimum(0)
        self.label_process.setFormat("%v/%m")
        self.label_process.setFixedWidth(150)

        self.label_status = QLabel()
        self.label_image_path = QLabel()
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
        
        self.load_image_button = QPushButton()
        self.load_image_button.setToolTip("Load image and read metadata")
        icon = QIcon.fromTheme("accessories-text-editor")
        self.load_image_button.setIcon(icon)
        self.load_image_button.clicked.connect(self.load_image_button_clicked)

        self.status_bar = self._create_status_bar()
        self.setStatusBar(self.status_bar)

    def _create_status_bar(self):
        status_bar = QStatusBar()
        status_bar.addWidget(self.label_process)
        status_bar.addWidget(self.label_status)
        status_bar.addWidget(self.label_image_path)

        # Add a spacer widget to push the next status bar widgets to the right.
        spacer = QWidget()
        status_bar.addWidget(spacer, 1)

        status_bar.addWidget(self.label_viewer_image_size)
        status_bar.addWidget(self.zoom_label)
        status_bar.addWidget(self.load_image_button)
        status_bar.addWidget(self.zoom_fit_button)
        status_bar.addWidget(self.zoom_orig_button)

        return status_bar
        
    def load_image_button_clicked(self):
        image_path = QFileDialog.getOpenFileName(self, "Model")[0]
        if image_path:
            self._load_image(image_path)
