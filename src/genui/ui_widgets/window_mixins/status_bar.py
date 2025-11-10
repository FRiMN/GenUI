from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QLabel, QPushButton, QStatusBar, QWidget, QProgressBar, QFileDialog, QHBoxLayout
from collections.abc import Callable
from typing import Dict, Optional

from ...utils import processing_time_estimator


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

        self.label_gpu_memory = QLabel()
        self.label_gpu_memory.setToolTip("Waiting GPU Memory Usage")
        self.label_gpu_memory.setText("VRAM: N/A")

        self.load_image_button = QPushButton()
        self.load_image_button.setToolTip("Load image and read metadata")
        icon = QIcon.fromTheme("accessories-text-editor")
        self.load_image_button.setIcon(icon)
        self.load_image_button.clicked.connect(self.load_image_button_clicked)

        eta_label = QLabel("ETA:")
        self.eta_secs = QLabel()
        # Create a combined widget for ETA display
        self.eta = QWidget()
        eta_layout = QHBoxLayout(self.eta)
        eta_layout.setContentsMargins(0, 0, 0, 0)
        eta_layout.addWidget(eta_label)
        eta_layout.addWidget(self.eta_secs)

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

        status_bar.addWidget(self.eta)
        status_bar.addWidget(self.label_memory_usage)
        status_bar.addWidget(self.label_gpu_memory)
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

    def update_gpu_memory_display(self, gpu_info: Optional[Dict] = None):
        """Update GPU memory display in status bar"""
        if not gpu_info:
            self.label_gpu_memory.setText("VRAM: N/A")
            self.label_gpu_memory.setToolTip("GPU Memory Usage: N/A")
            return

        try:
            used_gb = gpu_info['used_mem'] / (1024**3)  # Convert to GB
            total_gb = gpu_info['total_mem'] / (1024**3)  # Convert to GB
            allocated_gb = gpu_info['allocated'] / (1024**3)  # Convert to GB
            reserved_gb = gpu_info['reserved'] / (1024**3)  # Convert to GB

            # Format display text
            percent = (used_gb / total_gb) * 100
            display_text = f"VRAM: {percent:.1f}%"

            # Create detailed tooltip
            tooltip_text = (
                f"GPU Memory Usage:\n"
                f"Device: {gpu_info.get('device', 'N/A')}\n"
                f"Used: {used_gb:.2f} GB\n"
                f"Total: {total_gb:.2f} GB\n"
                f"Free: {gpu_info['free_mem'] / (1024**3):.2f} GB\n"
                f"Allocated: {allocated_gb:.2f} GB\n"
                f"Reserved: {reserved_gb:.2f} GB\n"
                f"Max Allocated: {gpu_info['max_allocated'] / (1024**3):.2f} GB\n"
                f"Max Reserved: {gpu_info['max_reserved'] / (1024**3):.2f} GB"
            )

            self.label_gpu_memory.setText(display_text)
            self.label_gpu_memory.setToolTip(tooltip_text)

            # Change color based on usage percentage
            usage_percent = (used_gb / total_gb) * 100
            if usage_percent > 90:
                self.label_gpu_memory.setStyleSheet("color: red")
            elif usage_percent > 75:
                self.label_gpu_memory.setStyleSheet("color: orange")
            else:
                self.label_gpu_memory.setStyleSheet(None)

        except (KeyError, TypeError, ZeroDivisionError) as e:
            print(f"Error updating GPU memory display: {e}")
            self.label_gpu_memory.setText("VRAM: Error")
            self.label_gpu_memory.setToolTip("Error reading GPU memory info")
            
    def update_eta(self, mpx: float):
        self.eta_secs.setText(f"{processing_time_estimator.eta(mpx).seconds:.2f} sec")
