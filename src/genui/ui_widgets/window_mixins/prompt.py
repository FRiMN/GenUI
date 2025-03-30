from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QDialog, QFileDialog, QLabel
from PyQt6.QtCore import Qt, QSize

from ...ui_widgets.editor_autocomplete import AutoCompleteTextEdit
from ...ui_widgets.lora_table import LoraTable
from pathlib import Path


class SquareButton(QPushButton):
    def sizeHint(self):
        size = super().sizeHint()
        side = min(size.width(), size.height()) + 3
        return QSize(side, side)
        
        
class CircleButton(QPushButton):
    def __init__(self, text, number, color, parent=None):
        super().__init__(parent)
        
        # container = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(8)
        
        size = super().sizeHint()
        print(size.height())
        fsize = super().font().pointSizeF()
        print(fsize)
        s = 20
        self.circle_label = QLabel(str(number))
        self.circle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.circle_label.setFixedSize(s, s)
        self.circle_label.setStyleSheet(f"""
            background-color: {color};
            border-radius: 3px;
            color: white;
            font-weight: bold;
            font-size: {fsize*1}pt;
        """)
        
        text_label = QLabel(text)
        
        layout.addWidget(text_label)
        layout.addWidget(self.circle_label)
        layout.addStretch()
        
        self.setLayout(layout)
        self.setFixedHeight(28)
    
    def setNumber(self, number):
        self.circle_label.setText(str(number))


class LoraWindow(QDialog):
    # FIXME: Close with main window.
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("LoRAs")
        
        self.lora_table = LoraTable()
        self.add_button = SquareButton("+")
        self.add_button.clicked.connect(self.handle_add_button)
        
        layout = QHBoxLayout()
        btn_layout = QVBoxLayout()
        btn_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        btn_layout.addWidget(self.add_button)
        layout.addWidget(self.lora_table)
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)
        
    def handle_add_button(self):
        filepath = QFileDialog.getOpenFileName(self, "LoRA Model")[0]
        if filepath:
            self.lora_table.add_lora(Path(filepath))


class PromptMixin:
    def __init__(self):
        super().__init__()

        self.prompt_editor = AutoCompleteTextEdit()
        self.prompt_editor.setToolTip("Positive prompt")
        self.negative_editor = AutoCompleteTextEdit()
        self.negative_editor.setToolTip("Negative prompt")
        
        self.lora_window = LoraWindow()
        
        self.open_lora_window_btn = CircleButton("LoRA", 0, "gray")
        self.open_lora_window_btn.clicked.connect(self.lora_window.show)
        
        self.lora_window.lora_table.updated.connect(self.open_lora_window_btn.setNumber)

    def _build_prompt_panel(self):
        panel = QVBoxLayout()
        panel.setContentsMargins(0, 0, 0, 0)
        panel.addWidget(self.open_lora_window_btn)
        panel.addWidget(self.prompt_editor)
        panel.addWidget(self.negative_editor)

        panel_box = QWidget()
        panel_box.setLayout(panel)
        return panel_box
