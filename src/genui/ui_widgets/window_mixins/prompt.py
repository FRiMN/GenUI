from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QDialog, QFileDialog
from PyQt6.QtCore import Qt, QSize, QRect
from PyQt6.QtGui import QColor, QBrush, QPainter, QIcon

from ...ui_widgets.editor_autocomplete import AutoCompleteTextEdit
from ...ui_widgets.lora_table import LoraTable
from pathlib import Path


class SquareButton(QPushButton):
    def sizeHint(self):
        size = super().sizeHint()
        side = min(size.width(), size.height()) + 3
        return QSize(side, side)
        
        
class CounterButton(QPushButton):    
    def __init__(self, text: str, number: int, total: int, parent=None):
        super().__init__(text, parent)
        
        self.text = text
        self.number = number
        self.total = total
    
    def setNumber(self, number: int, total: int | None = None):
        self.number = number
        if total is not None:
            self.total = total
        self.update()   # Emit paintEvent
        
    def paintEvent(self, event):
        super().paintEvent(event)
        self.setText(f"{self.text}: {self.number}/{self.total}")


class LoraWindow(QDialog):
    # FIXME: Close with main window.
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("LoRAs")
        
        self.lora_table = LoraTable()
        
        icon = QIcon.fromTheme(QIcon.ThemeIcon.ListAdd)
        self.add_button = add = SquareButton()
        add.setIcon(icon)
        add.setToolTip("Add a new LoRA")
        add.clicked.connect(self.handle_add_button)
        
        icon = QIcon.fromTheme(QIcon.ThemeIcon.ListRemove)
        self.remove_button = remove = SquareButton()
        remove.setIcon(icon)
        remove.setToolTip("Remove selected LoRAs")
        remove.clicked.connect(self.lora_table.remove_selected)
        
        icon = QIcon.fromTheme("user-trash")
        self.clear_button = clear = SquareButton()
        clear.setIcon(icon)
        clear.setToolTip("Clear all LoRAs")
        clear.clicked.connect(self.lora_table.clear)
        
        icon = QIcon.fromTheme("emblem-synchronizing")
        self.toggle_button = toggle = SquareButton()
        toggle.setIcon(icon)
        toggle.setToolTip("Toggle all LoRAs")
        toggle.clicked.connect(self.lora_table.toggle_all)
        
        layout = QHBoxLayout()
        btn_layout = QVBoxLayout()
        btn_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        btn_layout.addWidget(self.add_button)
        btn_layout.addWidget(self.remove_button)
        btn_layout.addWidget(self.clear_button)
        btn_layout.addWidget(self.toggle_button)
        
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
        
        self.open_lora_window_btn = CounterButton("LoRA", 0, 0)
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
