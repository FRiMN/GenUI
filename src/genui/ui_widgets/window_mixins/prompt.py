from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QDialog, QFileDialog
from PyQt6.QtCore import Qt, QSize, QRect
from PyQt6.QtGui import QColor, QBrush, QPainter

from ...ui_widgets.editor_autocomplete import AutoCompleteTextEdit
from ...ui_widgets.lora_table import LoraTable
from pathlib import Path


class SquareButton(QPushButton):
    def sizeHint(self):
        size = super().sizeHint()
        side = min(size.width(), size.height()) + 3
        return QSize(side, side)
        
        
class CounterButton(QPushButton):
    text_margin = 5
    
    def __init__(self, text, number, color, parent=None):
        super().__init__(text, parent)
        
        self.number = number
    
    def setNumber(self, number: int):
        self.number = number
        self.update()   # Emit paintEvent
        
    @property
    def _counter_radius(self) -> int:
        return self.height()//2
        
    @property
    def _counter_rect(self) -> QRect:
        text_width = self.fontMetrics().horizontalAdvance(self.text())
        text_start = (self.width() - text_width) // 2
        radius = self._counter_radius
        x = text_start + text_width + self.text_margin
        print(f"{self.width()=}; {text_width=}; {text_start=}; {x=}")
        y = (self.height()-radius)//2
        
        center = (x, y)
        size = (radius, radius)
        return QRect(*center, *size)
        
    @property
    def color(self) -> QColor:
        color = "gray" if self.number == 0 else "#f0a04b"
        return QColor(color)
        
    def paintEvent(self, event):
        super().paintEvent(event)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        font = painter.font()
        font.setPixelSize(self._counter_radius)
        painter.setFont(font)
        
        paint_rect = self._counter_rect
        
        painter.setBrush(QBrush(self.color))
        painter.drawEllipse(paint_rect)
        
        painter.setPen(QColor("white"))
        painter.drawText(
            paint_rect, 
            Qt.AlignmentFlag.AlignCenter, str(self.number)
        )


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
        
        self.open_lora_window_btn = CounterButton("LoRA", 0, "gray")
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
