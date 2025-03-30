from pathlib import Path
import sys
from PyQt6.QtWidgets import (QTableWidget, 
                            QTableWidgetItem, QVBoxLayout, QWidget, 
                            QPushButton, QCheckBox, QDoubleSpinBox, 
                            QHeaderView, QHBoxLayout, QFileDialog)
from PyQt6.QtCore import Qt, pyqtSignal


class LoraTable(QWidget):
    updated = pyqtSignal(int)
    
    header_labels = ["", "Name", "Weight"]
    files_paths: dict[str, Path] = {}
    
    def __init__(self, *args):
        super().__init__(*args)
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setMinimumSize(200, 200)
        
        self._set_columns()
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.table)
        self.setLayout(layout)
        
    def _set_columns(self):
        self.table.setHorizontalHeaderLabels(self.header_labels)
        
        rm = QHeaderView.ResizeMode
        self.table.horizontalHeader().setSectionResizeMode(0, rm.Fixed)
        self.table.setColumnWidth(0, 30)
        self.table.horizontalHeader().setSectionResizeMode(1, rm.Stretch)
        
        # self.table.verticalHeader().setVisible(True)
        
    def _build_active_checkbox(self):
        check_widget = QWidget()
        check_layout = QHBoxLayout(check_widget)
        check_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        checkbox = QCheckBox()
        checkbox.setChecked(True)
        checkbox.stateChanged.connect(self._handle_updates)
        
        check_layout.addWidget(checkbox)
        check_widget.setLayout(check_layout)
        return check_widget
        
    def _build_name(self, filepath: Path) -> str:
        return filepath.name.split(".")[0]
        
    def _build_weight_spinbox(self):
        spinbox = QDoubleSpinBox()
        spinbox.setMinimum(0)
        spinbox.setMaximum(1)
        spinbox.setValue(1.0)
        spinbox.setSingleStep(0.05)
        return spinbox
        
    def get_active_loras_indexes(self) -> list[int]:
        indexes = []
        for row in range(self.table.rowCount()):
            checkbox = self.table.cellWidget(row, 0).findChild(QCheckBox)
            if checkbox.isChecked():
                indexes.append(row)
        return indexes
        
    def _handle_updates(self):
        active_rows = self.get_active_loras_indexes()
        self.updated.emit(len(active_rows))
        
    def add_lora(self, filepath: Path):
        next_row = self.table.rowCount()
        self.table.insertRow(next_row)
        name = self._build_name(filepath)
        self.files_paths[name] = filepath
        
        self.table.setCellWidget(next_row, 0, self._build_active_checkbox())
        self.table.setItem(next_row, 1, QTableWidgetItem(name))
        self.table.setCellWidget(next_row, 2, self._build_weight_spinbox())
        
        self._handle_updates()
