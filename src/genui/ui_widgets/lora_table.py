from pathlib import Path
from typing import Any

from PyQt6.QtWidgets import (
                            QTableWidgetItem, QVBoxLayout, QWidget, 
                            QCheckBox, QDoubleSpinBox, QTableWidget,
                            QHeaderView, QHBoxLayout, QAbstractItemView)
from PyQt6.QtCore import QModelIndex, Qt, pyqtSignal

from ..generator.sdxl import LoRASettings


class LoraTable(QWidget):
    updated = pyqtSignal(int, int)
    
    header_labels = ["", "Name", "Weight"]
    files_paths: dict[str, Path] = {}
    
    cell_active = 0
    cell_name = 1
    cell_weight = 2
    
    def __init__(self, *args):
        super().__init__(*args)
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setMinimumSize(200, 200)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        
        self._set_columns()
        
        self.table.cellClicked.connect(self._handle_row_selection)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.table)
        self.setLayout(layout)
        
    def _set_columns(self):
        self.table.setHorizontalHeaderLabels(self.header_labels)
        
        rm = QHeaderView.ResizeMode
        self.table.horizontalHeader().setSectionResizeMode(self.cell_active, rm.Fixed)
        self.table.setColumnWidth(self.cell_active, 30)
        self.table.horizontalHeader().setSectionResizeMode(self.cell_name, rm.Stretch)
        
    def _handle_row_selection(self, row, column):
        """Always select the row when a cell is clicked"""
        self.table.selectRow(row)
        
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
        
    def _build_weight_spinbox(self) -> QDoubleSpinBox:
        spinbox = QDoubleSpinBox()
        spinbox.setMinimum(0)
        spinbox.setMaximum(1)
        spinbox.setValue(1.0)
        spinbox.setSingleStep(0.05)
        return spinbox
        
    def get_active_loras_indexes(self) -> list[int]:
        indexes = []
        for row in range(self.table.rowCount()):
            checkbox = self.get_active_checkbox(row)
            if checkbox.isChecked():
                indexes.append(row)
        return indexes
        
    def get_active_checkbox(self, row: int) -> QCheckBox:
        """Returns the checkbox widget from Active cell for the given row."""
        return self.table.cellWidget(row, self.cell_active).findChild(QCheckBox)
        
    def _handle_updates(self):
        active_rows = self.get_active_loras_indexes()
        total_rows = self.table.rowCount()
        self.updated.emit(len(active_rows), total_rows)
        
    def add_lora(self, filepath: Path):
        name = self._build_name(filepath)
        if name in self.files_paths:
            return
        
        next_row = self.table.rowCount()
        self.table.insertRow(next_row)
        self.files_paths[name] = filepath
        
        self.table.setCellWidget(next_row, self.cell_active, self._build_active_checkbox())
        self.table.setItem(next_row, self.cell_name, QTableWidgetItem(name))
        self.table.setCellWidget(next_row, self.cell_weight, self._build_weight_spinbox())
        
        self._handle_updates()
        
    def remove_lora(self, row: QModelIndex):
        name = self.table.item(row.row(), self.cell_name).text()
        self.table.removeRow(row.row())
        del self.files_paths[name]
        
        self._handle_updates()
        
    def remove_selected(self):
        """Removes the selected rows from the table."""
        selected_rows = self.table.selectionModel().selectedRows()
        for row in selected_rows:
            self.remove_lora(row)
        
    def clear(self):
        """Clears the table."""
        while self.table.rowCount() > 0:
            self.table.removeRow(0)
        self.files_paths.clear()
        
        self._handle_updates()
        
    def deactivate_all(self):
        """Deactivates all rows in the table."""
        active_rows = self.get_active_loras_indexes()
        for row in active_rows:
            checkbox = self.get_active_checkbox(row)
            checkbox.setChecked(False)
            
        self._handle_updates()
        
    def activate_all(self):
        """Activates all rows in the table."""
        for row in range(self.table.rowCount()):
            checkbox = self.get_active_checkbox(row)
            checkbox.setChecked(True)
            
        self._handle_updates()
        
    def toggle_all(self):
        """Toggle activation for all rows in the table."""
        active_rows = self.get_active_loras_indexes()
        new_state = len(active_rows) == 0
        for row in range(self.table.rowCount()):
            checkbox = self.get_active_checkbox(row)
            checkbox.setChecked(new_state)
                
        self._handle_updates()
        
    def get_loras(self) -> list[LoRASettings]:
        loras = []
        for row in range(self.table.rowCount()):
            name = self.table.item(row, self.cell_name).text()
            filepath = str(self.files_paths[name])
            weight = self.table.cellWidget(row, self.cell_weight).value()
            active = self.get_active_checkbox(row).isChecked()
            loras.append(LoRASettings(name, filepath, weight, active))
        return loras
        