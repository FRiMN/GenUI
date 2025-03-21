from collections.abc import Callable
from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt

from ...generator.sdxl import interrupt


class GenerationCommandMixin:
    _generate_method: Callable
    _validate_data_for_generation_method: Callable
    
    def __init__(self):
        super().__init__()

        self.button_generate = bg = QtWidgets.QPushButton("Generate", self)
        bg.setStyleSheet("background-color: darkblue")
        bg.clicked.connect(self.handle_generate)
        bg.setShortcut("Ctrl+Return")

        self.button_interrupt = bi = QtWidgets.QPushButton(self)
        bi.setText("Stop")
        bi.setStyleSheet("background-color: darkred")
        # Note: This button is disabled until generation starts
        bi.setDisabled(True)
        bi.clicked.connect(self.handle_interrupt)

        self.action_toolbar = self._create_action_toolbar()

    def _create_action_toolbar(self):
        action_toolbar = QtWidgets.QToolBar("Action", self)
        action_toolbar.addWidget(self.button_generate)
        action_toolbar.addWidget(self.button_interrupt)
        return action_toolbar

    def handle_generate(self):
        if not self._validate_data_for_generation_method():
            self.show_error_modal_dialog()
            return

        self.button_generate.setDisabled(True)
        self.button_interrupt.setDisabled(False)

        try:
            self._generate_method()
        except ValueError as e:
            self.show_error_modal_dialog(str(e))
            self.button_generate.setDisabled(False)
            self.button_interrupt.setDisabled(True)

    def handle_interrupt(self):
        self.button_interrupt.setDisabled(True)
        self.label_status.setText("Interrupting...")
        # FIXME: `self.model_path` can be changed. Need using prompt.
        interrupt(self.model_path)

    def show_error_modal_dialog(self, err_data: str | None = None):
        """Show a modal dialog with an error message."""
        # TODO: Extract to a separate mixin or function.

        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle("Error")

        if err_data:
            msg.setText(err_data)
        else:
            msg.setText("Data for generation not valid! Try check a model is chosen.")

        msg.setWindowModality(Qt.WindowModality.ApplicationModal)

        msg.exec()
