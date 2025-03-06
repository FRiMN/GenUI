from PyQt6.QtWidgets import QWidget, QVBoxLayout

from ...ui_widgets.editor_autocomplete import AwesomeTextEdit


class PromptMixin:
    def __init__(self):
        super().__init__()

        self.prompt_editor = AwesomeTextEdit()
        self.prompt_editor.setToolTip("Positive prompt")
        self.negative_editor = AwesomeTextEdit()
        self.negative_editor.setToolTip("Negative prompt")

    def _build_prompt_panel(self):
        panel = QVBoxLayout()
        panel.setContentsMargins(0, 0, 0, 0)
        panel.addWidget(self.prompt_editor)
        panel.addWidget(self.negative_editor)

        panel_box = QWidget()
        panel_box.setLayout(panel)
        return panel_box
