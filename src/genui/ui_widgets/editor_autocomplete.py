# From: https://stackoverflow.com/a/29268818/2404596
from importlib.resources import path as resource_path, open_text
from pathlib import Path

from pkg_resources import resource_string

from PyQt6 import QtCore
from PyQt6.QtWidgets import QCompleter, QPlainTextEdit, QWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QTextCursor, QPalette, QColor, QKeyEvent, QFocusEvent

from ..utils import Timer, BACKGROUND_COLOR_HEX


@Timer("Autocomplete words loader")
def load_words() -> list[str]:
    words = []
    # with resource_path("..resources", "autocomplete.txt") as file_path:
    #     with file_path.open() as f:
    #         lines = f.readlines()

    # current_file_path = Path(__file__).resolve()
    # parent_dir = current_file_path.parent
    # relative_path = Path("../resources")
    # absolute_path = (parent_dir / relative_path).resolve()
    # abs_mod_path = ".".join(absolute_path.parts[1:])
    # print(abs_mod_path)

    with open_text("genui.resources", "autocomplete.txt") as f:
        lines = f.readlines()

    for line in lines:
        word, frequency = line.split(",")
        words.append(word)

    return words


class LastSelectedCompleter(QCompleter):
    insert_text = QtCore.pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        QCompleter.__init__(self, *args, **kwargs)
        self.highlighted.connect(self.setHighlighted)

    def setHighlighted(self, text: str):
        self.lastSelected = text

    def getSelected(self):
        return self.lastSelected


class AwesomeTextEdit(QPlainTextEdit):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self.completer = LastSelectedCompleter(load_words(), parent)
        self.completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.completer.setWidget(self)
        self.completer.insert_text.connect(self.insert_completion)

        palette = self.palette()
        palette.setColor(QPalette.ColorGroup.All, QPalette.ColorRole.Base, QColor.fromString(BACKGROUND_COLOR_HEX))
        self.setPalette(palette)

    def insert_completion(self, completion: str):
        tc = self.textCursor()
        extra = (len(completion) - len(self.completer.completionPrefix()))
        tc.movePosition(QTextCursor.MoveOperation.Left)
        tc.movePosition(QTextCursor.MoveOperation.EndOfWord)
        tc.insertText(completion[-extra:])
        self.setTextCursor(tc)
        self.completer.popup().hide()

    def focusInEvent(self, event: QFocusEvent):
        if self.completer:
            self.completer.setWidget(self)
        QPlainTextEdit.focusInEvent(self, event)

    def keyPressEvent(self, event: QKeyEvent):
        # FIXME.

        tc = self.textCursor()
        popup = self.completer.popup()

        if not popup:
            return

        if (
                event.key() == Qt.Key.Key_Return and popup.isVisible()
                and not tc.hasSelection()
        ):
            self.completer.insert_text.emit(self.completer.getSelected())
            self.completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
            return

        if (
                event.key() == Qt.Key.Key_Up and event.modifiers() == Qt.KeyboardModifier.ControlModifier
                and not popup.isVisible() and tc.hasSelection()
        ):
            selected_text = tc.selectedText()
            # old_increaser = 0.0
            # increaser_delta = 0.05

            if (
                    selected_text.startswith("(")
                    and selected_text.endswith(")")
                    and ":" in selected_text
            ):
                # old_increaser = selected_text[-5:-1]
                selected_text = selected_text[1:-6]

            new_text = f"({selected_text}:0.15)"
            tc.insertText(new_text)
            return

        if tc.hasSelection():
            return

        QPlainTextEdit.keyPressEvent(self, event)
        tc.select(QTextCursor.SelectionType.WordUnderCursor)

        selected_text = tc.selectedText()
        if len(selected_text) > 2:  # noqa: PLR2004
            self.completer.setCompletionPrefix(selected_text)

            cur_index = self.completer.completionModel().index(0, 0)
            popup.setCurrentIndex(cur_index)

            cr = self.cursorRect()
            cr.setWidth(
                popup.sizeHintForColumn(0)
                + popup.verticalScrollBar().sizeHint().width()
            )
            self.completer.complete(cr)
        else:
            popup.hide()
