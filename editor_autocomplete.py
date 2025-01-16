# From: https://stackoverflow.com/a/29268818/2404596
from pathlib import Path

from PyQt6 import QtCore
from PyQt6.QtWidgets import QCompleter, QPlainTextEdit
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QTextCursor


def load_words() -> list[str]:
    words = []
    path = Path("./autocomplete.txt")
    with path.open() as f:
        lines = f.readlines()

    for line in lines:
        word, frequency = line.split(",")
        words.append(word)

    return words


class AwesomeTextEdit(QPlainTextEdit):
    def __init__(self, parent=None):
        super(AwesomeTextEdit, self).__init__(parent)

        self.completer = MyCompleter(load_words(), parent)
        self.completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.completer.setWidget(self)
        self.completer.insertText.connect(self.insertCompletion)

    def insertCompletion(self, completion):
        tc = self.textCursor()
        extra = (len(completion) - len(self.completer.completionPrefix()))
        tc.movePosition(QTextCursor.MoveOperation.Left)
        tc.movePosition(QTextCursor.MoveOperation.EndOfWord)
        tc.insertText(completion[-extra:])
        self.setTextCursor(tc)
        self.completer.popup().hide()

    def focusInEvent(self, event):
        if self.completer:
            self.completer.setWidget(self)
        QPlainTextEdit.focusInEvent(self, event)

    def keyPressEvent(self, event):
        # TODO: Fix partial word completion: exmpl. "reast".
        # print(f"{event.key()=}")

        tc = self.textCursor()
        # print(f"{tc.hasSelection()=}")

        if (
                event.key() == Qt.Key.Key_Return and self.completer.popup().isVisible()
                and not tc.hasSelection()
        ):
            self.completer.insertText.emit(self.completer.getSelected())
            self.completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
            return

        if (
                event.key() == Qt.Key.Key_Up and event.modifiers() == Qt.KeyboardModifier.ControlModifier
                and not self.completer.popup().isVisible() and tc.hasSelection()
        ):
            selected_text = tc.selectedText()
            print(f"{selected_text=}")
            old_increaser = 0.0
            increaser_delta = 0.05

            if (
                    selected_text.startswith("(")
                    and selected_text.endswith(")")
                    and ":" in selected_text
            ):
                old_increaser = selected_text[-5:-1]
                print(f"{old_increaser=}")
                selected_text = selected_text[1:-6]

            new_text = f"({selected_text}:0.15)"
            tc.insertText(new_text)
            return

        if tc.hasSelection():
            return

        QPlainTextEdit.keyPressEvent(self, event)
        tc.select(QTextCursor.SelectionType.WordUnderCursor)

        selected_text = tc.selectedText()
        if len(selected_text) > 2:
            self.completer.setCompletionPrefix(selected_text)
            popup = self.completer.popup()
            popup.setCurrentIndex(self.completer.completionModel().index(0, 0))

            cr = self.cursorRect()
            cr.setWidth(
                popup.sizeHintForColumn(0)
                + popup.verticalScrollBar().sizeHint().width()
            )
            self.completer.complete(cr)
        else:
            self.completer.popup().hide()


class MyCompleter(QCompleter):
    insertText = QtCore.pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        QCompleter.__init__(self, *args, **kwargs)
        self.highlighted.connect(self.setHighlighted)

    def setHighlighted(self, text):
        self.lastSelected = text

    def getSelected(self):
        return self.lastSelected
