from importlib.resources import open_text

from PyQt6.QtWidgets import QCompleter, QTextEdit, QAbstractItemView
from PyQt6.QtCore import Qt, QStringListModel
from PyQt6.QtGui import QTextCursor, QPalette, QColor, QKeyEvent

from ..utils import Timer, BACKGROUND_COLOR_HEX


@Timer("Autocomplete words loader")
def load_words() -> list[str]:
    words = []

    with open_text("genui.resources", "autocomplete.txt") as f:
        lines = f.readlines()

    for line in lines:
        word, frequency = line.split(",")
        words.append(word)

    return words


class AutoCompleteTextEdit(QTextEdit):
    min_word_length = 2
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_completer()
        
        palette = self.palette()
        palette.setColor(QPalette.ColorGroup.All, QPalette.ColorRole.Base, QColor.fromString(BACKGROUND_COLOR_HEX))
        self.setPalette(palette)

    def setup_completer(self):
        words = load_words()
        self.completer_model = QStringListModel(words)

        self.completer = QCompleter(self.completer_model, self)
        self.completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.completer.setWidget(self)

        self.completer.activated.connect(self.insert_completion)
        self.textChanged.connect(self.updateCompleter)
        
    def fix_special_symbols_selected(self, cursor: QTextCursor) -> None:
        """
        Fixes the error when selecting word before a comma or Compel operator (+ or -).
        
        Example:
            Expected: "<start selection>selected_word<cursor_position><end selection>,"
            Actual: "selected_word<start selection><cursor_position>,<end selection>"
        """
        word_under_cursor = cursor.selectedText()
        is_compel_operator = word_under_cursor.startswith("+") or word_under_cursor.startswith("-")
        is_comma = word_under_cursor == ","
        if is_comma or is_compel_operator:
            cursor.movePosition(QTextCursor.MoveOperation.Left, QTextCursor.MoveMode.MoveAnchor, 2)
            cursor.select(cursor.SelectionType.WordUnderCursor)

    def insert_completion(self, completion: str):
        """Sets the completion text in the editor."""
        cursor = self.textCursor()
        cursor.select(cursor.SelectionType.WordUnderCursor)
        self.fix_special_symbols_selected(cursor)

        cursor.removeSelectedText()
        cursor.insertText(completion)
        self.setTextCursor(cursor)  # Return cursor to its original position

    def reset_completer(self) -> None:
        self.completer.setCompletionPrefix("")
        self.completer.popup().hide()

    def updateCompleter(self):
        """Update the completer."""
        popup: QAbstractItemView = self.completer.popup()
        cursor = self.textCursor()
        cursor.select(cursor.SelectionType.WordUnderCursor)
        self.fix_special_symbols_selected(cursor)

        word_under_cursor = cursor.selectedText()

        if len(word_under_cursor) < self.min_word_length:
            self.reset_completer()
            return

        exist_prefix = self.completer.completionPrefix()
        if word_under_cursor != exist_prefix:
            self.completer.setCompletionPrefix(word_under_cursor)
            
        if self.completer.completionCount() == 0:
            self.reset_completer()
            return
            
        indx = self.completer.completionModel().index(0, 0) # First item
        
        if (
            self.completer.completionCount() == 1 and 
            indx.data() == word_under_cursor
        ):
            """Completion already accepted"""
            self.reset_completer()
            return

        popup.setCurrentIndex(indx) # Select first item
        
        cr = self.cursorRect()
        cr.setWidth(
            popup.sizeHintForColumn(0)
            + popup.verticalScrollBar().sizeHint().width()
        )
        self.completer.complete(cr) # Show popup
        
    def keyPressEvent(self, e: QKeyEvent):
        if self.completer.popup().isVisible():
            if e.key() == Qt.Key.Key_Return:
                e.ignore()
                return
                
        super().keyPressEvent(e)
