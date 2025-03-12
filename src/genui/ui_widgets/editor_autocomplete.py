from importlib.resources import open_text

from PyQt6.QtWidgets import QCompleter, QTextEdit, QAbstractItemView
from PyQt6.QtCore import Qt, QStringListModel, QRegularExpression
from PyQt6.QtGui import QTextCursor, QPalette, QColor, QKeyEvent, QSyntaxHighlighter, QTextCharFormat, QFont

from ..utils import Timer, BACKGROUND_COLOR_HEX
from ..settings import settings


@Timer("Autocomplete words loader")
def load_words() -> list[str]:
    words = []

    with open_text("genui.resources", "autocomplete.txt") as f:
        lines = f.readlines()

    for line in lines:
        word, frequency = line.split(",")
        words.append(word)

    return words
    
    
class PromptHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlighting_rules = []

        # self.add_rule(r"\,", Qt.GlobalColor.green, None)
        # self.add_rule(r"\.", Qt.GlobalColor.green, None)
        # # case: `apricots+`
        self.add_rule(r"\b\S+[\+\-]+", None, settings.prompt_editor.compel_font_weight)
        # case: `(picking apricots)++`
        self.add_rule(r"\(.+?\)[\+\-]+", None, settings.prompt_editor.compel_font_weight)
        # case: `(picking (apricots)1.3)1.1, (apricots)1.1`
        self.add_rule(r"\([^,]+\)\d\.\d\b", None, settings.prompt_editor.compel_font_weight)
        
    def add_rule(self, pattern, color: Qt.GlobalColor | None, weight: int | None):
        regex = QRegularExpression(pattern)
        format = QTextCharFormat()
        
        if weight is not None:
            format.setFontWeight(weight)
        if color:
            format.setForeground(QColor(color))
            
        self.highlighting_rules.append((regex, format))

    def highlightBlock(self, text):
        """Apply highlighting rules to the current block of text."""
        for regex, format in self.highlighting_rules:
            match_iterator = regex.globalMatch(text)
            while match_iterator.hasNext():
                match = match_iterator.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), format)
    
    
class WordsCompleter(QCompleter):
    words = load_words()
    completer_model = QStringListModel(words)
    
    def __init__(self, parent=None):
        super().__init__(self.completer_model, parent)
        
        self.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.setFilterMode(Qt.MatchFlag.MatchContains)


class AutoCompleteTextEdit(QTextEdit):
    min_word_length = 2
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_font()
        self.setup_completer()
        self.highlighter = PromptHighlighter(self.document())
        
        palette = self.palette()
        palette.setColor(QPalette.ColorGroup.All, QPalette.ColorRole.Base, QColor.fromString(BACKGROUND_COLOR_HEX))
        self.setPalette(palette)

    def setup_completer(self):
        self.completer = WordsCompleter(self)
        self.completer.setWidget(self)

        self.completer.activated.connect(self.insert_completion)
        self.textChanged.connect(self.updateCompleter)
        
    def setup_font(self):
        s = settings.prompt_editor
        
        font = QFont()
        font.setFamily(s.font_family or self.font().family())
        font.setPointSize(s.font_size)
        font.setWeight(s.font_weight)
        
        self.setFont(font)
        
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
