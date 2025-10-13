from __future__ import annotations
from typing import TYPE_CHECKING
from contextlib import suppress
from importlib.resources import open_text

from PyQt6.QtWidgets import QCompleter, QTextEdit, QAbstractItemView, QWidget
from PyQt6.QtCore import Qt, QStringListModel, QRegularExpression, QMimeData
from PyQt6.QtGui import QTextCursor, QPalette, QColor, QKeyEvent, QSyntaxHighlighter, QTextCharFormat, QFont
from compel.prompt_parser import PromptParser

from ..utils import BACKGROUND_COLOR_HEX
from ..common.trace import Timer
from ..settings import settings
from .window_mixins.propagate_events import PropagateEventsMixin

if TYPE_CHECKING:
    from compel.prompt_parser import Conjunction, Prompt, FlattenedPrompt, Fragment


@Timer("Autocomplete words loader")
def load_words() -> list[str]:
    words = []

    with open_text("genui.resources", "autocomplete.txt") as f:
        lines = f.readlines()

    for line in lines:
        word, frequency = line.split(",")
        words.append(word)

    return words


class CompelPromptHighlighter(QSyntaxHighlighter):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.parser = PromptParser()
        self.setup_highlighting_rules()

    def setup_highlighting_rules(self):
        # Базовые правила для ключевых слов и операторов
        self.break_highlight_rule = bh = QTextCharFormat()
        bh.setForeground(QColor(Qt.GlobalColor.gray))
        bh.setFontItalic(True)
        self.break_regex = QRegularExpression(r"\bBREAK\b")

        self.pos_highlight_rule = ph = QTextCharFormat()
        ph.setForeground(QColor(Qt.GlobalColor.green))

        self.neg_highlight_rule = nh = QTextCharFormat()
        nh.setForeground(QColor(Qt.GlobalColor.red))

    def highlightBlock(self, text: str):
        """Apply highlighting rules to the current block of text."""
        self.highlight_syntax(text)
        self.highlight_break(text)

    def highlight_break(self, text: str):
        match_iterator = self.break_regex.globalMatch(text)
        while match_iterator.hasNext():
            match = match_iterator.next()
            self.setFormat(match.capturedStart(), match.capturedLength(), self.break_highlight_rule)

    def highlight_syntax(self, text: str) -> None:
        """Расширенная подсветка на основе синтаксического анализа"""
        with suppress(Exception):
            # Парсим промпт для получения структуры
            conjunction = self.parser.parse_conjunction(text, verbose=False)
            # Рекурсивно обходим структуру для подсветки
            self.highlight_conjunction(conjunction, text)

    def highlight_conjunction(self, conjunction: Conjunction, full_text: str) -> None:
        """Подсвечивает элементы конъюнкции"""
        # Глобальный курсор. Необходим для повторяющихся слов в тексте.
        # Кейс: `<start_hl>word<end_hl>+, (more <start_hl>word<end_hl>s)`.
        global_pos = 0

        for prompt in conjunction.prompts:
            if hasattr(prompt, 'children'):
                global_pos = self.highlight_prompt_fragments(prompt, full_text, global_pos)

    def highlight_prompt_fragments(
        self,
        prompt: Prompt | FlattenedPrompt,
        full_text: str,
        global_pos: int
    ) -> int:
        """Подсвечивает элементы промпта"""
        for fragment in prompt.children:
            global_pos = self.highlight_fragment(fragment, full_text, global_pos)
        return global_pos

    def highlight_fragment(self, fragment: Fragment, full_text: str, global_pos: int) -> int:
        """Подсвечивает конкретный элемент синтаксиса"""
        fragment_text = self.get_fragment_text(fragment)

        elements = fragment_text.split(",")
        for element_text in elements:
            element_text = element_text.strip()
            if not element_text:
                # print(f"Warning: Empty element found in fragment '{fragment}'")
                continue

            pos, global_pos = self.find_element_in_text(full_text, element_text, global_pos)
            if pos == -1:
                print(f"Warning: Element '{fragment}' not found in text")
                return global_pos

            if hasattr(fragment, 'weight') and fragment.weight != 1.0:
                format_rule = (
                    self.pos_highlight_rule
                    if fragment.weight > 1.0
                    else self.neg_highlight_rule
                )
                self.setFormat(pos, len(element_text), format_rule)

        return global_pos

    def get_fragment_text(self, fragment: Fragment) -> str:
        """Извлекает текстовое представление элемента"""
        text = ""

        if hasattr(fragment, 'text'):
            text = fragment.text

        elif hasattr(fragment, '__repr__'):
            # TODO: may be not need.
            repr_text = repr(fragment)
            # Извлекаем текст из repr представления
            if 'Fragment:' in repr_text:
                text = repr_text.split("'")[1] if "'" in repr_text else repr_text.split(':')[1]

        text = text.strip(",").strip()
        return text
        
    def is_this_whole_word(self, full_text: str, pos: int, element: str) -> bool:
        """Проверяет, является ли элемент целым словом в тексте"""
        if pos > 0:
            if full_text[pos - 1] not in ("(", " ", ","):
                return False

        if pos < len(full_text)-1:
            if full_text[pos + len(element)] not in (")", " ", ",", "+", "-"):
                return False
                
        return True

    def find_element_in_text(self, full_text: str, element: str, global_pos: int) -> tuple[int, int]:
        while global_pos < len(full_text):
            pos = full_text.find(element, global_pos)
            if pos == -1:
                break
                
            if not self.is_this_whole_word(full_text, pos, element):
                global_pos = pos + len(element)
                continue

            global_pos = pos + len(element)
            return pos, global_pos
        return -1, global_pos


class WordsCompleter(QCompleter):
    words = load_words()
    completer_model = QStringListModel(words)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(self.completer_model, parent)

        self.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.setFilterMode(Qt.MatchFlag.MatchContains)


class AutoCompleteTextEdit(QTextEdit, PropagateEventsMixin):
    min_word_length = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_font()
        self.setup_completer()
        self.highlighter = CompelPromptHighlighter(self.document())

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
        is_compel_operator = word_under_cursor.startswith(("+", "-"))
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

    def insertFromMimeData(self, source: QMimeData, src_type: str = ""):
        """Insert text via drag&drop, Ctrl+V, menu and etc."""
        if not source.hasText():
            super().insertFromMimeData(source)

        self.insertPlainText(source.text())

    def keyPressEvent(self, e: QKeyEvent):
        if self.completer.popup().isVisible() and e.key() == Qt.Key.Key_Return:
            e.ignore()
            return

        super().keyPressEvent(e)
