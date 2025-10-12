import pytest
from mock import Mock, patch

from PyQt6.QtGui import QTextCharFormat

from genui.ui_widgets.editor_autocomplete import CompelPromptHighlighter

hl = CompelPromptHighlighter()
pos = hl.pos_highlight_rule
neg = hl.neg_highlight_rule
br = hl.break_highlight_rule

test_data = [
    ("word", []),
    ("another word", []),
    ("word+", [(0, 4, pos)]),
    ("word-", [(0, 4, neg)]),
    ("word++", [(0, 4, pos)]),
    ("word--", [(0, 4, neg)]),
    ("word+++", [(0, 4, pos)]),
    ("word---", [(0, 4, neg)]),
    ("word+,", [(0, 4, pos)]),  # with comma
    ("BREAK", [(0, 5, br)]),
    ("BREAK,", [(0, 5, br)]),   # with comma
    ("word+, (more words)", [(0, 4, pos)]),
    ("(more words), word+", [(14, 4, pos)]),
    ("(more words+), word", [(6, 5, pos)]),
    ("(more words)+, word", [(1, 10, pos)]),
    ("(more word), word+", [(13, 4, pos)]),
    ("(most compelling, most consistent)-", [(1, 15, neg), (18, 15, neg)]),
    ("(accurate, correct)1.5", [(1, 8, pos), (11, 7, pos)]),
    ("word+, word+word", [(0, 4, pos)]),
    ("word+, word-word", [(0, 4, pos)]),
    ("word+, word++word", [(0, 4, pos)]),
    ("word+, word--word", [(0, 4, pos)]),
    ("word+, word+-word", [(0, 4, pos)]),
    ("word+, word-+word", [(0, 4, pos)]),
]

@pytest.mark.parametrize(("text", "expected"), test_data)
@patch('genui.ui_widgets.editor_autocomplete.CompelPromptHighlighter.setFormat')
def test_highlight(mocked_setFormat: Mock, text: str, expected: list[tuple[int, int, QTextCharFormat]]):
    hl.highlightBlock(text)

    assert mocked_setFormat.call_count == len(expected)
    for i, call in enumerate(mocked_setFormat.call_args_list):
        assert call.args == expected[i]


@patch('genui.ui_widgets.editor_autocomplete.CompelPromptHighlighter.setFormat')
def test_long_text(mocked_setFormat: Mock):
    loop_count = 15
    pattern = "sketch-, lineart+, "
    text = pattern * loop_count

    formats = []
    for i in range(loop_count):
        shift = len(pattern) * i
        sketch_format =(
            0+shift,
            6,
            neg
        )
        lineart_format =(
            9+shift,
            7,
            pos
        )
        formats.extend([sketch_format, lineart_format])

    hl.highlightBlock(text)

    assert mocked_setFormat.call_count == len(formats)
    print(mocked_setFormat.call_args_list)
    for i, call in enumerate(mocked_setFormat.call_args_list):
        print(formats[i], i)
        assert call.args == formats[i]
