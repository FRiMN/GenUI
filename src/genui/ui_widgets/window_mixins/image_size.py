from PyQt6 import QtWidgets

from ...utils import TOOLBAR_MARGIN

ASPECT_RATIOS = (
    "1:1",
    "L 5:4",
    "L 4:3",
    "L 3:2",
    "L 16:10",
    "L 16:9",
    "L 21:9",
    "P 4:5",
    "P 3:4",
    "P 2:3",
    "P 9:10",
    "P 9:16",
    "P 9:21",
)


class ImageSizeMixin:
    def __init__(self):
        super().__init__()

        self.image_size = (0, 0)

        self.base_size_editor = bse = QtWidgets.QSpinBox()
        bse.setMinimum(512)
        bse.setMaximum(8192)
        bse.setSingleStep(128)
        bse.setValue(1024)
        bse.setToolTip("Base size")
        bse.valueChanged.connect(self.handle_change_base_size)

        self.label_size = QtWidgets.QLabel()

        self.size_aspect_ratio = sar = QtWidgets.QComboBox()
        sar.addItems(ASPECT_RATIOS)
        sar.currentTextChanged.connect(self.handle_change_size_aspect_ratio)
        sar.setCurrentText("P 4:5")

        self.size_toolbar = self._create_size_toolbar()

    def _create_size_toolbar(self):
        size_label = QtWidgets.QLabel("Size:")
        size_label.setContentsMargins(*TOOLBAR_MARGIN)

        size_toolbar = QtWidgets.QToolBar("Size", self)
        size_toolbar.addWidget(size_label)
        size_toolbar.addWidget(self.base_size_editor)
        size_toolbar.addSeparator()
        size_toolbar.addWidget(self.size_aspect_ratio)
        size_toolbar.addSeparator()
        size_toolbar.addWidget(self.label_size)
        return size_toolbar

    def handle_change_size_aspect_ratio(self, text: str):
        base_size = self.base_size_editor.value()

        if " " not in text:
            self.image_size = (base_size, base_size)
        else:
            algn, s = text.split(" ")
            w, h = s.split(":")
            w = int(w)
            h = int(h)

            ratio: float = h / w
            w = base_size
            h = base_size
            if ratio > 1:
                w /= ratio
            else:
                h *= ratio

            w = round(w)
            h = round(h)

            while w % 8:
                w += 1
            while h % 8:
                h += 1

            self.image_size = (w, h)

        self.label_size.setText(f"{self.image_size[0]} x {self.image_size[1]}")

    def handle_change_base_size(self, val: int):
        t = self.size_aspect_ratio.currentText()
        self.handle_change_size_aspect_ratio(t)
