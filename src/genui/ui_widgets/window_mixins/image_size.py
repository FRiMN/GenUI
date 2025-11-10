from PyQt6 import QtWidgets

from ...utils import TOOLBAR_MARGIN, get_aspect_ratios

# ASPECT_RATIOS = (
#     ("1:1", 1),
#     ("L 5:4", round(4/5, 2)),
#     ("L 4:3", round(3/4, 2)),
#     ("L 3:2", round(2/3, 2)),
#     ("L 16:10", round(10/16, 2)),
#     ("L 16:9", round(9/16, 2)),
#     ("L 21:9", round(9/21, 2)),
#     ("P 4:5", round(5/4, 2)),
#     ("P 3:4", round(4/3, 2)),
#     ("P 2:3", round(3/2, 2)),
#     ("P 9:10", round(10/9, 2)),
#     ("P 9:16", round(16/9, 2)),
#     ("P 9:21", round(21/9, 2)),
# )
# ASPECT_RATIOS_LABELS = [label for label, _ in ASPECT_RATIOS]
# ASPECT_RATIOS_VALUES = [value for _, value in ASPECT_RATIOS]

ASPECT_RATIOS_LABELS = [
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
]
ASPECT_RATIOS = get_aspect_ratios(ASPECT_RATIOS_LABELS)
ASPECT_RATIOS_VALUES = [value for _, value in ASPECT_RATIOS]


class ImageSizeMixin:
    def __init__(self):
        super().__init__()

        self.image_size = (0, 0)    # width, height

        self.base_size_editor = bse = QtWidgets.QSpinBox()
        bse.setMinimum(512)
        bse.setMaximum(8192)
        bse.setSingleStep(128)
        bse.setValue(1024)
        bse.setToolTip("Base size")

        self.label_size = QtWidgets.QLabel()
        self.label_resolution_mpx = QtWidgets.QLabel()
        self.sub_label_mpx = QtWidgets.QLabel("Mpx")
        self.sub_label_mpx.setStyleSheet("color: gray;")

        bse.valueChanged.connect(self.handle_change_base_size)

        self.size_aspect_ratio = sar = QtWidgets.QComboBox()
        sar.addItems(ASPECT_RATIOS_LABELS)
        sar.currentTextChanged.connect(self.handle_change_size_aspect_ratio)
        sar.setCurrentText("P 4:5")

        self.update_eta(self.image_mpx)

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
        size_toolbar.addSeparator()
        size_toolbar.addWidget(self.label_resolution_mpx)
        size_toolbar.addWidget(self.sub_label_mpx)
        return size_toolbar

    @property
    def image_mpx(self) -> float:
        return self.image_size[0] * self.image_size[1] / 1e6

    def handle_change_size_aspect_ratio(self, text: str):
        base_size = self.base_size_editor.value()

        if " " not in text:
            self.image_size = (base_size, base_size)
        else:
            ratio: float = next(iter([value for label, value in ASPECT_RATIOS if label == text]))
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
        self.label_resolution_mpx.setText(f"{self.image_mpx:.2f}")
        self.update_eta(self.image_mpx)

    def handle_change_base_size(self, val: int):
        t = self.size_aspect_ratio.currentText()
        self.handle_change_size_aspect_ratio(t)

    def set_image_size(self, size: tuple[int, int]):
        """Correctly set the image size on all widgets."""
        base_size = max(size)
        w, h = size
        aspect_ratio = round(h / w, 2)
        closest = min(ASPECT_RATIOS_VALUES, key=lambda x: abs(x - aspect_ratio))
        aspect_ratio_label = next(iter([label for label, value in ASPECT_RATIOS if value == closest]))

        self.base_size_editor.setValue(base_size)
        self.size_aspect_ratio.setCurrentText(aspect_ratio_label)
