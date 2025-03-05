import random

from PyQt6 import QtWidgets

from utils import TOOLBAR_MARGIN


class SeedMixin:
    def __init__(self):
        super().__init__()

        self.seed_editor = QtWidgets.QSpinBox()
        # Any 32-bit integer is a valid seed.
        self.seed_editor.setRange(0, 2_147_483_647)
        self.seed_editor.setToolTip("Seed")

        self.seed_random_btn = QtWidgets.QPushButton()
        self.seed_random_btn.setText("RND")
        self.seed_random_btn.clicked.connect(self.handle_random_seed)

        self.seed_toolbar = self._create_seed_toolbar()

    def _create_seed_toolbar(self):
        seed_label = QtWidgets.QLabel("Seed:")
        seed_label.setContentsMargins(*TOOLBAR_MARGIN)

        seed_toolbar = QtWidgets.QToolBar("Seed", self)
        seed_toolbar.addWidget(seed_label)
        seed_toolbar.addWidget(self.seed_editor)
        seed_toolbar.addWidget(self.seed_random_btn)
        return seed_toolbar

    def handle_random_seed(self, *args, **kwargs):
        val = random.randint(self.seed_editor.minimum(), self.seed_editor.maximum())    # noqa: S311
        self.seed_editor.setValue(val)
