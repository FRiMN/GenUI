import random

from PyQt6 import QtWidgets


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

    def handle_random_seed(self, *args, **kwargs):
        val = random.randint(self.seed_editor.minimum(), self.seed_editor.maximum())    # noqa: S311
        self.seed_editor.setValue(val)
