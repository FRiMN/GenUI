import random

from PyQt6 import QtWidgets


class SeedMixin:
    def __init__(self):
        print(f"init SeedMixin")
        super().__init__()

        self.seed_editor = QtWidgets.QSpinBox()
        self.seed_editor.setRange(0, 1_000_000_000)
        self.seed_editor.setToolTip("Seed")

        self.seed_random_btn = QtWidgets.QPushButton()
        self.seed_random_btn.setText("RND")
        self.seed_random_btn.clicked.connect(self.handle_random_seed)

    def handle_random_seed(self, *args, **kwargs):
        val = random.randint(self.seed_editor.minimum(), self.seed_editor.maximum())
        self.seed_editor.setValue(val)
