from PyQt6 import QtWidgets


class GenerationCommandMixin:
    def __init__(self):
        print(f"init GenerationCommandMixin")
        super().__init__()

        self._generate_method = None

        self.button_generate = bg = QtWidgets.QPushButton('Generate', self)
        bg.setStyleSheet("background-color: darkblue")
        bg.clicked.connect(self.handle_generate)
        bg.setShortcut("Ctrl+Return")

        self.button_interrupt = bi = QtWidgets.QPushButton(self)
        bi.setText('Stop')
        bi.setStyleSheet("background-color: darkred")
        bi.setDisabled(True)
        bi.clicked.connect(self.handle_interrupt)

    def handle_generate(self):
        # self.button_generate.setDisabled(True)
        # self.button_interrupt.setDisabled(False)
        # gen_worker = Worker(self)
        # gen_worker.run()
        # gen_worker.progress_preview.connect(self.repaint_image)

        self._generate_method()

        # self.generator.send(time.time())

    def handle_interrupt(self):
        # interrupt()
        self.gen_worker.stop()