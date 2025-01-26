import random
import time
from pathlib import Path

from PIL import Image
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import QObject, pyqtSignal, QThread

from ui_widgets.editor_autocomplete import AwesomeTextEdit
from generator.sdxl import load_pipline, generate, get_schedulers_map, set_scheduler, get_scheduler_config
from ui_widgets.photo_viewer import PhotoViewer

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


class Worker(QObject):
    finished = pyqtSignal()
    progress_preview = pyqtSignal(bytes, int, int, int, int)

    def __init__(self, window):
        super().__init__()
        self.window = window
        self.pipline = window.pipline
        self.step = 0
        self._isRunning = True

    def callback_preview(self, image: Image.Image, step: int):
        self.step = step
        image_data = image.tobytes()
        self.progress_preview.emit(image_data, step, 20, image.width, image.height)
        return not self._isRunning

    def run(self):
        image: Image.Image = generate(
            self.pipline,
            self.window.prompt_editor.toPlainText(),
            self.window.negative_editor.toPlainText(),
            seed=self.window.seed_editor.value(),
            size=self.window.size,
            clip_skip=self.window.clip_skip.value(),
            callback=self.callback_preview,
        )
        self.callback_preview(image, self.step)
        self.save_image(image)

        self.stop()

    def stop(self):
        print("stopping")
        self._isRunning = False
        self.finished.emit()

    def generate_filepath(self) -> Path:
        t = time.time()
        return Path(f"/media/frimn/archive31/ai/stable_diffusion/ComfyUI/output/genui/{t}.jpg")

    def save_image(self, image: Image.Image):
        p = self.generate_filepath()
        image.save(p)


class Window(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GenUI")
        self.size = (0, 0)

        self.pipline = load_pipline("/media/frimn/archive31/ai/stable_diffusion/ComfyUI/models/checkpoints/anime/illustrious/obsessionIllustrious_v31.safetensors")

        self.viewer = PhotoViewer(self)
        self.viewer.setZoomPinned(True)
        self.viewer.zoomed.connect(self.handle_zoomed)
        self.viewer.repainted.connect(self.handle_zoomed)

        self._build_status_widgets()
        self._build_prompt_editors()
        self._build_generation_widgets()
        self._build_seed_widgets()
        self._build_size_widgets()
        self._build_scheduler_widgets()

        panel_box = self._build_prompt_panel()

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(panel_box)
        splitter.addWidget(self.viewer)
        width = self.width()
        splitter.setSizes((
            round(1 / 3 * width),
            round(2 / 3 * width)
        ))

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(splitter)

        layout_box = QtWidgets.QWidget()
        layout_box.setLayout(layout)

        self.setCentralWidget(layout_box)

        self._createToolBars()

    def _build_generation_widgets(self):
        self.button_generate = bg = QtWidgets.QPushButton('Generate', self)
        bg.setStyleSheet("background-color: darkblue")
        bg.clicked.connect(self.handle_generate)
        bg.setShortcut("Ctrl+Return")

        self.button_interrupt = bi = QtWidgets.QPushButton(self)
        bi.setText('Stop')
        bi.setStyleSheet("background-color: darkred")
        bi.setDisabled(True)
        bi.clicked.connect(self.handle_interrupt)

    def _build_prompt_panel(self):
        panel = QtWidgets.QVBoxLayout()
        panel.addWidget(self.prompt_editor)
        panel.addWidget(self.negative_editor)

        panel_box = QtWidgets.QWidget()
        panel_box.setLayout(panel)
        return panel_box

    def _build_status_widgets(self):
        self.label_process = QtWidgets.QLabel(self)
        self.label_current_size = QtWidgets.QLabel(self)
        self.label_current_size.setToolTip("The actual size of the image. It may be less when previewing")
        self.label_status = QtWidgets.QLabel(self)

    def _build_seed_widgets(self):
        self.seed_editor = QtWidgets.QSpinBox()
        self.seed_editor.setRange(0, 1_000_000_000)
        self.seed_editor.setToolTip("Seed")

        self.seed_random_btn = QtWidgets.QPushButton()
        self.seed_random_btn.setText("RND")
        self.seed_random_btn.clicked.connect(self.handle_random_seed)

    def _build_size_widgets(self):
        self.base_size_editor = bse = QtWidgets.QSpinBox()
        bse.setMinimum(512)
        bse.setMaximum(8192)
        bse.setSingleStep(128)
        bse.setValue(1280)
        bse.setToolTip("Base size")
        bse.valueChanged.connect(self.handle_change_base_size)

        self.label_size = QtWidgets.QLabel()

        self.size_aspect_ratio = sar = QtWidgets.QComboBox()
        sar.addItems(ASPECT_RATIOS)
        sar.currentTextChanged.connect(self.handle_change_size_aspect_ratio)
        sar.setCurrentText("P 4:5")

    def _build_prompt_editors(self):
        self.prompt_editor = AwesomeTextEdit()
        self.negative_editor = AwesomeTextEdit()

    def _build_scheduler_widgets(self):
        self.scheduler_selector = ss = QtWidgets.QComboBox()
        self.schedulers_map = get_schedulers_map(self.pipline)
        self.scheduler_config = get_scheduler_config(self.pipline)
        schedulers = sorted(self.schedulers_map.keys())
        default_scheduler = next((x for x in schedulers if x.endswith("(Default)")))
        ss.addItems(schedulers)
        ss.setCurrentText(default_scheduler)
        ss.currentTextChanged.connect(self.handle_change_scheduler)

        self.clip_skip = cs = QtWidgets.QSpinBox()
        cs.setValue(1)

    def _createToolBars(self):
        action_toolbar = QtWidgets.QToolBar("Action", self)
        action_toolbar.addWidget(self.button_generate)
        action_toolbar.addWidget(self.button_interrupt)

        seed_toolbar = QtWidgets.QToolBar("Seed", self)
        seed_label = QtWidgets.QLabel("Seed:")
        seed_label.setContentsMargins(5, 0, 5, 0)
        seed_toolbar.addWidget(seed_label)
        seed_toolbar.addWidget(self.seed_editor)
        seed_toolbar.addWidget(self.seed_random_btn)

        size_label = QtWidgets.QLabel("Size:")
        size_label.setContentsMargins(5, 0, 5, 0)

        size_toolbar = QtWidgets.QToolBar("Size", self)
        size_toolbar.addWidget(size_label)
        size_toolbar.addWidget(self.base_size_editor)
        size_toolbar.addSeparator()
        size_toolbar.addWidget(self.size_aspect_ratio)
        size_toolbar.addSeparator()
        size_toolbar.addWidget(self.label_size)

        progress_toolbar = QtWidgets.QToolBar("Progress", self)
        progress_toolbar.addWidget(self.label_process)
        progress_toolbar.addSeparator()
        progress_toolbar.addWidget(self.label_current_size)
        progress_toolbar.addSeparator()
        progress_toolbar.addWidget(self.label_status)

        self.zoom_label = QtWidgets.QLabel()
        self.zoom_fit_button = QtWidgets.QPushButton()
        self.zoom_fit_button.setText("Fit")
        self.zoom_fit_button.clicked.connect(self.viewer.resetView)
        zoom_toolbar = QtWidgets.QToolBar("Zoom", self)
        zoom_toolbar.addWidget(self.zoom_label)
        zoom_toolbar.addWidget(self.zoom_fit_button)

        scheduler_toolbar = QtWidgets.QToolBar("Scheduler", self)
        scheduler_toolbar.addWidget(self.scheduler_selector)
        scheduler_toolbar.addWidget(self.clip_skip)

        self.addToolBar(action_toolbar)
        self.addToolBar(seed_toolbar)
        self.addToolBar(size_toolbar)
        self.addToolBar(scheduler_toolbar)

        self.addToolBar(QtCore.Qt.ToolBarArea.BottomToolBarArea, progress_toolbar)
        self.addToolBar(QtCore.Qt.ToolBarArea.BottomToolBarArea, zoom_toolbar)

    def handle_change_scheduler(self, text: str):
        set_scheduler(self.pipline, text, self.schedulers_map, self.scheduler_config)

    def handle_zoomed(self):
        self.zoom_label.setText(f"Zoom: {self.viewer.zoom_image_level():.2f}")

    def handle_random_seed(self, *args, **kwargs):
        val = random.randint(self.seed_editor.minimum(), self.seed_editor.maximum())
        self.seed_editor.setValue(val)

    def handle_change_base_size(self, val: int):
        t = self.size_aspect_ratio.currentText()
        self.handle_change_size_aspect_ratio(t)

    def handle_change_size_aspect_ratio(self, text):
        base_size = self.base_size_editor.value()

        if " " not in text:
            self.size = (base_size, base_size)
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

            self.size = (w, h)

        self.label_size.setText(f"{self.size[0]} x {self.size[1]}")

    # @Timer("Repaint")
    def repaint_image(
            self,
            image_bytes: bytes,
            step: int,
            steps: int,
            width: int,
            height: int
    ):
        self.label_process.setText(f"Step: {step}/{steps}")
        base_size = self.base_size_editor.value()
        image = Image.frombytes(
            "RGB",
            (width, height),
            image_bytes,
        )

        if image.width < base_size and image.height < base_size:
            # We need resize all "latent" images to real size,
            # otherwise position of zoomed image in widget will be reset.
            mw = self.size[0] / image.width
            mh = self.size[1] / image.height

            image = image.resize((int(image.width * mw), int(image.height * mh)))

        pixmap = image.toqpixmap()
        self.viewer.setPhoto(pixmap)

        s = pixmap.size()
        self.label_current_size.setText(f"{s.width()} x {s.height()}")

    def threated_generate(self):
        # if getattr(self, "thread") and self.thread is not None:
        #     del self.thread
        # if getattr(self, "worker") and self.worker is not None:
        #     del self.worker

        self.gen_thread = QThread(parent=self)
        self.gen_worker = Worker(self)
        self.gen_worker.moveToThread(self.gen_thread)

        self.gen_thread.started.connect(self.gen_worker.run)
        self.gen_worker.finished.connect(self.gen_thread.quit)
        self.gen_worker.finished.connect(self.gen_worker.deleteLater)
        self.gen_thread.finished.connect(self.gen_thread.deleteLater)

        self.gen_worker.progress_preview.connect(self.repaint_image)

        self.gen_worker.finished.connect(lambda: self.button_interrupt.setDisabled(True))
        self.gen_worker.finished.connect(lambda: self.button_generate.setDisabled(False))
        self.gen_worker.finished.connect(lambda: self.label_status.setText("Done."))

        self.gen_thread.start()

    def handle_generate(self):
        self.label_status.setText("Generation...")
        self.button_generate.setDisabled(True)
        self.button_interrupt.setDisabled(False)
        # gen_worker = Worker(self)
        # gen_worker.run()
        # gen_worker.progress_preview.connect(self.repaint_image)

        self.threated_generate()

    def handle_interrupt(self):
        # interrupt()
        self.gen_worker.stop()

    def closeEvent(self, event):
        event.accept()  # Close the app


if __name__ == '__main__':
    print("starting")
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.setGeometry(500, 300, 800, 600)
    window.show()
    sys.exit(app.exec())