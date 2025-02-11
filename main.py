from ui_widgets.window_mixins.generation_command import GenerationCommandMixin
from ui_widgets.window_mixins.image_size import ImageSizeMixin
from ui_widgets.window_mixins.seed import SeedMixin
from worker import Worker

from PIL import Image
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import QThread

from ui_widgets.editor_autocomplete import AwesomeTextEdit
from generator.sdxl import get_schedulers_map
from ui_widgets.photo_viewer import PhotoViewer


class Window(QtWidgets.QMainWindow, ImageSizeMixin, SeedMixin, GenerationCommandMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.setWindowTitle("GenUI")

        self.model_path = None
        self.model_name = None

        self._generate_method = self.threaded_generate

        self._build_threaded_worker()
        self._build_widgets()

    def closeEvent(self, event):
        self.gen_worker.stop()
        event.accept()  # Close the app

    def _build_threaded_worker(self):
        self.gen_thread = QThread(parent=self)
        self.gen_worker = Worker()
        self.gen_worker.moveToThread(self.gen_thread)

        self.gen_thread.started.connect(self.gen_worker.run)
        self.gen_worker.finished.connect(self.gen_thread.quit)
        self.gen_worker.finished.connect(self.gen_worker.deleteLater)
        self.gen_thread.finished.connect(self.gen_thread.deleteLater)

        self.gen_worker.progress_preview.connect(self.repaint_image)

        self.gen_thread.start()

    def _build_widgets(self):
        self.model_path_btn = QtWidgets.QPushButton("Model", self)
        self.model_path_btn.setToolTip("Model")
        self.model_path_btn.clicked.connect(self.handle_change_model)

        self.viewer = PhotoViewer(self)
        self.viewer.setZoomPinned(True)
        self.viewer.zoomed.connect(self.handle_zoomed)
        self.viewer.repainted.connect(self.handle_zoomed)

        self._build_status_widgets()
        self._build_scheduler_widgets()

        self.prompt_editor = AwesomeTextEdit()
        self.negative_editor = AwesomeTextEdit()

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

    def _build_scheduler_widgets(self):
        self.scheduler_selector = ss = QtWidgets.QComboBox()
        ss.setToolTip("Scheduler")
        # ss.currentTextChanged.connect(self.handle_change_scheduler)
        schedulers_map = get_schedulers_map()
        schedulers = sorted(schedulers_map.keys())
        ss.addItems(schedulers)

        self.clip_skip = cs = QtWidgets.QSpinBox()
        cs.setValue(1)

    # def update_scheduler_widgets(self):
    #     schedulers_map = get_schedulers_map(self.pipeline)
    #
    #     schedulers = sorted(schedulers_map.keys())
    #     default_scheduler = next((x for x in schedulers if x.endswith("(Default)")))
    #     print(f"{default_scheduler=}")
    #
    #     self.scheduler_selector.addItems(schedulers)
    #     self.scheduler_selector.setCurrentText(default_scheduler)

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
        scheduler_toolbar.addWidget(self.model_path_btn)

        self.addToolBar(action_toolbar)
        self.addToolBar(seed_toolbar)
        self.addToolBar(size_toolbar)
        self.addToolBar(scheduler_toolbar)

        self.addToolBar(QtCore.Qt.ToolBarArea.BottomToolBarArea, progress_toolbar)
        self.addToolBar(QtCore.Qt.ToolBarArea.BottomToolBarArea, zoom_toolbar)

    def handle_zoomed(self):
        self.zoom_label.setText(f"Zoom: {self.viewer.zoom_image_level():.2f}")

    def handle_change_model(self):
        print("handle_change_model")
        self.model_path = QtWidgets.QFileDialog.getOpenFileName(self, "Model")[0]
        self.model_name = self.model_path.split("/")[-1].split(".")[0]
        self.model_path_btn.setText(self.model_name)

        # load_modal_win = QtWidgets.QDialog()
        # load_modal_win.exec()
        # self.pipeline = load_pipeline(self.model_path)
        # self.update_scheduler_widgets()

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
            mw = self.image_size[0] / image.width
            mh = self.image_size[1] / image.height

            image = image.resize((int(image.width * mw), int(image.height * mh)))

        pixmap = image.toqpixmap()
        self.viewer.setPhoto(pixmap)

        s = pixmap.size()
        self.label_current_size.setText(f"{s.width()} x {s.height()}")

    def threaded_generate(self):
        self.label_status.setText("Generation...")

        # self.gen_worker.finished.connect(lambda: self.button_interrupt.setDisabled(True))
        # self.gen_worker.finished.connect(lambda: self.button_generate.setDisabled(False))
        # self.gen_worker.finished.connect(lambda: self.label_status.setText("Done."))

        self.gen_worker.parent_conn.send(dict(
            model_path=self.model_path,
            scheduler_name=self.scheduler_selector.currentText(),
            prompt=self.prompt_editor.toPlainText(),
            neg_prompt=self.negative_editor.toPlainText(),
            seed=self.seed_editor.value(),
            size=self.image_size,
            clip_skip=self.clip_skip.value(),
        ))


if __name__ == '__main__':
    print("starting")
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationDisplayName("GenUI")

    window = Window()
    window.setGeometry(500, 300, 1300, 600)
    window.show()

    sys.exit(app.exec())
