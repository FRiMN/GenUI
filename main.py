import datetime

from PIL import Image
from PyQt6 import QtWidgets
from PyQt6.QtCore import QThread, QSize
from PyQt6.QtGui import QIcon, QCloseEvent

from generator.sdxl import get_schedulers_map, GenerationPrompt, load_pipeline
from ui_widgets.editor_autocomplete import AwesomeTextEdit
from ui_widgets.photo_viewer import PhotoViewer, FastViewer
from ui_widgets.window_mixins.generation_command import GenerationCommandMixin
from ui_widgets.window_mixins.image_size import ImageSizeMixin
from ui_widgets.window_mixins.seed import SeedMixin
from worker import Worker


TOOLBAR_MARGIN = (3, 0, 3, 0)


class Window(QtWidgets.QMainWindow, ImageSizeMixin, SeedMixin, GenerationCommandMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_path = None  # Path to the model file
        self.model_name = None

        self._generate_method = self.threaded_generate
        self._validate_data_for_generation_method = self.validate_data_for_generation

        self._build_threaded_worker()
        self._build_widgets()

    def closeEvent(self, event: QCloseEvent):
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

        self.gen_worker.done.connect(lambda: self.button_interrupt.setDisabled(True))
        self.gen_worker.done.connect(lambda: self.button_generate.setDisabled(False))

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
        self.viewer.repainted.connect(self.handle_repainted)

        self.preview_viewer = FastViewer(self.viewer, QSize(200, 200))
        self.preview_viewer.move(10, 10)
        self.preview_viewer.setStyleSheet("border: 5px solid white; border-radius: 5px")

        self._build_status_widgets()
        self._build_scheduler_widgets()
        self._build_cache_widgets()

        self.prompt_editor = AwesomeTextEdit()
        self.prompt_editor.setToolTip("Positive prompt")
        self.negative_editor = AwesomeTextEdit()
        self.negative_editor.setToolTip("Negative prompt")

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
        panel.setContentsMargins(0, 0, 0, 0)
        panel.addWidget(self.prompt_editor)
        panel.addWidget(self.negative_editor)

        panel_box = QtWidgets.QWidget()
        panel_box.setLayout(panel)
        return panel_box

    def _build_status_widgets(self):
        self.label_process = QtWidgets.QProgressBar(self)
        self.label_process.setMinimum(0)
        self.label_process.setFormat("%v/%m")
        self.label_process.setFixedWidth(150)

        self.label_status = QtWidgets.QLabel(self)

    def _build_scheduler_widgets(self):
        self.scheduler_selector = ss = QtWidgets.QComboBox()
        ss.setToolTip("Scheduler")
        schedulers_map = get_schedulers_map()
        schedulers = sorted(schedulers_map.keys())
        ss.addItems(schedulers)

        self.cfg_editor = cfg = QtWidgets.QSpinBox()
        cfg.setMaximum(10)
        cfg.setMinimum(0)
        cfg.setValue(0)
        cfg.setToolTip("Guidance scale. 0 value is auto.")

        self.steps_editor = se = QtWidgets.QSpinBox()
        se.setMaximum(1000)
        se.setMinimum(1)
        se.setValue(20)
        se.setToolTip("Inference steps. Default is 50.")

    def _build_cache_widgets(self):
        self.deepcache_enabled_editor = dce = QtWidgets.QCheckBox()
        dce.setChecked(True)

    def _createToolBars(self):
        action_toolbar = self._create_action_toolbar()
        seed_toolbar = self._create_seed_toolbar()
        size_toolbar = self._create_size_toolbar()
        scheduler_toolbar = self._create_scheduler_toolbar()
        deepcache_toolbar = self._create_deepcache_toolbar()

        self.addToolBar(action_toolbar)
        self.addToolBar(seed_toolbar)
        self.addToolBar(size_toolbar)
        self.addToolBar(scheduler_toolbar)
        self.addToolBar(deepcache_toolbar)

        status_bar = self._create_status_bar()
        self.setStatusBar(status_bar)

    def _create_status_bar(self):
        self.zoom_label = QtWidgets.QLabel()

        icon = QIcon.fromTheme(QIcon.ThemeIcon.ZoomFitBest)
        self.zoom_fit_button = QtWidgets.QPushButton()
        self.zoom_fit_button.setIcon(icon)
        self.zoom_fit_button.setToolTip("Fit image to viewport")
        self.zoom_fit_button.clicked.connect(self.viewer.resetView)

        self.zoom_orig_button = QtWidgets.QPushButton()
        icon = QIcon.fromTheme("zoom-original")
        self.zoom_orig_button.setIcon(icon)
        self.zoom_orig_button.setToolTip("Set original size of image")
        self.zoom_orig_button.clicked.connect(self.viewer.origView)

        self.label_viewer_image_size = QtWidgets.QLabel()

        status_bar = QtWidgets.QStatusBar()
        status_bar.addWidget(self.label_process)
        status_bar.addWidget(self.label_status)

        # Add a spacer widget to push the next status bar widgets to the right.
        spacer = QtWidgets.QWidget()
        status_bar.addWidget(spacer, 1)

        status_bar.addWidget(self.label_viewer_image_size)
        status_bar.addWidget(self.zoom_label)
        status_bar.addWidget(self.zoom_fit_button)
        status_bar.addWidget(self.zoom_orig_button)

        return status_bar

    def _create_action_toolbar(self):
        action_toolbar = QtWidgets.QToolBar("Action", self)
        action_toolbar.addWidget(self.button_generate)
        action_toolbar.addWidget(self.button_interrupt)
        return action_toolbar

    def _create_scheduler_toolbar(self):
        cfg_label = QtWidgets.QLabel("CFG:")
        cfg_label.setContentsMargins(*TOOLBAR_MARGIN)
        steps_label = QtWidgets.QLabel("Steps:")
        steps_label.setContentsMargins(*TOOLBAR_MARGIN)

        scheduler_toolbar = QtWidgets.QToolBar("Scheduler", self)

        scheduler_toolbar.addWidget(self.scheduler_selector)
        scheduler_toolbar.addSeparator()

        scheduler_toolbar.addWidget(cfg_label)
        scheduler_toolbar.addWidget(self.cfg_editor)
        scheduler_toolbar.addSeparator()

        scheduler_toolbar.addWidget(steps_label)
        scheduler_toolbar.addWidget(self.steps_editor)
        scheduler_toolbar.addSeparator()

        scheduler_toolbar.addWidget(self.model_path_btn)
        return scheduler_toolbar

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

    def _create_seed_toolbar(self):
        seed_label = QtWidgets.QLabel("Seed:")
        seed_label.setContentsMargins(*TOOLBAR_MARGIN)

        seed_toolbar = QtWidgets.QToolBar("Seed", self)
        seed_toolbar.addWidget(seed_label)
        seed_toolbar.addWidget(self.seed_editor)
        seed_toolbar.addWidget(self.seed_random_btn)
        return seed_toolbar

    def _create_deepcache_toolbar(self):
        cache_label = QtWidgets.QLabel("DeepCache:")
        cache_label.setContentsMargins(*TOOLBAR_MARGIN)

        deepcache_toolbar = QtWidgets.QToolBar("DeepCache", self)
        deepcache_toolbar.addWidget(cache_label)
        deepcache_toolbar.addWidget(self.deepcache_enabled_editor)
        return deepcache_toolbar

    def handle_zoomed(self):
        prct = int(self.viewer.zoom_image_level() * 100)
        self.zoom_label.setText(f"Zoom: {prct}%")

    def handle_repainted(self):
        s = self.viewer.pixmap_size()
        self.label_viewer_image_size.setText(f"{s.width()} x {s.height()}")

    def handle_change_model(self):
        self.model_path = QtWidgets.QFileDialog.getOpenFileName(self, "Model")[0]
        self.model_name = self.model_path.split("/")[-1].split(".")[0]
        self.model_path_btn.setText(self.model_name)

    def repaint_image(
            self,
            image_bytes: bytes,
            step: int,
            steps: int,
            width: int,
            height: int,
            gen_time: datetime.timedelta
    ):
        self.label_process.setMaximum(steps)
        self.label_process.setValue(step)
        if gen_time:
            # TODO: extract to signal.
            self.label_status.setText(f"Done in {gen_time.seconds} sec.")

        # FIXME: base size can changed.
        base_size = self.base_size_editor.value()
        image = Image.frombytes(
            "RGB",
            (width, height),
            image_bytes,
        )

        pixmap = image.toqpixmap()
        # We copy pixmap for avoid set preview latent image to viewer (caching?).
        pixmap = pixmap.copy()

        # Latent image smaller than result image.
        is_latent_image = image.width < base_size and image.height < base_size

        if is_latent_image:
            self.preview_viewer.set_pixmap(pixmap)
        else:
            self.viewer.setPhoto(pixmap)
            self.preview_viewer.set_pixmap(None)

        pipe = load_pipeline(self.model_path)
        if pipe._interrupt:
            self.preview_viewer.set_pixmap(None)

    def threaded_generate(self):
        self.label_status.setText("Generation...")
        self.label_process.setMaximum(self.steps_editor.value())
        self.label_process.setValue(0)

        prompt = GenerationPrompt(
            model_path=self.model_path,
            scheduler_name=self.scheduler_selector.currentText(),
            prompt=self.prompt_editor.toPlainText(),
            neg_prompt=self.negative_editor.toPlainText(),
            seed=self.seed_editor.value(),
            size=self.image_size,
            guidance_scale=self.cfg_editor.value(),
            inference_steps=self.steps_editor.value(),
            deepcache_enabled=self.deepcache_enabled_editor.isChecked(),
        )
        # Send prompt to worker for start of generation.
        self.gen_worker.parent_conn.send(prompt)

    def validate_data_for_generation(self) -> bool:
        return bool(
            self.model_name
            and self.scheduler_selector.currentText()
            and self.image_size
        )


if __name__ == "__main__":
    print("starting")
    import sys

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationDisplayName("GenUI")

    window = Window()
    window.setGeometry(500, 300, 1300, 800)
    window.show()

    sys.exit(app.exec())
