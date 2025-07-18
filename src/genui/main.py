import datetime
import sys

from PIL import Image
from PyQt6 import QtWidgets
from PyQt6.QtCore import QThread, QSize
from PyQt6.QtGui import QCloseEvent, QDropEvent

from .generator.sdxl import GenerationPrompt, load_pipeline
from .ui_widgets.photo_viewer import PhotoViewer, FastViewer
from .ui_widgets.window_mixins.generation_command import GenerationCommandMixin
from .ui_widgets.window_mixins.image_size import ImageSizeMixin
from .ui_widgets.window_mixins.prompt import PromptMixin
from .ui_widgets.window_mixins.scheduler import SchedulerMixin
from .ui_widgets.window_mixins.seed import SeedMixin
from .ui_widgets.window_mixins.status_bar import StatusBarMixin
from .utils import TOOLBAR_MARGIN
from .worker import Worker
from .settings import settings
from .__version__ import __version__


class Window(
    QtWidgets.QMainWindow,
    ImageSizeMixin,
    SeedMixin,
    GenerationCommandMixin,
    PromptMixin,
    SchedulerMixin,
    StatusBarMixin
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._generate_method = self.threaded_generate
        self._validate_data_for_generation_method = self.validate_data_for_generation
        self._load_image = self.load_image

        self._build_threaded_worker()
        self._build_widgets()

        self.setAcceptDrops(True)

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

        self.gen_worker.done.connect(self.handle_done)

        self.gen_worker.progress_preview.connect(self.repaint_image)

        self.gen_worker.error.connect(self.handle_error)

        self.gen_thread.start()

    def _build_widgets(self):
        self.viewer = PhotoViewer(self)
        self.viewer.setZoomPinned(True)
        self.viewer.zoomed.connect(self.handle_zoomed)
        self.viewer.repainted.connect(self.handle_zoomed)
        self.viewer.repainted.connect(self.handle_repainted)

        self.zoom_fit_button.clicked.connect(self.viewer.resetView)
        self.zoom_orig_button.clicked.connect(self.viewer.origView)

        self.preview_viewer = FastViewer(self.viewer, QSize(200, 200))
        self.preview_viewer.move(10, 10)
        self.preview_viewer.setStyleSheet("border: 5px solid white; border-radius: 5px")

        self._build_cache_widgets()

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

        self._create_tool_bars()

    def _build_cache_widgets(self):
        self.deepcache_enabled_editor = dce = QtWidgets.QCheckBox()
        dce.setChecked(True)

    def _create_tool_bars(self):
        deepcache_toolbar = self._create_deepcache_toolbar()

        self.addToolBar(self.action_toolbar)
        self.addToolBar(self.seed_toolbar)
        self.addToolBar(self.size_toolbar)
        self.addToolBar(self.scheduler_toolbar)
        self.addToolBar(deepcache_toolbar)

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

    def handle_done(self):
        self.button_interrupt.setDisabled(True)
        self.button_generate.setDisabled(False)

        pipe = load_pipeline(self.model_path)
        if pipe._interrupt:
            self.label_status.setText("Interrupted")
            self.preview_viewer.set_pixmap(None)

    def handle_error(self, error: str):
        self.button_interrupt.setDisabled(True)
        self.button_generate.setDisabled(False)

        self.show_error_modal_dialog(error)

    def repaint_image(  # noqa: PLR0913
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
        self.setWindowTitle(f"Generation: {int(step*100/steps)}%")
        if gen_time:
            # TODO: extract to signal.
            self.label_status.setText(f"Done in {gen_time.seconds} sec.")
            self.setWindowTitle(None)

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

        # Latent image smaller (~103x128) than result image.
        is_latent_image = image.width < base_size and image.height < base_size

        if is_latent_image:
            self.preview_viewer.set_pixmap(pixmap)
        else:
            self.viewer.setPhoto(pixmap, self.prompt)
            self.preview_viewer.set_pixmap(None)

            if settings.autosave_image.enabled:
                filepath = self.viewer.save_image()
                self.label_image_path.setText(f"Image saved to `{filepath}`")

    def threaded_generate(self):
        self.label_status.setText("Generation...")
        self.label_process.setMaximum(self.steps_editor.value())
        self.label_process.setValue(0)
        self.label_image_path.setText("")

        self.prompt = GenerationPrompt(
            model_path=self.model_path,
            scheduler_name=self.scheduler_selector.currentText(),
            prompt=self.prompt_editor.toPlainText(),
            neg_prompt=self.negative_editor.toPlainText(),
            seed=self.seed_editor.value(),
            size=self.image_size,
            guidance_scale=self.cfg_editor.value(),
            inference_steps=self.steps_editor.value(),
            deepcache_enabled=self.deepcache_enabled_editor.isChecked(),
            use_karras_sigmas=self.karras_sigmas_editor.isChecked(),
            use_vpred=self.vpred_editor.isChecked(),
            loras=frozenset(self.get_loras()),
        )
        # Send prompt to worker for start of generation.
        self.gen_worker.parent_conn.send(self.prompt)

    def validate_data_for_generation(self) -> bool:
        return bool(
            self.model_name
            and self.scheduler_selector.currentText()
            and self.image_size
        )

    def load_image(self, image_path: str):
        from .common.metadata import get_prompt_from_metadata
        import pyexiv2
        import traceback

        try:
            with pyexiv2.Image(image_path) as img:
                metadata:dict = img.read_xmp()

            prompt: GenerationPrompt = get_prompt_from_metadata(metadata)
        except Exception:   # noqa: BLE001
            print(traceback.format_exc())
            self.show_error_modal_dialog("File does not contain a valid metadata")
            return

        self.prompt = prompt    # TODO: Safe?

        self.prompt_editor.setPlainText(prompt.prompt)
        self.negative_editor.setPlainText(prompt.neg_prompt)
        self.seed_editor.setValue(prompt.seed)
        self.set_image_size(prompt.size)
        self.scheduler_selector.setCurrentText(prompt.scheduler_name)
        self.cfg_editor.setValue(prompt.guidance_scale)
        self.steps_editor.setValue(prompt.inference_steps)
        self.deepcache_enabled_editor.setChecked(prompt.deepcache_enabled)
        self.karras_sigmas_editor.setChecked(prompt.use_karras_sigmas)
        self.vpred_editor.setChecked(prompt.use_vpred)

        orig_model_name = prompt.model_path.split(".safetensors")[0]
        if orig_model_name != self.model_name:
            prompt.model_path = self.model_path or ""
            self.show_error_modal_dialog(
                f"The model in the image (<b>{orig_model_name}</b>) "
                f"does not match the current model (<b>{self.model_name}</b>). "
                "Model not changed"
            )

        image = Image.open(image_path)
        pixmap = image.toqpixmap()
        self.viewer.setPhoto(pixmap, prompt, metadata)

        self.label_image_path.setText(f"Loaded Image: `{image_path}`")

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            image_path = url.toLocalFile()
            self.load_image(image_path)

        event.accept()


def main():
    print(f"Version: {__version__}")
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationDisplayName("GenUI")

    app.setStyle("Fusion")

    window = Window()
    window.setGeometry(500, 300, 1300, 800)
    window.show()

    app.exec()
