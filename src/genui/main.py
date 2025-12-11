import datetime
import sys
from pathlib import Path

from PIL import Image
from PyQt6 import QtWidgets
from PyQt6.QtCore import QThread, QSize, QRectF
from PyQt6.QtGui import QCloseEvent, QDropEvent

from .generator.sdxl import GenerationPrompt, ModelSchedulerConfig, LoRASettings
from .ui_widgets.photo_viewer import PhotoViewer, FastViewer
from .ui_widgets.window_mixins.generation_command import GenerationCommandMixin
from .ui_widgets.window_mixins.image_size import ImageSizeMixin
from .ui_widgets.window_mixins.prompt import PromptMixin
from .ui_widgets.window_mixins.scheduler import SchedulerMixin
from .ui_widgets.window_mixins.seed import SeedMixin
from .ui_widgets.window_mixins.status_bar import StatusBarMixin
from .ui_widgets.window_mixins.system_monitor import SystemMonitorMixin
from .utils import TOOLBAR_MARGIN, pixmap_to_bytes
from .operations import ImageGenerationOperation, OperationWorker
from .utils import processing_time_estimator

from .settings import settings
from .__version__ import __version__


class Window(
    QtWidgets.QMainWindow,
    ImageSizeMixin,
    SeedMixin,
    GenerationCommandMixin,
    PromptMixin,
    SchedulerMixin,
    StatusBarMixin,
    SystemMonitorMixin
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._generate_method = self.threaded_generate
        self._fix_method = self.threaded_fix
        self._validate_data_for_generation_method = self.validate_data_for_generation
        self._load_image = self.load_image

        self.gen_operation = ImageGenerationOperation()

        self._build_threaded_worker()
        self._build_widgets()

        self.setAcceptDrops(True)

    def closeEvent(self, event: QCloseEvent):
        print("Closing window...")

        # Stop system monitoring
        try:
            self.stop_system_monitoring()
        except Exception as e:
            print(f"Error stopping system monitoring: {e}")

        # Stop the worker first
        if hasattr(self, 'gen_worker') and self.gen_worker:
            self.gen_worker.stop()

        # Stop and cleanup the operation
        if hasattr(self, 'gen_operation') and self.gen_operation:
            try:
                self.gen_operation.cleanup()
            except Exception as e:
                print(f"Error cleaning up operation: {e}")

        # Wait for thread to finish properly
        if hasattr(self, 'gen_thread') and self.gen_thread and self.gen_thread.isRunning():
            print("Waiting for thread to finish...")
            self.gen_thread.quit()
            if not self.gen_thread.wait(3000):  # Wait up to 3 seconds
                print("Warning: Thread did not finish gracefully, terminating...")
                self.gen_thread.terminate()
                if not self.gen_thread.wait(1000):  # Wait up to 1 second for termination
                    print("Warning: Thread termination failed")

        print("Window cleanup complete")
        event.accept()  # Close the app

    def _build_threaded_worker(self):
        self.gen_thread = QThread(parent=self)
        self.gen_worker = OperationWorker(self.gen_operation)
        self.gen_worker.moveToThread(self.gen_thread)

        self.gen_thread.started.connect(self.gen_worker.run)
        self.gen_worker.finished.connect(self.gen_thread.quit)
        self.gen_worker.finished.connect(self.gen_worker.deleteLater)
        self.gen_thread.finished.connect(self.gen_thread.deleteLater)

        self.gen_operation.signals.done.connect(self.handle_done)
        self.gen_operation.signals.progress_preview.connect(self.repaint_image)
        self.gen_operation.signals.scheduler_config.connect(self.update_scheduler_config)
        self.gen_operation.signals.gpu_memory_info.connect(self.update_gpu_memory_display)

        self.gen_worker.error.connect(self.handle_error)
        # self.gen_worker.show_adetailer_rect.connect(self.show_adetailer_rect)
        # self.gen_worker.progress_adetailer.connect(self.update_adetailer_progress)

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

    # def _build_neg_condition_widgets(self):
    #     self.neg_condition_divider_editor = QtWidgets.QComboBox()
    #     self.neg_condition_divider_editor.addItems(["0", "1", "2", "3", "4"])
    #     self.neg_condition_divider_editor.setCurrentText("0")

    def _create_tool_bars(self):
        deepcache_toolbar = self._create_deepcache_toolbar()
        # neg_condition_toolbar = self._create_neg_condition_toolbar()

        self.addToolBar(self.action_toolbar)
        self.addToolBar(self.seed_toolbar)
        self.addToolBar(self.size_toolbar)
        self.addToolBar(self.scheduler_toolbar)
        self.addToolBar(deepcache_toolbar)
        # self.addToolBar(neg_condition_toolbar)

    def _create_deepcache_toolbar(self):
        cache_label = QtWidgets.QLabel("DeepCache:")
        cache_label.setContentsMargins(*TOOLBAR_MARGIN)

        deepcache_toolbar = QtWidgets.QToolBar("DeepCache", self)
        deepcache_toolbar.addWidget(cache_label)
        deepcache_toolbar.addWidget(self.deepcache_enabled_editor)
        return deepcache_toolbar

    # def _create_neg_condition_toolbar(self):
    #     self._build_neg_condition_widgets()

    #     neg_condition_label = QtWidgets.QLabel("Neg Condition Divider:")
    #     neg_condition_label.setContentsMargins(*TOOLBAR_MARGIN)

    #     neg_condition_toolbar = QtWidgets.QToolBar("Neg Condition Divider", self)
    #     neg_condition_toolbar.addWidget(neg_condition_label)
    #     neg_condition_toolbar.addWidget(self.neg_condition_divider_editor)
    #     return neg_condition_toolbar

    def handle_zoomed(self):
        prct = int(self.viewer.zoom_image_level() * 100)
        self.zoom_label.setText(f"Zoom: {prct}%")

    def handle_repainted(self):
        s = self.viewer.pixmap_size()
        self.label_viewer_image_size.setText(f"{s.width()} x {s.height()}")

    def handle_done(self):
        self.reset_command_buttons()

        is_interrupted = self.label_status.text().startswith("Interrupt") and not self.preview_viewer.isHidden()
        if is_interrupted:
            self.label_status.setText("Interrupted")
            self.preview_viewer.set_pixmap(None)

    def handle_error(self, error: str):
        self.button_interrupt.setDisabled(True)
        self.button_generate.setDisabled(False)

        self.reset_command_buttons()

        self.show_error_modal_dialog(error)

    def update_scheduler_config(self, config: frozenset):
        self.scheduler_config = config

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
        self.setWindowTitle(f"Generating: {int(step*100/steps)}%")
        if gen_time:
            # TODO: extract to signal.
            self.label_status.setText(f"Done in {gen_time.seconds} sec.")
            self.setWindowTitle(None)
            # processing_time_estimator.update((width, height), gen_time)
            # self.update_eta(width * height / 1e6)

        image = Image.frombytes(
            "RGB",
            (width, height),
            image_bytes,
        )

        pixmap = image.toqpixmap()
        # We copy pixmap for avoid set preview latent image to viewer (caching?).
        pixmap = pixmap.copy()

        # Latent image smaller (~103x128) than result image.
        is_latent_image = image.width < 500 and image.height < 500

        if is_latent_image:
            self.preview_viewer.set_pixmap(pixmap)
        else:
            self.viewer.setPhoto(pixmap, prompt=self.prompt, scheduler_config=self.scheduler_config)
            self.preview_viewer.set_pixmap(None)

            if settings.autosave_image.enabled:
                filepath = self.viewer.save_image()
                self.label_image_path.setText(f"Image saved to `{filepath}`")

    def show_adetailer_rect(self, x: int, y: int, x2: int, y2: int):
        width = x2 - x
        height = y2 - y
        rects = [QRectF(x, y, width, height)]
        self.viewer.set_rects(rects)
        self.label_status.setText(f"Found {len(self.viewer.rects)} rects. Inpainting...")

    def update_adetailer_progress(self, progress: int, total: int):
        multiplier = len(self.viewer.rects)
        all_total = total * multiplier
        self.label_process.setMaximum(all_total)
        self.label_process.setValue(progress)

    def get_prompt(self) -> GenerationPrompt:
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
            use_karras_sigmas=self.karras_sigmas_editor.isChecked(),
            use_vpred=self.vpred_editor.isChecked(),
            loras=frozenset(self.get_loras()),
            # neg_condition_divider=int(self.neg_condition_divider_editor.currentText()),
        )
        return prompt

    def threaded_generate(self):
        self.prompt = p = self.get_prompt()

        self.label_status.setText("Generation...")
        self.label_process.setMaximum(self.steps_editor.value())
        self.label_process.setValue(0)
        self.label_image_path.setText("")

        scheduler_conf_request = ModelSchedulerConfig(
            name=p.scheduler_name,
            model_path=p.model_path,
            use_karras_sigmas=p.use_karras_sigmas,
            use_vpred=p.use_vpred
        )

        # Send prompt to worker for start of generation.
        self.gen_worker.queue.put(scheduler_conf_request)
        self.gen_worker.queue.put(self.prompt)

    def threaded_fix(self):
        self.label_status.setText("Adetailer fix...")
        self.label_process.setMaximum(self.fix_steps)
        self.label_process.setValue(0)
        self.label_image_path.setText("")
        self.viewer.clear_rects()

        self.prompt = self.get_prompt()

        if self.viewer.prompt:
            is_same_model = self.viewer.prompt.model_path == self.prompt.model_path
            is_empty_model = self.viewer.prompt.model_path == ""
            is_same_pathless_model = self.prompt.model_path.endswith(self.viewer.prompt.model_path)

            if not is_same_model and (is_empty_model or is_same_pathless_model):
                self.viewer.prompt.model_path = self.prompt.model_path

            if self.prompt == self.viewer.prompt:
                print("same prompt")
                self.prompt.image = pixmap_to_bytes(self.viewer._photo.pixmap())

        self.prompt.use_adetailer = True
        # Send prompt to worker for start of fixing image.
        self.gen_worker.send_task(self.prompt)

    def validate_data_for_generation(self) -> bool:
        return bool(
            self.model_name
            and self.scheduler_selector.currentText()
            and self.image_size
        )

    def find_local_model(self, path: str) -> str | None:
        """Find a model by its file name (like `model.safetensors`) in directory recursively."""
        import os

        # TODO: перенести проверку в settings
        assert settings.autofind_model.path, "Auto-find model path is not set"
        for root, dirs, files in os.walk(settings.autofind_model.path):
            for file in files:
                if file == path:
                    return os.path.join(root, file)
        return None

    def find_local_lora(self, name: str) -> str | None:
        import os

        # TODO: перенести проверку в settings
        assert settings.autofind_loras.path, "Auto-find LoRA path is not set"
        for root, dirs, files in os.walk(settings.autofind_loras.path):
            for file in files:
                if file.lower().endswith(".safetensors") and Path(file).stem == name:
                    return os.path.join(root, file)
        return None

    def load_image(self, image_path: str):
        from .common.metadata import get_prompt_from_metadata
        import pyexiv2
        import traceback

        try:
            with pyexiv2.Image(image_path) as img:
                metadata: dict = img.read_xmp()
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
        # self.neg_condition_divider_editor.setCurrentText(str(prompt.neg_condition_divider))

        errors = []

        orig_model_name = prompt.model_path.split(".safetensors")[0]

        if settings.autofind_model.enabled and self.model_path is None:
            exist_local_model = self.find_local_model(prompt.model_path)
            if exist_local_model:
                self.set_model(exist_local_model)

        if orig_model_name != self.model_name:
            prompt.model_path = self.model_path or ""
            errors.append(
                f"The model in the image (<b>{orig_model_name}</b>) "
                f"does not match the current model (<b>{self.model_name}</b>). "
                "Model not changed."
            )

        if settings.autofind_loras.enabled:
            found_loras: list[LoRASettings] = []
            missing_loras: list[tuple[str, float]] = []
            for lora in prompt.loras:
                local_path = self.find_local_lora(lora.name)
                if local_path:
                    found_loras.append(LoRASettings(name=lora.name, weight=lora.weight, filepath=local_path, active=True))
                else:
                    missing_loras.append((lora.name, lora.weight))

            if found_loras:
                self.lora_window.lora_table.clear()
                for lora in found_loras:
                    self.lora_window.lora_table.add_lora(Path(lora.filepath), weight=lora.weight)

            if missing_loras:
                errors.append(
                    "The following LoRAs from the image were not found locally:<br><b>"
                    + "</b>, <b>".join([f"{l[0]} (weight={l[1]})" for l in missing_loras])
                    + "</b>"
                )
        else:
            exist_loras = frozenset(self.lora_window.lora_table.get_loras())
            prompt_loras_names = set([(l.name, l.weight) for l in prompt.loras if l.active])
            exist_loras_names = set([(l.name, l.weight) for l in exist_loras if l.active])
            if len(exist_loras) != len(prompt.loras) or prompt_loras_names != exist_loras_names:
                prompt.loras = exist_loras
                errors.append(
                    "The LoRAs in the image do not match the current LoRAs. "
                    f"LoRAs (with weights) in image: <b>{prompt_loras_names}</b>. "
                    "LoRAs not changed."
                )

        if errors:
            self.show_error_modal_dialog(
                "\n".join(errors)
            )

        image = Image.open(image_path)
        pixmap = image.toqpixmap()
        self.viewer.setPhoto(pixmap, prompt, metadata)

        self.label_image_path.setText(f"Loaded Image: `{image_path}`")

    def load_model(self, model_path: str):
        self.set_model(model_path)

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            file_path = url.toLocalFile()
            if file_path.lower().endswith(".safetensors"):
                self.load_model(file_path)
            else:
                self.load_image(file_path)

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
