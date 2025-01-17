import random
from io import BytesIO
from pathlib import Path

import websocket    #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
from PIL import Image
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QObject, pyqtSignal, QThread
from PyQt6.QtGui import QPainter

from editor_autocomplete import AwesomeTextEdit
from workflows import get_workflows, load_workflow
from ws_api import server_address, client_id, get_images, interrupt

SCALE_FACTOR = 1.05
MAX_SCALE = 100
MIN_SCALE = 100

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


class PhotoViewer(QtWidgets.QGraphicsView):
    repainted = QtCore.pyqtSignal()
    zoomed = QtCore.pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent)
        self._zoom = 0
        self._pinned = False
        self._empty = True
        self._original_size = (0, 0)

        self._scene = QtWidgets.QGraphicsScene(self)
        self._photo = QtWidgets.QGraphicsPixmapItem()
        self._photo.setShapeMode(QtWidgets.QGraphicsPixmapItem.ShapeMode.HeuristicMaskShape)
        self._photo.setTransformationMode(QtCore.Qt.TransformationMode.SmoothTransformation)
        self._scene.addItem(self._photo)
        self.setScene(self._scene)

        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)

    def hasPhoto(self):
        return not self._empty

    def resetView(self, scale: float = 1.0):
        rect = QtCore.QRectF(self._photo.pixmap().rect())
        if rect.isNull():
            return

        self.setSceneRect(rect)
        if (scale := max(1.0, scale)) == 1:
            self._zoom = 0

        if self.hasPhoto():
            viewrect = self.viewport().rect()
            scenerect = self.transform().mapRect(rect)

            factor = min(
                viewrect.width() / scenerect.width(),
                viewrect.height() / scenerect.height()
            ) * scale
            self.scale(factor, factor)

            if not self.zoomPinned():
                self.centerOn(self._photo)

            self._scene.views()[0].viewport().repaint()
            self.repainted.emit()

    def setPhoto(self, pixmap=None):
        if pixmap and not pixmap.isNull():
            self._empty = False
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
            self._photo.setPixmap(QtGui.QPixmap())

        if not (self.zoomPinned() and self.hasPhoto()):
            self._zoom = 0

        rect = QtCore.QRectF(self._photo.pixmap().rect())
        self._original_size = (rect.width(), rect.height())
        self.resetView(SCALE_FACTOR ** self._zoom)

    def zoom_scene_level(self):
        return SCALE_FACTOR ** self._zoom

    def zoom_image_level(self):
        rect = QtCore.QRectF(self._photo.pixmap().rect())
        scenerect = self.transform().mapRect(rect)
        return scenerect.width() / self._original_size[0]

    def zoomPinned(self):
        return self._pinned

    def setZoomPinned(self, enable):
        self._pinned = bool(enable)

    def zoom(self, step):
        zoom = self._zoom + (step := int(step))
        if zoom == self._zoom:
            return

        if not (MIN_SCALE < zoom < MAX_SCALE):
            return

        self._zoom = zoom
        if step > 0:
            factor = SCALE_FACTOR ** step
        else:
            factor = 1 / SCALE_FACTOR ** abs(step)
        self.scale(factor, factor)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        self.zoom(delta and delta // abs(delta))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resetView()

    def scale(self, sx, sy):
        self.zoomed.emit()
        return super().scale(sx, sy)

    def toggleDragMode(self):
        if self.dragMode() == QtWidgets.QGraphicsView.DragMode.ScrollHandDrag:
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
        elif not self._photo.pixmap().isNull():
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)

    # def mouseMoveEvent(self, event):
    #     self.updateCoordinates(event.position().toPoint())
    #     super().mouseMoveEvent(event)

    # def leaveEvent(self, event):
    #     self.coordinatesChanged.emit(QtCore.QPoint())
    #     super().leaveEvent(event)


class Worker(QObject):
    finished = pyqtSignal()
    progress_preview = pyqtSignal(bytes, int, int)

    def __init__(self, ws, prompt):
        super().__init__()
        self.ws = ws
        self.prompt = prompt
        self.steps_count = int(prompt["116"]["inputs"]["steps"])
        self.step = 0
        self._isRunning = True

    def callback_preview(self, image_data: bytes, step: int):
        self.step = step
        self.progress_preview.emit(image_data, step, self.steps_count)
        return not self._isRunning

    def run(self):
        images = get_images(
            self.ws, self.prompt,
            preview_callback=self.callback_preview,
        )
        for node_id in images:
            for image_data in images[node_id]:
                self.callback_preview(image_data, self.step)

        self.stop()

    def stop(self):
        print("stopping")
        self._isRunning = False
        self.finished.emit()


class Window(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("A new window title")
        self.size = (0, 0)
        self.loaded_workflow = None

        self.ws = websocket.WebSocket()
        self.ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))

        self.viewer = PhotoViewer(self)
        self.viewer.setZoomPinned(True)
        self.viewer.zoomed.connect(self.handle_zoomed)
        self.viewer.repainted.connect(self.handle_zoomed)

        self._build_status_widgets()
        self._build_prompt_editors()
        self._build_generation_widgets()
        self._build_seed_widgets()
        self._build_size_widgets()

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
        self.workflow_selector = QtWidgets.QComboBox()
        # TODO: to relative paths
        workflows = tuple(map(str, get_workflows()))
        self.workflow_selector.addItems(workflows)
        self.workflow_selector.currentTextChanged.connect(self.handle_change_workflow)
        self.workflow_selector.setCurrentText(workflows[0])
        self.workflow_selector.currentTextChanged.emit(workflows[0])

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
        bse.setValue(1024)
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

    def _createToolBars(self):
        action_toolbar = QtWidgets.QToolBar("Action", self)
        action_toolbar.addWidget(self.button_generate)
        action_toolbar.addWidget(self.button_interrupt)

        workflow_toolbar = QtWidgets.QToolBar("Workflow", self)
        workflow_toolbar.addWidget(self.workflow_selector)

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
        zoom_toolbar = QtWidgets.QToolBar("Zoom", self)
        zoom_toolbar.addWidget(self.zoom_label)

        self.addToolBar(action_toolbar)
        self.addToolBar(seed_toolbar)
        self.addToolBar(size_toolbar)

        self.addToolBar(QtCore.Qt.ToolBarArea.BottomToolBarArea, progress_toolbar)
        self.addToolBar(QtCore.Qt.ToolBarArea.BottomToolBarArea, workflow_toolbar)
        self.addToolBar(QtCore.Qt.ToolBarArea.BottomToolBarArea, zoom_toolbar)

    def set_workflow(self, workflow: Path):
        self.loaded_workflow = load_workflow(workflow)

        self.prompt_editor.setPlainText(self.loaded_workflow["121"]["inputs"]["text"])
        self.negative_editor.setPlainText(self.loaded_workflow["7"]["inputs"]["text"])

        self.label_status.setText(f"Loaded workflow {workflow}")

    def handle_change_workflow(self, text: str):
        workflows = get_workflows()
        for workflow in workflows:
            if str(workflow) == text:
                self.set_workflow(workflow)
                break

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
            # if algn == "P":
            #     print(f"{w/h=}; {w/h * base_size}; {round(w/h * base_size)}")
            #     self.size = (
            #         round(w/h * base_size),
            #         base_size
            #     )
            # else:
            #     self.size = (
            #         base_size,
            #         round(h/w * base_size)
            #     )

            ratio: float = h / w
            w = base_size
            h = base_size
            if ratio > 1:
                w /= ratio
            else:
                h *= ratio

            self.size = (
                round(w),
                round(h)
            )

        self.label_size.setText(f"{self.size[0]} x {self.size[1]}")

    def repaint_image(self, image_bytes: bytes, step: int, steps: int):
        self.label_process.setText(f"Step: {step}/{steps}")
        bytes_data = BytesIO(image_bytes)
        image = Image.open(bytes_data)
        base_size = self.base_size_editor.value()

        if image.width < base_size and image.height < base_size:
            # We need resize all "latent" images to real size,
            # otherwise position of zoomed image in widget will be reset.
            mw = self.size[0] / image.width
            mh = self.size[1] / image.height
            resized_data = BytesIO()

            ni = image.resize((int(image.width * mw), int(image.height * mh)))
            ni.save(resized_data, "PNG")
            bytes_data = resized_data

        pixmap = QtGui.QPixmap()
        pixmap.loadFromData(bytes_data.getvalue())
        self.viewer.setPhoto(pixmap)

        s = pixmap.size()
        self.label_current_size.setText(f"{s.width()} x {s.height()}")

    def threated_generate(self, prompt: dict):
        self.thread = QThread()
        self.worker = Worker(self.ws, prompt)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.progress_preview.connect(self.repaint_image)

        self.worker.finished.connect(lambda: self.button_interrupt.setDisabled(True))
        self.worker.finished.connect(lambda: self.button_generate.setDisabled(False))
        self.worker.finished.connect(lambda: self.label_status.setText("Done."))

        self.thread.start()

    def handle_generate(self):
        self.label_status.setText("Generation...")
        self.button_generate.setDisabled(True)
        self.button_interrupt.setDisabled(False)

        # set the text prompt for our positive CLIPTextEncode
        self.loaded_workflow["121"]["inputs"]["text"] = self.prompt_editor.toPlainText()
        self.loaded_workflow["7"]["inputs"]["text"] = self.negative_editor.toPlainText()

        # set the seed for our KSampler node
        self.loaded_workflow["37"]["inputs"]["seed"] = self.seed_editor.value()

        self.loaded_workflow["32"]["inputs"]["width"] = str(self.size[0])
        self.loaded_workflow["32"]["inputs"]["height"] = str(self.size[1])

        self.threated_generate(self.loaded_workflow)

    def handle_interrupt(self):
        interrupt()
        self.worker.stop()

    def closeEvent(self, event):
        self.ws.close()
        event.accept()  # Close the app


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.setGeometry(500, 300, 800, 600)
    window.show()
    sys.exit(app.exec())