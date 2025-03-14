import dataclasses
import json
from typing import Any
from pathlib import Path

from PyQt6 import QtWidgets, QtCore
from PyQt6.QtCore import Qt, QSize, QPoint
from PyQt6.QtGui import QPainter, QColor, QPixmap, QBrush, QMouseEvent, QResizeEvent, QWheelEvent, QContextMenuEvent
from PyQt6.QtWidgets import QApplication
import pyexiv2

from ..generator.sdxl import GenerationPrompt
from ..utils import BACKGROUND_COLOR_HEX, generate_image_filepath


SCALE_FACTOR = 1.05
MAX_SCALE = 100
MIN_SCALE = -100

pyexiv2.registerNs('GenUI namespace', 'genui')


class PhotoViewer(QtWidgets.QGraphicsView):
    """PhotoViewer is a custom QGraphicsView widget that displays a image and allows zooming and panning."""

    repainted = QtCore.pyqtSignal()
    zoomed = QtCore.pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent)
        self._zoom = 0
        self._pinned = False
        self._empty = True
        self._original_size = (0, 0)

        self._photo = QtWidgets.QGraphicsPixmapItem()
        self._photo.setShapeMode(QtWidgets.QGraphicsPixmapItem.ShapeMode.HeuristicMaskShape)
        self._photo.setTransformationMode(QtCore.Qt.TransformationMode.SmoothTransformation)
        
        self.prompt: GenerationPrompt | None = None

        self._build_context_menu()

        self._scene = QtWidgets.QGraphicsScene(self)
        self._scene.addItem(self._photo)
        self.setScene(self._scene)

        va = QtWidgets.QGraphicsView.ViewportAnchor
        sb_policy = QtCore.Qt.ScrollBarPolicy
        rh = QPainter.RenderHint

        self.setTransformationAnchor(va.AnchorUnderMouse)
        self.setResizeAnchor(va.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(sb_policy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(sb_policy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor.fromString(BACKGROUND_COLOR_HEX)))
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setRenderHints(rh.Antialiasing | rh.SmoothPixmapTransform)

    def _build_context_menu(self):
        self.context_menu = menu = QtWidgets.QMenu(self)

        save_action = menu.addAction("Save Image")
        save_action.triggered.connect(self.save_image_manual)

    def save_image_manual(self):
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Image",
            "",
            "Images JPEG (*.jpg *.jpeg);;All Files (*)",
        )

        if file_name:
            self.save_image(file_name)
            
    def save_image(self, file_path: str | Path | None = None) -> str:
        if file_path is None:
            file_path = generate_image_filepath()
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
        self._photo.pixmap().save(str(file_path))
        self._save_metadata_to_image(file_path)
        return str(file_path)
        
    def _save_metadata_to_image(self, file_path: Path | str):
        d = dataclasses.asdict(self.prompt)
        d.pop("callback")
        d.pop("model_path")
        
        metadata = {
            "Xmp.genui.prompt": json.dumps(d)
        }
        
        with pyexiv2.Image(str(file_path)) as img:
            img.modify_xmp(metadata)

    def contextMenuEvent(self, event: QContextMenuEvent):
        if self.hasPhoto():
            self.context_menu.exec(event.globalPos())

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

            # print(f"{id(self)}; {viewrect=}; {scenerect=}")

            factor = min(
                viewrect.width() / scenerect.width(),
                viewrect.height() / scenerect.height()
            ) * scale
            self.scale(factor, factor)

            if not self.zoomPinned():
                self.centerOn(self._photo)

            self._scene.views()[0].viewport().repaint()
            self.repainted.emit()

    def origView(self):
        """Reset the view to the original size."""

        rect = QtCore.QRectF(self._photo.pixmap().rect())
        if rect.isNull():
            return

        self.setSceneRect(rect)

        if self.hasPhoto():
            scenerect = self.transform().mapRect(rect)

            factor = min(
                rect.width() / scenerect.width(),
                rect.height() / scenerect.height()
            )
            self.scale(factor, factor)

            if not self.zoomPinned():
                self.centerOn(self._photo)

            self._scene.views()[0].viewport().repaint()
            self.repainted.emit()

    def setPhoto(self, pixmap: QPixmap | None = None, prompt: GenerationPrompt | None = None):
        self.prompt = prompt
        
        if pixmap and not pixmap.isNull():
            self._empty = False
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
            self._photo.setPixmap(QPixmap())

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
        return scenerect.width() / self._original_size[0] if self._original_size[0] else 0

    def zoomPinned(self):
        return self._pinned

    def setZoomPinned(self, enable: bool | Any):
        self._pinned = bool(enable)

    def zoom(self, step: int):
        zoom = self._zoom + (step := int(step))
        if zoom == self._zoom:
            return

        if not (MIN_SCALE < zoom < MAX_SCALE):
            return

        self._zoom = zoom
        if step > 0:    # noqa: SIM108
            factor = SCALE_FACTOR ** step
        else:
            factor = 1 / SCALE_FACTOR ** abs(step)
        self.scale(factor, factor)

    def pixmap_size(self) -> QSize:
        return self._photo.pixmap().size()

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        self.zoom(delta and delta // abs(delta))

    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        self.resetView()

    def scale(self, sx: float, sy: float):
        self.zoomed.emit()
        return super().scale(sx, sy)

    def toggleDragMode(self):
        dm = QtWidgets.QGraphicsView.DragMode

        if self.dragMode() == dm.ScrollHandDrag:
            self.setDragMode(dm.NoDrag)
        elif not self._photo.pixmap().isNull():
            self.setDragMode(dm.ScrollHandDrag)


class FastViewer(QtWidgets.QLabel):
    """FastViewer is a custom QLabel widget that displays a preview image and allows expanding."""

    def __init__(self, parent: QtWidgets.QWidget, max_size: QSize):
        super().__init__(parent)

        # Max size of not expanded widget.
        self.max_size = max_size
        self.setHidden(True)
        self.expanded = False

    def set_pixmap(self, pixmap: QPixmap | None = None):
        if not pixmap:
            self.setHidden(True)
            self.setPixmap(QPixmap())
            self.expanded = False
            return

        if not self.expanded:
            scaled_pixmap = pixmap.scaled(self.max_size, Qt.AspectRatioMode.KeepAspectRatio)
            self.setPixmap(scaled_pixmap)
            self.setFixedSize(scaled_pixmap.size())
        else:
            pos: QPoint = self.pos()
            size = self.parent().size()
            size.setWidth(size.width() - pos.x() - 5)
            size.setHeight(size.height() - pos.y() - 5)

            scaled_pixmap = pixmap.scaled(size, Qt.AspectRatioMode.KeepAspectRatio)
            self.setPixmap(scaled_pixmap)
            self.setFixedSize(scaled_pixmap.size())

        self.setHidden(False)

    def mousePressEvent(self, ev: QMouseEvent):
        self.expanded = not self.expanded
        self.set_pixmap(self.pixmap())

    def enterEvent(self, ev: QMouseEvent):
        QApplication.setOverrideCursor(Qt.CursorShape.PointingHandCursor)

    def leaveEvent(self, ev: QMouseEvent):
        QApplication.restoreOverrideCursor()
