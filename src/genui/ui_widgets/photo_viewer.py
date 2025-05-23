from typing import Any
from pathlib import Path

from PyQt6 import QtWidgets, QtCore
from PyQt6.QtCore import Qt, QSize, QPoint, QPropertyAnimation, QEasingCurve, QRectF, pyqtProperty, QObject
from PyQt6.QtGui import QPainter, QColor, QPixmap, QBrush, QMouseEvent, QResizeEvent, QWheelEvent, QContextMenuEvent
from PyQt6.QtWidgets import QApplication, QGraphicsObject
import pyexiv2

from ..generator.sdxl import GenerationPrompt
from ..utils import BACKGROUND_COLOR_HEX, generate_image_filepath
from ..common.metadata import get_metadata_from_prompt
from .window_mixins.propagate_events import PropagateEventsMixin


SCALE_FACTOR = 1.05
MAX_SCALE = 100
MIN_SCALE = -100

pyexiv2.registerNs('GenUI namespace', 'genui')


class AnimatedPixmapItem(QGraphicsObject):
    duration: int = 500
    easing: QEasingCurve.Type = QEasingCurve.Type.Linear
    _opacity = 1.0

    def __init__(self, pixmap: QPixmap | None = None):
        super().__init__()
        self._pixmap = QPixmap(pixmap) if pixmap else QPixmap()

    def boundingRect(self):
        return QRectF(self._pixmap.rect())

    def paint(self, painter, option, widget=None):  # noqa: ANN001
        painter.setOpacity(self._opacity)
        painter.drawPixmap(0, 0, self._pixmap)

    def setPixmap(self, pixmap: QPixmap):
        self._pixmap = QPixmap(pixmap)
        self.update()

    def pixmap(self):
        return self._pixmap

    @pyqtProperty(float)
    def opacity(self) -> float:
        return self._opacity

    @opacity.setter
    def opacity(self, value: float):
        self._opacity = value
        self.update()

    def fade_in(self):
        anim = QPropertyAnimation(self, b"opacity")
        anim.setDuration(self.duration)
        anim.setStartValue(0)
        anim.setEndValue(1)
        anim.setEasingCurve(self.easing)
        return anim

    def fade_out(self):
        anim = QPropertyAnimation(self, b"opacity")
        anim.setDuration(self.duration)
        anim.setStartValue(1)
        anim.setEndValue(0)
        anim.setEasingCurve(self.easing)
        return anim


class ImageTransitionManager(QObject):
    """Manages transitions between images in a QGraphicsScene."""
    def __init__(self, scene: QtWidgets.QGraphicsScene):
        super().__init__()
        self.scene = scene
        self.current_item = None
        self.next_item = None
        # Storing references to active animations.
        # QPropertyAnimation objects are not preserved if there are no references to them,
        # and they are destroyed by the garbage collector before completion.
        self._animations: list[QPropertyAnimation] = []

    def set_image(self, pixmap: QPixmap):
        new_item = AnimatedPixmapItem(pixmap)

        if self.current_item is None or self.current_item.pixmap().isNull():
            new_item.opacity = 1
            self.scene.addItem(new_item)
            self.current_item = new_item
        else:
            new_item.opacity = 0
            self.scene.addItem(new_item)
            self.next_item = new_item
            self._start_transition()

    def _clear_animations(self):
        for anim in self._animations:
            anim.stop()
            try:
                anim.finished.disconnect()
            except Exception as e:  # noqa: BLE001
                print(f"Error on disconnect finished event: {e}")
        self._animations.clear()

    def _start_transition(self):
        self._clear_animations()

        fade_in = self.next_item.fade_in()
        fade_in.finished.connect(self._transition_completed)

        self._animations.extend([fade_in])

        fade_in.start()

    def _transition_completed(self):
        if self.current_item and self.current_item in self.scene.items():
            self.scene.removeItem(self.current_item)
        self.current_item = self.next_item
        self.next_item = None
        self._clear_animations()


class PhotoViewer(QtWidgets.QGraphicsView, PropagateEventsMixin):
    """PhotoViewer is a custom QGraphicsView widget that displays a image and allows zooming and panning."""

    repainted = QtCore.pyqtSignal()
    zoomed = QtCore.pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent)
        self.prompt: GenerationPrompt | None = None
        self.metadata: dict | None = None

        self._zoom = 0
        self._pinned = False
        self._empty = True
        self._original_size = (0, 0)
        self._pixmap = QPixmap()
        self._photo = AnimatedPixmapItem()

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

        self._transition_manager = ImageTransitionManager(self._scene)
        self._transition_manager.current_item = self._photo

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

        self._pixmap.save(str(file_path))
        self._save_metadata_to_image(file_path)
        return str(file_path)

    def _save_metadata_to_image(self, file_path: Path | str):
        metadata = self.metadata or get_metadata_from_prompt(self.prompt)

        with pyexiv2.Image(str(file_path)) as img:
            img.modify_xmp(metadata)

    def contextMenuEvent(self, event: QContextMenuEvent):
        if self.hasPhoto():
            self.context_menu.exec(event.globalPos())

    def hasPhoto(self):
        return not self._empty

    def resetView(self, scale: float = 1.0):
        rect = QtCore.QRectF(self._pixmap.rect())
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

    def setPhoto(
        self,
        pixmap: QPixmap | None = None,
        prompt: GenerationPrompt | None = None,
        metadata: dict | None = None
    ):
        self.prompt = prompt
        self.metadata = metadata
        self._pixmap = pixmap
        
        self.resetView(SCALE_FACTOR ** self._zoom)

        if pixmap and not pixmap.isNull():
            self._empty = False
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)

            self._transition_manager.set_image(pixmap)
            self._photo = self._transition_manager.current_item
        else:
            self._empty = True
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
            self._photo.setPixmap(QPixmap())

        if not (self.zoomPinned() and self.hasPhoto()):
            self._zoom = 0

        rect = QtCore.QRectF(self._pixmap.rect())
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
