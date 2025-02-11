from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtGui import QPainter

SCALE_FACTOR = 1.05
MAX_SCALE = 100
MIN_SCALE = -100


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

        va = QtWidgets.QGraphicsView.ViewportAnchor
        sb_policy = QtCore.Qt.ScrollBarPolicy
        rh = QPainter.RenderHint

        self.setTransformationAnchor(va.AnchorUnderMouse)
        self.setResizeAnchor(va.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(sb_policy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(sb_policy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setRenderHints(rh.Antialiasing | rh.SmoothPixmapTransform)

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
            print(f"equal")
            return

        if not (MIN_SCALE < zoom < MAX_SCALE):
            print(f"outside")
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
