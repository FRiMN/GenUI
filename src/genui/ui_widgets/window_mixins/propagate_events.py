from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QDropEvent, QDragMoveEvent

class PropagateEventsMixin(QWidget):
    """Mixin for propagating events to parent widgets.
    
    Need for widgets, who catch events and need to propagate them to parent widgets (ex. QGraphicsView, QTextEdit).
    """
    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        # Need for drag and drop
        event.accept()
        
    def dropEvent(self, event: QDropEvent):
        # Search for the parent widget that accepts drops
        p = self.parentWidget()
        while not p.acceptDrops():
            p = p.parentWidget()
        p.dropEvent(event)
