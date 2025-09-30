import psutil
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QLabel



class SystemMonitorMixin:
    """Mixin to provide periodic system monitoring (CPU and memory usage)."""

    def __init__(self):
        super().__init__()

        # System monitoring labels
        self.label_memory_usage = QLabel()
        self.label_memory_usage.setToolTip("Waiting System Memory Usage")
        self.label_memory_usage.setText("RAM: N/A")

        # Timer for periodic updates
        self._system_monitor_timer = QTimer()
        self._system_monitor_timer.timeout.connect(self._update_system_info)
        self._system_monitor_timer.setInterval(2000)  # Update every 2 seconds
        self._system_monitor_timer.start()

        # Initialize with current values
        self._update_system_info()

    def _update_system_info(self):
        """Update CPU and memory usage information."""
        try:
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            memory_percent = memory.percent

            # Update memory display
            self._update_memory_display(memory_used_gb, memory_total_gb, memory_percent, memory)
        except Exception as e:
            print(f"Error updating system info: {e}")
            self.label_memory_usage.setText("RAM: Error")

    def _update_memory_display(self, used_gb: float, total_gb: float, percent: float, memory_info):
        """Update memory usage display with color coding."""
        # display_text = f"RAM: {used_gb:.1f} / {total_gb:.1f}"
        display_text = f"RAM: {percent:.1f}%"
        self.label_memory_usage.setText(display_text)

        # Detailed tooltip
        try:
            available_gb = memory_info.available / (1024**3)
            free_gb = memory_info.free / (1024**3)

            tooltip_text = (
                f"System Memory Usage:\n"
                f"Used: {used_gb:.2f} GB ({percent:.1f}%)\n"
                f"Available: {available_gb:.2f} GB\n"
                f"Free: {free_gb:.2f} GB\n"
                f"Total: {total_gb:.2f} GB"
            )

            # Add buffer/cache info if available (Linux/macOS)
            if hasattr(memory_info, 'buffers') and hasattr(memory_info, 'cached'):
                buffers_gb = memory_info.buffers / (1024**3)
                cached_gb = memory_info.cached / (1024**3)
                tooltip_text += f"\nBuffers: {buffers_gb:.2f} GB\nCached: {cached_gb:.2f} GB"

            self.label_memory_usage.setToolTip(tooltip_text)
        except:
            self.label_memory_usage.setToolTip(f"Memory Usage: {percent:.1f}%")

        # Color coding based on usage percentage
        if percent > 90:
            self.label_memory_usage.setStyleSheet("color: red")
        elif percent > 80:
            self.label_memory_usage.setStyleSheet("color: orange")
        # elif percent > 70:
        #     self.label_memory_usage.setStyleSheet("color: yellow")
        else:
            self.label_memory_usage.setStyleSheet("")

    def stop_system_monitoring(self):
        """Stop the system monitoring timer."""
        if hasattr(self, '_system_monitor_timer'):
            self._system_monitor_timer.stop()

    def start_system_monitoring(self, interval_ms: int = 2000):
        """Start or restart system monitoring with specified interval."""
        if hasattr(self, '_system_monitor_timer'):
            self._system_monitor_timer.stop()
            self._system_monitor_timer.setInterval(interval_ms)
            self._system_monitor_timer.start()

    def set_system_monitoring_interval(self, interval_ms: int):
        """Change the monitoring interval."""
        if hasattr(self, '_system_monitor_timer'):
            self._system_monitor_timer.setInterval(interval_ms)
