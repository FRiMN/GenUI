from PyQt6.QtWidgets import QComboBox, QSpinBox, QCheckBox, QLabel, QToolBar, QPushButton, QFileDialog

from ...generator.sdxl import get_schedulers_map
from ...utils import TOOLBAR_MARGIN


class SchedulerMixin:
    def __init__(self):
        super().__init__()

        self.model_path = None  # Path to the model file
        self.model_name = None

        self.model_path_btn = QPushButton("Model", self)
        self.model_path_btn.setToolTip("Model")
        self.model_path_btn.clicked.connect(self.handle_change_model)

        self.scheduler_selector = ss = QComboBox()
        ss.setToolTip("Scheduler")
        schedulers_map = get_schedulers_map()
        schedulers = sorted(schedulers_map.keys())
        ss.addItems(schedulers)

        self.cfg_editor = cfg = QSpinBox()
        cfg.setMaximum(10)
        cfg.setMinimum(0)
        cfg.setValue(0)
        cfg.setToolTip("Guidance scale. 0 value is auto.")

        self.steps_editor = se = QSpinBox()
        se.setMaximum(1000)
        se.setMinimum(1)
        se.setValue(50)
        se.setToolTip("Inference steps. Default is 50.")

        self.karras_sigmas_editor = kse = QCheckBox()
        kse.setChecked(True)

        self.scheduler_toolbar = self._create_scheduler_toolbar()
        
    def _create_scheduler_toolbar(self):
        cfg_label = QLabel("CFG:")
        cfg_label.setContentsMargins(*TOOLBAR_MARGIN)
        steps_label = QLabel("Steps:")
        steps_label.setContentsMargins(*TOOLBAR_MARGIN)
        karras_sigmas_label = QLabel("Karras sigmas:")
        karras_sigmas_label.setContentsMargins(*TOOLBAR_MARGIN)

        scheduler_toolbar = QToolBar("Scheduler", self)

        scheduler_toolbar.addWidget(self.scheduler_selector)
        scheduler_toolbar.addSeparator()

        scheduler_toolbar.addWidget(cfg_label)
        scheduler_toolbar.addWidget(self.cfg_editor)
        scheduler_toolbar.addSeparator()

        scheduler_toolbar.addWidget(steps_label)
        scheduler_toolbar.addWidget(self.steps_editor)
        scheduler_toolbar.addSeparator()

        scheduler_toolbar.addWidget(karras_sigmas_label)
        scheduler_toolbar.addWidget(self.karras_sigmas_editor)
        scheduler_toolbar.addSeparator()

        scheduler_toolbar.addWidget(self.model_path_btn)
        return scheduler_toolbar

    def handle_change_model(self):
        self.model_path = QFileDialog.getOpenFileName(self, "Model")[0]
        self.model_name = self.model_path.split("/")[-1].split(".")[0]
        self.model_path_btn.setText(self.model_name)
