"""Launch upstream labcams with Priya-lab rig compatibility patches.

Run with the labcams conda environment, for example:

    python -m labcams_ps.gui path\to\labcams_widefield_pco_only.json -w

Current patches:
- opt-in PCO offline placeholder via ``"allow_missing_camera": true``
- an Alignment Preview dock for reference-image overlays during live preview
- a Session Save dock for output folder and timestamped prefix selection
"""

from __future__ import annotations

from datetime import datetime
from multiprocessing import Event, Lock, Queue, Value
import os

import numpy as np


def _display(message: str) -> None:
    try:
        from labcams.utils import display
    except Exception:
        print(message)
    else:
        display(message)


class UnavailableCam:
    """Small camera-shaped placeholder used only for offline GUI testing."""

    drivername = "PCO-offline"
    dtype = np.uint16

    def __init__(self, cam_id=None, name="pco_offline", width=128, height=128):
        self.cam_id = cam_id
        self.name = name
        self.h = Value("i", int(height))
        self.w = Value("i", int(width))
        self.nchan = Value("i", 1)
        self.nframes = Value("i", 0)
        self.nbuffers = Value("i", 1)
        self.fs = Value("d", 0.0)
        self.camera_ready = Event()
        self.camera_ready.set()
        self.close_event = Event()
        self.stop_trigger = Event()
        self.start_trigger = Event()
        self.save_trigger = Event()
        self.eventsQ = Queue()
        self.membuffer_lock = Lock()
        self.imgs = np.zeros((1, self.h.value, self.w.value, 1), dtype=self.dtype)

    def start(self):
        self.camera_ready.set()

    def stop_acquisition(self):
        self.start_trigger.clear()

    def stop_saving(self):
        self.save_trigger.clear()

    def close(self):
        self.close_event.set()
        try:
            self.eventsQ.close()
        except Exception:
            pass

    def join(self, timeout=None):
        return None

    def terminate(self):
        self.close_event.set()

    def is_alive(self):
        return False

    def get_img(self, frame_index=None):
        return self.imgs[0]


_ORIGINAL_INIT_PCO_CAM = None


def _patch_offline_pco() -> None:
    """Allow opt-in PCO placeholder cameras when hardware is unavailable."""

    global _ORIGINAL_INIT_PCO_CAM
    import labcams.cams as cams

    if getattr(cams.Camera, "_ps_offline_pco_patch", False):
        return

    _ORIGINAL_INIT_PCO_CAM = cams.Camera._init_pco_cam

    def _init_pco_cam_with_offline_fallback(self, parameters):
        allow_missing = bool(parameters.pop("allow_missing_camera", False))
        try:
            return _ORIGINAL_INIT_PCO_CAM(self, parameters)
        except Exception as err:
            if not allow_missing:
                raise
            _display(
                "[labcams_ps] WARNING: PCO camera unavailable; "
                "opening GUI with an offline placeholder. Recording is disabled "
                "for this camera. Original error: {0}".format(err)
            )
            self.cam = UnavailableCam(cam_id=self.cam_id, name=self.name)
            self.recorder_parameters["format"] = "daq"
            return None

    cams.Camera._init_pco_cam = _init_pco_cam_with_offline_fallback
    cams.Camera._ps_offline_pco_patch = True

_ORIGINAL_PCO_CAM_INIT = None


def _patch_pco_hwio4_status_expos() -> None:
    """Prefer the pco.python helper for enabling PCO line 4 exposure output."""

    global _ORIGINAL_PCO_CAM_INIT
    try:
        import labcams.pco as lab_pco
    except Exception:
        return

    if getattr(lab_pco.PCOCam, "_ps_hwio4_patch", False):
        return

    _ORIGINAL_PCO_CAM_INIT = lab_pco.PCOCam._cam_init

    def _cam_init_with_status_expos(self):
        _ORIGINAL_PCO_CAM_INIT(self)
        configure = getattr(self.cam, "configureHWIO_4_statusExpos", None)
        if configure is None:
            return
        try:
            ok = configure(True, "high level", "status expos", "all lines")
        except Exception as err:
            _display(
                "[labcams_ps] Could not set PCO HWIO4 Status Expos with all-lines timing; "
                "trying default timing. Original error: {0}".format(err)
            )
            try:
                ok = configure(True, "high level", "status expos", None)
            except Exception as err2:
                _display("[labcams_ps] WARNING: Could not configure PCO HWIO4 Status Expos: {0}".format(err2))
                return
        _display("[labcams_ps] PCO HWIO4 configured for Status Expos output: {0}".format(ok))

    lab_pco.PCOCam._cam_init = _cam_init_with_status_expos
    lab_pco.PCOCam._ps_hwio4_patch = True

def _patch_gui_docks() -> None:
    """Add Priya-rig workflow docks to the labcams GUI."""

    import labcams.gui as gui
    from PyQt5.QtCore import QTimer, Qt
    from PyQt5.QtWidgets import (
        QFileDialog,
        QComboBox,
        QDockWidget,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    if getattr(gui.LabCamsGUI, "_ps_gui_docks_patch", False):
        return

    original_init_ui = gui.LabCamsGUI.initUI

    def hide_upstream_led_dock(self):
        upstream_led_dock = getattr(self, "camstim_tab", None)
        if upstream_led_dock is None:
            return
        self.removeDockWidget(upstream_led_dock)
        upstream_led_dock.hide()
        upstream_led_dock.setParent(None)

    def init_ui_with_ps_docks(self):
        original_init_ui(self)
        self._ps_hide_upstream_led_dock()
        self._ps_add_session_save_dock()
        self._ps_add_preview_dock()
        self._ps_add_led_control_dock()
        self._ps_add_alignment_dock()
        QTimer.singleShot(0, self._ps_hide_upstream_led_dock)
        QTimer.singleShot(500, self._ps_hide_upstream_led_dock)

    def add_session_save_dock(self):
        dock = QDockWidget("Session Save", self)
        dock.setObjectName("ps_session_save")
        widget = QWidget()
        layout = QVBoxLayout(widget)

        info = QLabel(
            "Choose the output folder and filename prefix before recording. "
            "Apply creates prefix_YYYYMMDD_HHMMSS for this session."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        prefix_row = QHBoxLayout()
        prefix_edit = QLineEdit("session")
        prefix_row.addWidget(QLabel("Prefix"))
        prefix_row.addWidget(prefix_edit)
        layout.addLayout(prefix_row)

        folder_row = QHBoxLayout()
        folder_edit = QLineEdit(str(self.parameters.get("recorder_path", "")))
        browse_button = QPushButton("Browse")
        folder_row.addWidget(QLabel("Folder"))
        folder_row.addWidget(folder_edit)
        folder_row.addWidget(browse_button)
        layout.addLayout(folder_row)

        session_label = QLabel("Session name not applied")
        session_label.setWordWrap(True)
        layout.addWidget(session_label)

        apply_button = QPushButton("Apply Save Name")
        layout.addWidget(apply_button)

        def choose_folder():
            folder = QFileDialog.getExistingDirectory(
                self,
                "Choose labcams output folder",
                folder_edit.text() or str(self.parameters.get("recorder_path", "")),
            )
            if folder:
                folder_edit.setText(folder)

        def update_writer_folder(cam, folder):
            cam.recorder_path = folder
            cam.recorder_parameters["datafolder"] = folder
            if cam.writer is not None:
                cam.writer.datafolder = folder
                cam.writer.path_keys["datafolder"] = folder
                cam.writer.path_keys["recorder_path"] = folder
            if hasattr(cam.cam, "recorderpar") and cam.cam.recorderpar is not None:
                cam.cam.recorderpar["datafolder"] = folder

        def apply_save_name():
            folder = folder_edit.text().strip()
            prefix = prefix_edit.text().strip() or "session"
            safe_prefix = "_".join(prefix.replace("/", "_").replace("\\", "_").split())
            session_name = "{0}_{1}".format(safe_prefix, datetime.now().strftime("%Y%m%d_%H%M%S"))
            if folder:
                os.makedirs(folder, exist_ok=True)
                self.parameters["recorder_path"] = folder
                for cam in self.cams:
                    update_writer_folder(cam, folder)
            self.set_experiment_name(session_name)
            session_label.setText("{0} -> {1}".format(folder or "configured folder", session_name))
            _display("[labcams_ps] Save target set: {0} / {1}".format(folder, session_name))

        browse_button.clicked.connect(choose_folder)
        apply_button.clicked.connect(apply_save_name)

        dock.setWidget(widget)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self.ps_session_save_dock = dock
    def add_preview_dock(self):
        dock = QDockWidget("Preview", self)
        dock.setObjectName("ps_preview")
        widget = QWidget()
        layout = QVBoxLayout(widget)

        info = QLabel("Start live camera preview without saving frames to disk.")
        info.setWordWrap(True)
        layout.addWidget(info)

        button_row = QHBoxLayout()
        start_button = QPushButton("Start Preview")
        stop_button = QPushButton("Stop Preview")
        snapshot_button = QPushButton("Snapshot")
        button_row.addWidget(start_button)
        button_row.addWidget(stop_button)
        button_row.addWidget(snapshot_button)
        layout.addLayout(button_row)

        status = QLabel("Preview stopped")
        status.setWordWrap(True)
        layout.addWidget(status)

        def start_preview():
            self.recController.saveOnStartToggle.setChecked(False)
            self.recController.softTriggerToggle.setChecked(True)
            status.setText("Preview running; not saving")
            _display("[labcams_ps] Preview started without saving")

        def stop_preview():
            self.recController.softTriggerToggle.setChecked(False)
            status.setText("Preview stopped")
            _display("[labcams_ps] Preview stopped")

        def take_snapshot():
            self.recController.snapshotButton.click()

        start_button.clicked.connect(start_preview)
        stop_button.clicked.connect(stop_preview)
        snapshot_button.clicked.connect(take_snapshot)

        dock.setWidget(widget)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self.ps_preview_dock = dock
    def add_led_control_dock(self):
        trigger = getattr(self, "excitation_trigger", None)
        if trigger is None:
            return

        dock = QDockWidget("LED Control", self)
        dock.setObjectName("ps_led_control")
        widget = QWidget()
        layout = QVBoxLayout(widget)

        info = QLabel("Select which LED channel is gated by camera exposure pulses.")
        info.setWordWrap(True)
        layout.addWidget(info)

        mode_row = QHBoxLayout()
        mode_combo = QComboBox()
        mode_combo.addItem("Violet / 415 nm", 1)
        mode_combo.addItem("Blue / 470 nm", 2)
        mode_combo.addItem("Alternating 415/470", 3)
        mode_combo.setCurrentIndex(2)
        mode_row.addWidget(QLabel("Mode"))
        mode_row.addWidget(mode_combo)
        layout.addLayout(mode_row)

        button_row = QHBoxLayout()
        arm_button = QPushButton("Arm LEDs")
        disarm_button = QPushButton("Disarm LEDs")
        button_row.addWidget(arm_button)
        button_row.addWidget(disarm_button)
        layout.addLayout(button_row)

        status = QLabel("Ready")
        status.setWordWrap(True)
        layout.addWidget(status)

        def apply_mode(index):
            if index < 0:
                return
            trigger.set_mode(int(mode_combo.currentData()))
            trigger.check_nchannels()
            status.setText("Mode: {0}".format(mode_combo.currentText()))
            _display("[labcams_ps] LED mode set to {0}".format(mode_combo.currentText()))

        def arm_leds():
            trigger.arm()
            status.setText("Armed: {0}".format(mode_combo.currentText()))
            _display("[labcams_ps] LED trigger armed")

        def disarm_leds():
            trigger.disarm()
            status.setText("Disarmed")
            _display("[labcams_ps] LED trigger disarmed")

        mode_combo.currentIndexChanged.connect(apply_mode)
        arm_button.clicked.connect(arm_leds)
        disarm_button.clicked.connect(disarm_leds)
        apply_mode(mode_combo.currentIndex())

        dock.setWidget(widget)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self.ps_led_control_dock = dock

    def add_alignment_dock(self):
        if not getattr(self, "camwidgets", None):
            return

        dock = QDockWidget("Alignment Preview", self)
        dock.setObjectName("ps_alignment_preview")
        widget = QWidget()
        layout = QVBoxLayout(widget)

        info = QLabel(
            "Load a previous alignment snapshot as a red reference overlay; "
            "live preview is shown in green. This affects display only."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        row = QHBoxLayout()
        camera_select = QComboBox()
        for i, cam in enumerate(self.cams):
            camera_select.addItem("{0}: {1}".format(i, cam.name), i)
        row.addWidget(QLabel("Camera"))
        row.addWidget(camera_select)
        layout.addLayout(row)

        status = QLabel("No reference loaded")
        status.setWordWrap(True)
        layout.addWidget(status)

        button_row = QHBoxLayout()
        load_button = QPushButton("Load Reference")
        clear_button = QPushButton("Clear")
        button_row.addWidget(load_button)
        button_row.addWidget(clear_button)
        layout.addLayout(button_row)

        def current_widget():
            idx = int(camera_select.currentData())
            return self.camwidgets[idx]

        def load_reference_image(cam_widget, filename):
            try:
                from tifffile import imread
                reference = imread(filename)
            except Exception:
                from PIL import Image
                reference = np.asarray(Image.open(filename))

            reference = np.asarray(reference).squeeze()
            if reference.ndim == 3:
                if reference.shape[-1] <= 4:
                    reference = reference[..., :3].mean(axis=-1)
                else:
                    reference = reference[0]
            reference = reference.astype(np.float32, copy=False)
            reference -= np.nanmin(reference)
            max_ref = np.nanmax(reference)
            if max_ref > 0:
                reference /= max_ref

            image = getattr(cam_widget.view, "image", None)
            if image is not None:
                target_shape = image.shape[:2]
            else:
                target_shape = (cam_widget.cam.cam.h.value, cam_widget.cam.cam.w.value)
            if reference.shape[:2] != tuple(target_shape):
                import cv2
                reference = cv2.resize(
                    reference,
                    (int(target_shape[1]), int(target_shape[0])),
                    interpolation=cv2.INTER_AREA,
                )
            cam_widget.parameters["reference_channel"] = reference
            cam_widget.reference_toggle.value = True
            cam_widget.reference_toggle.checkbox.setChecked(True)
            live_image = getattr(cam_widget.view, "image", None)
            if live_image is not None:
                try:
                    cam_widget.image(np.asarray(live_image), cam_widget.lastnFrame + 1)
                except Exception as err:
                    _display("[labcams_ps] Loaded reference, but immediate overlay refresh failed: {0}".format(err))

        def load_reference():
            filename, _ = QFileDialog.getOpenFileName(
                self,
                "Load alignment reference image",
                "",
                "Images (*.tif *.tiff *.png *.jpg *.jpeg);;All files (*.*)",
            )
            if filename:
                load_reference_image(current_widget(), filename)
                status.setText(os.path.basename(filename))
                _display("[labcams_ps] Loaded alignment reference: {0}".format(filename))

        def clear_reference():
            cw = current_widget()
            if cw.parameters.get("reference_channel") is not None:
                cw.toggle_reference("")
            status.setText("No reference loaded")
            _display("[labcams_ps] Cleared alignment reference")

        load_button.clicked.connect(load_reference)
        clear_button.clicked.connect(clear_reference)

        dock.setWidget(widget)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self.ps_alignment_dock = dock

    gui.LabCamsGUI.initUI = init_ui_with_ps_docks
    gui.LabCamsGUI._ps_hide_upstream_led_dock = hide_upstream_led_dock
    gui.LabCamsGUI._ps_add_session_save_dock = add_session_save_dock
    gui.LabCamsGUI._ps_add_preview_dock = add_preview_dock
    gui.LabCamsGUI._ps_add_led_control_dock = add_led_control_dock
    gui.LabCamsGUI._ps_add_alignment_dock = add_alignment_dock
    gui.LabCamsGUI._ps_gui_docks_patch = True


def apply_patches() -> None:
    """Patch upstream labcams in memory for this process only."""

    _patch_offline_pco()
    _patch_pco_hwio4_status_expos()
    _patch_gui_docks()


def main() -> None:
    apply_patches()
    from labcams.gui import main as labcams_main

    labcams_main()


if __name__ == "__main__":
    main()