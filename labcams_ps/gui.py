"""Launch upstream labcams with Priya-lab rig compatibility patches.

Run with the labcams conda environment, for example:

    python -m labcams_ps.gui path\to\labcams_widefield_pco_only.json -w

The only current patch is an opt-in PCO offline placeholder.  Add
``"allow_missing_camera": true`` to a PCO camera entry to let the labcams GUI
open when the camera is disconnected or powered off.  Without that config flag,
PCO initialization errors still fail loudly.
"""

from __future__ import annotations

from multiprocessing import Event, Queue, Value
from multiprocessing import Lock
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


def apply_patches() -> None:
    """Patch upstream labcams in memory for this process only."""

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


def main() -> None:
    apply_patches()
    from labcams.gui import main as labcams_main

    labcams_main()


if __name__ == "__main__":
    main()