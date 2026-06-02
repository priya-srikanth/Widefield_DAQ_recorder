"""Treadmill voltage calibration and running-bout detection helpers.

Ported for this repository from the treadmill-processing pattern described in
``stroke_orofacial_pipeline``:

raw analog voltage -> calibrated velocity (mm/s) -> Gaussian smoothing ->
threshold/gap/min-duration running bout mask.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


def calibrate_treadmill(
    signal_in_volts: np.ndarray,
    offset_in_volts: float,
    volt_sec_per_rot: float,
    mm_per_rot: float,
) -> np.ndarray:
    """Convert treadmill voltage to velocity in mm/s.

    The calibration follows the legacy equation:

    ``(volts - offset) * (1 / volt_sec_per_rot) * mm_per_rot``.
    """
    if volt_sec_per_rot == 0:
        raise ValueError("volt_sec_per_rot must be non-zero")
    signal = np.asarray(signal_in_volts, dtype=np.float64)
    return (signal - float(offset_in_volts)) * (1.0 / float(volt_sec_per_rot)) * float(mm_per_rot)


def smooth_treadmill(speed_mm_s: np.ndarray, sample_rate_hz: float, sigma_sec: float) -> np.ndarray:
    """Gaussian smooth treadmill velocity."""
    speed = np.asarray(speed_mm_s, dtype=np.float64)
    sigma_samples = max(0.0, float(sigma_sec) * float(sample_rate_hz))
    if sigma_samples <= 0:
        return speed.copy()
    return gaussian_filter1d(speed, sigma=sigma_samples, mode="nearest")


def set_short_true_runs_false(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Set True runs shorter than ``min_len`` samples to False."""
    out = np.asarray(mask, dtype=bool).copy()
    min_len = int(min_len)
    if min_len <= 1 or out.size == 0:
        return out
    padded = np.r_[False, out, False]
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    for start, stop in zip(starts, stops):
        if stop - start < min_len:
            out[start:stop] = False
    return out


def find_running_bouts(
    treadmill_smoothed_mm_s: np.ndarray,
    sample_rate_hz: float,
    thresh_speed: float,
    max_gap_duration: float,
    min_duration: float,
) -> np.ndarray:
    """Return a boolean running mask from smoothed treadmill velocity.

    Steps:
    1. threshold speed > ``thresh_speed``
    2. fill short below-threshold gaps up to ``max_gap_duration``
    3. drop running bouts shorter than ``min_duration``
    """
    speed = np.asarray(treadmill_smoothed_mm_s, dtype=np.float64)
    running = speed > float(thresh_speed)
    gap_n = int(round(float(sample_rate_hz) * float(max_gap_duration)))
    if gap_n > 0:
        running = ~set_short_true_runs_false(~running, gap_n)
    run_n = int(round(float(sample_rate_hz) * float(min_duration)))
    if run_n > 0:
        running = set_short_true_runs_false(running, run_n)
    return running.astype(bool)


def bout_edges(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return start/stop sample indices for True runs in ``mask``."""
    arr = np.asarray(mask, dtype=bool)
    padded = np.r_[False, arr, False]
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1).astype(np.int64)
    stops = np.flatnonzero(changes == -1).astype(np.int64)
    return starts, stops


def cumulative_distance_mm(speed_mm_s: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    """Integrate velocity into cumulative distance in mm."""
    speed = np.asarray(speed_mm_s, dtype=np.float64)
    return np.cumsum(speed) / float(sample_rate_hz)
