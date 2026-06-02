"""Analog lick detection utilities.

Ported for this repository from the algorithm described for
``stroke_orofacial_pipeline/src/stroke_orofacial/spout_behavior/lick_detection.py``.

The lick signal sits high at rest and drops on lick contact. Detection uses
double-threshold hysteresis plus an offset-locked debounce window.
"""

from __future__ import annotations

import numpy as np


def find_upward_lick_indices(signal: np.ndarray, thresh_upper: float) -> np.ndarray:
    """Find lick onsets where the analog signal crosses below ``thresh_upper``."""
    arr = np.asarray(signal, dtype=np.float64)
    below = (arr < thresh_upper).astype(float)
    return np.flatnonzero(np.diff(below, prepend=0) > 0.5).astype(np.int64)


def find_downward_lick_indices(signal: np.ndarray, thresh_lower: float) -> np.ndarray:
    """Find lick offsets where the analog signal crosses above ``thresh_lower``."""
    arr = np.asarray(signal, dtype=np.float64)
    above = (arr > thresh_lower).astype(float)
    return np.flatnonzero(np.diff(above, prepend=0) > 0.5).astype(np.int64)


def clean_lick_indices(
    onset_idx: np.ndarray,
    offset_idx: np.ndarray,
    lockout_samples: tuple[int, int],
) -> np.ndarray:
    """Drop onset events inside the post-offset lockout/debounce windows."""
    onset_idx = np.asarray(onset_idx, dtype=np.int64)
    offset_idx = np.asarray(offset_idx, dtype=np.int64)
    left, right = lockout_samples
    if onset_idx.size == 0 or offset_idx.size == 0:
        return onset_idx
    keep = np.ones(onset_idx.shape, dtype=bool)
    for offset in offset_idx:
        lo = int(offset) + int(left)
        hi = int(offset) + int(right)
        keep &= ~((onset_idx >= lo) & (onset_idx < hi))
    return onset_idx[keep]


def detect_licks(
    signal: np.ndarray,
    fs: float,
    thresh_upper: float = 2.5,
    thresh_lower: float = 1.0,
    lockout_s: tuple[float, float] = (0.001, 0.020),
    refractory_s: float | None = None,
) -> dict[str, np.ndarray | float | tuple[float, float]]:
    """Detect analog lick onsets using hysteresis and optional refractory pruning.

    Parameters
    ----------
    signal:
        Analog lick voltage trace, high at rest and low during contact.
    fs:
        Sampling rate in Hz.
    thresh_upper:
        Onset threshold. A lick begins when voltage drops below this level.
    thresh_lower:
        Offset threshold. A lick ends when voltage rises above this level.
    lockout_s:
        Window relative to each offset in which new onsets are dropped. The
        legacy default is roughly ``(0.001, 0.020)`` seconds after offset.
    refractory_s:
        Optional minimum spacing between retained onsets. Useful for collapsing
        sustained bouts into coarser events for imaging averages.
    """
    if thresh_upper <= thresh_lower:
        raise ValueError("thresh_upper should be greater than thresh_lower for a high-rest lick signal.")
    onset = find_upward_lick_indices(signal, thresh_upper)
    offset = find_downward_lick_indices(signal, thresh_lower)
    lockout_samples = (int(round(lockout_s[0] * fs)), int(round(lockout_s[1] * fs)))
    cleaned = clean_lick_indices(onset, offset, lockout_samples)
    if refractory_s is not None and refractory_s > 0 and cleaned.size:
        min_gap = int(round(refractory_s * fs))
        kept = [int(cleaned[0])]
        last = int(cleaned[0])
        for sample in cleaned[1:]:
            sample = int(sample)
            if sample - last >= min_gap:
                kept.append(sample)
                last = sample
        cleaned = np.asarray(kept, dtype=np.int64)
    return {
        "lick_onsets": cleaned.astype(np.int64),
        "raw_onsets": onset.astype(np.int64),
        "offsets": offset.astype(np.int64),
        "fs": float(fs),
        "thresh_upper": float(thresh_upper),
        "thresh_lower": float(thresh_lower),
        "lockout_s": tuple(float(v) for v in lockout_s),
        "refractory_s": np.nan if refractory_s is None else float(refractory_s),
    }
