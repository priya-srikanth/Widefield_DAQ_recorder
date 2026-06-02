"""Sync-pulse alignment utilities for DAQ/widefield camera timebases.

Ported for this repository from the algorithm described for
``stroke_orofacial_pipeline/src/stroke_orofacial/alignment/frame_sync.py``,
itself a canonical port of Rich's legacy
``make_alignment_templates_20251216.py``.

Algorithm:

1. Robust min-max normalize sync signals and detect rising edges.
2. Normalize edge times to [0, 1] for pattern matching.
3. Match edge sequences by sliding-window L-p distance on inter-edge intervals.
4. Fit affine maps with ``np.polyfit``.
5. Interpolate dense lookup tables in both directions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def normalize_minmax_robust(x: np.ndarray, p_lo: float = 5, p_hi: float = 95) -> np.ndarray:
    """Robustly normalize an array to approximately [0, 1]."""
    arr = np.asarray(x, dtype=np.float64)
    lo, hi = np.nanpercentile(arr, [p_lo, p_hi])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = np.nanmin(arr)
        hi = np.nanmax(arr)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=np.float64)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def find_rising_edges(arr: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """Return rising-edge sample indices after thresholding an array."""
    above = np.asarray(arr) >= thr
    return np.flatnonzero((~above[:-1]) & above[1:]) + 1


def _norm_to_01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    span = x[-1] - x[0]
    if span <= 0 or not np.isfinite(span):
        return np.zeros_like(x)
    return (x - x[0]) / span


def align_edge_sequences(
    s1: np.ndarray,
    s2: np.ndarray,
    window: int = 20,
    p: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match two edge-time sequences by sliding-window interval similarity.

    Parameters
    ----------
    s1, s2:
        Normalized edge-time sequences.
    window:
        Number of consecutive inter-edge intervals used for local matching.
    p:
        L-p exponent used by the legacy matcher. Values below 1 emphasize
        relative pattern shape and are intentionally preserved.

    Returns
    -------
    idx1, idx2, distance:
        Strictly monotonic matched edge indices into ``s1`` and ``s2`` plus the
        local matching distance for each retained pair.
    """
    s1 = np.asarray(s1, dtype=np.float64)
    s2 = np.asarray(s2, dtype=np.float64)
    if s1.size < window + 1 or s2.size < window + 1:
        raise ValueError("Not enough sync edges for requested alignment window.")

    d1 = np.diff(s1)
    d2 = np.diff(s2)
    candidates = []
    for i in range(0, d1.size - window + 1):
        ref = d1[i : i + window]
        best_j = None
        best_dist = np.inf
        for j in range(0, d2.size - window + 1):
            cmp = d2[j : j + window]
            dist = float(np.sum(np.abs(ref - cmp) ** p) ** (1.0 / p))
            if dist < best_dist:
                best_dist = dist
                best_j = j
        candidates.append((i, int(best_j), best_dist))

    # Greedy monotonic prune. Keep only increasing matches in both sequences.
    kept = []
    last_i = -1
    last_j = -1
    for i, j, dist in sorted(candidates, key=lambda row: row[2]):
        if i > last_i and j > last_j:
            kept.append((i, j, dist))
            last_i = i
            last_j = j
    kept.sort(key=lambda row: row[0])

    idx1 = np.asarray([row[0] for row in kept], dtype=np.int64)
    idx2 = np.asarray([row[1] for row in kept], dtype=np.int64)
    dist = np.asarray([row[2] for row in kept], dtype=np.float64)
    return idx1, idx2, dist


def safe_interp1d(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Interpolate while sorting, deduplicating, and guarding non-finite data."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xq = np.asarray(xq, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    if x.size == 0:
        return np.full_like(xq, np.nan, dtype=np.float64)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    unique_x, unique_idx = np.unique(x, return_index=True)
    unique_y = y[unique_idx]
    if unique_x.size == 1:
        return np.full_like(xq, unique_y[0], dtype=np.float64)
    f = interp1d(
        unique_x,
        unique_y,
        bounds_error=False,
        fill_value=(unique_y[0], unique_y[-1]),
        assume_sorted=True,
    )
    return np.asarray(f(xq), dtype=np.float64)


@dataclass
class AlignmentParams:
    csv_idx_sync: int
    csv_idx_time: int
    key_sync: str = "Sync_signal"
    fps_cam: float | None = None
    window: int = 20
    p: float = 0.1
    min_matched_edges: int = 5
    edge_threshold: float = 0.5

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "AlignmentParams":
        return cls(
            csv_idx_sync=int(params["csv_idx_sync"]),
            csv_idx_time=int(params["csv_idx_time"]),
            key_sync=str(params.get("key_sync", "Sync_signal")),
            fps_cam=None if params.get("fps_cam") is None else float(params["fps_cam"]),
            window=int(params.get("window", 20)),
            p=float(params.get("p", 0.1)),
            min_matched_edges=int(params.get("min_matched_edges", 5)),
            edge_threshold=float(params.get("edge_threshold", 0.5)),
        )


def make_alignment_template(
    ws_signals: dict[str, np.ndarray | float],
    cam_csv: pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, np.ndarray | float | int | str]:
    """Make a sync-pulse alignment template.

    ``ws_signals`` should contain a sync vector named by ``key_sync`` and a
    sampling rate under ``sample_rate_input`` or ``Fs``. ``cam_csv`` is the
    no-header camera timestamp table, with sync and time columns selected by
    ``csv_idx_sync`` and ``csv_idx_time``.
    """
    p = AlignmentParams.from_dict(params)
    ws_sync = np.asarray(ws_signals[p.key_sync], dtype=np.float64)
    fs = float(ws_signals.get("sample_rate_input", ws_signals.get("Fs")))
    cam_sync = np.asarray(cam_csv.iloc[:, p.csv_idx_sync], dtype=np.float64)
    cam_time = np.asarray(cam_csv.iloc[:, p.csv_idx_time], dtype=np.float64)

    ws_norm = normalize_minmax_robust(ws_sync)
    cam_norm = normalize_minmax_robust(cam_sync)
    ws_edge_idx = find_rising_edges(ws_norm, p.edge_threshold)
    cam_edge_idx = find_rising_edges(cam_norm, p.edge_threshold)
    ws_edge_time = ws_edge_idx.astype(np.float64) / fs
    cam_edge_time = cam_time[cam_edge_idx].astype(np.float64)

    ws_edge_norm = _norm_to_01(ws_edge_time)
    cam_edge_norm = _norm_to_01(cam_edge_time)
    cam_match_idx, ws_match_idx, match_distance = align_edge_sequences(
        cam_edge_norm, ws_edge_norm, window=p.window, p=p.p
    )
    if cam_match_idx.size < p.min_matched_edges:
        raise ValueError(
            f"Only {cam_match_idx.size} matched sync edges; need at least {p.min_matched_edges}."
        )

    matched_cam_edge_idx = cam_edge_idx[cam_match_idx]
    matched_ws_edge_idx = ws_edge_idx[ws_match_idx]
    matched_cam_edge_time = cam_edge_time[cam_match_idx]
    matched_ws_edge_time = ws_edge_time[ws_match_idx]

    slope_ws_per_cam_frame, intercept_ws_samples = np.polyfit(
        matched_cam_edge_idx.astype(np.float64),
        matched_ws_edge_idx.astype(np.float64),
        deg=1,
    )
    slope_ws_time_per_cam_time, intercept_ws_time = np.polyfit(
        matched_cam_edge_time,
        matched_ws_edge_time,
        deg=1,
    )

    sig_camIdx__idx_ws = safe_interp1d(
        matched_ws_edge_idx,
        matched_cam_edge_idx,
        np.arange(ws_sync.size, dtype=np.float64),
    )
    sig_wsIdx__idx_cam = safe_interp1d(
        matched_cam_edge_idx,
        matched_ws_edge_idx,
        np.arange(len(cam_csv), dtype=np.float64),
    )

    expected = np.nan
    if p.fps_cam:
        expected = fs / p.fps_cam

    return {
        "sample_rate_input": fs,
        "fps_cam": np.nan if p.fps_cam is None else p.fps_cam,
        "key_sync": p.key_sync,
        "csv_idx_sync": p.csv_idx_sync,
        "csv_idx_time": p.csv_idx_time,
        "edge_threshold": p.edge_threshold,
        "window": p.window,
        "p": p.p,
        "min_matched_edges": p.min_matched_edges,
        "expected_ws_samples_per_camFrame": expected,
        "slope_ws_per_camFrame": float(slope_ws_per_cam_frame),
        "intercept_ws_samples": float(intercept_ws_samples),
        "slope_ws_time_per_cam_time": float(slope_ws_time_per_cam_time),
        "intercept_ws_time": float(intercept_ws_time),
        "ws_edge_idx": ws_edge_idx,
        "cam_edge_idx": cam_edge_idx,
        "ws_edge_time": ws_edge_time,
        "cam_edge_time": cam_edge_time,
        "matched_ws_edge_idx": matched_ws_edge_idx,
        "matched_cam_edge_idx": matched_cam_edge_idx,
        "matched_ws_edge_time": matched_ws_edge_time,
        "matched_cam_edge_time": matched_cam_edge_time,
        "match_distance": match_distance,
        "sig_camIdx__idx_ws": sig_camIdx__idx_ws,
        "sig_wsIdx__idx_cam": sig_wsIdx__idx_cam,
    }
