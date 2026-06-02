"""Plot cue-aligned wfield trial averages by spout position.

Uses DAQ cue/strobe/bit timing to classify trials, then reconstructs
hemodynamic-corrected maps from wfield U/SVTcorr for pre- and post-cue windows.
Allen region outlines are overlaid from the saved atlas label map.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


POSITION_NAMES = {
    0: "close_center",
    1: "close_L",
    2: "close_R",
    3: "far_center",
    4: "far_L",
    5: "far_R",
}
DISPLAY_ORDER = [1, 0, 2, 4, 3, 5]


def _rising_edges(x: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.diff(x.astype(np.int8), prepend=0) == 1)


def _load_daq_events(h5_path: Path) -> dict:
    with h5py.File(h5_path, "r") as f:
        sr = float(f.attrs["sample_rate_hz"])
        created_at = str(f.attrs["created_at"])
        names = [name.decode() for name in f["digital/channel_names"][:]]
        packed = f["digital/packed_samples"][:, 0]
        bits = np.unpackbits(packed[:, None], axis=1, bitorder="little")[:, : len(names)]

    idx = {name: names.index(name) for name in names}
    cue = _rising_edges(bits[:, idx["cue"]])
    strobe = _rising_edges(bits[:, idx["spout_strobe"]])
    pco = _rising_edges(bits[:, idx["pco_exposure"]])

    bit0 = bits[:, idx["spout_bit0"]]
    bit1 = bits[:, idx["spout_bit1"]]
    bit2 = bits[:, idx["spout_bit2"]]
    codes_at_strobe = (
        bit0[strobe].astype(np.int16)
        + 2 * bit1[strobe].astype(np.int16)
        + 4 * bit2[strobe].astype(np.int16)
    )

    return {
        "sample_rate_hz": sr,
        "created_at": created_at,
        "cue_samples": cue,
        "strobe_samples": strobe,
        "pco_samples": pco,
        "strobe_codes": codes_at_strobe,
    }


def _event_frame_indices_from_pco(event_samples: np.ndarray, pco_samples: np.ndarray) -> np.ndarray:
    """Map DAQ event samples to raw camera frame indices using DAQ PCO exposure pulses."""
    if pco_samples.size < 2:
        raise ValueError("Need at least two DAQ pco_exposure pulses for pulse-order frame alignment.")
    insertion = np.searchsorted(pco_samples, event_samples, side="left")
    insertion = np.clip(insertion, 1, len(pco_samples) - 1)
    prev_dist = np.abs(event_samples - pco_samples[insertion - 1])
    next_dist = np.abs(pco_samples[insertion] - event_samples)
    return np.where(prev_dist <= next_dist, insertion - 1, insertion).astype(np.int64)


def _load_camlog_frame_times(camlog: Path) -> np.ndarray:
    times = []
    with camlog.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",", 1)
            if len(parts) != 2:
                continue
            try:
                int(parts[0])
                times.append(datetime.fromisoformat(parts[1]))
            except ValueError:
                continue
    if not times:
        raise ValueError(f"No frame timestamps found in {camlog}")
    t0 = times[0]
    return np.array([(t - t0).total_seconds() for t in times], dtype=np.float64)


def _frame_qc(
    pco_samples: np.ndarray,
    svt_frame_count: int,
    camlog: Path | None,
    raw_event_frames: np.ndarray,
) -> dict:
    camlog_frame_count = None
    if camlog is not None:
        try:
            camlog_frame_count = int(len(_load_camlog_frame_times(camlog)))
        except Exception as exc:
            camlog_frame_count = f"error: {exc}"
    expected_raw_from_svt = int(svt_frame_count * 2)
    pco_count = int(len(pco_samples))
    pco_minus_expected = pco_count - expected_raw_from_svt
    qc = {
        "alignment_mode_note": "Primary frame mapping should use DAQ pco_exposure pulse order when available.",
        "daq_pco_exposure_count": pco_count,
        "camlog_frame_count": camlog_frame_count,
        "svt_corrected_frame_count": int(svt_frame_count),
        "expected_raw_frames_from_svt_pairs": expected_raw_from_svt,
        "daq_pco_minus_expected_raw_frames": int(pco_minus_expected),
        "raw_event_frame_min": int(np.min(raw_event_frames)) if raw_event_frames.size else None,
        "raw_event_frame_max": int(np.max(raw_event_frames)) if raw_event_frames.size else None,
        "corrected_event_frame_min": int(np.min(raw_event_frames // 2)) if raw_event_frames.size else None,
        "corrected_event_frame_max": int(np.max(raw_event_frames // 2)) if raw_event_frames.size else None,
    }
    if isinstance(camlog_frame_count, int):
        qc["daq_pco_minus_camlog_frames"] = int(pco_count - camlog_frame_count)
    return qc


def _cue_frame_indices_from_camlog(
    cue_samples: np.ndarray,
    sample_rate_hz: float,
    daq_created_at: str,
    camlog: Path,
) -> np.ndarray:
    cam_seconds = _load_camlog_frame_times(camlog)
    daq_t0 = datetime.fromisoformat(daq_created_at)
    # Use the first frame timestamp as camlog t=0, parsed independently above.
    first_frame = None
    with camlog.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",", 1)
            if len(parts) == 2:
                try:
                    int(parts[0])
                    first_frame = datetime.fromisoformat(parts[1])
                    break
                except ValueError:
                    pass
    if first_frame is None:
        raise ValueError(f"No first frame timestamp found in {camlog}")
    cue_abs_seconds = np.array(
        [
            (daq_t0 + timedelta(seconds=float(s) / sample_rate_hz) - first_frame).total_seconds()
            for s in cue_samples
        ],
        dtype=np.float64,
    )
    insertion = np.searchsorted(cam_seconds, cue_abs_seconds, side="left")
    insertion = np.clip(insertion, 1, len(cam_seconds) - 1)
    prev_dist = np.abs(cue_abs_seconds - cam_seconds[insertion - 1])
    next_dist = np.abs(cam_seconds[insertion] - cue_abs_seconds)
    return np.where(prev_dist <= next_dist, insertion - 1, insertion).astype(np.int64)


def _classify_cues(cue_samples: np.ndarray, strobe_samples: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Assign each cue the most recent strobe code."""
    strobe_idx = np.searchsorted(strobe_samples, cue_samples, side="right") - 1
    out = np.full(cue_samples.shape, -1, dtype=np.int16)
    valid = strobe_idx >= 0
    out[valid] = codes[strobe_idx[valid]]
    out[(out < 0) | (out > 5)] = -1
    return out


def _weighted_map(U: np.ndarray, svt_mean: np.ndarray) -> np.ndarray:
    return np.tensordot(U, svt_mean, axes=([2], [0])).astype(np.float32)


def _region_edges(atlas: np.ndarray) -> np.ndarray:
    valid = np.isfinite(atlas) & (atlas != 0)
    edges = np.zeros_like(valid, dtype=bool)
    edges[:-1, :] |= atlas[:-1, :] != atlas[1:, :]
    edges[:, :-1] |= atlas[:, :-1] != atlas[:, 1:]
    edges &= valid
    return ndimage.binary_dilation(edges, iterations=1)


def _overlay_regions(ax, edges: np.ndarray) -> None:
    overlay = np.zeros((*edges.shape, 4), dtype=np.float32)
    overlay[edges] = (0, 0, 0, 0.65)
    ax.imshow(overlay, interpolation="nearest")


def main() -> int:
    parser = argparse.ArgumentParser(description="Cue-aligned spout-position trial averages.")
    parser.add_argument("--daq-h5", type=Path, required=True)
    parser.add_argument("--wfield-results", type=Path, required=True)
    parser.add_argument("--allen-dir", type=Path, required=True)
    parser.add_argument("--camlog", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="PS94")
    parser.add_argument("--pre-s", type=float, default=1.0)
    parser.add_argument("--post-s", type=float, default=1.0)
    parser.add_argument("--fs", type=float, default=31.23)
    parser.add_argument(
        "--frame-align",
        choices=("pco", "camlog"),
        default="pco",
        help="Map DAQ events to imaging frames by DAQ pco_exposure pulse order (default) or legacy camlog wall-clock timestamps.",
    )
    parser.add_argument(
        "--activity-percentile",
        type=float,
        default=99.0,
        help="Symmetric color scale percentile for pre/post activity maps.",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    events = _load_daq_events(args.daq_h5)
    U = np.load(args.allen_dir / "U_atlas.npy", mmap_mode="r")
    SVTcorr = np.load(args.wfield_results / "SVTcorr.npy", mmap_mode="r")
    atlas = np.load(args.allen_dir / "allen_area_atlas_native_grid.npy")
    edges = _region_edges(atlas)

    if args.frame_align == "camlog":
        if args.camlog is None:
            raise ValueError("--frame-align camlog requires --camlog")
        raw_cue_frames = _cue_frame_indices_from_camlog(
            events["cue_samples"],
            events["sample_rate_hz"],
            events["created_at"],
            args.camlog,
        )
        cue_frames = raw_cue_frames // 2
        frame_mapping = (
            "DAQ cue wall-clock times mapped to nearest labcams camlog frame timestamp; "
            "individual camera-frame indices were divided by 2 to match paired 415/470 SVTcorr timepoints."
        )
    else:
        raw_cue_frames = _event_frame_indices_from_pco(events["cue_samples"], events["pco_samples"])
        cue_frames = raw_cue_frames // 2
        frame_mapping = (
            "DAQ cue samples mapped to nearest DAQ pco_exposure rising-edge index; "
            "that raw camera-frame index was divided by 2 to match paired 415/470 SVTcorr timepoints."
        )
    frame_qc = _frame_qc(events["pco_samples"], SVTcorr.shape[1], args.camlog, raw_cue_frames)
    cue_codes = _classify_cues(
        events["cue_samples"], events["strobe_samples"], events["strobe_codes"]
    )

    pre_n = int(round(args.pre_s * args.fs))
    post_n = int(round(args.post_s * args.fs))
    valid_window = (cue_frames - pre_n >= 0) & (cue_frames + post_n <= SVTcorr.shape[1])
    valid = valid_window & (cue_codes >= 0)

    maps = {}
    counts = {}
    for code in DISPLAY_ORDER:
        trial_frames = cue_frames[valid & (cue_codes == code)]
        counts[POSITION_NAMES[code]] = int(len(trial_frames))
        if len(trial_frames) == 0:
            continue
        pre_sum = np.zeros(SVTcorr.shape[0], dtype=np.float64)
        post_sum = np.zeros(SVTcorr.shape[0], dtype=np.float64)
        for frame in trial_frames:
            pre_sum += np.asarray(SVTcorr[:, frame - pre_n : frame]).mean(axis=1)
            post_sum += np.asarray(SVTcorr[:, frame : frame + post_n]).mean(axis=1)
        pre_mean = (pre_sum / len(trial_frames)).astype(np.float32)
        post_mean = (post_sum / len(trial_frames)).astype(np.float32)
        pre_map = _weighted_map(U, pre_mean)
        post_map = _weighted_map(U, post_mean)
        maps[POSITION_NAMES[code]] = {
            "pre": pre_map,
            "post": post_map,
            "delta": (post_map - pre_map).astype(np.float32),
        }

    all_prepost = np.concatenate(
        [v["pre"].ravel() for v in maps.values()] + [v["post"].ravel() for v in maps.values()]
    )
    activity_lim = float(np.nanpercentile(np.abs(all_prepost), args.activity_percentile))
    all_delta = np.concatenate([v["delta"].ravel() for v in maps.values()])
    delta_lim = float(np.nanpercentile(np.abs(all_delta), 99.0))
    activity_lim = max(activity_lim, 1e-6)
    delta_lim = max(delta_lim, 1e-6)

    fig, axes = plt.subplots(6, 3, figsize=(13, 18), constrained_layout=True)
    for row, code in enumerate(DISPLAY_ORDER):
        name = POSITION_NAMES[code]
        for col, key in enumerate(("pre", "post", "delta")):
            ax = axes[row, col]
            ax.set_axis_off()
            if name not in maps:
                ax.set_title(f"{name}: no trials")
                continue
            lim = delta_lim if key == "delta" else activity_lim
            im = ax.imshow(maps[name][key], cmap="RdBu_r", vmin=-lim, vmax=lim)
            _overlay_regions(ax, edges)
            label = {"pre": "1 s pre-cue", "post": "1 s post-cue", "delta": "post - pre"}[key]
            ax.set_title(f"{name} n={counts[name]} | {label}", fontsize=10)
        fig.colorbar(im, ax=axes[row, :], shrink=0.7, pad=0.01)

    fig.suptitle(
        f"{args.label} motion-corrected, hemodynamic-corrected cue averages with Allen region outlines",
        fontsize=14,
    )
    png = args.output / f"{args.label}_spout_positions_1s_pre_post_delta_allen_overlay.png"
    fig.savefig(png, dpi=180)
    plt.close(fig)

    npz_payload = {}
    for name, vals in maps.items():
        for key, arr in vals.items():
            npz_payload[f"{name}_{key}"] = arr
    np.savez_compressed(
        args.output / f"{args.label}_spout_positions_1s_pre_post_delta_maps.npz",
        **npz_payload,
    )

    summary = {
        "daq_h5": str(args.daq_h5),
        "wfield_results": str(args.wfield_results),
        "allen_dir": str(args.allen_dir),
        "camlog": str(args.camlog) if args.camlog else None,
        "output": str(args.output),
        "pre_s": args.pre_s,
        "post_s": args.post_s,
        "fs": args.fs,
        "cue_count": int(len(events["cue_samples"])),
        "pco_exposure_count": int(len(events["pco_samples"])),
        "frame_align": args.frame_align,
        "frame_alignment_qc": frame_qc,
        "valid_cues_with_windows": int(valid.sum()),
        "counts_by_position": counts,
        "activity_display_limit": activity_lim,
        "delta_display_limit": delta_lim,
        "classification": "Most recent spout_strobe before cue; code = bit0 + 2*bit1 + 4*bit2 sampled at strobe rising edge.",
        "frame_mapping": frame_mapping,
    }
    (args.output / f"{args.label}_spout_positions_1s_pre_post_delta_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {png}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
