"""Plot post-lick wfield averages grouped by spout position.

This is analogous to ``plot_spout_trial_averages.py`` except events are falling
threshold crossings on the analog lick channel rather than cue TTLs. Lick maps
are post-event only because licks can occur in bouts and a pre-lick baseline can
contain earlier lick-related activity.
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

try:
    from .lick_detection import detect_licks
except ImportError:  # Allow direct script execution.
    from lick_detection import detect_licks


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


def _decode_analog_channel(f, channel_name: str) -> np.ndarray:
    names = [name.decode() for name in f["analog/channel_names"][:]]
    if channel_name not in names:
        raise ValueError(f"Analog channel {channel_name!r} not found. Available: {names}")
    idx = names.index(channel_name)
    if "samples_int16" in f["analog"]:
        raw = f["analog/samples_int16"][:, idx]
        scale = float(f["analog/int16_scale_volts_per_count"][idx])
        offset = float(f["analog/int16_offset_volts"][idx])
        return raw.astype(np.float32) * scale + offset
    return np.asarray(f["analog/samples"][:, idx], dtype=np.float32)


def _load_daq_events(
    h5_path: Path,
    lick_channel: str,
    thresh_upper: float,
    thresh_lower: float,
    lockout_s: tuple[float, float],
    refractory_s: float,
) -> dict:
    with h5py.File(h5_path, "r") as f:
        sr = float(f.attrs["sample_rate_hz"])
        created_at = str(f.attrs["created_at"])
        lick = _decode_analog_channel(f, lick_channel)
        names = [name.decode() for name in f["digital/channel_names"][:]]
        packed = f["digital/packed_samples"][:, 0]
        bits = np.unpackbits(packed[:, None], axis=1, bitorder="little")[:, : len(names)]

    lick_detection = detect_licks(
        lick,
        sr,
        thresh_upper=thresh_upper,
        thresh_lower=thresh_lower,
        lockout_s=lockout_s,
        refractory_s=refractory_s,
    )
    lick_samples = np.asarray(lick_detection["lick_onsets"], dtype=np.int64)
    idx = {name: names.index(name) for name in names}
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
        "lick_samples": lick_samples,
        "strobe_samples": strobe,
        "strobe_codes": codes_at_strobe,
        "pco_samples": pco,
        "lick_voltage_percentiles": np.percentile(lick, [0.1, 1, 5, 50, 95, 99, 99.9]).tolist(),
        "lick_detection": {
            "raw_onset_count": int(np.asarray(lick_detection["raw_onsets"]).size),
            "offset_count": int(np.asarray(lick_detection["offsets"]).size),
            "cleaned_onset_count": int(lick_samples.size),
            "thresh_upper": float(thresh_upper),
            "thresh_lower": float(thresh_lower),
            "lockout_s": [float(v) for v in lockout_s],
            "refractory_s": float(refractory_s),
        },
    }


def _load_camlog_frame_times(camlog: Path) -> tuple[np.ndarray, datetime]:
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
    return np.array([(t - t0).total_seconds() for t in times], dtype=np.float64), t0


def _event_frame_indices_from_pco(event_samples: np.ndarray, pco_samples: np.ndarray) -> np.ndarray:
    """Map DAQ event samples to raw camera frame indices using DAQ PCO exposure pulses."""
    if pco_samples.size < 2:
        raise ValueError("Need at least two DAQ pco_exposure pulses for pulse-order frame alignment.")
    insertion = np.searchsorted(pco_samples, event_samples, side="left")
    insertion = np.clip(insertion, 1, len(pco_samples) - 1)
    prev_dist = np.abs(event_samples - pco_samples[insertion - 1])
    next_dist = np.abs(pco_samples[insertion] - event_samples)
    return np.where(prev_dist <= next_dist, insertion - 1, insertion).astype(np.int64)


def _frame_qc(
    pco_samples: np.ndarray,
    svt_frame_count: int,
    camlog: Path | None,
    raw_event_frames: np.ndarray,
) -> dict:
    camlog_frame_count = None
    if camlog is not None:
        try:
            camlog_frame_count = int(len(_load_camlog_frame_times(camlog)[0]))
        except Exception as exc:
            camlog_frame_count = f"error: {exc}"
    expected_raw_from_svt = int(svt_frame_count * 2)
    pco_count = int(len(pco_samples))
    qc = {
        "alignment_mode_note": "Primary frame mapping should use DAQ pco_exposure pulse order when available.",
        "daq_pco_exposure_count": pco_count,
        "camlog_frame_count": camlog_frame_count,
        "svt_corrected_frame_count": int(svt_frame_count),
        "expected_raw_frames_from_svt_pairs": expected_raw_from_svt,
        "daq_pco_minus_expected_raw_frames": int(pco_count - expected_raw_from_svt),
        "raw_event_frame_min": int(np.min(raw_event_frames)) if raw_event_frames.size else None,
        "raw_event_frame_max": int(np.max(raw_event_frames)) if raw_event_frames.size else None,
        "corrected_event_frame_min": int(np.min(raw_event_frames // 2)) if raw_event_frames.size else None,
        "corrected_event_frame_max": int(np.max(raw_event_frames // 2)) if raw_event_frames.size else None,
    }
    if isinstance(camlog_frame_count, int):
        qc["daq_pco_minus_camlog_frames"] = int(pco_count - camlog_frame_count)
    return qc


def _event_frame_indices_from_camlog(
    event_samples: np.ndarray,
    sample_rate_hz: float,
    daq_created_at: str,
    camlog: Path,
) -> np.ndarray:
    cam_seconds, first_frame = _load_camlog_frame_times(camlog)
    daq_t0 = datetime.fromisoformat(daq_created_at)
    event_abs_seconds = np.array(
        [
            (daq_t0 + timedelta(seconds=float(s) / sample_rate_hz) - first_frame).total_seconds()
            for s in event_samples
        ],
        dtype=np.float64,
    )
    insertion = np.searchsorted(cam_seconds, event_abs_seconds, side="left")
    insertion = np.clip(insertion, 1, len(cam_seconds) - 1)
    prev_dist = np.abs(event_abs_seconds - cam_seconds[insertion - 1])
    next_dist = np.abs(cam_seconds[insertion] - event_abs_seconds)
    return np.where(prev_dist <= next_dist, insertion - 1, insertion).astype(np.int64)


def _classify_events(event_samples: np.ndarray, strobe_samples: np.ndarray, codes: np.ndarray) -> np.ndarray:
    strobe_idx = np.searchsorted(strobe_samples, event_samples, side="right") - 1
    out = np.full(event_samples.shape, -1, dtype=np.int16)
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
    return ndimage.binary_dilation(edges & valid, iterations=1)


def _overlay_regions(ax, edges: np.ndarray) -> None:
    overlay = np.zeros((*edges.shape, 4), dtype=np.float32)
    overlay[edges] = (0, 0, 0, 0.65)
    ax.imshow(overlay, interpolation="nearest")


def _shared_limit(maps: dict[str, np.ndarray], percentile: float) -> float:
    vals = []
    for arr in maps.values():
        vals.append(arr.ravel())
    vals = np.concatenate(vals)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1e-6
    return max(float(np.nanpercentile(np.abs(vals), percentile)), 1e-6)


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-lick averages by spout position.")
    parser.add_argument("--daq-h5", type=Path, required=True)
    parser.add_argument("--wfield-results", type=Path, required=True)
    parser.add_argument("--allen-dir", type=Path, required=True)
    parser.add_argument("--camlog", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="PS95")
    parser.add_argument("--lick-channel", default="lick_analog")
    parser.add_argument("--lick-thresh-upper-v", type=float, default=2.5)
    parser.add_argument("--lick-thresh-lower-v", type=float, default=1.0)
    parser.add_argument("--lockout-s", type=float, nargs=2, default=(0.001, 0.020))
    parser.add_argument("--refractory-s", type=float, default=0.10)
    parser.add_argument("--post-s", type=float, default=0.150)
    parser.add_argument("--fs", type=float, default=31.23)
    parser.add_argument(
        "--frame-align",
        choices=("pco", "camlog"),
        default="pco",
        help="Map DAQ licks to imaging frames by DAQ pco_exposure pulse order (default) or legacy camlog wall-clock timestamps.",
    )
    parser.add_argument("--display-percentile", type=float, default=99.0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    events = _load_daq_events(
        args.daq_h5,
        args.lick_channel,
        args.lick_thresh_upper_v,
        args.lick_thresh_lower_v,
        tuple(args.lockout_s),
        args.refractory_s,
    )
    U = np.load(args.allen_dir / "U_atlas.npy", mmap_mode="r")
    SVTcorr = np.load(args.wfield_results / "SVTcorr.npy", mmap_mode="r")
    atlas = np.load(args.allen_dir / "allen_area_atlas_native_grid.npy")
    edges = _region_edges(atlas)

    if args.frame_align == "camlog":
        if args.camlog is None:
            raise ValueError("--frame-align camlog requires --camlog")
        raw_frames = _event_frame_indices_from_camlog(
            events["lick_samples"],
            events["sample_rate_hz"],
            events["created_at"],
            args.camlog,
        )
        frame_mapping = (
            "DAQ lick wall-clock times mapped to nearest labcams camlog frame timestamp; "
            "individual camera-frame indices were divided by 2 to match paired 415/470 SVTcorr timepoints."
        )
    else:
        raw_frames = _event_frame_indices_from_pco(events["lick_samples"], events["pco_samples"])
        frame_mapping = (
            "DAQ lick samples mapped to nearest DAQ pco_exposure rising-edge index; "
            "that raw camera-frame index was divided by 2 to match paired 415/470 SVTcorr timepoints."
        )
    lick_frames = raw_frames // 2
    frame_qc = _frame_qc(events["pco_samples"], SVTcorr.shape[1], args.camlog, raw_frames)
    codes = _classify_events(events["lick_samples"], events["strobe_samples"], events["strobe_codes"])

    post_n = max(1, int(round(args.post_s * args.fs)))
    valid_window = (lick_frames + post_n <= SVTcorr.shape[1])
    valid = valid_window & (codes >= 0)

    maps = {}
    counts = {}
    for code in DISPLAY_ORDER:
        event_frames = lick_frames[valid & (codes == code)]
        name = POSITION_NAMES[code]
        counts[name] = int(event_frames.size)
        if event_frames.size == 0:
            continue
        post_sum = np.zeros(SVTcorr.shape[0], dtype=np.float64)
        for frame in event_frames:
            post_sum += np.asarray(SVTcorr[:, frame : frame + post_n]).mean(axis=1)
        maps[name] = _weighted_map(U, (post_sum / event_frames.size).astype(np.float32))

    lim = _shared_limit(maps, args.display_percentile)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    im = None
    for ax, code in zip(axes.ravel(), DISPLAY_ORDER):
        name = POSITION_NAMES[code]
        ax.set_axis_off()
        if name not in maps:
            ax.set_title(f"{name}: no licks")
            continue
        im = ax.imshow(maps[name], cmap="RdBu_r", vmin=-lim, vmax=lim)
        _overlay_regions(ax, edges)
        ax.set_title(f"{name} n={counts[name]} | {args.post_s * 1000:.0f} ms post-lick", fontsize=10)
    if im is not None:
        fig.colorbar(
            im,
            ax=axes.ravel().tolist(),
            shrink=0.78,
            pad=0.01,
            label=f"Shared scale across all panels (±{lim:.4g})",
        )
    fig.suptitle(f"{args.label} post-lick hemodynamic-corrected averages by spout position", fontsize=14)
    png = args.output / f"{args.label}_lick_aligned_{int(round(args.post_s * 1000))}ms_post_by_spout.png"
    fig.savefig(png, dpi=180)
    plt.close(fig)

    npz_payload = {}
    for name, arr in maps.items():
        npz_payload[f"{name}_post"] = arr
    np.savez_compressed(
        args.output / f"{args.label}_lick_aligned_{int(round(args.post_s * 1000))}ms_post_by_spout_maps.npz",
        **npz_payload,
    )
    summary = {
        "label": args.label,
        "daq_h5": str(args.daq_h5),
        "wfield_results": str(args.wfield_results),
        "allen_dir": str(args.allen_dir),
        "camlog": str(args.camlog) if args.camlog is not None else None,
        "frame_align": args.frame_align,
        "lick_channel": args.lick_channel,
        "lick_thresh_upper_v": args.lick_thresh_upper_v,
        "lick_thresh_lower_v": args.lick_thresh_lower_v,
        "lockout_s": [float(v) for v in args.lockout_s],
        "refractory_s": args.refractory_s,
        "post_s": args.post_s,
        "fs": args.fs,
        "detected_lick_count": int(events["lick_samples"].size),
        "valid_licks_with_windows": int(valid.sum()),
        "counts_by_position": counts,
        "display_limit": lim,
        "lick_voltage_percentiles": events["lick_voltage_percentiles"],
        "lick_detection": events["lick_detection"],
        "frame_mapping": frame_mapping,
        "frame_alignment_qc": frame_qc,
        "output_png": str(png),
    }
    (args.output / f"{args.label}_lick_aligned_{int(round(args.post_s * 1000))}ms_post_by_spout_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
