"""Plot hemodynamic-corrected widefield activity during running vs not running."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

try:
    from .treadmill import bout_edges, calibrate_treadmill, find_running_bouts, smooth_treadmill
except ImportError:  # Allow direct script execution.
    from treadmill import bout_edges, calibrate_treadmill, find_running_bouts, smooth_treadmill


def _rising_edges(x: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.diff(x.astype(np.int8), prepend=0) == 1).astype(np.int64)


def _decode_analog_channel(f: h5py.File, channel_name: str) -> np.ndarray:
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


def _load_daq(h5_path: Path, treadmill_channel: str) -> dict:
    with h5py.File(h5_path, "r") as f:
        sample_rate_hz = float(f.attrs["sample_rate_hz"])
        created_at = str(f.attrs.get("created_at", "")) or None
        treadmill_v = _decode_analog_channel(f, treadmill_channel)
        names = [name.decode() for name in f["digital/channel_names"][:]]
        packed = f["digital/packed_samples"][:, 0]
        bits = np.unpackbits(packed[:, None], axis=1, bitorder="little")[:, : len(names)]
    idx = {name: i for i, name in enumerate(names)}
    if "pco_exposure" not in idx:
        raise ValueError("Digital channel 'pco_exposure' is required.")
    return {
        "sample_rate_hz": sample_rate_hz,
        "created_at": created_at,
        "treadmill_v": treadmill_v,
        "pco_samples": _rising_edges(bits[:, idx["pco_exposure"]]),
    }


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


def _weighted_map(U: np.ndarray, svt_mean: np.ndarray) -> np.ndarray:
    return np.tensordot(U, svt_mean, axes=([2], [0])).astype(np.float32)


def _display_limit(arrays: list[np.ndarray], percentile: float) -> float:
    vals = np.concatenate([arr.ravel() for arr in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1e-6
    return max(float(np.nanpercentile(np.abs(vals), percentile)), 1e-6)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot running vs not-running widefield maps.")
    parser.add_argument("--label", required=True)
    parser.add_argument("--daq-h5", type=Path, required=True)
    parser.add_argument("--wfield-results", type=Path, required=True)
    parser.add_argument("--allen-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channel", default="treadmill")
    parser.add_argument("--offset-v", default="auto")
    parser.add_argument("--volt-sec-per-rot", type=float, required=True)
    parser.add_argument("--mm-per-rot", type=float, required=True)
    parser.add_argument("--smoothing-sigma-s", type=float, default=0.100)
    parser.add_argument("--thresh-speed", type=float, default=10.0)
    parser.add_argument("--max-gap-duration", type=float, default=0.250)
    parser.add_argument("--min-duration", type=float, default=0.500)
    parser.add_argument("--frame-margin-s", type=float, default=0.0)
    parser.add_argument("--percentile", type=float, default=99.0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    daq = _load_daq(args.daq_h5, args.channel)
    treadmill_v = daq["treadmill_v"]
    sample_rate_hz = daq["sample_rate_hz"]
    if str(args.offset_v).lower() == "auto":
        offset_v = float(np.median(treadmill_v))
        offset_source = "median voltage from this recording"
    else:
        offset_v = float(args.offset_v)
        offset_source = "command line"

    speed = calibrate_treadmill(treadmill_v, offset_v, args.volt_sec_per_rot, args.mm_per_rot)
    smooth_speed = smooth_treadmill(speed, sample_rate_hz, args.smoothing_sigma_s)
    running = find_running_bouts(
        smooth_speed,
        sample_rate_hz,
        args.thresh_speed,
        args.max_gap_duration,
        args.min_duration,
    )
    if args.frame_margin_s > 0:
        margin = int(round(args.frame_margin_s * sample_rate_hz))
        if margin > 0:
            running = ndimage.binary_erosion(running, iterations=margin, border_value=0)

    U = np.load(args.allen_dir / "U_atlas.npy", mmap_mode="r")
    SVTcorr = np.load(args.wfield_results / "SVTcorr.npy", mmap_mode="r")
    atlas = np.load(args.allen_dir / "allen_area_atlas_native_grid.npy")
    edges = _region_edges(atlas)

    pco_samples = daq["pco_samples"]
    raw_frame_for_corrected = 2 * np.arange(SVTcorr.shape[1], dtype=np.int64)
    valid = raw_frame_for_corrected < pco_samples.size
    corrected_sample = pco_samples[raw_frame_for_corrected[valid]]
    frame_running = np.zeros(SVTcorr.shape[1], dtype=bool)
    frame_running[np.flatnonzero(valid)] = running[np.clip(corrected_sample, 0, running.size - 1)]
    frame_not_running = valid & ~frame_running

    if frame_running.sum() == 0 or frame_not_running.sum() == 0:
        raise ValueError(
            f"Need both running and not-running frames; got running={int(frame_running.sum())}, "
            f"not_running={int(frame_not_running.sum())}."
        )

    running_svt = np.asarray(SVTcorr[:, frame_running]).mean(axis=1)
    not_running_svt = np.asarray(SVTcorr[:, frame_not_running]).mean(axis=1)
    running_map = _weighted_map(U, running_svt)
    not_running_map = _weighted_map(U, not_running_svt)
    delta_map = running_map - not_running_map

    lim = _display_limit([running_map, not_running_map, delta_map], args.percentile)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)
    panels = [
        ("running", running_map, int(frame_running.sum())),
        ("not running", not_running_map, int(frame_not_running.sum())),
        ("running - not running", delta_map, int(frame_running.sum())),
    ]
    im = None
    for ax, (title, arr, n) in zip(axes, panels):
        im = ax.imshow(arr, cmap="RdBu_r", vmin=-lim, vmax=lim)
        _overlay_regions(ax, edges)
        ax.set_axis_off()
        ax.set_title(f"{title}\nn={n} corrected frames")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, pad=0.01, label=f"shared scale (+/-{lim:.4g})")
    fig.suptitle(f"{args.label}: hemodynamic-corrected activity by treadmill state", fontsize=14)
    png = args.output / f"{args.label}_running_vs_not_running_activity_maps.png"
    fig.savefig(png, dpi=180)
    plt.close(fig)

    starts, stops = bout_edges(running)
    durations_s = (stops - starts) / sample_rate_hz
    npz_path = args.output / f"{args.label}_running_vs_not_running_activity_maps.npz"
    np.savez_compressed(
        npz_path,
        running_map=running_map,
        not_running_map=not_running_map,
        running_minus_not_running=delta_map,
        frame_running=frame_running,
        smooth_speed_mm_s=smooth_speed.astype(np.float32),
        sample_rate_hz=np.asarray(sample_rate_hz, dtype=np.float64),
    )
    summary = {
        "label": args.label,
        "daq_h5": str(args.daq_h5),
        "wfield_results": str(args.wfield_results),
        "allen_dir": str(args.allen_dir),
        "channel": args.channel,
        "offset_v": offset_v,
        "offset_source": offset_source,
        "volt_sec_per_rot": args.volt_sec_per_rot,
        "mm_per_rot": args.mm_per_rot,
        "smoothing_sigma_s": args.smoothing_sigma_s,
        "thresh_speed": args.thresh_speed,
        "max_gap_duration": args.max_gap_duration,
        "min_duration": args.min_duration,
        "frame_margin_s": args.frame_margin_s,
        "daq_pco_exposure_count": int(pco_samples.size),
        "svt_corrected_frame_count": int(SVTcorr.shape[1]),
        "valid_corrected_frame_count": int(valid.sum()),
        "running_corrected_frame_count": int(frame_running.sum()),
        "not_running_corrected_frame_count": int(frame_not_running.sum()),
        "n_running_bouts": int(starts.size),
        "running_bout_duration_s_median": float(np.median(durations_s)) if durations_s.size else None,
        "running_bout_duration_s_max": float(np.max(durations_s)) if durations_s.size else None,
        "display_limit": lim,
        "outputs": {"png": str(png), "npz": str(npz_path)},
    }
    summary_path = args.output / f"{args.label}_running_vs_not_running_activity_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
