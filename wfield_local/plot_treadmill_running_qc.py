"""Generate treadmill velocity and running-bout QC plots from DAQ recorder HDF5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .treadmill import (
        bout_edges,
        calibrate_treadmill,
        cumulative_distance_mm,
        find_running_bouts,
        smooth_treadmill,
    )
except ImportError:  # Allow direct script execution.
    from treadmill import (
        bout_edges,
        calibrate_treadmill,
        cumulative_distance_mm,
        find_running_bouts,
        smooth_treadmill,
    )


DEFAULT_OFFSET_V = 1.2587643276652853
DEFAULT_VOLT_SEC_PER_ROT = 0.382
DEFAULT_MM_PER_ROT = 29.25
DEFAULT_SMOOTHING_SIGMA_S = 0.15
DEFAULT_THRESH_SPEED_MM_S = 5.0
DEFAULT_MAX_GAP_DURATION_S = 0.3
DEFAULT_MIN_DURATION_S = 2.0


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


def _load_treadmill(h5_path: Path, channel_name: str) -> tuple[np.ndarray, float, str | None]:
    with h5py.File(h5_path, "r") as f:
        sample_rate_hz = float(f.attrs["sample_rate_hz"])
        created_at = str(f.attrs.get("created_at", "")) or None
        signal = _decode_analog_channel(f, channel_name)
    return signal, sample_rate_hz, created_at


def _stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "min": float(np.min(values)),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def _choose_example_windows(
    running: np.ndarray,
    sample_rate_hz: float,
    window_s: float,
) -> list[tuple[str, int, int]]:
    starts, stops = bout_edges(running)
    win_n = int(round(window_s * sample_rate_hz))
    examples: list[tuple[str, int, int]] = []
    if starts.size:
        durations = stops - starts
        order = np.argsort(durations)[::-1]
        for rank, idx in enumerate(order[:3], start=1):
            center = int((starts[idx] + stops[idx]) // 2)
            lo = max(0, center - win_n // 2)
            hi = min(running.size, lo + win_n)
            lo = max(0, hi - win_n)
            examples.append((f"running bout {rank}", lo, hi))

    not_running = ~running
    nr_starts, nr_stops = bout_edges(not_running)
    if nr_starts.size:
        durations = nr_stops - nr_starts
        order = np.argsort(durations)[::-1]
        for rank, idx in enumerate(order[:3], start=1):
            center = int((nr_starts[idx] + nr_stops[idx]) // 2)
            lo = max(0, center - win_n // 2)
            hi = min(running.size, lo + win_n)
            lo = max(0, hi - win_n)
            examples.append((f"not running {rank}", lo, hi))
    return examples


def _plot_overview(
    output: Path,
    label: str,
    voltage: np.ndarray,
    speed: np.ndarray,
    smooth_speed: np.ndarray,
    running: np.ndarray,
    sample_rate_hz: float,
    thresh_speed: float,
) -> Path:
    t = np.arange(voltage.size) / sample_rate_hz
    starts, stops = bout_edges(running)

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    axes[0].plot(t, voltage, linewidth=0.5, color="#4c78a8")
    axes[0].set_ylabel("volts")
    axes[0].set_title(f"{label}: treadmill voltage")

    axes[1].plot(t, speed, linewidth=0.35, alpha=0.35, color="#999999", label="raw calibrated")
    axes[1].plot(t, smooth_speed, linewidth=0.8, color="#f58518", label="smoothed")
    axes[1].axhline(thresh_speed, color="black", linestyle="--", linewidth=0.8, label="running threshold")
    axes[1].set_ylabel("mm/s")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_title("calibrated treadmill speed")

    axes[2].plot(t, smooth_speed, linewidth=0.6, color="#4c78a8")
    ymin, ymax = axes[2].get_ylim()
    for start, stop in zip(starts, stops):
        axes[2].axvspan(start / sample_rate_hz, stop / sample_rate_hz, color="#54a24b", alpha=0.25)
    axes[2].set_ylim(ymin, ymax)
    axes[2].set_ylabel("mm/s")
    axes[2].set_xlabel("time (s)")
    axes[2].set_title(f"running bouts (n={starts.size})")

    png = output / f"{label}_treadmill_running_bout_overview.png"
    fig.savefig(png, dpi=180)
    plt.close(fig)
    return png


def _plot_examples(
    output: Path,
    label: str,
    voltage: np.ndarray,
    smooth_speed: np.ndarray,
    running: np.ndarray,
    sample_rate_hz: float,
    thresh_speed: float,
    window_s: float,
) -> Path:
    examples = _choose_example_windows(running, sample_rate_hz, window_s)
    if not examples:
        raise ValueError("No running or non-running windows available for example plot.")
    fig, axes = plt.subplots(len(examples), 1, figsize=(12, 2.2 * len(examples)), sharex=False, constrained_layout=True)
    if len(examples) == 1:
        axes = [axes]
    for ax, (title, lo, hi) in zip(axes, examples):
        t = (np.arange(lo, hi) - lo) / sample_rate_hz
        ax.plot(t, smooth_speed[lo:hi], color="#f58518", linewidth=1.0, label="smoothed speed")
        ax.axhline(thresh_speed, color="black", linestyle="--", linewidth=0.8)
        ax2 = ax.twinx()
        ax2.plot(t, voltage[lo:hi], color="#4c78a8", linewidth=0.45, alpha=0.45, label="voltage")
        ax.fill_between(
            t,
            ax.get_ylim()[0],
            ax.get_ylim()[1],
            where=running[lo:hi],
            color="#54a24b",
            alpha=0.18,
            transform=ax.get_xaxis_transform(),
        )
        ax.set_title(f"{title}: samples {lo}-{hi}")
        ax.set_ylabel("mm/s")
        ax2.set_ylabel("V")
    axes[-1].set_xlabel("window time (s)")
    png = output / f"{label}_treadmill_running_not_running_examples.png"
    fig.savefig(png, dpi=180)
    plt.close(fig)
    return png


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot treadmill running-bout QC from DAQ HDF5.")
    parser.add_argument("--label", required=True)
    parser.add_argument("--daq-h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channel", default="treadmill")
    parser.add_argument(
        "--offset-v",
        default=str(DEFAULT_OFFSET_V),
        help="Voltage offset to subtract, or 'auto' to use the median voltage. Default matches legacy cohort config.",
    )
    parser.add_argument("--volt-sec-per-rot", type=float, default=DEFAULT_VOLT_SEC_PER_ROT)
    parser.add_argument("--mm-per-rot", type=float, default=DEFAULT_MM_PER_ROT)
    parser.add_argument("--smoothing-sigma-s", type=float, default=DEFAULT_SMOOTHING_SIGMA_S)
    parser.add_argument("--thresh-speed", type=float, default=DEFAULT_THRESH_SPEED_MM_S)
    parser.add_argument("--max-gap-duration", type=float, default=DEFAULT_MAX_GAP_DURATION_S)
    parser.add_argument("--min-duration", type=float, default=DEFAULT_MIN_DURATION_S)
    parser.add_argument("--example-window-s", type=float, default=10.0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    voltage, sample_rate_hz, created_at = _load_treadmill(args.daq_h5, args.channel)
    if str(args.offset_v).lower() == "auto":
        offset_v = float(np.median(voltage))
        offset_source = "median voltage from this recording"
    else:
        offset_v = float(args.offset_v)
        offset_source = "command line"

    speed = calibrate_treadmill(voltage, offset_v, args.volt_sec_per_rot, args.mm_per_rot)
    smooth_speed = smooth_treadmill(speed, sample_rate_hz, args.smoothing_sigma_s)
    running = find_running_bouts(
        smooth_speed,
        sample_rate_hz,
        args.thresh_speed,
        args.max_gap_duration,
        args.min_duration,
    )
    starts, stops = bout_edges(running)
    durations_s = (stops - starts) / sample_rate_hz
    distance = cumulative_distance_mm(speed, sample_rate_hz)

    overview_png = _plot_overview(
        args.output,
        args.label,
        voltage,
        speed,
        smooth_speed,
        running,
        sample_rate_hz,
        args.thresh_speed,
    )
    examples_png = _plot_examples(
        args.output,
        args.label,
        voltage,
        smooth_speed,
        running,
        sample_rate_hz,
        args.thresh_speed,
        args.example_window_s,
    )

    npz_path = args.output / f"{args.label}_treadmill_running_bouts.npz"
    np.savez_compressed(
        npz_path,
        voltage=voltage.astype(np.float32),
        speed_mm_s=speed.astype(np.float32),
        smooth_speed_mm_s=smooth_speed.astype(np.float32),
        running_mask=running,
        bout_start_samples=starts,
        bout_stop_samples=stops,
        cumulative_distance_mm=distance.astype(np.float32),
        sample_rate_hz=np.asarray(sample_rate_hz, dtype=np.float64),
    )

    summary = {
        "label": args.label,
        "daq_h5": str(args.daq_h5),
        "created_at": created_at,
        "channel": args.channel,
        "sample_rate_hz": sample_rate_hz,
        "offset_v": offset_v,
        "offset_source": offset_source,
        "volt_sec_per_rot": args.volt_sec_per_rot,
        "mm_per_rot": args.mm_per_rot,
        "smoothing_sigma_s": args.smoothing_sigma_s,
        "thresh_speed": args.thresh_speed,
        "max_gap_duration": args.max_gap_duration,
        "min_duration": args.min_duration,
        "n_samples": int(voltage.size),
        "duration_s": float(voltage.size / sample_rate_hz),
        "running_fraction": float(np.mean(running)),
        "n_running_bouts": int(starts.size),
        "running_bout_duration_s_stats": _stats(durations_s),
        "voltage_stats": _stats(voltage),
        "speed_mm_s_stats": _stats(speed),
        "smooth_speed_mm_s_stats": _stats(smooth_speed),
        "total_distance_mm": float(distance[-1]) if distance.size else 0.0,
        "outputs": {
            "overview_png": str(overview_png),
            "examples_png": str(examples_png),
            "npz": str(npz_path),
        },
    }
    summary_path = args.output / f"{args.label}_treadmill_running_bout_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
