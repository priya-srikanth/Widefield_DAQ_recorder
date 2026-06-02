"""Compare old camlog wall-clock frame mapping against DAQ PCO pulse mapping.

The DAQ PCO exposure pulse order is treated as the hardware reference because
those pulses are sampled by the same DAQ clock as behavior events. The older
method maps DAQ event wall-clock times to labcams camlog timestamps. This script
plots the difference between those two mappings for selected event types.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .lick_detection import detect_licks
except ImportError:  # Allow direct script execution.
    from lick_detection import detect_licks


DEFAULT_DIGITAL_EVENTS = ("cue", "trial_start", "spout_strobe", "reward_ttl")


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


def _load_daq_events(
    h5_path: Path,
    digital_events: tuple[str, ...],
    include_licks: bool,
    lick_channel: str,
    lick_thresh_upper_v: float,
    lick_thresh_lower_v: float,
    lockout_s: tuple[float, float],
    refractory_s: float,
) -> dict:
    with h5py.File(h5_path, "r") as f:
        sr = float(f.attrs["sample_rate_hz"])
        created_at = str(f.attrs["created_at"])
        names = [name.decode() for name in f["digital/channel_names"][:]]
        packed = f["digital/packed_samples"][:, 0]
        bits = np.unpackbits(packed[:, None], axis=1, bitorder="little")[:, : len(names)]

        events: dict[str, np.ndarray] = {}
        name_to_idx = {name: i for i, name in enumerate(names)}
        for name in digital_events:
            if name not in name_to_idx:
                continue
            events[name] = _rising_edges(bits[:, name_to_idx[name]])
        if "pco_exposure" not in name_to_idx:
            raise ValueError("Digital channel 'pco_exposure' is required for DAQ PCO frame alignment.")
        pco_samples = _rising_edges(bits[:, name_to_idx["pco_exposure"]])

        lick_detection = None
        if include_licks:
            lick = _decode_analog_channel(f, lick_channel)
            lick_detection = detect_licks(
                lick,
                sr,
                thresh_upper=lick_thresh_upper_v,
                thresh_lower=lick_thresh_lower_v,
                lockout_s=lockout_s,
                refractory_s=refractory_s,
            )
            events["lick"] = np.asarray(lick_detection["lick_onsets"], dtype=np.int64)

    return {
        "sample_rate_hz": sr,
        "created_at": created_at,
        "events": events,
        "pco_samples": pco_samples,
        "lick_detection": lick_detection,
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
    return np.asarray([(t - t0).total_seconds() for t in times], dtype=np.float64), t0


def _event_raw_frames_from_camlog(
    event_samples: np.ndarray,
    sample_rate_hz: float,
    daq_created_at: str,
    camlog_seconds: np.ndarray,
    first_frame_time: datetime,
) -> np.ndarray:
    daq_t0 = datetime.fromisoformat(daq_created_at)
    event_abs_seconds = np.asarray(
        [
            (daq_t0 + timedelta(seconds=float(s) / sample_rate_hz) - first_frame_time).total_seconds()
            for s in event_samples
        ],
        dtype=np.float64,
    )
    insertion = np.searchsorted(camlog_seconds, event_abs_seconds, side="left")
    insertion = np.clip(insertion, 1, len(camlog_seconds) - 1)
    prev_dist = np.abs(event_abs_seconds - camlog_seconds[insertion - 1])
    next_dist = np.abs(camlog_seconds[insertion] - event_abs_seconds)
    return np.where(prev_dist <= next_dist, insertion - 1, insertion).astype(np.int64)


def _event_raw_frames_from_pco(event_samples: np.ndarray, pco_samples: np.ndarray) -> np.ndarray:
    if pco_samples.size < 2:
        raise ValueError("Need at least two DAQ pco_exposure pulses for pulse-order frame alignment.")
    insertion = np.searchsorted(pco_samples, event_samples, side="left")
    insertion = np.clip(insertion, 1, len(pco_samples) - 1)
    prev_dist = np.abs(event_samples - pco_samples[insertion - 1])
    next_dist = np.abs(pco_samples[insertion] - event_samples)
    return np.where(prev_dist <= next_dist, insertion - 1, insertion).astype(np.int64)


def _stats(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def _plot_event(ax_hist, ax_trace, event_name: str, corrected_delta: np.ndarray, time_delta_ms: np.ndarray) -> None:
    if corrected_delta.size == 0:
        ax_hist.set_title(f"{event_name}: no events")
        ax_trace.set_axis_off()
        return
    bins = np.arange(np.floor(corrected_delta.min()) - 0.5, np.ceil(corrected_delta.max()) + 1.5)
    ax_hist.hist(corrected_delta, bins=bins, color="#4c78a8", edgecolor="white")
    ax_hist.axvline(0, color="black", linewidth=1)
    ax_hist.set_title(
        f"{event_name}: n={corrected_delta.size}, median={np.median(corrected_delta):.1f} corrected frames"
    )
    ax_hist.set_xlabel("old camlog - new PCO (corrected frames)")
    ax_hist.set_ylabel("event count")

    ax_trace.plot(corrected_delta, ".", markersize=3, color="#f58518", label="corrected frames")
    ax_trace.axhline(0, color="black", linewidth=1)
    ax_trace.set_ylabel("corrected-frame delta")
    ax_trace.set_xlabel("event order")
    ax_trace_ms = ax_trace.twinx()
    ax_trace_ms.plot(time_delta_ms, ".", markersize=2, alpha=0.25, color="#54a24b", label="ms")
    ax_trace_ms.set_ylabel("raw-frame time delta (ms)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare camlog and DAQ PCO event-to-frame mappings.")
    parser.add_argument("--label", required=True)
    parser.add_argument("--daq-h5", type=Path, required=True)
    parser.add_argument("--camlog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", nargs="+", default=list(DEFAULT_DIGITAL_EVENTS))
    parser.add_argument("--include-licks", action="store_true")
    parser.add_argument("--lick-channel", default="lick_analog")
    parser.add_argument("--lick-thresh-upper-v", type=float, default=2.5)
    parser.add_argument("--lick-thresh-lower-v", type=float, default=1.0)
    parser.add_argument("--lockout-s", type=float, nargs=2, default=(0.001, 0.020))
    parser.add_argument("--refractory-s", type=float, default=0.10)
    parser.add_argument(
        "--raw-frame-rate",
        type=float,
        default=None,
        help="Optional raw camera frame rate for ms conversion. Defaults to median DAQ pco_exposure interval.",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    daq = _load_daq_events(
        args.daq_h5,
        tuple(args.events),
        args.include_licks,
        args.lick_channel,
        args.lick_thresh_upper_v,
        args.lick_thresh_lower_v,
        tuple(args.lockout_s),
        args.refractory_s,
    )
    cam_seconds, first_frame_time = _load_camlog_frame_times(args.camlog)
    pco_samples = daq["pco_samples"]
    sample_rate_hz = float(daq["sample_rate_hz"])
    if args.raw_frame_rate is None:
        raw_frame_rate = sample_rate_hz / float(np.median(np.diff(pco_samples)))
    else:
        raw_frame_rate = float(args.raw_frame_rate)

    rows = []
    summary = {
        "label": args.label,
        "daq_h5": str(args.daq_h5),
        "camlog": str(args.camlog),
        "sample_rate_hz": sample_rate_hz,
        "daq_created_at": daq["created_at"],
        "camlog_first_frame_time": first_frame_time.isoformat(),
        "daq_pco_exposure_count": int(pco_samples.size),
        "camlog_frame_count": int(cam_seconds.size),
        "estimated_raw_frame_rate_hz": raw_frame_rate,
        "method": "delta = old camlog wall-clock frame index - new DAQ pco_exposure pulse-order frame index",
        "events": {},
    }

    event_results = {}
    for event_name, samples in daq["events"].items():
        if samples.size == 0:
            continue
        pco_frames = _event_raw_frames_from_pco(samples, pco_samples)
        camlog_frames = _event_raw_frames_from_camlog(
            samples,
            sample_rate_hz,
            daq["created_at"],
            cam_seconds,
            first_frame_time,
        )
        raw_delta = camlog_frames - pco_frames
        corrected_delta = (camlog_frames // 2) - (pco_frames // 2)
        time_delta_ms = raw_delta / raw_frame_rate * 1000.0
        event_results[event_name] = {
            "raw_delta": raw_delta,
            "corrected_delta": corrected_delta,
            "time_delta_ms": time_delta_ms,
        }
        summary["events"][event_name] = {
            "raw_frame_delta_stats": _stats(raw_delta.astype(np.float64)),
            "corrected_frame_delta_stats": _stats(corrected_delta.astype(np.float64)),
            "time_delta_ms_stats": _stats(time_delta_ms.astype(np.float64)),
        }
        for i, sample in enumerate(samples):
            rows.append(
                {
                    "label": args.label,
                    "event": event_name,
                    "event_index": i,
                    "daq_sample": int(sample),
                    "pco_raw_frame": int(pco_frames[i]),
                    "camlog_raw_frame": int(camlog_frames[i]),
                    "raw_frame_delta_old_minus_new": int(raw_delta[i]),
                    "corrected_frame_delta_old_minus_new": int(corrected_delta[i]),
                    "time_delta_ms_old_minus_new": float(time_delta_ms[i]),
                }
            )

    n_events = max(1, len(event_results))
    fig, axes = plt.subplots(n_events, 2, figsize=(13, 3.4 * n_events), constrained_layout=True)
    if n_events == 1:
        axes = np.asarray([axes])
    for row_idx, (event_name, result) in enumerate(event_results.items()):
        _plot_event(
            axes[row_idx, 0],
            axes[row_idx, 1],
            event_name,
            result["corrected_delta"],
            result["time_delta_ms"],
        )
    fig.suptitle(f"{args.label}: old camlog vs new DAQ PCO frame alignment", fontsize=15)
    png = args.output / f"{args.label}_frame_alignment_old_camlog_vs_new_pco.png"
    fig.savefig(png, dpi=180)
    plt.close(fig)

    csv_path = args.output / f"{args.label}_frame_alignment_old_camlog_vs_new_pco_events.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["label"])
        writer.writeheader()
        writer.writerows(rows)

    summary["output_png"] = str(png)
    summary["output_csv"] = str(csv_path)
    if daq["lick_detection"] is not None:
        summary["lick_detection"] = {
            "raw_onset_count": int(np.asarray(daq["lick_detection"]["raw_onsets"]).size),
            "cleaned_onset_count": int(np.asarray(daq["lick_detection"]["lick_onsets"]).size),
            "thresh_upper": float(daq["lick_detection"]["thresh_upper"]),
            "thresh_lower": float(daq["lick_detection"]["thresh_lower"]),
            "refractory_s": float(daq["lick_detection"]["refractory_s"]),
        }
    summary_path = args.output / f"{args.label}_frame_alignment_old_camlog_vs_new_pco_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
