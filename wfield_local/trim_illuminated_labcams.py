"""Trim a labcams DAT file to DAQ-confirmed illuminated 415/470 frame pairs.

This rescues sessions where the camera saved continuously but LEDs were gated.
It uses DAQ pco_exposure, led415_ttl, and led470_ttl channels to label each
saved physical camera frame, drops dark frames, keeps clean adjacent 415/470
pairs, and writes a normal two-channel labcams DAT for wfield processing.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import h5py
import numpy as np


DAT_RE = re.compile(r"_(?P<nchan>\d+)_(?P<h>\d+)_(?P<w>\d+)_(?P<dtype>uint16|int16|uint8|float32|float64)\.dat$")


def parse_labcams_dat_name(path: Path) -> tuple[int, int, int, np.dtype]:
    match = DAT_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not parse labcams DAT shape from filename: {path.name}")
    nchan = int(match.group("nchan"))
    height = int(match.group("h"))
    width = int(match.group("w"))
    dtype = np.dtype(match.group("dtype"))
    return nchan, height, width, dtype


def rising_falling_edges(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    rises = np.flatnonzero((~mask[:-1]) & mask[1:]) + 1
    falls = np.flatnonzero(mask[:-1] & (~mask[1:])) + 1
    if mask[0]:
        rises = np.r_[0, rises]
    if mask[-1]:
        falls = np.r_[falls, len(mask)]
    if len(falls) and len(rises) and falls[0] < rises[0]:
        falls = falls[1:]
    if len(rises) and len(falls) and rises[-1] > falls[-1]:
        rises = rises[:-1]
    n = min(len(rises), len(falls))
    return rises[:n], falls[:n]


def analog_ttl_mask(volts: np.ndarray, fixed_threshold: float | None) -> tuple[np.ndarray, float, float, float]:
    lo, hi = np.percentile(volts, [1, 99.9])
    if fixed_threshold is None:
        threshold = lo + 0.5 * (hi - lo)
        if hi - lo < 0.5:
            threshold = 2.5
    else:
        threshold = fixed_threshold
    return volts > threshold, float(threshold), float(lo), float(hi)


def load_daq_labels(
    h5_path: Path,
    physical_frame_count: int,
    offset: int | None,
    led_threshold: float | None,
) -> tuple[np.ndarray, dict]:
    with h5py.File(h5_path, "r") as h5:
        fs = float(h5.attrs["sample_rate_hz"])
        digital_names = [x.decode() for x in h5["digital/channel_names"][()]]
        analog_names = [x.decode() for x in h5["analog/channel_names"][()]]
        packed = h5["digital/packed_samples"][()][:, 0]
        pco_bit = digital_names.index("pco_exposure")
        pco = ((packed >> pco_bit) & 1).astype(bool)

        samples_i16 = h5["analog/samples_int16"][()].astype(np.float32)
        scale = h5["analog/int16_scale_volts_per_count"][()]
        zero = h5["analog/int16_offset_volts"][()]
        volts = samples_i16 * scale + zero
        led415 = volts[:, analog_names.index("led415_ttl")]
        led470 = volts[:, analog_names.index("led470_ttl")]

    pco_rise, pco_fall = rising_falling_edges(pco)
    b415, thr415, lo415, hi415 = analog_ttl_mask(led415, led_threshold)
    b470, thr470, lo470, hi470 = analog_ttl_mask(led470, led_threshold)

    labels_all = np.zeros(len(pco_rise), dtype=np.int16)
    for i, (start, stop) in enumerate(zip(pco_rise, pco_fall)):
        window = slice(max(0, start - 1), min(len(pco), stop + 1))
        has415 = bool(b415[window].any())
        has470 = bool(b470[window].any())
        if has415 and has470:
            labels_all[i] = 3
        elif has415:
            labels_all[i] = 415
        elif has470:
            labels_all[i] = 470

    candidate_offsets = [offset] if offset is not None else [0, 1]
    best = None
    for off in candidate_offsets:
        if off < 0 or off + physical_frame_count > len(labels_all):
            continue
        labels = labels_all[off : off + physical_frame_count]
        illum = labels[labels != 0]
        same_adjacent = int(np.sum(illum[1:] == illum[:-1])) if len(illum) > 1 else 0
        both = int(np.sum(illum == 3))
        score = (len(illum), -same_adjacent, -both)
        if best is None or score > best[0]:
            best = (score, off, labels)
    if best is None:
        raise ValueError("No valid DAQ exposure-label offset for DAT frame count")
    _, chosen_offset, labels = best

    meta = {
        "sample_rate_hz": fs,
        "daq_pco_exposure_count": int(len(labels_all)),
        "dat_physical_frame_count": int(physical_frame_count),
        "chosen_exposure_offset": int(chosen_offset),
        "led415_threshold_v": thr415,
        "led470_threshold_v": thr470,
        "led415_p1_v": lo415,
        "led415_p999_v": hi415,
        "led470_p1_v": lo470,
        "led470_p999_v": hi470,
        "labels_415": int(np.sum(labels == 415)),
        "labels_470": int(np.sum(labels == 470)),
        "labels_both": int(np.sum(labels == 3)),
        "labels_dark": int(np.sum(labels == 0)),
    }
    return labels, meta


def make_clean_pairs(labels: np.ndarray, order: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    illuminated = np.flatnonzero(labels != 0)
    pairs: list[tuple[int, int]] = []
    pair_labels: list[tuple[int, int]] = []
    skipped: list[int] = []
    i = 0
    while i + 1 < len(illuminated):
        a = int(illuminated[i])
        b = int(illuminated[i + 1])
        la = int(labels[a])
        lb = int(labels[b])
        if (la, lb) in ((415, 470), (470, 415)):
            if order == "415-470":
                pair = (a, b) if (la, lb) == (415, 470) else (b, a)
                lab = (415, 470)
            elif order == "470-415":
                pair = (a, b) if (la, lb) == (470, 415) else (b, a)
                lab = (470, 415)
            else:
                pair = (a, b)
                lab = (la, lb)
            pairs.append(pair)
            pair_labels.append(lab)
            i += 2
        else:
            skipped.append(a)
            i += 1
    if i < len(illuminated):
        skipped.append(int(illuminated[i]))
    return np.asarray(pairs, dtype=np.int64), np.asarray(pair_labels, dtype=np.int16), np.asarray(skipped, dtype=np.int64)


def write_trimmed_dat(
    source: Path,
    output: Path,
    pairs: np.ndarray,
    height: int,
    width: int,
    dtype: np.dtype,
    chunk_pairs: int,
) -> None:
    src = np.memmap(source, mode="r", dtype=dtype, shape=(source.stat().st_size // (height * width * dtype.itemsize), height, width))
    out = np.memmap(output, mode="w+", dtype=dtype, shape=(len(pairs), 2, height, width))
    for start in range(0, len(pairs), chunk_pairs):
        stop = min(start + chunk_pairs, len(pairs))
        idx = pairs[start:stop]
        out[start:stop, 0] = src[idx[:, 0]]
        out[start:stop, 1] = src[idx[:, 1]]
        out.flush()
        print(f"wrote pairs {stop}/{len(pairs)}", flush=True)
    del out
    del src


def main() -> int:
    parser = argparse.ArgumentParser(description="Trim labcams DAT to illuminated 415/470 pairs using DAQ TTLs.")
    parser.add_argument("dat", type=Path)
    parser.add_argument("daq_h5", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="illuminated")
    parser.add_argument("--offset", type=int, default=None, help="DAQ pco_exposure offset relative to DAT frames. Default chooses 0/1 automatically.")
    parser.add_argument("--led-threshold", type=float, default=None)
    parser.add_argument("--order", choices=("415-470", "470-415", "as-acquired"), default="415-470")
    parser.add_argument("--chunk-pairs", type=int, default=256)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    nchan, height, width, dtype = parse_labcams_dat_name(args.dat)
    if nchan != 2:
        raise ValueError(f"Expected a two-channel labcams DAT filename, got nchan={nchan}")
    physical_frames = args.dat.stat().st_size // (height * width * dtype.itemsize)
    if physical_frames * height * width * dtype.itemsize != args.dat.stat().st_size:
        raise ValueError("DAT size is not an integer number of physical frames")

    labels, meta = load_daq_labels(args.daq_h5, int(physical_frames), args.offset, args.led_threshold)
    pairs, pair_labels, skipped = make_clean_pairs(labels, args.order)
    out_dat = args.output_dir / f"{args.dat.stem}_{args.label}_cleanpairs_2_{height}_{width}_{dtype.name}.dat"
    map_npz = args.output_dir / f"{args.dat.stem}_{args.label}_cleanpairs_frame_map.npz"
    map_csv = args.output_dir / f"{args.dat.stem}_{args.label}_cleanpairs_frame_map.csv"
    summary_path = args.output_dir / f"{args.dat.stem}_{args.label}_cleanpairs_summary.json"

    print(f"source physical frames: {physical_frames:,}")
    print(f"clean pairs: {len(pairs):,}; skipped illuminated singleton/problem frames: {len(skipped):,}")
    print(f"output: {out_dat}")
    write_trimmed_dat(args.dat, out_dat, pairs, height, width, dtype, args.chunk_pairs)

    np.savez_compressed(
        map_npz,
        pair_index=np.arange(len(pairs), dtype=np.int64),
        original_frame_index_ch0=pairs[:, 0],
        original_frame_index_ch1=pairs[:, 1],
        channel_label_ch0=pair_labels[:, 0],
        channel_label_ch1=pair_labels[:, 1],
        labels_per_original_frame=labels,
        skipped_original_frame_index=skipped,
    )
    with map_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["pair_index", "original_frame_index_ch0", "channel_label_ch0", "original_frame_index_ch1", "channel_label_ch1"])
        for i, ((a, b), (la, lb)) in enumerate(zip(pairs, pair_labels)):
            writer.writerow([i, int(a), int(la), int(b), int(lb)])

    summary = {
        **meta,
        "source_dat": str(args.dat),
        "daq_h5": str(args.daq_h5),
        "output_dat": str(out_dat),
        "frame_map_npz": str(map_npz),
        "frame_map_csv": str(map_csv),
        "output_shape": [int(len(pairs)), 2, int(height), int(width)],
        "output_dtype": dtype.name,
        "channel_order": args.order,
        "clean_pairs": int(len(pairs)),
        "skipped_illuminated_frames": int(len(skipped)),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
