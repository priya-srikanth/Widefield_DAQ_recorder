"""Relabel/trim a labcams DAT to DAQ-confirmed illuminated 415/470 frame pairs.

labcams writes camera frames to the .dat in raw arrival order; the 415/470
"channels" are only an interpretation imposed by the ``_N_H_W_`` filename at read
time (frame i -> channel i % N). When acquisition is trial-gated, the LED phase
and the running frame counter can drift apart between trials (a trial with an odd
frame count flips which wavelength lands in "channel 0"), so a naive reshape mixes
415 and 470 across trials. Continuously-saved-but-LED-gated sessions additionally
contain dark inter-trial frames.

This module uses the DAQ ``pco_exposure`` / ``led415_ttl`` / ``led470_ttl``
channels as ground truth to label every saved physical frame, drop dark frames,
keep clean adjacent 415/470 pairs, and write a standard two-channel labcams DAT
(channel 0 = 415, channel 1 = 470) plus a frame map and summary. This is the
canonical step to run BEFORE motion correction and SVD on trial-gated data.

Modes:
  rescue          - continuously-saved sessions (expects many dark inter-trial
                    frames; drops them). Default.
  acquire-enable  - PCO Acquire-Enable gated sessions (expects ~no dark frames;
                    warns if many are found, which would indicate a gating fault).
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


def relabel_dat_from_daq(
    dat: Path,
    daq_h5: Path,
    output_dir: Path,
    label: str = "illuminated",
    offset: int | None = None,
    led_threshold: float | None = None,
    order: str = "415-470",
    chunk_pairs: int = 256,
    mode: str = "rescue",
) -> dict:
    """Relabel a labcams DAT to DAQ-confirmed 415/470 pairs and write a new DAT.

    Returns a summary dict; ``summary['output_dat']`` is the relabeled .dat path,
    suitable as input to motion correction. Importable for pipeline use.
    """
    dat = Path(dat)
    daq_h5 = Path(daq_h5)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nchan, height, width, dtype = parse_labcams_dat_name(dat)
    if nchan != 2:
        raise ValueError(f"Expected a two-channel labcams DAT filename, got nchan={nchan}")
    frame_bytes = height * width * dtype.itemsize
    physical_frames = dat.stat().st_size // frame_bytes
    if physical_frames * frame_bytes != dat.stat().st_size:
        raise ValueError("DAT size is not an integer number of physical frames")

    labels, meta = load_daq_labels(daq_h5, int(physical_frames), offset, led_threshold)
    dark_frac = meta["labels_dark"] / max(int(physical_frames), 1)
    if mode == "acquire-enable" and dark_frac > 0.05:
        print(f"WARNING [acquire-enable mode]: {meta['labels_dark']} dark frames "
              f"({100*dark_frac:.1f}%) found; acquire-enable gating should produce ~none. "
              f"Check that the camera only acquired during trials.", flush=True)

    pairs, pair_labels, skipped = make_clean_pairs(labels, order)
    out_dat = output_dir / f"{dat.stem}_{label}_cleanpairs_2_{height}_{width}_{dtype.name}.dat"
    map_npz = output_dir / f"{dat.stem}_{label}_cleanpairs_frame_map.npz"
    map_csv = output_dir / f"{dat.stem}_{label}_cleanpairs_frame_map.csv"
    summary_path = output_dir / f"{dat.stem}_{label}_cleanpairs_summary.json"

    print(f"[relabel:{mode}] source physical frames: {physical_frames:,}", flush=True)
    print(f"[relabel:{mode}] clean pairs: {len(pairs):,}; skipped singleton/problem frames: {len(skipped):,}; "
          f"dark dropped: {meta['labels_dark']:,}", flush=True)
    print(f"[relabel:{mode}] output: {out_dat}", flush=True)
    write_trimmed_dat(dat, out_dat, pairs, height, width, dtype, chunk_pairs)

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
        "mode": mode,
        "source_dat": str(dat),
        "daq_h5": str(daq_h5),
        "output_dat": str(out_dat),
        "frame_map_npz": str(map_npz),
        "frame_map_csv": str(map_csv),
        "output_shape": [int(len(pairs)), 2, int(height), int(width)],
        "output_dtype": dtype.name,
        "channel_order": order,
        "clean_pairs": int(len(pairs)),
        "skipped_illuminated_frames": int(len(skipped)),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Relabel labcams DAT to DAQ-confirmed 415/470 pairs.")
    parser.add_argument("dat", type=Path)
    parser.add_argument("daq_h5", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="illuminated")
    parser.add_argument("--mode", choices=("rescue", "acquire-enable"), default="rescue",
                        help="rescue: drop dark inter-trial frames (continuous saving). "
                             "acquire-enable: warn if dark frames found (gated acquisition).")
    parser.add_argument("--offset", type=int, default=None, help="DAQ pco_exposure offset relative to DAT frames. Default chooses 0/1 automatically.")
    parser.add_argument("--led-threshold", type=float, default=None)
    parser.add_argument("--order", choices=("415-470", "470-415", "as-acquired"), default="415-470")
    parser.add_argument("--chunk-pairs", type=int, default=256)
    args = parser.parse_args()
    relabel_dat_from_daq(
        args.dat, args.daq_h5, args.output_dir,
        label=args.label, offset=args.offset, led_threshold=args.led_threshold,
        order=args.order, chunk_pairs=args.chunk_pairs, mode=args.mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
