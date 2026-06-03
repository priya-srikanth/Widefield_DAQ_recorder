"""Run wfield motion correction on a labcams DAT file.

This wrapper avoids the installed wfield CLI's confusing output-path behavior
and writes motion-corrected data plus shift summaries into an explicit folder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wfield.io import mmap_dat
from wfield.registration import motion_correct


def _load_relabel_fn():
    # Imported lazily (only when --daq-h5 is given) so the common no-relabel path
    # does not import h5py.
    try:
        from .trim_illuminated_labcams import relabel_dat_from_daq
    except ImportError:  # direct script execution
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from trim_illuminated_labcams import relabel_dat_from_daq
    return relabel_dat_from_daq


def main() -> int:
    parser = argparse.ArgumentParser(description="Motion-correct a wfield/labcams DAT.")
    parser.add_argument("dat", type=Path, help="DAT file ending in _N_H_W_dtype.dat")
    parser.add_argument("--output", type=Path, required=True, help="Output folder")
    parser.add_argument(
        "--mode",
        choices=("2d", "ecc"),
        default="2d",
        help="2d is translation-only and faster; ecc estimates rigid-body shifts.",
    )
    parser.add_argument("--chunksize", type=int, default=256)
    parser.add_argument("--nreference", type=int, default=60)
    parser.add_argument(
        "--limit-frames",
        type=int,
        default=None,
        help="Optional short run for testing. Uses only the first N frames.",
    )
    # TTL-based LED relabeling BEFORE motion correction (and thus before SVD).
    # For trial-gated recordings the saved .dat channel split (frame parity) can
    # drift from the true 415/470 LED across trials; relabeling from the DAQ
    # makes channel 0 = 415, channel 1 = 470 deterministically.
    parser.add_argument(
        "--daq-h5",
        type=Path,
        default=None,
        help="DAQ recorder .h5. If given, relabel the DAT to DAQ-confirmed 415/470 "
        "pairs first, then motion-correct the relabeled DAT.",
    )
    parser.add_argument(
        "--relabel-mode",
        choices=("rescue", "acquire-enable"),
        default="acquire-enable",
        help="Relabel mode when --daq-h5 is given (default acquire-enable).",
    )
    parser.add_argument("--relabel-led-threshold", type=float, default=None)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    dat_path = args.dat
    relabel_summary = None
    if args.daq_h5 is not None:
        print(f"[pipeline] TTL relabel before motion correction (mode={args.relabel_mode})", flush=True)
        relabel_dat_from_daq = _load_relabel_fn()
        relabel_summary = relabel_dat_from_daq(
            args.dat, args.daq_h5, args.output,
            label="daq_led", mode=args.relabel_mode,
            led_threshold=args.relabel_led_threshold,
        )
        dat_path = Path(relabel_summary["output_dat"])
        print(f"[pipeline] relabeled DAT -> {dat_path}", flush=True)

    dat = mmap_dat(str(dat_path), mode="r")
    if args.limit_frames is not None:
        dat = dat[: args.limit_frames]

    nframes, nchannels, height, width = dat.shape
    dtype = np.dtype(dat.dtype).name
    corrected_path = args.output / (
        f"motioncorrect_{nchannels}_{height}_{width}_{dtype}.bin"
    )

    print(f"Input: {dat_path}", flush=True)
    print(f"Input shape: {dat.shape}, dtype={dat.dtype}", flush=True)
    print(f"Output: {corrected_path}", flush=True)
    corrected = np.memmap(
        corrected_path,
        mode="w+",
        dtype=dat.dtype,
        shape=dat.shape,
    )

    (yshifts, xshifts), rshifts = motion_correct(
        dat,
        out=corrected,
        chunksize=args.chunksize,
        nreference=args.nreference,
        mode=args.mode,
        apply_shifts=True,
    )
    corrected.flush()
    del corrected

    shifts = np.rec.array(
        [yshifts, xshifts],
        dtype=[("y", "float32"), ("x", "float32")],
    )
    np.save(args.output / "motion_correction_shifts.npy", shifts)
    np.save(args.output / "motion_correction_rotation.npy", rshifts)

    summary = {
        "source_dat": str(dat_path),
        "original_dat": str(args.dat),
        "daq_h5": str(args.daq_h5) if args.daq_h5 is not None else None,
        "ttl_relabeled": relabel_summary is not None,
        "relabel_mode": args.relabel_mode if relabel_summary is not None else None,
        "motion_corrected_file": str(corrected_path),
        "shape": [int(nframes), int(nchannels), int(height), int(width)],
        "dtype": dtype,
        "mode": args.mode,
        "chunksize": args.chunksize,
        "nreference": args.nreference,
        "limit_frames": args.limit_frames,
        "y_shift_min_max": [float(np.nanmin(yshifts)), float(np.nanmax(yshifts))],
        "x_shift_min_max": [float(np.nanmin(xshifts)), float(np.nanmax(xshifts))],
        "rotation_min_max": [float(np.nanmin(rshifts)), float(np.nanmax(rshifts))],
    }
    (args.output / "motion_correction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("Motion correction complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
