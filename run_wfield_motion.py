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
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    dat = mmap_dat(str(args.dat), mode="r")
    if args.limit_frames is not None:
        dat = dat[: args.limit_frames]

    nframes, nchannels, height, width = dat.shape
    dtype = np.dtype(dat.dtype).name
    corrected_path = args.output / (
        f"motioncorrect_{nchannels}_{height}_{width}_{dtype}.bin"
    )

    print(f"Input: {args.dat}", flush=True)
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
        "source_dat": str(args.dat),
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
