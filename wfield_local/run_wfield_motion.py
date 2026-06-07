"""Run wfield motion correction on a labcams DAT file.

This wrapper avoids the installed wfield CLI's confusing output-path behavior
and writes motion-corrected data plus shift summaries into an explicit folder.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from wfield.io import mmap_dat
# Sign-corrected motion correction (wfield 0.4.2 registration_upsample doubles drift
# due to a phase-correlation sign error). See wfield_local/motion_correct_fixed.py.
from wfield_local.motion_correct_fixed import motion_correct


def _run_relabel_subprocess(dat: Path, daq_h5: Path, output: Path, mode: str, led_threshold) -> Path:
    """Run the TTL relabel in a SEPARATE process and return the relabeled .dat.

    h5py and wfield cannot be imported in the same process on this rig (their
    HDF5 DLLs conflict), so the relabel (h5py) must run in its own interpreter,
    separate from motion correction (wfield).
    """
    import json as _json
    import subprocess
    import sys

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [sys.executable, "-m", "wfield_local.trim_illuminated_labcams",
           str(dat), str(daq_h5), "--output-dir", str(output),
           "--label", "daq_led", "--mode", mode]
    if led_threshold is not None:
        cmd += ["--led-threshold", str(led_threshold)]
    print("[pipeline] TTL relabel (separate process): " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=repo_root)
    summary_path = output / f"{Path(dat).stem}_daq_led_cleanpairs_summary.json"
    summary = _json.loads(summary_path.read_text(encoding="utf-8"))
    return Path(summary["output_dat"])


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
    relabeled = False
    if args.daq_h5 is not None:
        print(f"[pipeline] TTL relabel before motion correction (mode={args.relabel_mode})", flush=True)
        dat_path = _run_relabel_subprocess(
            args.dat, args.daq_h5, args.output, args.relabel_mode, args.relabel_led_threshold)
        relabeled = True
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
        "ttl_relabeled": relabeled,
        "relabel_mode": args.relabel_mode if relabeled else None,
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
