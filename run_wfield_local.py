"""Run local wfield SVD + hemodynamic correction on labcams DAT files.

This is a local fallback for the NeuroCAAS wfield-preprocess step.  It uses the
installed ``wfield`` package but exposes memory-sensitive decomposition
parameters that the stock CLI keeps hard-coded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wfield.decomposition import approximate_svd
from wfield.hemocorrection import hemodynamic_correction
from wfield.io import mmap_dat


def _mean_by_chunks(dat, chunk_size: int) -> np.ndarray:
    total = np.zeros(dat.shape[1:], dtype=np.float64)
    n = 0
    for start in range(0, dat.shape[0], chunk_size):
        stop = min(start + chunk_size, dat.shape[0])
        block = np.asarray(dat[start:stop], dtype=np.float32)
        total += block.sum(axis=0, dtype=np.float64)
        n += stop - start
        print(f"baseline mean: {stop}/{dat.shape[0]} frames", flush=True)
    return total / max(n, 1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Local wfield decomposition and hemodynamic correction."
    )
    parser.add_argument("dat", type=Path, help="DAT file ending in _N_H_W_dtype.dat")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("-k", type=int, default=100, help="Number of SVD components")
    parser.add_argument("--fs", type=float, default=31.23, help="Per-channel frame rate")
    parser.add_argument(
        "--functional-channel",
        type=int,
        default=1,
        help="Functional channel index. For this rig/labcams data, 1 is 470 nm.",
    )
    parser.add_argument("--nbinned-frames", type=int, default=1000)
    parser.add_argument("--nframes-per-bin", type=int, default=15)
    parser.add_argument("--nframes-per-chunk", type=int, default=500)
    parser.add_argument("--baseline-chunk-size", type=int, default=256)
    parser.add_argument("--skip-correction", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Loading DAT as memmap: {args.dat}", flush=True)
    dat = mmap_dat(str(args.dat), mode="r")
    print(f"Data shape: {dat.shape}, dtype={dat.dtype}", flush=True)

    frames_average_path = args.output / "frames_average.npy"
    if frames_average_path.exists():
        print(f"Loading existing baseline: {frames_average_path}", flush=True)
        frames_average = np.load(frames_average_path)
    else:
        print("Computing baseline mean by chunks...", flush=True)
        frames_average = _mean_by_chunks(dat, args.baseline_chunk_size)
        np.save(frames_average_path, frames_average)

    print("Running approximate_svd...", flush=True)
    U, SVT = approximate_svd(
        dat,
        frames_average,
        k=args.k,
        nframes_per_bin=args.nframes_per_bin,
        nbinned_frames=args.nbinned_frames,
        nframes_per_chunk=args.nframes_per_chunk,
    )
    np.save(args.output / "U.npy", U)
    np.save(args.output / "SVT.npy", SVT)
    print(f"Saved U {U.shape} and SVT {SVT.shape}", flush=True)

    summary = {
        "dat": str(args.dat),
        "output": str(args.output),
        "data_shape": list(dat.shape),
        "k": args.k,
        "fs": args.fs,
        "functional_channel": args.functional_channel,
        "nbinned_frames": args.nbinned_frames,
        "nframes_per_bin": args.nframes_per_bin,
        "nframes_per_chunk": args.nframes_per_chunk,
        "baseline_chunk_size": args.baseline_chunk_size,
    }

    if not args.skip_correction and dat.shape[1] == 2:
        print("Running hemodynamic correction...", flush=True)
        svt_470 = SVT[:, args.functional_channel :: 2]
        svt_415 = SVT[:, (args.functional_channel + 1) % 2 :: 2]
        SVTcorr, rcoeffs, T = hemodynamic_correction(U, svt_470, svt_415, fs=args.fs)
        np.save(args.output / "SVTcorr.npy", SVTcorr)
        np.save(args.output / "rcoeffs.npy", rcoeffs)
        np.save(args.output / "T.npy", T)
        summary["SVTcorr_shape"] = list(SVTcorr.shape)
        print(f"Saved SVTcorr {SVTcorr.shape}", flush=True)

    (args.output / "local_wfield_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
