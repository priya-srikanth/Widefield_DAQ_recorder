"""Estimate widefield storage and ROI / trial-gating savings.

Scans a labcams data root for `.dat` files (size, frame shape, and duration at a
nominal frame rate) and, optionally, computes the trial duty cycle from a DAQ
behavior session to estimate combined ROI + trial-gating savings versus
full-FOV continuous acquisition.

  saved_size = baseline * (roi_pixels / full_pixels) * (in-trial_time / total_time)

Usage:
  python tools/estimate_storage.py --labcams-dir E:\\labcams_data
  python tools/estimate_storage.py --full 540x640 --roi 487x480 --daq path\\to\\behavior.h5

Run in an environment with h5py + numpy (e.g. the `wfield` conda env).
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import h5py
import numpy as np

DAT_RE = re.compile(r"_(\d+)_(\d+)_(\d+)_(uint16|int16|uint8|float32|float64)\.dat$")
ITEMSIZE = {"uint16": 2, "int16": 2, "uint8": 1, "float32": 4, "float64": 8}


def _hxw(s: str) -> tuple[int, int]:
    h, w = re.split(r"[xX,]", s)
    return int(h), int(w)


def scan_dats(root: Path, fps: float, min_gb: float):
    rows = []
    for d in glob.glob(str(root / "**" / "*.dat"), recursive=True):
        m = DAT_RE.search(os.path.basename(d))
        if not m:
            continue
        _nch, H, W, dt = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
        sz = os.path.getsize(d)
        frames = sz / (H * W * ITEMSIZE[dt])
        rows.append((sz, H, W, frames, frames / fps, os.path.relpath(d, root)))
    rows.sort(reverse=True)
    print(f"=== labcams .dat files under {root} (>= {min_gb} GB) ===")
    for sz, H, W, frames, dur, rel in rows:
        if sz < min_gb * 1e9:
            continue
        print(f"  {sz/1e9:7.1f} GB  {H}x{W} ({H*W} px)  {frames:.0f} frames  ~{dur/60:.1f} min  {rel}")
    return rows


def duty_cycle(h5_path: Path, ttl_thresh: float):
    with h5py.File(h5_path, "r") as f:
        fs = float(f.attrs["sample_rate_hz"])
        di = [s.decode() for s in f["digital/channel_names"][:]]
        an = [s.decode() for s in f["analog/channel_names"][:]]
        bits = np.unpackbits(f["digital/packed_samples"][:, 0][:, None], axis=1, bitorder="little")
        raw = f["analog/samples_int16"][:].astype(np.float32)
        volts = raw * f["analog/int16_scale_volts_per_count"][:] + f["analog/int16_offset_volts"][:]
    N = bits.shape[0]
    T = N / fs

    def rises(sig, thr):
        b = (sig > thr).astype(np.int8)
        return np.flatnonzero(np.diff(b) == 1) + 1

    def find(names, *pats):
        for i, n in enumerate(names):
            if any(re.search(p, n, re.IGNORECASE) for p in pats):
                return i
        return None

    ts_i = find(di, r"trial_?start")
    te_i_an = find(an, r"trial_?end", r"trial_?stop")
    te_i_di = find(di, r"trial_?end", r"trial_?stop")
    if ts_i is None or (te_i_an is None and te_i_di is None):
        print(f"  {h5_path.name}: missing trial_start/trial_end channels; cannot compute duty cycle")
        return None
    st = rises(bits[:, ts_i].astype(float), 0.5) / fs
    en = (rises(volts[:, te_i_an], ttl_thresh) if te_i_an is not None
          else rises(bits[:, te_i_di].astype(float), 0.5)) / fs
    wins = []
    for s0 in st:
        later = en[en > s0]
        if len(later):
            wins.append((s0, later[0]))
    if not wins:
        print(f"  {h5_path.name}: no paired trial windows")
        return None
    in_trial = sum(b - a for a, b in wins)
    duty = in_trial / T
    print(f"  {h5_path.name}: {T/60:.1f} min, {len(wins)} trials, mean {np.mean([b-a for a,b in wins]):.2f}s, "
          f"in-trial {in_trial:.0f}s -> duty {100*duty:.1f}%")
    return duty


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labcams-dir", type=Path, default=None, help="root to scan for .dat files")
    ap.add_argument("--fps", type=float, default=62.5, help="total camera fps (both LEDs) for duration estimate")
    ap.add_argument("--min-gb", type=float, default=1.0, help="only list .dat files at least this large")
    ap.add_argument("--full", type=str, default="540x640", help="full-FOV frame HxW for the baseline")
    ap.add_argument("--roi", type=str, default=None, help="ROI frame HxW to compare (e.g. 487x480)")
    ap.add_argument("--daq", type=Path, nargs="*", default=None, help="DAQ behavior HDF5(s) for trial duty cycle")
    ap.add_argument("--ttl-thresh", type=float, default=1.5)
    args = ap.parse_args()

    if args.labcams_dir:
        scan_dats(args.labcams_dir, args.fps, args.min_gb)

    fh, fw = _hxw(args.full)
    full_px = fh * fw
    baseline_gb_hr = full_px * 2 * args.fps * 3600 / 1e9
    print(f"\n=== Rates (uint16, {args.fps:g} fps) ===")
    print(f"  full FOV {fh}x{fw} continuous = {baseline_gb_hr:.0f} GB/hr (baseline)")

    roi_factor = 1.0
    if args.roi:
        rh, rw = _hxw(args.roi)
        roi_factor = (rh * rw) / full_px
        print(f"  ROI {rh}x{rw} continuous   = {baseline_gb_hr*roi_factor:.0f} GB/hr "
              f"({100*roi_factor:.1f}% of full, {100*(1-roi_factor):.0f}% saved by ROI)")

    duty = None
    if args.daq:
        print("\n=== Trial duty cycle ===")
        ds = [duty_cycle(Path(p), args.ttl_thresh) for p in args.daq]
        ds = [d for d in ds if d is not None]
        if ds:
            duty = float(np.mean(ds))

    print("\n=== Combined estimate (per hour of session) ===")
    print(f"  full FOV + continuous : {baseline_gb_hr:.0f} GB/hr")
    if args.roi:
        print(f"  ROI only              : {baseline_gb_hr*roi_factor:.0f} GB/hr  ({100*(1-roi_factor):.0f}% saved)")
    if duty is not None:
        print(f"  trial-gating only     : {baseline_gb_hr*duty:.0f} GB/hr  ({100*(1-duty):.0f}% saved, duty {100*duty:.1f}%)")
        combo = roi_factor * duty
        print(f"  ROI + trial-gating    : {baseline_gb_hr*combo:.0f} GB/hr  ({100*(1-combo):.0f}% saved)")
    elif args.roi:
        print("  (add --daq <behavior.h5> to factor in trial-gating duty cycle)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
