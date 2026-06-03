"""Dump a DAQ recorder HDF5 structure and/or check a labcams .dat/.camlog.

Usage:
  python tools/inspect_recording.py --daq path\\to\\daq.h5
  python tools/inspect_recording.py --camlog run.camlog --dat run_..._2_540_640_uint16.dat

Run in an environment with h5py + numpy (e.g. the `wfield` conda env).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
from pathlib import Path

import h5py
import numpy as np

DAT_RE = re.compile(r"_(\d+)_(\d+)_(\d+)_(uint16|int16|uint8|float32|float64)\.dat$")
ITEMSIZE = {"uint16": 2, "int16": 2, "uint8": 1, "float32": 4, "float64": 8}


def dump_h5(p: Path):
    print(f"=== HDF5 {p.name} ===")
    with h5py.File(p, "r") as f:
        for k, v in f.attrs.items():
            sv = str(v)
            print(f"  attr {k} = {sv[:200]}{'...' if len(sv) > 200 else ''}")

        def show(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  DS  {name}  shape={obj.shape} dtype={obj.dtype}")
            else:
                print(f"  GRP {name}")
        f.visititems(show)


def check_camlog(camlog: Path, dat: Path | None, gap_factor: float):
    fids, ts = [], []
    for line in camlog.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split(",", 1)
        try:
            fids.append(int(parts[0]))
            ts.append(_dt.datetime.fromisoformat(parts[1]))
        except Exception:
            pass
    print(f"=== camlog {camlog.name} ===")
    print(f"  frames={len(fids)}  id range {min(fids)}..{max(fids)}  span {(ts[-1]-ts[0]).total_seconds():.2f}s")
    if len(ts) > 1:
        dt = np.array([(ts[i] - ts[i - 1]).total_seconds() * 1000 for i in range(1, len(ts))])
        med = float(np.median(dt))
        gaps = np.flatnonzero(dt > gap_factor * med)
        print(f"  inter-frame ms: median={med:.3f} min={dt.min():.3f} max={dt.max():.3f}  gaps>{gap_factor}x={len(gaps)}")
    if dat and dat.exists():
        m = DAT_RE.search(dat.name)
        if m:
            H, W, ddt = int(m.group(2)), int(m.group(3)), m.group(4)
            sz = os.path.getsize(dat)
            fb = H * W * ITEMSIZE[ddt]
            print(f"=== dat {dat.name} === {H}x{W} {ddt} size={sz} frames={sz/fb:.3f} "
                  f"matches camlog={sz // fb == len(fids)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--daq", type=Path, default=None)
    ap.add_argument("--camlog", type=Path, default=None)
    ap.add_argument("--dat", type=Path, default=None)
    ap.add_argument("--gap-factor", type=float, default=1.8)
    args = ap.parse_args()
    if not any([args.daq, args.camlog, args.dat]):
        ap.error("provide at least one of --daq / --camlog / --dat")
    if args.daq:
        dump_h5(args.daq)
    if args.camlog:
        check_camlog(args.camlog, args.dat, args.gap_factor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
