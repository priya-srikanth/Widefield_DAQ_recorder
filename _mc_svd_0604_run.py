"""Driver: motion correction + SVD/hemo for the three 2026-06-04 sessions.

Mirrors the 6/3 recipe (regime B, ROI recordings):
  1. run_wfield_motion  --daq-h5 <h5> --relabel-mode rescue --mode 2d
       -> TTL relabel (cleanpairs) then 2D motion correction
       -> motioncorrect_2_460_480_uint16.bin + shifts + summary
  2. run_wfield_local   <motioncorrect.bin> -k 100 --functional-channel 1
       --fs 31.23 --freq-highpass 0.1 --freq-lowpass 14.0
       -> U.npy / SVT.npy / SVTcorr.npy + summary

PS92 uses the force-split session's CONCATENATED camera .dat + DAQ .h5
(see wfield_local/concat_split_session.py).

Idempotent: skips motion if the .bin exists, skips SVD if SVTcorr.npy exists.
Usage:  python _mc_svd_0604_run.py [PS92_0604 PS94_0604 PS95_0604]
"""
import os
import sys
import subprocess
from pathlib import Path

PY = r"C:\ProgramData\anaconda3\envs\wfield\python.exe"
REPO = r"C:\Github\Widefield_DAQ_recorder"
D = r"E:\labcams_data\20260604"
Q = r"E:\DAQ_recorder_output\20250604"
DIMS = "2_460_480"

SESSIONS = {
    "PS92_0604": dict(
        raw=fr"{D}\PS92_20260604_132934\raw_widefield_data_concat\pco_edge_run000_00000000_{DIMS}_uint16.dat",
        daq=fr"{Q}\PS92_20260604_concat.h5",
        motion=fr"{D}\PS92_20260604_132934\motion_corrected",
    ),
    "PS94_0604": dict(
        raw=fr"{D}\PS94_20260604_151516\raw_widefield_data\pco_edge_run000_00000000_{DIMS}_uint16.dat",
        daq=fr"{Q}\PS94_20260604_152103.h5",
        motion=fr"{D}\PS94_20260604_151516\motion_corrected",
    ),
    "PS95_0604": dict(
        raw=fr"{D}\PS95_20260604_165712\raw_widefield_data\pco_edge_run000_00000000_{DIMS}_uint16.dat",
        daq=fr"{Q}\PS95_20260604_170729.h5",
        motion=fr"{D}\PS95_20260604_165712\motion_corrected",
    ),
}


def run(cmd):
    env = dict(os.environ, PYTHONPATH=REPO)
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([PY, "-m", *cmd], cwd=REPO, env=env)
    if r.returncode != 0:
        raise SystemExit(f"step failed (exit {r.returncode}): {cmd}")


def do_session(key):
    s = SESSIONS[key]
    motion = Path(s["motion"])
    bin_path = motion / f"motioncorrect_{DIMS}_uint16.bin"
    results = motion / "wfield_local_results"
    svtcorr = results / "SVTcorr.npy"
    print(f"\n================ {key} ================", flush=True)
    if not Path(s["raw"]).exists():
        raise SystemExit(f"raw .dat missing: {s['raw']}")
    if not Path(s["daq"]).exists():
        raise SystemExit(f"daq .h5 missing: {s['daq']}")

    # 1. motion correction (relabel rescue, 2d)
    if bin_path.exists():
        print(f"[skip] motion bin exists: {bin_path}", flush=True)
    else:
        run(["wfield_local.run_wfield_motion", s["raw"], "--output", s["motion"],
             "--daq-h5", s["daq"], "--relabel-mode", "rescue", "--mode", "2d"])

    # 2. SVD + hemodynamic correction
    if svtcorr.exists():
        print(f"[skip] SVTcorr exists: {svtcorr}", flush=True)
    else:
        run(["wfield_local.run_wfield_local", str(bin_path), "--output", str(results),
             "-k", "100", "--functional-channel", "1", "--fs", "31.23",
             "--freq-highpass", "0.1", "--freq-lowpass", "14.0"])
    print(f"================ {key} DONE ================", flush=True)


if __name__ == "__main__":
    keys = sys.argv[1:] or list(SESSIONS)
    for k in keys:
        do_session(k)
    print("\nALL DONE:", keys, flush=True)
