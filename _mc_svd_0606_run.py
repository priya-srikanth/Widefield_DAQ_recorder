"""Driver: motion correction + SVD/hemo for the four 2026-06-06 sessions.

Relabel mode = rescue for all (the relabel only uses mode for a warning; frame
selection is identical, so rescue is safe regardless of trial-gated vs continuous;
the printed dark-frame count reveals the recording type). Then 2D motion correction
and SVD k=100 + hemo (hp 0.1 / lp 14, functional channel 1).

DAQ files are loose in E:\\DAQ_recorder_output\\ this day (no date subfolder).
Idempotent: skips motion if the .bin exists, SVD if SVTcorr.npy exists.
Usage:  python _mc_svd_0606_run.py [PS92_0606 PS93_0606 PS94_0606 PS95_0606]
"""
import os
import sys
import subprocess
from pathlib import Path

PY = r"C:\ProgramData\anaconda3\envs\wfield\python.exe"
REPO = r"C:\Github\Widefield_DAQ_recorder"
D = r"E:\labcams_data\20260606"
Q = r"E:\DAQ_recorder_output"
DIMS = "2_460_480"

SESSIONS = {
    "PS92_0606": dict(sess="PS92_20260606_122451", daq=fr"{Q}\PS92_20260606_122508.h5"),
    "PS93_0606": dict(sess="PS93_20260606_180117", daq=fr"{Q}\PS93_20260606_180219.h5"),
    "PS94_0606": dict(sess="PS94_20260606_140854", daq=fr"{Q}\PS94_20260606_140912.h5"),
    "PS95_0606": dict(sess="PS95_20260606_160806", daq=fr"{Q}\PS95_20260606_160825.h5"),
}


def run(cmd):
    env = dict(os.environ, PYTHONPATH=REPO)
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([PY, "-m", *cmd], cwd=REPO, env=env)
    if r.returncode != 0:
        raise SystemExit(f"step failed (exit {r.returncode}): {cmd}")


def do_session(key):
    s = SESSIONS[key]
    raw = fr"{D}\{s['sess']}\raw_widefield_data\pco_edge_run000_00000000_{DIMS}_uint16.dat"
    motion = Path(fr"{D}\{s['sess']}\motion_corrected")
    bin_path = motion / f"motioncorrect_{DIMS}_uint16.bin"
    results = motion / "wfield_local_results"
    svtcorr = results / "SVTcorr.npy"
    print(f"\n================ {key} ================", flush=True)
    if not Path(raw).exists():
        raise SystemExit(f"raw .dat missing: {raw}")
    if not Path(s["daq"]).exists():
        raise SystemExit(f"daq .h5 missing: {s['daq']}")

    if bin_path.exists():
        print(f"[skip] motion bin exists: {bin_path}", flush=True)
    else:
        run(["wfield_local.run_wfield_motion", raw, "--output", str(motion),
             "--daq-h5", s["daq"], "--relabel-mode", "rescue", "--mode", "2d"])

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
