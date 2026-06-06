"""Driver: motion correction + SVD/hemo for the four 2026-06-05 sessions.

Per-session relabel mode reflects the recording type:
  PS92 = trial-triggered  -> --relabel-mode rescue        (drop dark inter-trial frames)
  PS93/PS94/PS95 = continuous -> --relabel-mode acquire-enable (expect ~no dark frames)
All relabel via the DAQ (led415/led470/pco_exposure TTLs) so channel identity
(0=415, 1=470) is correct, then 2D motion correction, then SVD k=100 + hemo
(hp 0.1 / lp 14, functional channel 1). PS93 is a new animal.

Idempotent: skips motion if the .bin exists, SVD if SVTcorr.npy exists.
Usage:  python _mc_svd_0605_run.py [PS92_0605 PS93_0605 PS94_0605 PS95_0605]
"""
import os
import sys
import subprocess
from pathlib import Path

PY = r"C:\ProgramData\anaconda3\envs\wfield\python.exe"
REPO = r"C:\Github\Widefield_DAQ_recorder"
D = r"E:\labcams_data\20260605"
Q = r"E:\DAQ_recorder_output\20260605"
DIMS = "2_460_480"

SESSIONS = {
    "PS92_0605": dict(sess="PS92_20260605_125023", daq=fr"{Q}\PS92_20260605_125301.h5",
                      relabel="rescue"),         # trial-triggered
    "PS93_0605": dict(sess="PS93_20260605_174659", daq=fr"{Q}\PS93_20260605_175452.h5",
                      relabel="acquire-enable"),  # continuous (new animal)
    "PS94_0605": dict(sess="PS94_20260605_142009", daq=fr"{Q}\PS94_20260605_142249.h5",
                      relabel="acquire-enable"),  # continuous
    "PS95_0605": dict(sess="PS95_20260605_163102", daq=fr"{Q}\PS95_20260605_163405.h5",
                      relabel="acquire-enable"),  # continuous
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
    print(f"\n================ {key} (relabel={s['relabel']}) ================", flush=True)
    if not Path(raw).exists():
        raise SystemExit(f"raw .dat missing: {raw}")
    if not Path(s["daq"]).exists():
        raise SystemExit(f"daq .h5 missing: {s['daq']}")

    if bin_path.exists():
        print(f"[skip] motion bin exists: {bin_path}", flush=True)
    else:
        run(["wfield_local.run_wfield_motion", raw, "--output", str(motion),
             "--daq-h5", s["daq"], "--relabel-mode", s["relabel"], "--mode", "2d"])

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
