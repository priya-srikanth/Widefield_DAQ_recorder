"""Motion correction (SIGN-FIXED) + SVD/hemo for the four 2026-06-07 sessions.

Uses the fixed motion correction (run_wfield_motion -> motion_correct_fixed).
relabel rescue (safe universal; dark count reveals recording type). DAQ loose in
E:\\DAQ_recorder_output\\. Idempotent.
Usage:  python _mc_svd_0607_run.py [PS92_0607 ...]
"""
import os, sys, subprocess
from pathlib import Path
PY = r"C:\ProgramData\anaconda3\envs\wfield\python.exe"
REPO = r"C:\Github\Widefield_DAQ_recorder"
D = r"E:\labcams_data\20260607"; Q = r"E:\DAQ_recorder_output"; DIMS = "2_460_480"
SESSIONS = {
    "PS92_0607": dict(sess="PS92_20260607_121538", daq=fr"{Q}\PS92_20260607_121551.h5"),
    "PS93_0607": dict(sess="PS93_20260607_174844", daq=fr"{Q}\PS93_20260607_174854.h5"),
    "PS94_0607": dict(sess="PS94_20260607_140731", daq=fr"{Q}\PS94_20260607_140813.h5"),
    "PS95_0607": dict(sess="PS95_20260607_155000", daq=fr"{Q}\PS95_20260607_155400.h5"),
}
def run(cmd):
    env = dict(os.environ, PYTHONPATH=REPO); print("\n$ " + " ".join(map(str, cmd)), flush=True)
    if subprocess.run([PY, "-m", *cmd], cwd=REPO, env=env).returncode: raise SystemExit(f"fail {cmd}")
def do(key):
    s = SESSIONS[key]; raw = fr"{D}\{s['sess']}\raw_widefield_data\pco_edge_run000_00000000_{DIMS}_uint16.dat"
    mc = Path(fr"{D}\{s['sess']}\motion_corrected"); binp = mc / f"motioncorrect_{DIMS}_uint16.bin"
    results = mc / "wfield_local_results"
    print(f"\n================ {key} ================", flush=True)
    if binp.exists(): print(f"[skip] bin exists", flush=True)
    else: run(["wfield_local.run_wfield_motion", raw, "--output", str(mc), "--daq-h5", s["daq"], "--relabel-mode", "rescue", "--mode", "2d"])
    if (results / "SVTcorr.npy").exists(): print("[skip] SVTcorr exists", flush=True)
    else: run(["wfield_local.run_wfield_local", str(binp), "--output", str(results), "-k", "100", "--functional-channel", "1", "--fs", "31.23", "--freq-highpass", "0.1", "--freq-lowpass", "14.0"])
    print(f"================ {key} DONE ================", flush=True)
if __name__ == "__main__":
    for k in (sys.argv[1:] or list(SESSIONS)): do(k)
    print("\nALL DONE", flush=True)
