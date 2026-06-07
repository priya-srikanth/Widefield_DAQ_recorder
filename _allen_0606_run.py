"""Driver: Allen (CCF) landmark transform for the four 2026-06-06 sessions.

Runs after _mc_svd_0606_run.py. 8-pt-affine landmarks v1, 540x640 grid.
Idempotent: skips a session whose allen output already has U_atlas.npy.
"""
import os
import sys
import subprocess
from pathlib import Path

PY = r"C:\ProgramData\anaconda3\envs\wfield\python.exe"
REPO = r"C:\Github\Widefield_DAQ_recorder"
D = r"E:\labcams_data\20260606"
TAG = "affine8v1"

SESSIONS = {
    "PS92_0606": "PS92_20260606_122451",
    "PS93_0606": "PS93_20260606_180117",
    "PS94_0606": "PS94_20260606_140854",
    "PS95_0606": "PS95_20260606_160806",
}


def run(cmd):
    env = dict(os.environ, PYTHONPATH=REPO)
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([PY, "-m", *cmd], cwd=REPO, env=env)
    if r.returncode != 0:
        raise SystemExit(f"step failed (exit {r.returncode}): {cmd}")


def do_session(key):
    sess = SESSIONS[key]
    results = Path(fr"{D}\{sess}\motion_corrected\wfield_local_results")
    landmarks = fr"{D}\{sess}\raw_widefield_data\dorsal_cortex_landmarks_v1.json"
    allen = results / f"allen_aligned_{TAG}"
    print(f"\n================ {key} (allen transform) ================", flush=True)
    if not (results / "U.npy").exists():
        raise SystemExit(f"SVD U.npy missing: {results}")
    if not Path(landmarks).exists():
        raise SystemExit(f"landmark JSON missing: {landmarks}")
    if (allen / "U_atlas.npy").exists():
        print(f"[skip] allen output exists: {allen}", flush=True)
        return
    run(["wfield_local.apply_allen_transform", str(results),
         "--landmarks", landmarks, "--output", str(allen)])
    print(f"================ {key} DONE ================", flush=True)


if __name__ == "__main__":
    keys = sys.argv[1:] or list(SESSIONS)
    for k in keys:
        do_session(k)
    print("\nALL DONE:", keys, flush=True)
