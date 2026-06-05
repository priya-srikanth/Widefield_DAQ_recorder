"""Driver: Allen landmark transform for the three 2026-06-04 sessions.

Runs AFTER _mc_svd_0604_run.py (needs wfield_local_results/U.npy etc.). Mirrors
the affine8v1 batch step 1:
  apply_allen_transform <wfield_local_results> --landmarks <8pt JSON>
      --output <wfield_local_results/allen_aligned_affine8v1>
-> U_atlas.npy, frames_average_atlas.npy, allen_area_atlas_native_grid.npy,
   allen_brain_mask_native_grid.npy (+ outlines) on the 540x640 Allen grid.

Landmarks are the 8-pt-affine dorsal_cortex_landmarks_v1.json placed in the GUI.
Idempotent: skips a session whose allen output already has U_atlas.npy.
Usage:  python _allen_0604_run.py [PS92_0604 PS94_0604 PS95_0604]
"""
import os
import sys
import subprocess
from pathlib import Path

PY = r"C:\ProgramData\anaconda3\envs\wfield\python.exe"
REPO = r"C:\Github\Widefield_DAQ_recorder"
D = r"E:\labcams_data\20260604"
TAG = "affine8v1"

SESSIONS = {
    "PS92_0604": dict(
        results=fr"{D}\PS92_20260604_132934\motion_corrected\wfield_local_results",
        landmarks=fr"{D}\PS92_20260604_132934\raw_widefield_data_concat\dorsal_cortex_landmarks_v1.json",
    ),
    "PS94_0604": dict(
        results=fr"{D}\PS94_20260604_151516\motion_corrected\wfield_local_results",
        landmarks=fr"{D}\PS94_20260604_151516\raw_widefield_data\dorsal_cortex_landmarks_v1.json",
    ),
    "PS95_0604": dict(
        results=fr"{D}\PS95_20260604_165712\motion_corrected\wfield_local_results",
        landmarks=fr"{D}\PS95_20260604_165712\raw_widefield_data\dorsal_cortex_landmarks_v1.json",
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
    results = Path(s["results"])
    allen = results / f"allen_aligned_{TAG}"
    print(f"\n================ {key} (allen transform) ================", flush=True)
    if not (results / "U.npy").exists():
        raise SystemExit(f"SVD U.npy missing (motion+SVD not done?): {results}")
    if not Path(s["landmarks"]).exists():
        raise SystemExit(f"landmark JSON missing: {s['landmarks']}")
    if (allen / "U_atlas.npy").exists():
        print(f"[skip] allen output exists: {allen}", flush=True)
        return
    run(["wfield_local.apply_allen_transform", str(results),
         "--landmarks", s["landmarks"], "--output", str(allen)])
    print(f"================ {key} DONE ================", flush=True)


if __name__ == "__main__":
    keys = sys.argv[1:] or list(SESSIONS)
    for k in keys:
        do_session(k)
    print("\nALL DONE:", keys, flush=True)
