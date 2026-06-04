"""Batch LocaNMF at fixed (chosen) params across many sessions -- kill-safe, resumable.

Runs ``run_locanmf`` once per session listed in a JSON manifest, all at the same
``r2_thresh``/``loc_thresh``/``maxrank``, each into its own output dir. Like
``sweep_locanmf.py`` it writes a session's outputs only on completion and SKIPS any
session whose ``<label>_locanmf_summary.json`` already exists -- so the batch can be
terminated anytime (e.g. to free the machine) and resumes where it stopped.

To avoid overwriting/deleting any existing MICROSCOPE outputs, point ``output`` at a
NEW per-session folder (e.g. ``locanmf_affine8v1_final``); the runner only creates/writes.

Manifest JSON = list of {"allen_dir","label","output"[, "svt"]}:
    python -m wfield_local.batch_locanmf --manifest sessions.json \
        --r2 0.95 --loc 80 --maxrank 20
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _stamp(msg):
    return f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path, help="JSON list of {allen_dir,label,output[,svt]}")
    ap.add_argument("--r2", type=float, default=0.95)
    ap.add_argument("--loc", type=float, default=80.0)
    ap.add_argument("--maxrank", type=int, default=20)
    ap.add_argument("--mode", default="locanmf", choices=("locanmf", "snmf", "both"))
    ap.add_argument("--log", type=Path, default=None, help="master log (default: alongside manifest)")
    args = ap.parse_args()

    sessions = json.loads(args.manifest.read_text())
    master_path = args.log or args.manifest.with_suffix(".batch.log")
    master = open(master_path, "a", buffering=1)

    def log(msg):
        line = _stamp(msg)
        print(line, flush=True)
        master.write(line + "\n")

    log(f"batch start: {len(sessions)} sessions  params r2={args.r2} loc_thresh={args.loc} maxrank={args.maxrank}")
    done = 0
    for s in sessions:
        label = s["label"]
        outdir = Path(s["output"])
        summary = outdir / f"{label}_locanmf_summary.json"
        if summary.exists():
            log(f"SKIP {label} (already complete: {summary})")
            done += 1
            continue
        outdir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", "-m", "wfield_local.run_locanmf",
               "--allen-dir", str(s["allen_dir"]), "--label", label,
               "--output", str(outdir), "--mode", args.mode, "--maxrank", str(args.maxrank),
               "--loc-thresh", str(args.loc), "--r2-thresh", str(args.r2), "--device", "auto"]
        if s.get("svt"):
            cmd += ["--svt", str(s["svt"])]
        log(f"START {label} -> {outdir}")
        t0 = time.time()
        runlog = open(outdir / "run.log", "w", buffering=1)
        try:
            rc = subprocess.call(cmd, stdout=runlog, stderr=subprocess.STDOUT)
        except KeyboardInterrupt:
            runlog.close()
            log(f"INTERRUPTED during {label} after {time.time()-t0:.0f}s; {done} session(s) saved. Exiting.")
            master.close()
            return 130
        runlog.close()
        if rc == 0 and summary.exists():
            log(f"DONE {label}: exit 0 in {time.time()-t0:.0f}s")
            done += 1
        else:
            log(f"FAILED {label}: exit {rc} in {time.time()-t0:.0f}s (see {outdir/'run.log'})")

    log(f"batch complete: {done}/{len(sessions)} sessions have outputs")
    master.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
