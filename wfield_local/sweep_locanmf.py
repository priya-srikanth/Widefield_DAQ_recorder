"""Parameter sweep for LocaNMF -- kill-safe, resumable, live-logged.

Runs ``run_locanmf`` for each (r2_thresh, loc_thresh) combination SEQUENTIALLY, each
into its own output dir under ``--sweep-root``. ``run_locanmf`` writes a combo's outputs
only when that combo finishes, so terminating the sweep (Ctrl-C or a hard kill) loses at
most the single in-progress combo -- every COMPLETED combo's outputs stay on disk.
Re-running skips any combo whose ``<label>_locanmf_summary.json`` already exists, so it
RESUMES where it stopped. This lets you stop the sweep anytime to free the machine (e.g.
to run behavior) without losing finished work.

Each combo's stdout/stderr streams live (subprocess uses ``python -u``, log file is
line-buffered) to ``<sweep-root>/<combo>/run.log``; a master timeline goes to
``<sweep-root>/sweep.log``.

    python -m wfield_local.sweep_locanmf \
        --allen-dir "...\\allen_aligned_affine8v1" --label PS94_0601 \
        --sweep-root "...\\locanmf_sweep" \
        --r2 0.95 0.99 --loc 70 80 --maxrank 20
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def _stamp(msg):
    return f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allen-dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--sweep-root", required=True, type=Path)
    ap.add_argument("--svt", default=None)
    ap.add_argument("--r2", nargs="+", type=float, default=[0.95, 0.99])
    ap.add_argument("--loc", nargs="+", type=float, default=[70.0, 80.0])
    ap.add_argument("--maxrank", type=int, default=20)
    ap.add_argument("--mode", default="locanmf", choices=("locanmf", "snmf", "both"))
    args = ap.parse_args()

    args.sweep_root.mkdir(parents=True, exist_ok=True)
    master = open(args.sweep_root / "sweep.log", "a", buffering=1)

    def log(msg):
        line = _stamp(msg)
        print(line, flush=True)
        master.write(line + "\n")

    combos = [(r2, loc) for r2 in args.r2 for loc in args.loc]
    log(f"sweep start: {len(combos)} combos = {combos}  (resumable; skips finished combos)")

    done, ran = 0, 0
    for r2, loc in combos:
        tag = f"r2{r2:g}_loc{int(loc)}"
        outdir = args.sweep_root / tag
        summary = outdir / f"{args.label}_locanmf_summary.json"
        if summary.exists():
            log(f"SKIP {tag} (already complete)")
            done += 1
            continue
        outdir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", "-m", "wfield_local.run_locanmf",
               "--allen-dir", str(args.allen_dir), "--label", args.label,
               "--output", str(outdir), "--mode", args.mode, "--maxrank", str(args.maxrank),
               "--loc-thresh", str(loc), "--r2-thresh", str(r2), "--device", "auto"]
        if args.svt:
            cmd += ["--svt", str(args.svt)]
        log(f"START {tag}")
        t0 = time.time()
        runlog = open(outdir / "run.log", "w", buffering=1)
        try:
            rc = subprocess.call(cmd, stdout=runlog, stderr=subprocess.STDOUT)
        except KeyboardInterrupt:
            runlog.close()
            log(f"INTERRUPTED during {tag} after {time.time()-t0:.0f}s; "
                f"{done} combo(s) saved, {tag} incomplete (will re-run on resume). Exiting.")
            master.close()
            return 130
        runlog.close()
        if rc == 0 and summary.exists():
            log(f"DONE {tag}: exit 0 in {time.time()-t0:.0f}s")
            done += 1
        else:
            log(f"FAILED {tag}: exit {rc} in {time.time()-t0:.0f}s (no summary; see {outdir/'run.log'})")
        ran += 1

    log(f"sweep complete: {done}/{len(combos)} combos have outputs ({ran} run this pass)")
    master.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
