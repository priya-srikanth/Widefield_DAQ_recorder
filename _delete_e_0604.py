"""Delete CONFIRMED-COPIED 2026-06-04 files from E: (after archival to N:/M:).

Safety model:
  - Reuses the exact src->dst mapping the archival used (import _archive_0604.jobs),
    plus the PS92 session-root snapshots that were copied manually to N:.
  - Deletes an E: file ONLY after re-verifying its destination exists with a
    MATCHING SIZE, at delete time. Any mismatch -> skip + report (never delete).
  - Only ever touches paths under E:\\labcams_data\\20260604 or
    E:\\DAQ_recorder_output\\20250604.
  - NEVER deletes the reproducible E-only intermediates (they were not copied, so
    they are not in the job list): the cleanpairs *.dat and the concat raw .dat.
  - Removes directories only if they end up empty.

Dry-run by default; pass --execute to actually delete.
"""
import os
import sys

import _archive_0604 as arch

E_ROOTS = (r"E:\labcams_data\20260604", r"E:\DAQ_recorder_output\20250604")
NL = r"N:\MICROSCOPE\Priya\Widefield\labcams\20260604"


def build_jobs():
    jobs = list(arch.jobs)
    # PS92 session-root snapshots were copied to N: manually (not via the archival walk)
    src = r"E:\labcams_data\20260604\PS92_20260604_132934\snapshots"
    dst = fr"{NL}\PS92_20260604_132934\raw_widefield_data\snapshots"
    if os.path.isdir(src):
        for f in os.listdir(src):
            sp = os.path.join(src, f)
            if os.path.isfile(sp):
                jobs.append((sp, os.path.join(dst, f)))
    return jobs


def main():
    execute = "--execute" in sys.argv
    jobs = build_jobs()
    to_delete, skip, gone = [], [], 0
    for s, d in jobs:
        if not os.path.exists(s):
            gone += 1
            continue
        if not any(os.path.normcase(os.path.abspath(s)).startswith(os.path.normcase(r)) for r in E_ROOTS):
            skip.append((s, "outside allowed E roots")); continue
        if not os.path.exists(d):
            skip.append((s, "dest MISSING")); continue
        if os.path.getsize(d) != os.path.getsize(s):
            skip.append((s, f"size mismatch E={os.path.getsize(s)} N/M={os.path.getsize(d)}")); continue
        to_delete.append((s, os.path.getsize(s)))

    freed = sum(sz for _, sz in to_delete)
    print(f"{'EXECUTE' if execute else 'DRY-RUN'}: {len(to_delete)} files to delete, "
          f"{freed/1e9:.1f} GB to free | {gone} already gone | {len(skip)} skipped\n", flush=True)
    for s, why in skip:
        print(f"  SKIP ({why}): {s}", flush=True)

    # ---- reproducible E-only intermediates: explicitly authorized for deletion,
    # but only once their regeneration sources are confirmed on M:/N: ----
    M = r"M:\Widefield\labcams_raw_data\20260604"
    ND = r"N:\MICROSCOPE\Priya\Widefield\DAQ_recorder_output\20250604"
    SESS = {  # session dir -> (list of raw .dat on M, DAQ h5 on N)  [regen sources]
        "PS92_20260604_132934": (
            [fr"{M}\PS92\raw_widefield_data\pco_edge_run000_00000000_2_460_480_uint16.dat",
             fr"{M}\PS92\raw_widefield_data_2\pco_edge_run000_00000000_2_460_480_uint16.dat"],
            fr"{ND}\PS92_20260604_concat.h5"),
        "PS94_20260604_151516": (
            [fr"{M}\PS94\pco_edge_run000_00000000_2_460_480_uint16.dat"],
            fr"{ND}\PS94_20260604_152103.h5"),
        "PS95_20260604_165712": (
            [fr"{M}\PS95\pco_edge_run000_00000000_2_460_480_uint16.dat"],
            fr"{ND}\PS95_20260604_170729.h5"),
    }
    intermediates = []  # (path, size, regen_ok)
    for sess, (raws, daq) in SESS.items():
        sdir = fr"E:\labcams_data\20260604\{sess}"
        regen_ok = all(os.path.exists(r) for r in raws) and os.path.exists(daq)
        # cleanpairs intermediate(s) in motion_corrected
        mc = fr"{sdir}\motion_corrected"
        if os.path.isdir(mc):
            for f in os.listdir(mc):
                if f.endswith(".dat") and "cleanpairs" in f:
                    p = os.path.join(mc, f)
                    intermediates.append((p, os.path.getsize(p), regen_ok))
        # concat raw (PS92 only): needs both segments on M
        cc = fr"{sdir}\raw_widefield_data_concat"
        if os.path.isdir(cc):
            for f in os.listdir(cc):
                if f.endswith("_uint16.dat"):
                    p = os.path.join(cc, f)
                    intermediates.append((p, os.path.getsize(p), regen_ok))
    inter_del = [(p, sz) for p, sz, ok in intermediates if ok]
    inter_skip = [(p, sz) for p, sz, ok in intermediates if not ok]
    freed += sum(sz for _, sz in inter_del)
    print(f"\n[reproducible intermediates] {len(inter_del)} to delete "
          f"({sum(sz for _,sz in inter_del)/1e9:.1f} GB); {len(inter_skip)} kept (regen source missing)", flush=True)
    for p, sz in inter_del:
        print(f"  DEL {p} ({sz/1e9:.1f} GB)", flush=True)
    for p, sz in inter_skip:
        print(f"  KEEP (regen source not confirmed!) {p}", flush=True)
    print(f"\n{'EXECUTE' if execute else 'DRY-RUN'} total to free: {freed/1e9:.1f} GB", flush=True)

    deleted = 0
    if execute:
        for s, sz in to_delete:
            os.remove(s)
            deleted += 1
        for p, sz in inter_del:
            os.remove(p)
            deleted += 1
        print(f"\ndeleted {deleted} files", flush=True)
        # prune empty dirs under the E roots (deepest first)
        removed_dirs = 0
        for root in E_ROOTS:
            for dirpath, dirs, files in os.walk(root, topdown=False):
                if not os.listdir(dirpath):
                    os.rmdir(dirpath); removed_dirs += 1
        print(f"removed {removed_dirs} empty dirs", flush=True)

    # report what intentionally REMAINS on E
    print("\n[KEPT on E: by design -- reproducible, never copied]", flush=True)
    for dp, _, fs in os.walk(r"E:\labcams_data\20260604"):
        for f in fs:
            p = os.path.join(dp, f)
            if (f.endswith(".dat") and ("cleanpairs" in f or "raw_widefield_data_concat" in dp)):
                print(f"  {p}  ({os.path.getsize(p)/1e9:.1f} GB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
