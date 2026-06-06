"""End-of-day archival + cleanup for a widefield recording day (reusable).

Daily policy this implements:
  * RAW camera movies            -> M: standby   (cold, immutable originals)
  * MOTION-CORRECTED video (.bin) -> M: standby  (huge; kept OFF MICROSCOPE to
    save space, under <date>\\<animal>\\motion_corrected\\). LocaNMF doesn't need
    it (it uses SVTcorr + the atlas), so this doesn't affect the GPU.
  * ALL other outputs + DAQ      -> N: MICROSCOPE (SVD: U/SVT/SVTcorr, Allen
    transform, event/lick/quiet maps, motion QC, DAQ h5)
  * Once every copy is size-verified, the copied E: files PLUS the reproducible
    E:-only intermediates (cleanpairs ``*.dat`` and any ``*_concat`` raw) may be
    deleted from E: to reclaim space.

Nothing is hardcoded to a session/date/dimension: pass --date YYYYMMDD and the
tool walks E:\\labcams_data\\<date>, mirrors the tree to M:/N:, and (on clean)
re-verifies each destination before removing the E: copy.

Classification of every file under E:\\labcams_data\\<date>:
  - ``*_uint16.dat`` whose name contains ``cleanpairs``            -> E-only intermediate
  - ``*_uint16.dat`` inside a ``*concat*`` folder                  -> E-only intermediate (concat raw)
  - any other ``*_uint16.dat``                                     -> RAW  -> M: (mirrors tree)
  - everything else (npy/npz/png/json/csv/camlog/tif/bin/...)      -> OUTPUT -> N: (mirrors tree)
DAQ h5 files containing <date> anywhere under E:\\DAQ_recorder_output -> N: DAQ
(mirroring their parent folder name).

Subcommands
  archive   copy E: -> M:/N: (idempotent, size-verified; LocaNMF inputs first so a
            GPU can start immediately; M: raw copied last). Writes a manifest on N:.
  verify    report whether every E: file has a confirmed (size-matched) copy on M:/N:.
  clean     delete confirmed-copied E: files + the reproducible intermediates.
            DRY-RUN by default; pass --execute to delete. Each delete re-verifies
            the destination size at delete time; an intermediate is removed only
            once its regeneration sources are confirmed archived (the session's
            raw on M: and a DAQ h5 for the date on N:). Empty dirs are pruned.

Examples
  python -m wfield_local.archive_day archive --date 20260604
  python -m wfield_local.archive_day verify  --date 20260604
  python -m wfield_local.archive_day clean   --date 20260604            # dry-run
  python -m wfield_local.archive_day clean   --date 20260604 --execute

Drive roots default to this rig's mounts; override with --e-lab/--e-daq/--m-raw/
--n-lab/--n-daq if they differ.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

BUF = 64 * 1024 * 1024

DEFAULTS = dict(
    e_lab=r"E:\labcams_data",
    e_daq=r"E:\DAQ_recorder_output",
    m_raw=r"M:\Widefield\labcams_raw_data",
    n_lab=r"N:\MICROSCOPE\Priya\Widefield\labcams",
    n_daq=r"N:\MICROSCOPE\Priya\Widefield\DAQ_recorder_output",
)


def _size(p):
    try:
        return os.path.getsize(p)
    except OSError:
        return -1


def discover(cfg, date):
    """Return (jobs, intermediates, daq_jobs) for one recording day.

    jobs:          dicts(src,dst,kind in {raw,output},session) -- copied E->M/N
    intermediates: dicts(src,session,kind) -- reproducible, E-only (never copied)
    daq_jobs:      dicts(src,dst,kind=daq)
    """
    e_date = os.path.join(cfg["e_lab"], date)
    if not os.path.isdir(e_date):
        raise SystemExit(f"no labcams data for {date}: {e_date}")
    jobs, inter = [], []
    for root, _dirs, files in os.walk(e_date):
        parent = os.path.basename(root)
        for f in files:
            src = os.path.join(root, f)
            rel = os.path.relpath(src, e_date)
            session = rel.split(os.sep)[0]
            if f.endswith("_uint16.dat"):
                if "cleanpairs" in f:
                    inter.append(dict(src=src, session=session, kind="cleanpairs"))
                elif "concat" in parent.lower():
                    inter.append(dict(src=src, session=session, kind="concat_raw"))
                else:
                    jobs.append(dict(src=src, dst=os.path.join(cfg["m_raw"], date, rel),
                                     kind="raw", session=session))
            elif f.startswith("motioncorrect_") and f.endswith(".bin"):
                animal = session.split("_")[0]   # PS92_20260605_125023 -> PS92
                jobs.append(dict(src=src, kind="mcbin", session=session,
                                 dst=os.path.join(cfg["m_raw"], date, animal, "motion_corrected", f)))
            else:
                jobs.append(dict(src=src, dst=os.path.join(cfg["n_lab"], date, rel),
                                 kind="output", session=session))
    daq = []
    if os.path.isdir(cfg["e_daq"]):
        for root, _dirs, files in os.walk(cfg["e_daq"]):
            for f in files:
                if f.endswith(".h5") and date in f:
                    daq.append(dict(src=os.path.join(root, f),
                                    dst=os.path.join(cfg["n_daq"], os.path.basename(root), f),
                                    kind="daq", session=None))
    return jobs, inter, daq


def _priority(job):
    """Copy order: LocaNMF inputs first, DAQ, other N: outputs, huge M: files last."""
    if job["kind"] in ("raw", "mcbin"):
        return 4   # huge cold files -> M: standby, least urgent
    if job["kind"] == "daq":
        return 2
    d = job["dst"].lower()
    if "wfield_local_results" in d or "allen_aligned" in d:
        return 0   # SVD (SVTcorr) + Allen transform = LocaNMF inputs
    return 3       # other N: outputs (maps, QC, shifts, summaries, frame_map, ...)


def _copy_one(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    ssz = _size(src)
    if os.path.exists(dst) and _size(dst) == ssz:
        return "skip"
    tmp = dst + ".part"
    with open(src, "rb", buffering=0) as fi, open(tmp, "wb", buffering=0) as fo:
        while True:
            b = fi.read(BUF)
            if not b:
                break
            fo.write(b)
    os.replace(tmp, dst)
    return "ok" if _size(dst) == ssz else "FAIL"


def cmd_archive(cfg, date):
    jobs, inter, daq = discover(cfg, date)
    allj = sorted(jobs + daq, key=lambda j: (_priority(j), -_size(j["src"])))
    total = sum(_size(j["src"]) for j in allj)
    print(f"[archive {date}] {len(allj)} files, {total/1e9:.1f} GB "
          f"(LocaNMF inputs first, raw->M last)\n", flush=True)
    res = {"ok": 0, "skip": 0, "FAIL": 0}
    fails, done = [], 0
    for i, j in enumerate(allj, 1):
        sz = _size(j["src"])
        try:
            st = _copy_one(j["src"], j["dst"])
        except Exception as ex:
            st = "FAIL"
            print(f"   EXCEPTION {j['src']}: {ex}", flush=True)
        res[st] = res.get(st, 0) + 1
        if st == "FAIL":
            fails.append(j)
        done += sz
        print(f"[{i}/{len(allj)}] {sz/1e9:6.2f} GB {st:4}  {j['dst']}  ({done/1e9:.1f}/{total/1e9:.1f})",
              flush=True)
    print(f"\n[archive {date}] ok={res['ok']} skip={res['skip']} FAIL={res['FAIL']}", flush=True)
    for j in fails:
        print(f"  FAILED: {j['src']} -> {j['dst']}", flush=True)
    if inter:
        gb = sum(_size(j["src"]) for j in inter) / 1e9
        print(f"[archive {date}] {len(inter)} reproducible intermediates left on E "
              f"({gb:.1f} GB) -- removed by `clean`, not copied.", flush=True)
    # manifest on N
    man = dict(date=date, n_files=len(allj), total_gb=round(total / 1e9, 2),
               result=res, intermediates=[j["src"] for j in inter],
               jobs=[{"src": j["src"], "dst": j["dst"], "kind": j["kind"]} for j in allj])
    mandir = os.path.join(cfg["n_lab"], date)
    try:
        os.makedirs(mandir, exist_ok=True)
        with open(os.path.join(mandir, f"_archive_manifest_{date}.json"), "w") as fh:
            json.dump(man, fh, indent=2)
    except OSError as ex:
        print(f"  (manifest not written: {ex})", flush=True)
    return 1 if fails else 0


def _verify(cfg, date):
    jobs, inter, daq = discover(cfg, date)
    allj = jobs + daq
    ok, missing, mismatch = [], [], []
    for j in allj:
        s, d = _size(j["src"]), _size(j["dst"])
        if d < 0:
            missing.append(j)
        elif d != s:
            mismatch.append((j, s, d))
        else:
            ok.append(j)
    return allj, ok, missing, mismatch, inter, daq


def cmd_verify(cfg, date):
    allj, ok, missing, mismatch, inter, _ = _verify(cfg, date)
    print(f"[verify {date}] {len(ok)}/{len(allj)} confirmed copied (size-matched) on M:/N:")
    for j in missing:
        print(f"  MISSING on dest: {j['src']} -> {j['dst']}")
    for j, s, d in mismatch:
        print(f"  SIZE MISMATCH: {j['src']} E={s} dest={d}")
    print(f"[verify {date}] reproducible E-only intermediates (removed by clean): {len(inter)}")
    for j in inter:
        print(f"  {j['kind']}: {j['src']} ({_size(j['src'])/1e9:.1f} GB)")
    return 0 if not missing and not mismatch else 1


def cmd_clean(cfg, date, execute):
    allj, ok_jobs, missing, mismatch, inter, daq = _verify(cfg, date)
    # confirmed regeneration sources per session
    raw_ok = set()
    for j in ok_jobs:
        if j["kind"] == "raw":
            raw_ok.add(j["session"])
    daq_ok = any(_size(j["dst"]) == _size(j["src"]) for j in daq)

    to_delete = [j for j in ok_jobs]                         # confirmed-copied E: files
    inter_del, inter_skip = [], []
    for j in inter:
        if j["session"] in raw_ok and daq_ok:
            inter_del.append(j)
        else:
            inter_skip.append(j)

    freed = sum(_size(j["src"]) for j in to_delete) + sum(_size(j["src"]) for j in inter_del)
    print(f"{'EXECUTE' if execute else 'DRY-RUN'} clean {date}: "
          f"delete {len(to_delete)} copied + {len(inter_del)} intermediates "
          f"= {freed/1e9:.1f} GB | not-yet-copied (kept): {len(missing)+len(mismatch)} | "
          f"intermediates kept (regen unconfirmed): {len(inter_skip)}", flush=True)
    for j in missing:
        print(f"  KEEP (dest missing): {j['src']}")
    for j, s, d in mismatch:
        print(f"  KEEP (size mismatch): {j['src']}")
    for j in inter_skip:
        print(f"  KEEP (regen source not confirmed): {j['src']}")

    if not execute:
        print("\n(dry-run; pass --execute to delete)")
        return 0

    deleted = 0
    for j in to_delete:
        if _size(j["dst"]) == _size(j["src"]):          # re-verify at delete time
            os.remove(j["src"])
            deleted += 1
        else:
            print(f"  SKIP at-delete (dest changed): {j['src']}")
    for j in inter_del:
        os.remove(j["src"])
        deleted += 1
    # prune empty dirs under the day's E tree + any emptied DAQ parent dirs
    pruned = 0
    roots = [os.path.join(cfg["e_lab"], date)] + sorted({os.path.dirname(j["src"]) for j in daq})
    for r in roots:
        if not os.path.isdir(r):
            continue
        for dp, _dirs, _files in os.walk(r, topdown=False):
            try:
                if not os.listdir(dp):
                    os.rmdir(dp)
                    pruned += 1
            except OSError:
                pass
    print(f"\ndeleted {deleted} files; pruned {pruned} empty dirs; freed ~{freed/1e9:.1f} GB")
    return 0


def main():
    ap = argparse.ArgumentParser(description="End-of-day widefield archival + cleanup.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("archive", "verify", "clean"))
    ap.add_argument("--date", required=True, help="recording day, YYYYMMDD (labcams folder name)")
    ap.add_argument("--execute", action="store_true", help="(clean) actually delete; default dry-run")
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), default=v)
    args = ap.parse_args()
    cfg = {k: getattr(args, k) for k in DEFAULTS}
    if args.command == "archive":
        return cmd_archive(cfg, args.date)
    if args.command == "verify":
        return cmd_verify(cfg, args.date)
    return cmd_clean(cfg, args.date, args.execute)


if __name__ == "__main__":
    sys.exit(main())
