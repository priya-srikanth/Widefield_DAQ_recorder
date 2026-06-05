"""Archive the 2026-06-04 sessions: raw -> M: standby, analyzed+DAQ -> N: MICROSCOPE.

COPY ONLY. Deletes nothing (server-safety: deletions are a separate explicit step).
Idempotent: skips any destination file that already exists with a matching size.
Verifies every copy by size; prints an OK/FAIL summary at the end.

  M: raw  -> M:\\Widefield\\labcams_raw_data\\20260604\\<ANIMAL>\\...
  N: lab  -> N:\\MICROSCOPE\\Priya\\Widefield\\labcams\\20260604\\<session>\\...
  N: DAQ  -> N:\\MICROSCOPE\\Priya\\Widefield\\DAQ_recorder_output\\20250604\\...

Excluded from N: the cleanpairs intermediate .dat (not archived for 6/3 either).
Excluded from M: the 90 GB concat raw (reproducible from the two segments).
"""
import os
import shutil
import sys

BUF = 64 * 1024 * 1024
E = r"E:\labcams_data\20260604"
EQ = r"E:\DAQ_recorder_output\20250604"
M = r"M:\Widefield\labcams_raw_data\20260604"
NLAB = r"N:\MICROSCOPE\Priya\Widefield\labcams\20260604"
NDAQ = r"N:\MICROSCOPE\Priya\Widefield\DAQ_recorder_output\20250604"

SESS = {
    "PS92": "PS92_20260604_132934",
    "PS94": "PS94_20260604_151516",
    "PS95": "PS95_20260604_165712",
}

jobs = []  # (src, dst)


def add(src, dst):
    if os.path.exists(src):
        jobs.append((src, dst))
    else:
        print(f"[warn] source missing, skipped: {src}", flush=True)


def add_tree(src_dir, dst_dir, skip_pred=None):
    for root, _dirs, files in os.walk(src_dir):
        rel = os.path.relpath(root, src_dir)
        for f in files:
            if skip_pred and skip_pred(f):
                continue
            s = os.path.join(root, f)
            d = os.path.join(dst_dir, rel, f) if rel != "." else os.path.join(dst_dir, f)
            add(s, d)


# ---------------- RAW -> M: (two segments only for PS92) ----------------
# PS92: two segments, same filename -> keep subfolders to disambiguate
add(rf"{E}\PS92_20260604_132934\raw_widefield_data\pco_edge_run000_00000000_2_460_480_uint16.dat",
    rf"{M}\PS92\raw_widefield_data\pco_edge_run000_00000000_2_460_480_uint16.dat")
add(rf"{E}\PS92_20260604_132934\raw_widefield_data\pco_edge_run000_00000000.camlog",
    rf"{M}\PS92\raw_widefield_data\pco_edge_run000_00000000.camlog")
add(rf"{E}\PS92_20260604_132934\raw_widefield_data_2\pco_edge_run000_00000000_2_460_480_uint16.dat",
    rf"{M}\PS92\raw_widefield_data_2\pco_edge_run000_00000000_2_460_480_uint16.dat")
add(rf"{E}\PS92_20260604_132934\raw_widefield_data_2\pco_edge_run000_00000000.camlog",
    rf"{M}\PS92\raw_widefield_data_2\pco_edge_run000_00000000.camlog")
# PS94, PS95: single raw .dat + camlog
for a in ("PS94", "PS95"):
    sd = rf"{E}\{SESS[a]}\raw_widefield_data"
    add(rf"{sd}\pco_edge_run000_00000000_2_460_480_uint16.dat",
        rf"{M}\{a}\pco_edge_run000_00000000_2_460_480_uint16.dat")
    add(rf"{sd}\pco_edge_run000_00000000.camlog",
        rf"{M}\{a}\pco_edge_run000_00000000.camlog")

# ---------------- ANALYZED -> N: per session ----------------
for a, s in SESS.items():
    edir = rf"{E}\{s}"
    ndir = rf"{NLAB}\{s}"
    # raw_widefield_data metadata (NO raw .dat): landmark json, camlog, snapshots
    # PS92 landmark json lives in raw_widefield_data_concat; others in raw_widefield_data
    for sub in ("raw_widefield_data", "raw_widefield_data_2", "raw_widefield_data_concat"):
        srcsub = rf"{edir}\{sub}"
        if not os.path.isdir(srcsub):
            continue
        # landmark JSON + camlog -> N raw_widefield_data
        for f in os.listdir(srcsub):
            p = rf"{srcsub}\{f}"
            if f.endswith(".json") and "landmark" in f.lower():
                add(p, rf"{ndir}\raw_widefield_data\{f}")
            elif f.endswith(".camlog"):
                tag = "" if sub == "raw_widefield_data" else ("_2" if sub.endswith("_2") else "_concat")
                add(p, rf"{ndir}\raw_widefield_data\pco_edge_run000_00000000{tag}.camlog")
            elif f.endswith("_manifest.json"):
                add(p, rf"{ndir}\motion_corrected\{f}")
        snp = rf"{srcsub}\snapshots"
        if os.path.isdir(snp):
            add_tree(snp, rf"{ndir}\raw_widefield_data\snapshots")
    # motion_corrected: everything EXCEPT the cleanpairs intermediate .dat
    add_tree(rf"{edir}\motion_corrected", rf"{ndir}\motion_corrected",
             skip_pred=lambda f: f.endswith(".dat"))

# ---------------- DAQ -> N: ----------------
for f in ("PS92_20260604_133714.h5", "PS92_20260604_140742.h5", "PS92_20260604_concat.h5",
          "PS94_20260604_152103.h5", "PS95_20260604_170729.h5"):
    add(rf"{EQ}\{f}", rf"{NDAQ}\{f}")


def copy_one(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    ssz = os.path.getsize(src)
    if os.path.exists(dst) and os.path.getsize(dst) == ssz:
        return ("skip", ssz)
    tmp = dst + ".part"
    with open(src, "rb", buffering=0) as fi, open(tmp, "wb", buffering=0) as fo:
        while True:
            b = fi.read(BUF)
            if not b:
                break
            fo.write(b)
    os.replace(tmp, dst)
    dsz = os.path.getsize(dst)
    return ("ok" if dsz == ssz else "FAIL", ssz)


def _priority(dst):
    """Lower = copied first. LocaNMF inputs on N: go first so the GPU can start;
    M: raw (cold standby) goes last."""
    dl = dst.lower()
    on_n = dl.startswith("n:")
    if on_n and ("wfield_local_results" in dl or "allen_aligned" in dl):
        return 0   # SVD (SVTcorr) + Allen transform (U_atlas/atlas/mask) = LocaNMF inputs
    if on_n and dl.endswith(".bin"):
        return 1   # corrected video
    if on_n and "daq_recorder_output" in dl:
        return 2   # DAQ
    if on_n:
        return 3   # other N: metadata (shifts, summaries, frame_map, camlogs, snapshots)
    return 4       # M: raw standby (largest, least urgent)


def main():
    jobs.sort(key=lambda j: (_priority(j[1]), -os.path.getsize(j[0])))
    total = sum(os.path.getsize(s) for s, _ in jobs)
    print(f"[archive] {len(jobs)} files, {total/1e9:.1f} GB total (N-analyzed first, M-raw last)\n", flush=True)
    done = 0
    res = {"ok": 0, "skip": 0, "FAIL": 0}
    fails = []
    for i, (s, d) in enumerate(jobs, 1):
        sz = os.path.getsize(s)
        print(f"[{i}/{len(jobs)}] {sz/1e9:6.2f} GB  {os.path.basename(s)} -> {d}", flush=True)
        try:
            st, _ = copy_one(s, d)
        except Exception as ex:
            st = "FAIL"
            print(f"    EXCEPTION: {ex}", flush=True)
        res[st] = res.get(st, 0) + 1
        if st == "FAIL":
            fails.append((s, d))
        done += sz
        print(f"    {st}   ({done/1e9:.1f}/{total/1e9:.1f} GB)", flush=True)
    print(f"\n[archive] DONE  ok={res['ok']} skip={res['skip']} FAIL={res['FAIL']}", flush=True)
    for s, d in fails:
        print(f"  FAILED: {s} -> {d}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
