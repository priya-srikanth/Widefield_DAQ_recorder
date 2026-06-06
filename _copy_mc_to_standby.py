"""Copy motion-corrected videos (motioncorrect_*.bin) to M: standby for backup.

For every session on MICROSCOPE (N:) that has a motioncorrect_*.bin, copy it to
the mirrored session path on standby:
  M:\\Widefield\\labcams\\<date>\\<session>\\motion_corrected\\<binname>
Source auto-selection: if the same bin still exists on E: (faster local read),
copy from E:; otherwise copy from N:. Idempotent + size-verified (skips a dest
that already byte-matches). Copy-only; deletes nothing.

Usage:
  python _copy_mc_to_standby.py                       # all dates
  python _copy_mc_to_standby.py --dates 20260605      # only these dates
"""
import argparse
import os
import re
import sys

BUF = 64 * 1024 * 1024
N_ROOT = r"N:\MICROSCOPE\Priya\Widefield\labcams"
E_ROOT = r"E:\labcams_data"
M_ROOT = r"M:\Widefield\labcams"


def _sz(p):
    try:
        return os.path.getsize(p)
    except OSError:
        return -1


def discover(dates):
    jobs = []
    for date in sorted(os.listdir(N_ROOT)):
        dp = os.path.join(N_ROOT, date)
        if not (os.path.isdir(dp) and re.fullmatch(r"\d{8}", date)):
            continue
        if dates and date not in dates:
            continue
        for root, _d, files in os.walk(dp):
            for f in files:
                if f.startswith("motioncorrect_") and f.endswith(".bin"):
                    npath = os.path.join(root, f)
                    m = re.search(r"(PS\d+)", os.path.relpath(npath, dp))
                    animal = m.group(1) if m else "UNKNOWN"
                    ecand = npath.replace(N_ROOT, E_ROOT)
                    src = ecand if os.path.exists(ecand) else npath
                    dst = npath.replace(N_ROOT, M_ROOT)  # mirror session path into M: labcams
                    jobs.append(dict(date=date, animal=animal, src=src,
                                     from_="E" if src == ecand else "N", dst=dst))
    return jobs


def copy_one(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    s = _sz(src)
    if os.path.exists(dst) and _sz(dst) == s:
        return "skip"
    tmp = dst + ".part"
    with open(src, "rb", buffering=0) as fi, open(tmp, "wb", buffering=0) as fo:
        while True:
            b = fi.read(BUF)
            if not b:
                break
            fo.write(b)
    os.replace(tmp, dst)
    return "ok" if _sz(dst) == s else "FAIL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="*", default=None, help="restrict to these YYYYMMDD dates")
    args = ap.parse_args()
    jobs = discover(args.dates)
    total = sum(_sz(j["src"]) for j in jobs)
    print(f"[mc->standby] {len(jobs)} bins, {total/1e9:.0f} GB "
          f"(source: {sum(1 for j in jobs if j['from_']=='E')} from E, "
          f"{sum(1 for j in jobs if j['from_']=='N')} from N)\n", flush=True)
    res = {"ok": 0, "skip": 0, "FAIL": 0}
    fails, done = [], 0
    for i, j in enumerate(jobs, 1):
        s = _sz(j["src"])
        print(f"[{i}/{len(jobs)}] {s/1e9:6.1f} GB  {j['from_']}->M  {j['date']}/{j['animal']}", flush=True)
        try:
            st = copy_one(j["src"], j["dst"])
        except Exception as ex:
            st = "FAIL"; print(f"   EXCEPTION: {ex}", flush=True)
        res[st] = res.get(st, 0) + 1
        if st == "FAIL":
            fails.append(j)
        done += s
        print(f"    {st}  ({done/1e9:.0f}/{total/1e9:.0f} GB)  -> {j['dst']}", flush=True)
    print(f"\n[mc->standby] ok={res['ok']} skip={res['skip']} FAIL={res['FAIL']}", flush=True)
    for j in fails:
        print(f"  FAILED: {j['src']} -> {j['dst']}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
