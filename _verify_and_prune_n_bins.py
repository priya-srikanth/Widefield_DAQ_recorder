"""Verify motion-corrected bins are BYTE-IDENTICAL on M: standby, then (optionally)
delete them from N: MICROSCOPE to free space.

For each motioncorrect_*.bin under N:\\...\\labcams, find its mirror on
M:\\Widefield\\labcams, hash BOTH (blake2b, full read) and compare. With
--execute, delete an N: bin only when its M: copy is byte-identical (hash match).
Dry-run by default. Never deletes when the M: copy is missing or differs.

LocaNMF doesn't use the bin (it uses SVTcorr + the atlas), so removing it from N:
does not affect the GPU; the bin remains on M: standby.
"""
import argparse
import hashlib
import os
import sys

N_ROOT = r"N:\MICROSCOPE\Priya\Widefield\labcams"
M_ROOT = r"M:\Widefield\labcams"
CHUNK = 64 * 1024 * 1024


def _hash(path):
    h = hashlib.blake2b()
    with open(path, "rb", buffering=0) as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def discover():
    bins = []
    for root, _d, files in os.walk(N_ROOT):
        for f in files:
            if f.startswith("motioncorrect_") and f.endswith(".bin"):
                npath = os.path.join(root, f)
                bins.append((npath, npath.replace(N_ROOT, M_ROOT)))
    return sorted(bins)


def main():
    execute = "--execute" in sys.argv
    bins = discover()
    print(f"{'EXECUTE' if execute else 'DRY-RUN'}: {len(bins)} bins on N: to verify vs M:\n", flush=True)
    identical, deleted, freed = [], 0, 0
    for npath, mpath in bins:
        nsz = os.path.getsize(npath)
        if not os.path.exists(mpath):
            print(f"  KEEP (no M: copy): {npath}", flush=True); continue
        if os.path.getsize(mpath) != nsz:
            print(f"  KEEP (size differs N={nsz} M={os.path.getsize(mpath)}): {npath}", flush=True); continue
        nh = _hash(npath); mh = _hash(mpath)
        if nh == mh:
            identical.append((npath, nsz))
            print(f"  OK byte-identical ({nsz/1e9:.1f} GB): {os.path.relpath(npath, N_ROOT)}", flush=True)
            if execute:
                os.remove(npath); deleted += 1; freed += nsz
        else:
            print(f"  KEEP (HASH DIFFERS!): {npath}\n    N:{nh[:16]} M:{mh[:16]}", flush=True)
    print(f"\n{len(identical)}/{len(bins)} byte-identical on M:. "
          f"{'deleted '+str(deleted)+' from N: ('+format(freed/1e9,'.0f')+' GB freed)' if execute else '(dry-run; pass --execute to delete from N:)'}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
