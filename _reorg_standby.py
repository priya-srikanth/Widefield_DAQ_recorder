"""Reorganize M: standby to mirror MICROSCOPE's session structure.

  OLD: M:\\Widefield\\labcams_raw_data\\<date>\\<animal-or-session>\\...
  NEW: M:\\Widefield\\labcams\\<date>\\<session>\\{raw_widefield_data,motion_corrected}\\...

- Root renamed labcams_raw_data -> labcams (match the N: 'labcams' name).
- Every file lands under its canonical <session> folder; the motion-corrected
  .bin goes in <session>\\motion_corrected\\ (same parent as raw_widefield_data).
- Moves are os.rename within the M: volume (instant, no data copy). *.part files
  (interrupted copies) are deleted, not moved.

Dry-run by default; --execute to perform. Post-verifies every moved file by size.
"""
import os
import re
import sys

OLD = r"M:\Widefield\labcams_raw_data"
NEW = r"M:\Widefield\labcams"

# canonical session folder per (date, animal), from the N: 'labcams' tree
SESSION = {
    ("20260601", "PS92"): "PS92_20260601",
    ("20260601", "PS94"): "PS94_20260601_141614",
    ("20260601", "PS95"): "PS95_20260601_153653",
    ("20260602", "PS92"): "PS92_20260602_151820",
    ("20260603", "PS92"): "PS92_20260603_104008",
    ("20260603", "PS94"): "PS94_20260603",
    ("20260603", "PS95"): "PS95_20260603_194442",
    ("20260604", "PS92"): "PS92_20260604_132934",
    ("20260604", "PS94"): "PS94_20260604_151516",
    ("20260604", "PS95"): "PS95_20260604_165712",
    ("20260605", "PS92"): "PS92_20260605_125023",
    ("20260605", "PS93"): "PS93_20260605_174659",
    ("20260605", "PS94"): "PS94_20260605_142009",
    ("20260605", "PS95"): "PS95_20260605_163102",
}
SUBDIRS = ("raw_widefield_data", "raw_widefield_data_2", "motion_corrected", "snapshots")


def _sz(p):
    try:
        return os.path.getsize(p)
    except OSError:
        return -1


def plan():
    moves, parts, problems = [], [], []
    for date in sorted(os.listdir(OLD)):
        dp = os.path.join(OLD, date)
        if not (os.path.isdir(dp) and re.fullmatch(r"\d{8}", date)):
            continue
        for top in sorted(os.listdir(dp)):
            tp = os.path.join(dp, top)
            if not os.path.isdir(tp):
                continue
            is_session = bool(re.search(r"_\d{8}", top)) or top in SESSION.values()
            animal_m = re.fullmatch(r"(PS\d+)", top)
            for root, _d, files in os.walk(tp):
                rel = os.path.relpath(os.path.join(root, ""), tp).strip(".\\/")
                for f in files:
                    src = os.path.join(root, f)
                    if f.endswith(".part"):
                        parts.append(src); continue
                    if is_session:
                        dst = os.path.join(NEW, date, top, rel, f) if rel else os.path.join(NEW, date, top, f)
                    elif animal_m:
                        animal = animal_m.group(1)
                        sess = SESSION.get((date, animal))
                        if not sess:
                            problems.append(src); continue
                        relparts = rel.split(os.sep) if rel else []
                        if relparts and relparts[0] in SUBDIRS:
                            dst = os.path.join(NEW, date, sess, rel, f)
                        elif f.startswith("motioncorrect_") and f.endswith(".bin"):
                            dst = os.path.join(NEW, date, sess, "motion_corrected", f)
                        else:  # flat raw/camlog/json in the animal folder
                            dst = os.path.join(NEW, date, sess, "raw_widefield_data", f)
                    else:
                        problems.append(src); continue
                    moves.append((src, dst))
    return moves, parts, problems


def main():
    execute = "--execute" in sys.argv
    moves, parts, problems = plan()
    gb = sum(_sz(s) for s, _ in moves) / 1e9
    print(f"{'EXECUTE' if execute else 'DRY-RUN'}: {len(moves)} files ({gb:.0f} GB) -> {NEW}")
    print(f"  .part to delete: {len(parts)} | unmappable (left in place): {len(problems)}")
    for s, d in moves[:6] + ([("...", "...")] if len(moves) > 6 else []):
        print(f"   {s}\n     -> {d}")
    for p in problems:
        print(f"  PROBLEM (no session mapping): {p}")
    if not execute:
        print("\n(dry-run; pass --execute)")
        return 0
    if problems:
        print("\nRefusing to execute with unmappable files; resolve first.")
        return 1
    for s in parts:
        os.remove(s)
    moved = 0
    for s, d in moves:
        os.makedirs(os.path.dirname(d), exist_ok=True)
        sz = _sz(s)
        os.replace(s, d)
        if _sz(d) != sz:
            print(f"  SIZE MISMATCH after move: {d}")
        else:
            moved += 1
    # prune empty dirs in OLD, then remove OLD root if empty
    for root, _d, _f in os.walk(OLD, topdown=False):
        try:
            if not os.listdir(root):
                os.rmdir(root)
        except OSError:
            pass
    print(f"\nmoved {moved}/{len(moves)} files; deleted {len(parts)} .part")
    print(f"OLD root still exists: {os.path.exists(OLD)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
