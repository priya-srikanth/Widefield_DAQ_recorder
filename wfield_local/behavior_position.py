"""Behavior-log spout-position BACKUP.

When the DAQ spout-strobe bits are faulty (the 2026-08 sessions had DAQ line5 `spout_bit1` dead — 0 edges —
which dropped close_R + far_center and mis-coded them onto other positions), the per-cue position from
`_classify_cues` is wrong. The behavior task controller logs the TRUE position per trial in `trials.csv`
(`pos_idx`, same 0-5 convention as POSITION_NAMES).

The behavior session starts AFTER the DAQ recording and the two use independent trial counters, so cue 0 is
not trial 0. We find the integer trial-offset that MAXIMIZES agreement with the positions the DAQ DID get
right — that offset absorbs the behavior-vs-DAQ startup difference; the alignment is then drift-free
(index-based, not a wall-clock offset that would slip over a long session). Cross-checked on 8/5: the DAQ
has exactly one cue per behavior trial, and the best offset reproduces the behavior log's own per-position
counts at 100% agreement on the 4 good positions. Only overrides when it validates (>=0.9 on the DAQ's good
positions) AND the DAQ is actually broken (<6 positions); otherwise returns the DAQ codes unchanged. If the
DAQ-cue and behavior-trial counts don't correspond 1:1 (e.g. the DAQ stopped before the behavior session
ended), validation fails and we keep the DAQ codes — so good sessions are never altered.

Behavior logs: M:/MICROSCOPE/Priya/Behavior_logs/Widefield/<mouse>_<YYYYMMDD>_*/trials.csv .
"""
from __future__ import annotations

import csv
import glob
import os
import re

import numpy as np

from wfield_local.plot_spout_trial_averages import _classify_cues

BEH_ROOT = "M:/MICROSCOPE/Priya/Behavior_logs/Widefield"


def _behavior_dir(session):
    lab = session["label"]; mouse = lab[:4]
    m = re.search(r"/labcams/(\d{8})/", session["mc"].replace("\\", "/"))
    if not m:
        return None
    hits = sorted(glob.glob(f"{BEH_ROOT}/{mouse}_{m.group(1)}_*"))
    return hits[0] if hits else None


def _behavior_positions(session):
    """True per-trial pos_idx (0-5) from the task controller's trials.csv, real trials only."""
    d = _behavior_dir(session)
    if not d:
        return None
    f = os.path.join(d, "trials.csv")
    if not os.path.exists(f):
        return None
    try:
        rows = list(csv.DictReader(open(f)))
        real = [r for r in rows if r.get("device_trial_start_ms") != r.get("device_trial_end_ms")]
        b = np.array([int(r["pos_idx"]) for r in real], dtype=np.int16)
        return b if b.size >= 10 else None
    except Exception:
        return None


def classify_cues_with_backup(session, cue, verbose=True):
    daq = _classify_cues(cue["cue_samples"], cue["strobe_samples"], cue["strobe_codes"])
    good = sorted(set(int(x) for x in daq if x >= 0))
    if len(good) >= 6:
        return daq                                    # DAQ has all 6 positions -> trust it
    b = _behavior_positions(session)
    if b is None:
        return daq
    N = len(daq); rng = max(20, abs(len(b) - N) + 10)
    best = (-1.0, 0, None)
    for off in range(-rng, rng + 1):                  # behavior[j] -> daq[j+off]
        out = np.full(N, -1, dtype=np.int16)
        k = np.arange(len(b)) + off; ok = (k >= 0) & (k < N)
        out[k[ok]] = b[ok]
        val = (out >= 0) & np.isin(daq, good) & np.isin(out, good)
        if val.sum() < 20:
            continue
        sc = float((daq[val] == out[val]).mean())
        if sc > best[0]:
            best = (sc, off, out)
    sc, off, out = best
    if out is not None and sc >= 0.9:
        res = out.copy(); miss = res < 0; res[miss] = daq[miss]
        if verbose:
            print(f"  [behavior-backup] {session['label']}: DAQ had {good} (<6); behavior log aligned at "
                  f"trial-offset {off:+d} (good-pos agreement {sc:.2f}, n_cues={N}, n_beh={len(b)}); "
                  f"positions now {sorted(set(int(x) for x in res if x >= 0))}", flush=True)
        return res
    if verbose:
        print(f"  [behavior-backup] {session['label']}: DAQ missing positions but behavior alignment could "
              f"not be validated (best {sc:.2f}, n_cues={N}, n_beh={len(b) if b is not None else 0}) -> "
              f"kept DAQ codes", flush=True)
    return daq
