"""Decode intended spout position from LocaNMF component activity (per session).

Baseline validation of the per-position cortical model for the stroke study: can the 6 spout
positions be decoded from cortical activity, and WHICH areas carry the code? Features =
footprint-scaled dF/F per component in a post-first-lick window (pre-cue baseline-subtracted);
labels = spout position. ENGAGED trials only (cue followed by a lick within --max-rt; no-lick
trials excluded -- baseline no-lick = disengagement). Multinomial logistic regression,
StratifiedKFold CV; reports accuracy vs chance, confusion matrix, and per-area decoding
(SSp-only, MO-only, all) -- the interpretable LocaNMF payoff.

    python -m wfield_local.locanmf_position_decoder --date 0603 --output "<dir>"
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

from wfield_local.locanmf_cue_lick_analysis import SESSIONS
from wfield_local.plot_lick_aligned_averages import _load_daq_events, _event_frame_indices_from_pco, POSITION_NAMES, DISPLAY_ORDER
from wfield_local.plot_spout_trial_averages import _load_daq_events as _load_cue_events, _classify_cues
from wfield_local.locanmf_lick_aligned import _corrected_frame_samples, _nearest_corrected_frame
from wfield_local.locanmf_crossanimal_dff import _footprint_scale, _frames

POS = [POSITION_NAMES[c] for c in DISPLAY_ORDER]


def _trial_features(s, args):
    mc = s["mc"]
    C = np.load(f"{mc}/locanmf_affine8v1_final/{s['label']}_locanmf_C.npy").astype(np.float64)
    reg = np.load(f"{mc}/locanmf_affine8v1_final/{s['label']}_locanmf_regions.npy")
    A = np.load(f"{mc}/locanmf_affine8v1_final/{s['label']}_locanmf_A.npy", mmap_mode="r")
    ncomp, T = C.shape
    dff = _footprint_scale(A, ncomp)[:, None] * C
    cue = _load_cue_events(s["h5"]); lk = _load_daq_events(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    cue_f, lick_f, csmp = _frames(s, cue, lk)
    codes = _classify_cues(cue["cue_samples"], cue["strobe_samples"], cue["strobe_codes"])
    pre_n = int(round(args.pre_s * args.fs)); post_n = int(round(args.post_s * args.fs))
    maxrt_n = int(round(args.max_rt * args.fs))
    ls = np.sort(lick_f); j = np.searchsorted(ls, cue_f, side="right")
    first = np.where(j < ls.size, ls[np.clip(j, 0, ls.size - 1)], -1)
    rt = first - cue_f
    X, y = [], []
    for k in range(cue_f.size):
        if codes[k] < 0:
            continue
        if not (first[k] > 0 and 0 < rt[k] <= maxrt_n):   # ENGAGED: cue followed by a lick
            continue
        fl = int(first[k]); c0 = int(cue_f[k])
        if fl + post_n > T or c0 - pre_n < 0:
            continue
        base = dff[:, c0 - pre_n:c0].mean(1)
        feat = dff[:, fl:fl + post_n].mean(1) - base       # post-first-lick minus pre-cue baseline
        X.append(feat); y.append(int(codes[k]))
    return np.array(X), np.array(y), reg


def _decode(X, y, seed=0):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    pred = cross_val_predict(clf, X, y, cv=cv)
    return accuracy_score(y, pred), pred


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default="0603", help="session-label date suffix to include (e.g. 0603)")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--fs", type=float, default=31.23)
    ap.add_argument("--pre-s", type=float, default=1.0)
    ap.add_argument("--post-s", type=float, default=0.4, help="post-first-lick feature window")
    ap.add_argument("--max-rt", type=float, default=2.0)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sess = [s for s in SESSIONS if s["label"].endswith(args.date)]
    print(f"sessions: {[s['label'] for s in sess]}  (6 positions, chance=16.7%)", flush=True)

    groups = {"all": None, "SSp": "SSp", "MO": ("MOp", "MOs")}
    fig, axes = plt.subplots(1, len(sess), figsize=(5 * len(sess), 4.6), squeeze=False)
    summary = {}
    for si, s in enumerate(sess):
        X, y, reg = _trial_features(s, args)
        names = {int(k): v for k, v in json.load(
            open(glob.glob(f"{s['mc']}/wfield_local_results/allen_aligned_affine8v1/allen_area_names.json")[0]))}
        accs = {}
        for g, pref in groups.items():
            if pref is None:
                cols = np.arange(X.shape[1])
            else:
                prefs = (pref,) if isinstance(pref, str) else pref
                cols = np.array([i for i in range(X.shape[1])
                                 if any(names.get(int(reg[i]), "").startswith(p) for p in prefs)])
            if cols.size == 0:
                accs[g] = float("nan"); continue
            acc, pred = _decode(X[:, cols], y)
            accs[g] = acc
            if g == "all":
                cm = confusion_matrix(y, pred, labels=DISPLAY_ORDER)
                cmn = cm / cm.sum(1, keepdims=True)
        summary[s["label"]] = {"n_trials": int(X.shape[0]), "n_comp": int(X.shape[1]), "acc": accs}
        print(f"{s['label']}: n={X.shape[0]} trials, {X.shape[1]} comps | "
              + "  ".join(f"{g}={a:.2f}" for g, a in accs.items()), flush=True)
        ax = axes[0][si]
        im = ax.imshow(cmn, vmin=0, vmax=1, cmap="magma")
        ax.set_xticks(range(6)); ax.set_xticklabels([POSITION_NAMES[c] for c in DISPLAY_ORDER], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(6)); ax.set_yticklabels([POSITION_NAMES[c] for c in DISPLAY_ORDER], fontsize=7)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{s['label']}  acc={accs['all']:.2f} (chance .17)\nSSp={accs['SSp']:.2f} MO={accs['MO']:.2f}",
                     fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(f"Spout-position decoding from LocaNMF dF/F (engaged trials), {args.date} sessions", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.output / f"locanmf_position_decoder_{args.date}.png", dpi=130); plt.close(fig)
    (args.output / f"locanmf_position_decoder_{args.date}_summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote", args.output / f"locanmf_position_decoder_{args.date}.png", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
