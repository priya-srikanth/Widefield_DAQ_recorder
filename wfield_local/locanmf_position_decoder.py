"""Decode intended spout position from cortical activity (LocaNMF components or SVD/Allen ROIs).

Baseline validation of the per-position model for the stroke study. Features = activity in a
CUE-aligned window (pre-cue baseline-subtracted) on ENGAGED trials (cue followed by a lick
within --max-rt; no-lick excluded). Cue-aligned so the same decoder later applies to ALL
trials incl post-stroke no-lick failed attempts. Validation = 5-fold (20% held out per fold),
each trial scored on held-out data. Multinomial logistic regression; reports accuracy vs
chance, confusion matrix, and per-area decoding (SSp / MO / all).

  --source locanmf : footprint-scaled dF/F per LocaNMF component (session-specific basis)
  --source roi     : SVD Allen-region ROI dF/F (mean_pixels(U_region) @ SVTcorr; atlas-anchored)
  --align cue|lick : feature window relative to cue (default) or first lick

    python -m wfield_local.locanmf_position_decoder --date 0603 --source locanmf --output "<dir>"
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
from wfield_local.plot_lick_aligned_averages import _load_daq_events, POSITION_NAMES, DISPLAY_ORDER
from wfield_local.plot_spout_trial_averages import _load_daq_events as _load_cue_events, _classify_cues
from wfield_local.locanmf_crossanimal_dff import _footprint_scale, _frames


def _build_signal(s, source):
    """Return (signal [nfeat,T], feat_region_labels [nfeat])."""
    mc = s["mc"]
    if source == "locanmf":
        C = np.load(f"{mc}/locanmf_affine8v1_final/{s['label']}_locanmf_C.npy").astype(np.float64)
        reg = np.load(f"{mc}/locanmf_affine8v1_final/{s['label']}_locanmf_regions.npy")
        A = np.load(f"{mc}/locanmf_affine8v1_final/{s['label']}_locanmf_A.npy", mmap_mode="r")
        return _footprint_scale(A, C.shape[0])[:, None] * C, reg
    ad = glob.glob(f"{mc}/wfield_local_results/allen_aligned_affine8v1")[0]
    U = np.load(f"{ad}/U_atlas.npy"); SVT = np.load(f"{mc}/wfield_local_results/SVTcorr.npy")
    atlas = np.load(f"{ad}/allen_area_atlas_native_grid.npy")
    mask = np.load(f"{ad}/allen_brain_mask_native_grid.npy").astype(bool)
    Uf = U.reshape(-1, U.shape[2]); at = atlas.reshape(-1); mk = mask.reshape(-1)
    rois, regs = [], []
    for l in np.unique(at):
        if l == 0:
            continue
        pix = (at == l) & mk
        if pix.sum() < 20:
            continue
        rois.append(np.nanmean(Uf[pix], 0) @ SVT); regs.append(int(l))
    return np.array(rois), np.array(regs)


def _trial_features(s, args):
    sig, feat_reg = _build_signal(s, args.source)
    nfeat, T = sig.shape
    cue = _load_cue_events(s["h5"]); lk = _load_daq_events(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    cue_f, lick_f, csmp = _frames(s, cue, lk)
    codes = _classify_cues(cue["cue_samples"], cue["strobe_samples"], cue["strobe_codes"])
    pre_n = int(round(args.pre_s * args.fs)); post_n = int(round(args.post_s * args.fs))
    maxrt_n = int(round(args.max_rt * args.fs))
    ls = np.sort(lick_f); j = np.searchsorted(ls, cue_f, side="right")
    first = np.where(j < ls.size, ls[np.clip(j, 0, ls.size - 1)], -1); rt = first - cue_f
    X, y = [], []
    for k in range(cue_f.size):
        if codes[k] < 0 or not (first[k] > 0 and 0 < rt[k] <= maxrt_n):   # engaged: cue + lick
            continue
        c0 = int(cue_f[k]); align = c0 if args.align == "cue" else int(first[k])
        if align + post_n > T or c0 - pre_n < 0:
            continue
        feat = sig[:, align:align + post_n].mean(1) - sig[:, c0 - pre_n:c0].mean(1)
        X.append(feat); y.append(int(codes[k]))
    return np.array(X), np.array(y), feat_reg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default="0603")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--source", choices=("locanmf", "roi"), default="locanmf")
    ap.add_argument("--align", choices=("cue", "lick"), default="cue")
    ap.add_argument("--fs", type=float, default=31.23)
    ap.add_argument("--pre-s", type=float, default=1.0)
    ap.add_argument("--post-s", type=float, default=1.0, help="feature window after the alignment event")
    ap.add_argument("--max-rt", type=float, default=2.0)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sess = [s for s in SESSIONS if s["label"].endswith(args.date)]
    print(f"source={args.source} align={args.align}  sessions={[s['label'] for s in sess]}  chance=0.167", flush=True)

    groups = {"all": None, "SSp": ("SSp",), "MO": ("MOp", "MOs")}
    fig, axes = plt.subplots(1, len(sess), figsize=(5 * len(sess), 4.6), squeeze=False)
    summary = {}
    for si, s in enumerate(sess):
        X, y, feat_reg = _trial_features(s, args)
        names = {int(k): v for k, v in json.load(
            open(glob.glob(f"{s['mc']}/wfield_local_results/allen_aligned_affine8v1/allen_area_names.json")[0]))}
        accs = {}; recall = {}; cmn = None
        for g, prefs in groups.items():
            cols = (np.arange(X.shape[1]) if prefs is None else
                    np.array([i for i in range(X.shape[1]) if any(names.get(int(feat_reg[i]), "").startswith(p) for p in prefs)]))
            if cols.size == 0:
                accs[g] = float("nan"); recall[g] = [float("nan")] * 6; continue
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
            pred = cross_val_predict(clf, X[:, cols], y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
            accs[g] = accuracy_score(y, pred)
            cm = confusion_matrix(y, pred, labels=DISPLAY_ORDER); cmg = cm / np.maximum(cm.sum(1, keepdims=True), 1)
            recall[g] = np.diag(cmg).tolist()      # per-position recall = the pre/post-diffable quantity
            if g == "all":
                cmn = cmg
        # trials per position (for weighting the pre/post comparison)
        npos = {POSITION_NAMES[c]: int((y == c).sum()) for c in DISPLAY_ORDER}
        summary[s["label"]] = {"n_trials": int(X.shape[0]), "n_feat": int(X.shape[1]), "acc": accs,
                               "positions": [POSITION_NAMES[c] for c in DISPLAY_ORDER],
                               "recall_by_position": recall, "n_per_position": npos,
                               "source": args.source, "align": args.align}
        print(f"{s['label']}: n={X.shape[0]} {args.source}feat={X.shape[1]} | "
              + "  ".join(f"{g}={a:.2f}" for g, a in accs.items()), flush=True)
        ax = axes[0][si]; im = ax.imshow(cmn, vmin=0, vmax=1, cmap="magma")
        labs = [POSITION_NAMES[c] for c in DISPLAY_ORDER]
        ax.set_xticks(range(6)); ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(6)); ax.set_yticklabels(labs, fontsize=7)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{s['label']}  acc={accs['all']:.2f} (chance .17)\nSSp={accs['SSp']:.2f} MO={accs['MO']:.2f}", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(f"Spout-position decoding [{args.source}, {args.align}-aligned, engaged trials, 5-fold], {args.date}",
                 fontsize=12)
    fig.tight_layout()
    tag = f"{args.source}_{args.align}"
    fig.savefig(args.output / f"locanmf_position_decoder_{args.date}_{tag}.png", dpi=130); plt.close(fig)

    # ---- per-position recall (the quantity to diff pre vs post), by feature group ----
    posnames = [POSITION_NAMES[c] for c in DISPLAY_ORDER]
    fig2, axes2 = plt.subplots(1, len(sess), figsize=(5 * len(sess), 4), squeeze=False, sharey=True)
    for si, s in enumerate(sess):
        ax = axes2[0][si]; rec = summary[s["label"]]["recall_by_position"]; x = np.arange(6); w = 0.27
        for gi, g in enumerate(["all", "SSp", "MO"]):
            ax.bar(x + (gi - 1) * w, rec.get(g, [np.nan] * 6), w, label=g)
        ax.axhline(1 / 6, color="grey", ls="--", lw=0.8, label="chance")
        ax.set_xticks(x); ax.set_xticklabels(posnames, rotation=45, ha="right", fontsize=7)
        ax.set_title(s["label"], fontsize=10); ax.set_ylim(0, 1)
        if si == 0:
            ax.set_ylabel("decoding recall"); ax.legend(fontsize=7)
    fig2.suptitle(f"Per-position decoding recall [{tag}] — pre/post comparison quantity, {args.date}", fontsize=12)
    fig2.tight_layout()
    fig2.savefig(args.output / f"locanmf_position_recall_{args.date}_{tag}.png", dpi=130); plt.close(fig2)

    (args.output / f"locanmf_position_decoder_{args.date}_{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote", args.output / f"locanmf_position_decoder_{args.date}_{tag}.png",
          "+ recall fig + summary", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
