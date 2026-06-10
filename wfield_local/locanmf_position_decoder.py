"""Decode intended spout position from cortical activity (LocaNMF components or SVD/Allen ROIs).

Baseline validation of the per-position model for the stroke study. Features = mean activity
in a window after the cue (default) or first lick, on ENGAGED trials (cue followed by a lick
within --max-rt; no-lick held aside). Multinomial logistic regression; reports accuracy vs
chance, confusion matrix, and per-area decoding (SSp / MO / all).

  --source locanmf : footprint-scaled dF/F per LocaNMF component (session-specific basis)
  --source roi     : SVD Allen-region ROI dF/F (mean_pixels(U_region) @ SVTcorr; atlas-anchored)
  --align cue|lick : feature window relative to cue (default) or first lick
  --baseline none|precue : per-trial pre-cue subtraction. DEFAULT none -- a per-trial pre-cue
      baseline over-subtracts genuine (anticipatory) position signal, and a *session-constant*
      baseline (e.g. quiet-period) is invisible to a standardized decoder anyway.
  --cv block|random : DEFAULT block. Spout positions are presented in BLOCKS (~6 trials), so
      random k-fold leaks each block's slow-drift fingerprint across train/test (trials in the
      same block are not independent) -- inflating accuracy, especially with no baseline.
      Block-aware CV (leave-whole-blocks-out) forces generalization to unseen blocks.

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
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_predict
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
    # block id: consecutive same-position cues = one block (positions are presented in ~6-trial blocks)
    blk_id = np.full(cue_f.size, -1, dtype=int); b = -1; prev = None
    for k in range(cue_f.size):
        if codes[k] < 0:
            continue
        if prev is None or codes[k] != prev:
            b += 1
        blk_id[k] = b; prev = int(codes[k])
    pre_n = int(round(args.pre_s * args.fs)); post_n = int(round(args.post_s * args.fs))
    maxrt_n = int(round(args.max_rt * args.fs))
    ls = np.sort(lick_f); j = np.searchsorted(ls, cue_f, side="right")
    first = np.where(j < ls.size, ls[np.clip(j, 0, ls.size - 1)], -1); rt = first - cue_f
    subtract = args.baseline == "precue"
    X, y, g, Xn, yn = [], [], [], [], []
    for k in range(cue_f.size):
        if codes[k] < 0:
            continue
        c0 = int(cue_f[k])
        # cue/precue-referenced window start (precue = the post_n window ending at the cue)
        ref0 = c0 - post_n if args.align == "precue" else c0
        if ref0 < 0 or ref0 + post_n > T:
            continue
        if subtract:
            if c0 - pre_n < 0:
                continue
            base = sig[:, c0 - pre_n:c0].mean(1)
        else:
            base = 0.0
        if first[k] > 0 and 0 < rt[k] <= maxrt_n:           # ENGAGED: cue + lick
            w0 = int(first[k]) if args.align == "lick" else ref0
            if w0 < 0 or w0 + post_n > T:
                continue
            X.append(sig[:, w0:w0 + post_n].mean(1) - base); y.append(int(codes[k])); g.append(int(blk_id[k]))
        else:                                               # NO-LICK: cue/precue-referenced (no lick to align)
            Xn.append(sig[:, ref0:ref0 + post_n].mean(1) - base); yn.append(int(codes[k]))
    return np.array(X), np.array(y), np.array(g), np.array(Xn), np.array(yn), feat_reg


def _save_session_fig(label, cmn, sm, labs, args, tag):
    """One compact (confusion | recall) figure for a single session, for the animal-first deck."""
    fig, (axc, axr) = plt.subplots(1, 2, figsize=(8.4, 3.8))
    if cmn is not None:
        im = axc.imshow(cmn, vmin=0, vmax=1, cmap="magma")
        axc.set_xticks(range(6)); axc.set_xticklabels(labs, rotation=45, ha="right", fontsize=7)
        axc.set_yticks(range(6)); axc.set_yticklabels(labs, fontsize=7)
        axc.set_xlabel("predicted"); axc.set_ylabel("true")
        fig.colorbar(im, ax=axc, shrink=0.75)
    axc.set_title(f"{label}  {args.align} 0-{args.post_s:g}s\nacc={sm['acc']['all']:.2f} "
                  f"SSp={sm['acc']['SSp']:.2f} MO={sm['acc']['MO']:.2f} (chance .17)", fontsize=9)
    x = np.arange(6); nl_ok = args.align in ("cue", "precue") and not np.isnan(sm["acc_nolick"]["all"])
    w = 0.38 if nl_ok else 0.6
    axr.bar(x - (w / 2 if nl_ok else 0), sm["recall_by_position"].get("all", [np.nan] * 6), w,
            color="tab:blue", label=f"engaged ({args.cv}-CV)")
    if nl_ok:
        axr.bar(x + w / 2, sm["recall_nolick_by_position"].get("all", [np.nan] * 6), w,
                color="tab:red", label=f"no-lick (n={sm['n_nolick']})")
    axr.axhline(1 / 6, color="grey", ls="--", lw=0.8, label="chance")
    axr.set_xticks(x); axr.set_xticklabels(labs, rotation=45, ha="right", fontsize=7); axr.set_ylim(0, 1)
    nls = f"  no-lick={sm['acc_nolick']['all']:.2f}" if nl_ok else ""
    axr.set_title(f"per-position recall  eng={sm['acc']['all']:.2f}{nls}", fontsize=9)
    axr.set_ylabel("recall"); axr.legend(fontsize=7)
    fig.tight_layout()
    p = args.output / f"locanmf_position_session_{label}_{tag}.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    print("  wrote", p.name, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default="0603")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--source", choices=("locanmf", "roi"), default="locanmf")
    ap.add_argument("--align", choices=("cue", "lick", "precue"), default="cue")
    ap.add_argument("--baseline", choices=("none", "precue"), default="none")
    ap.add_argument("--cv", choices=("block", "random"), default="block")
    ap.add_argument("--fs", type=float, default=31.23)
    ap.add_argument("--pre-s", type=float, default=1.0)
    ap.add_argument("--post-s", type=float, default=2.0, help="feature window after the alignment event "
                    "(2.0 = empirical optimum; spans the lick bout. >~2.5s dilutes the transient, see window sweep)")
    ap.add_argument("--max-rt", type=float, default=2.0)
    ap.add_argument("--per-session", action="store_true",
                    help="also write one compact confusion+recall figure per session "
                         "(locanmf_position_session_{label}_{tag}.png) for the animal-first deck")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sess = [s for s in SESSIONS if s["label"].endswith(args.date)]
    print(f"source={args.source} align={args.align} baseline={args.baseline} cv={args.cv}  "
          f"sessions={[s['label'] for s in sess]}  chance=0.167", flush=True)

    def _cv_predict(clf, Xc, yv, gv):
        if args.cv == "block":
            ng = min(5, int(np.unique(gv).size))
            return cross_val_predict(clf, Xc, yv, cv=GroupKFold(ng), groups=gv)
        return cross_val_predict(clf, Xc, yv, cv=StratifiedKFold(5, shuffle=True, random_state=0))

    groups = {"all": None, "SSp": ("SSp",), "MO": ("MOp", "MOs")}
    fig, axes = plt.subplots(1, len(sess), figsize=(5 * len(sess), 4.6), squeeze=False)
    summary = {}
    for si, s in enumerate(sess):
        X, y, gblk, Xnl, ynl, feat_reg = _trial_features(s, args)
        names = {int(k): v for k, v in json.load(
            open(glob.glob(f"{s['mc']}/wfield_local_results/allen_aligned_affine8v1/allen_area_names.json")[0]))}
        nl_ok = args.align in ("cue", "precue") and Xnl.shape[0] >= 6   # no-lick valid for cue/precue (no lick needed)
        accs = {}; recall = {}; acc_nl = {}; recall_nl = {}; cmn = None
        for g, prefs in groups.items():
            cols = (np.arange(X.shape[1]) if prefs is None else
                    np.array([i for i in range(X.shape[1]) if any(names.get(int(feat_reg[i]), "").startswith(p) for p in prefs)]))
            if cols.size == 0:
                accs[g] = recall[g] = float("nan"); acc_nl[g] = float("nan"); recall_nl[g] = [float("nan")] * 6; continue
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
            pred = _cv_predict(clf, X[:, cols], y, gblk)
            accs[g] = accuracy_score(y, pred)
            cmg = confusion_matrix(y, pred, labels=DISPLAY_ORDER); cmg = cmg / np.maximum(cmg.sum(1, keepdims=True), 1)
            recall[g] = np.diag(cmg).tolist()      # engaged per-position recall (5-fold, out-of-sample)
            if g == "all":
                cmn = cmg
            if nl_ok:                              # train on engaged, apply to held-aside no-lick trials
                clf2 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)).fit(X[:, cols], y)
                pnl = clf2.predict(Xnl[:, cols]); acc_nl[g] = accuracy_score(ynl, pnl)
                cmnl = confusion_matrix(ynl, pnl, labels=DISPLAY_ORDER)
                recall_nl[g] = (np.diag(cmnl) / np.maximum(cmnl.sum(1), 1)).tolist()
            else:
                acc_nl[g] = float("nan"); recall_nl[g] = [float("nan")] * 6
        npos = {POSITION_NAMES[c]: int((y == c).sum()) for c in DISPLAY_ORDER}
        summary[s["label"]] = {"n_trials": int(X.shape[0]), "n_feat": int(X.shape[1]), "acc": accs,
                               "positions": [POSITION_NAMES[c] for c in DISPLAY_ORDER],
                               "recall_by_position": recall, "n_per_position": npos,
                               "n_nolick": int(Xnl.shape[0]), "acc_nolick": acc_nl,
                               "recall_nolick_by_position": recall_nl,
                               "confusion_all": cmn.tolist() if cmn is not None else None,
                               "source": args.source, "align": args.align}
        nlstr = (f" | NO-LICK(n={Xnl.shape[0]}) " + "  ".join(f"{g}={a:.2f}" for g, a in acc_nl.items())) if nl_ok \
            else f" | no-lick n={Xnl.shape[0]} (skipped: needs --align cue)"
        print(f"{s['label']}: engaged n={X.shape[0]} {args.source}feat={X.shape[1]} | "
              + "  ".join(f"{g}={a:.2f}" for g, a in accs.items()) + nlstr, flush=True)
        ax = axes[0][si]; im = ax.imshow(cmn, vmin=0, vmax=1, cmap="magma")
        labs = [POSITION_NAMES[c] for c in DISPLAY_ORDER]
        ax.set_xticks(range(6)); ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(6)); ax.set_yticklabels(labs, fontsize=7)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{s['label']}  acc={accs['all']:.2f} (chance .17)\nSSp={accs['SSp']:.2f} MO={accs['MO']:.2f}", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.7)
        if args.per_session:
            _save_session_fig(s["label"], cmn, summary[s["label"]], labs, args, tag=f"{args.source}_{args.align}_base-{args.baseline}_cv-{args.cv}")
    fig.suptitle(f"Spout-position decoding [{args.source}, {args.align}-aligned 0-{args.post_s:g}s, "
                 f"baseline={args.baseline}, {args.cv}-CV, engaged], {args.date}", fontsize=12)
    fig.tight_layout()
    tag = f"{args.source}_{args.align}_base-{args.baseline}_cv-{args.cv}"
    fig.savefig(args.output / f"locanmf_position_decoder_{args.date}_{tag}.png", dpi=130); plt.close(fig)

    # ---- per-position recall: engaged (5-fold) vs no-lick (trained on engaged) ----
    posnames = [POSITION_NAMES[c] for c in DISPLAY_ORDER]
    fig2, axes2 = plt.subplots(1, len(sess), figsize=(5 * len(sess), 4.2), squeeze=False, sharey=True)
    for si, s in enumerate(sess):
        ax = axes2[0][si]; sm = summary[s["label"]]; x = np.arange(6); w = 0.38
        ax.bar(x - w / 2, sm["recall_by_position"].get("all", [np.nan] * 6), w, color="tab:blue",
               label="engaged (5-fold)")
        ax.bar(x + w / 2, sm["recall_nolick_by_position"].get("all", [np.nan] * 6), w, color="tab:red",
               label=f"no-lick (n={sm['n_nolick']})")
        ax.axhline(1 / 6, color="grey", ls="--", lw=0.8, label="chance")
        ax.set_xticks(x); ax.set_xticklabels(posnames, rotation=45, ha="right", fontsize=7); ax.set_ylim(0, 1)
        nls = f"{sm['acc_nolick']['all']:.2f}" if not np.isnan(sm['acc_nolick']['all']) else "n/a"
        ax.set_title(f"{s['label']}  eng={sm['acc']['all']:.2f}  no-lick={nls}", fontsize=9)
        if si == 0:
            ax.set_ylabel("per-position recall"); ax.legend(fontsize=7)
    fig2.suptitle(f"Per-position recall — engaged vs no-lick [{tag}], {args.date} "
                  f"(no-lick = baseline disengagement here; post-stroke = failed attempts)", fontsize=11)
    fig2.tight_layout()
    fig2.savefig(args.output / f"locanmf_position_recall_{args.date}_{tag}.png", dpi=130); plt.close(fig2)

    (args.output / f"locanmf_position_decoder_{args.date}_{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote", args.output / f"locanmf_position_decoder_{args.date}_{tag}.png",
          "+ recall fig + summary", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
