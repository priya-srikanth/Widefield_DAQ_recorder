"""Persist per-session position decoders and apply a FROZEN decoder across days — the
confirmatory arm of the stroke study (train pre-stroke, apply post-stroke).

Two transfer paths for the cross-day problem (LocaNMF components are session-specific):
  - ROI (default, registration-free): Allen-region features are atlas-anchored, so a decoder
    trained on day A applies directly to day B (features aligned by region label).
  - Fixed-A / refit-C (scaffold, for the LocaNMF basis): keep pre-stroke footprints A_ref and
    project a new session onto them, C_new = pinv(A_ref) @ (U_new @ SVT_new), on the shared
    Allen-aligned grid. Valid because the stroke is subcortical (cortical map intact).

    python -m wfield_local.locanmf_frozen_decoder --save            # save all baseline decoders
    python -m wfield_local.locanmf_frozen_decoder --transfer --output "<dir>"   # cross-day ROI demo
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix

from wfield_local.locanmf_cue_lick_analysis import SESSIONS
from wfield_local.locanmf_position_decoder import _trial_features
from wfield_local.plot_lick_aligned_averages import POSITION_NAMES, DISPLAY_ORDER

FS = 31.23
BASELINE_DAYS = ("0601", "0602", "0603", "0604")


def _sess(label):
    return next(s for s in SESSIONS if s["label"] == label)


def _args(source="locanmf", align="lick", post_s=2.0, baseline="none"):
    return SimpleNamespace(source=source, align=align, baseline=baseline,
                           pre_s=1.0, post_s=post_s, fs=FS, max_rt=2.0)


def _pipe():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))


def _decoder_dir(s):
    d = Path(s["mc"]) / "decoder_affine8v1"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_session_decoder(label, source="locanmf", align="lick", post_s=2.0):
    """Fit on ALL engaged trials and persist the pipeline + metadata to MICROSCOPE."""
    s = _sess(label)
    X, y, g, Xnl, ynl, feat_reg = _trial_features(s, _args(source, align, post_s))
    pipe = _pipe().fit(X, y)
    # honest block-CV accuracy for the record
    ng = min(5, int(np.unique(g).size))
    cv_acc = accuracy_score(y, cross_val_predict(_pipe(), X, y, cv=GroupKFold(ng), groups=g)) if ng >= 2 else float("nan")
    dd = _decoder_dir(s)
    stem = f"{label}_decoder_{source}_{align}_2s"
    joblib.dump({"pipeline": pipe, "feat_region": feat_reg, "classes": pipe.named_steps["logisticregression"].classes_,
                 "source": source, "align": align, "post_s": post_s, "baseline": "none"}, dd / f"{stem}.joblib")
    meta = {"label": label, "source": source, "align": align, "post_s": post_s, "cv_block_accuracy": cv_acc,
            "n_engaged": int(X.shape[0]), "n_features": int(X.shape[1]), "n_nolick": int(Xnl.shape[0]),
            "positions": [POSITION_NAMES[c] for c in DISPLAY_ORDER], "chance": 1 / 6,
            "feature_region_labels": [int(r) for r in feat_reg]}
    (dd / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    return dd / f"{stem}.joblib", cv_acc


def _aligned(Xtr, rtr, Xte, rte):
    """Subset two ROI feature matrices to their common region labels, in a shared order."""
    common = [r for r in rtr if r in set(rte)]
    itr = [list(rtr).index(r) for r in common]; ite = [list(rte).index(r) for r in common]
    return Xtr[:, itr], Xte[:, ite]


def cross_day_transfer(train_label, test_label, source="roi", align="lick", post_s=2.0):
    """Train a frozen decoder on train_label, apply to test_label (ROI: registration-free)."""
    Xtr, ytr, _, _, _, rtr = _trial_features(_sess(train_label), _args(source, align, post_s))
    Xte, yte, _, Xnl, ynl, rte = _trial_features(_sess(test_label), _args(source, align, post_s))
    if source == "roi":
        Xtr, Xte = _aligned(Xtr, rtr, Xte, rte)
    elif Xtr.shape[1] != Xte.shape[1]:
        raise ValueError("LocaNMF bases differ across sessions; use source='roi' or the fixed-A path.")
    pipe = _pipe().fit(Xtr, ytr)
    return accuracy_score(yte, pipe.predict(Xte))


def project_C_fixed_A(a_ref_label, new_label):
    """SCAFFOLD (post-stroke LocaNMF path): project a new session onto fixed pre-stroke footprints.
    C_new = pinv(A_ref) @ (U_new @ SVT_new) on the shared Allen-aligned grid. Returns C_new so a
    frozen LocaNMF decoder (trained on A_ref's components) can be applied to the new session.
    Requires both sessions on the same affine8v1 pixel grid; not exercised until post-stroke data."""
    ref = _sess(a_ref_label); new = _sess(new_label)
    A = np.load(f"{ref['mc']}/locanmf_affine8v1_final/{a_ref_label}_locanmf_A.npy")  # (npix, ncomp)
    ad = glob.glob(f"{new['mc']}/wfield_local_results/allen_aligned_affine8v1")[0]
    U = np.load(f"{ad}/U_atlas.npy"); SVT = np.load(f"{new['mc']}/wfield_local_results/SVTcorr.npy")
    dff = U.reshape(-1, U.shape[2]) @ SVT                    # (npix, T) on the shared grid
    A2 = A.reshape(dff.shape[0], -1)
    return np.linalg.pinv(A2) @ dff                          # (ncomp, T) = refit C on the frozen basis


def _transfer_matrix_fig(out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), squeeze=False)
    posn = [POSITION_NAMES[c] for c in DISPLAY_ORDER]
    for ax, an in zip(axes[0], ("PS92", "PS94", "PS95")):
        days = [s["label"] for s in SESSIONS if s["label"].startswith(an) and s["label"][-4:] in BASELINE_DAYS]
        n = len(days); M = np.full((n, n), np.nan)
        for i, tr in enumerate(days):
            for j, te in enumerate(days):
                try:
                    if i == j:  # within-day: honest block-CV
                        X, y, g, _, _, r = _trial_features(_sess(tr), _args("roi", "lick", 2.0))
                        ng = min(5, int(np.unique(g).size))
                        M[i, j] = accuracy_score(y, cross_val_predict(_pipe(), X, y, cv=GroupKFold(ng), groups=g))
                    else:
                        M[i, j] = cross_day_transfer(tr, te, source="roi")
                except Exception as e:
                    print(f"  {an} {tr}->{te} failed: {str(e)[:50]}")
        im = ax.imshow(M, vmin=1 / 6, vmax=0.9, cmap="viridis")
        for i in range(n):
            for j in range(n):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                            color="w" if M[i, j] < 0.6 else "k", fontsize=10)
        dl = [d[-4:-2] + "/" + d[-2:] for d in days]
        ax.set_xticks(range(n)); ax.set_xticklabels(dl, fontsize=8); ax.set_yticks(range(n)); ax.set_yticklabels(dl, fontsize=8)
        ax.set_xlabel("test day"); ax.set_ylabel("train day"); ax.set_title(an, fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle("Frozen ROI decoder cross-day transfer (diagonal = within-day block-CV; off-diagonal = train->test). "
                 "The off-diagonal drop is the baseline cost of freezing a decoder across days.", fontsize=10.5)
    fig.tight_layout(); p = out / "locanmf_frozen_decoder_crossday_roi.png"; fig.savefig(p, dpi=130); plt.close(fig)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save", action="store_true", help="fit + persist decoders for all baseline sessions to MICROSCOPE")
    ap.add_argument("--transfer", action="store_true", help="cross-day ROI transfer demo + figure")
    ap.add_argument("--output", type=Path, default=Path("."))
    args = ap.parse_args()
    labels = [s["label"] for s in SESSIONS if s["label"][-4:] in BASELINE_DAYS]
    if args.save:
        for lab in labels:
            for source in ("locanmf", "roi"):
                try:
                    p, acc = save_session_decoder(lab, source=source)
                    print(f"saved {p.name}  (block-CV acc={acc:.2f})", flush=True)
                except Exception as e:
                    print(f"FAILED {lab} {source}: {str(e)[:60]}", flush=True)
    if args.transfer:
        args.output.mkdir(parents=True, exist_ok=True)
        print("wrote", _transfer_matrix_fig(args.output), flush=True)
    if not (args.save or args.transfer):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
