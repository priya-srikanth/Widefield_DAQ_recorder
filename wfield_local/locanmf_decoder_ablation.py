"""Region-importance for the position decoder by ABLATION (the principled alternative to univariate/
|weight| ranking). Two figures:
  fig_region_ablation: per Allen region, region-only accuracy (sufficiency) + leave-one-region-out drop
    (necessity), and by region GROUP with leave-group-out (handles within-group redundancy).
  fig_grouped_ablation: unilateral SENSORY (SSp/SSs L,R) and MOTOR (MOp/MOs L,R) -- region-only,
    leave-ONE-group-out, leave-TWO-groups-out (reveals necessity hidden by cross-hemisphere redundancy).
All block-CV, LocaNMF, no-baseline, first-lick 2s.

    python -m wfield_local.locanmf_decoder_ablation --date 0603 --output "<dir>"
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score

from wfield_local.locanmf_cue_lick_analysis import SESSIONS
from wfield_local.locanmf_position_decoder import _trial_features

FS = 31.23


def _args():
    return SimpleNamespace(source="locanmf", align="lick", baseline="none", pre_s=1.0, post_s=2.0, fs=FS, max_rt=2.0)


def _acc(X, y, g):
    if X.shape[1] == 0:
        return np.nan
    ng = min(5, int(np.unique(g).size))
    return accuracy_score(y, cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)),
                                               X, y, cv=GroupKFold(ng), groups=g))


def _group(rn):
    if rn.startswith(("SSp", "SSs")):
        return "SSp/SSs"
    if rn.startswith("MO"):
        return "MOp/MOs"
    if rn.startswith("VIS"):
        return "Visual"
    if rn.startswith("AUD"):
        return "Auditory"
    if rn.startswith("RSP"):
        return "Retrosplenial"
    if rn.startswith(("PL", "FRP", "ORB", "ACA", "ILA")):
        return "Frontal/Cing"
    return "Other"


def _load(label):
    s = next(x for x in SESSIONS if x["label"] == label)
    X, y, g, _, _, reg = _trial_features(s, _args())
    names = {int(k): v for k, v in json.load(open(glob.glob(
        f"{s['mc']}/wfield_local_results/allen_aligned_affine8v1/allen_area_names.json")[0]))}
    rn = np.array([names.get(int(reg[i]), "?") for i in range(X.shape[1])])
    return X, y, g, rn


def fig_region_ablation(label, out):
    X, y, g, rn = _load(label); grp = np.array([_group(r) for r in rn]); full = _acc(X, y, g)
    regs = [r for r in sorted(set(rn)) if (rn == r).sum() >= 2]
    only_r = {r: _acc(X[:, rn == r], y, g) for r in regs}
    loo_r = {r: full - _acc(X[:, rn != r], y, g) for r in regs}
    only_g = {gg: _acc(X[:, grp == gg], y, g) for gg in sorted(set(grp))}
    loo_g = {gg: full - _acc(X[:, grp != gg], y, g) for gg in sorted(set(grp))}
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]; gg = sorted(only_g, key=only_g.get, reverse=True); x = np.arange(len(gg)); w = 0.4
    ax.bar(x - w / 2, [only_g[k] for k in gg], w, label="region-only (sufficiency)", color="#2980b9")
    ax.bar(x + w / 2, [loo_g[k] for k in gg], w, label="leave-group-out drop (necessity)", color="#c0392b")
    ax.axhline(1 / 6, color="k", ls="--", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(gg, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("accuracy / drop"); ax.set_title(f"By region GROUP (full={full:.2f})"); ax.legend(fontsize=9)
    ax = axes[1]; top = sorted(regs, key=only_r.get, reverse=True)[:16]
    cols = ["#c0392b" if r.startswith(("SSp", "SSs")) else "#2980b9" if r.startswith("MO") else "#888" for r in top]
    ax.barh(range(len(top))[::-1], [only_r[r] for r in top], 0.4, color=cols, label="region-only")
    ax.barh([t + 0.4 for t in range(len(top))][::-1], [loo_r[r] for r in top], 0.4, color="k", alpha=0.5, label="leave-one-out drop")
    ax.set_yticks(range(len(top))[::-1]); ax.set_yticklabels(top, fontsize=8); ax.axvline(1 / 6, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("accuracy (bar) / drop (dark)"); ax.set_title("Top regions by region-only accuracy"); ax.legend(fontsize=8)
    fig.suptitle(f"{label}: REGION IMPORTANCE by ablation (block-CV). region-only=sufficiency, leave-out=unique. chance=0.17", fontsize=12)
    fig.tight_layout(); p = out / f"locanmf_region_ablation_{label}.png"; fig.savefig(p, dpi=130); plt.close(fig)
    return p


def _gh(rn):
    h = "L" if rn.endswith("_left") else "R" if rn.endswith("_right") else "?"
    if rn.startswith(("SSp", "SSs")):
        return f"SEN_{h}"
    if rn.startswith(("MOp", "MOs")):
        return f"MOT_{h}"
    return "OTHER"


def fig_grouped_ablation(labels, out, tag):
    fig, axes = plt.subplots(1, len(labels), figsize=(7.5 * len(labels), 5.6), squeeze=False)
    G = ["SEN_L", "SEN_R", "MOT_L", "MOT_R"]
    pairs = [("SEN_L", "SEN_R"), ("MOT_L", "MOT_R"), ("SEN_L", "MOT_L"), ("SEN_R", "MOT_R"), ("SEN_L", "MOT_R"), ("SEN_R", "MOT_L")]
    for ax, label in zip(axes[0], labels):
        X, y, g, rn = _load(label); gr = np.array([_gh(r) for r in rn]); full = _acc(X, y, g)
        def msk(groups):
            return np.isin(gr, groups)
        loo = {k: full - _acc(X[:, ~msk([k])], y, g) for k in G}
        ltwo = {f"{a}+{b}": full - _acc(X[:, ~msk([a, b])], y, g) for a, b in pairs}
        labs = G + list(ltwo); vals = [loo[k] for k in G] + [ltwo[k] for k in ltwo]
        cols = ["#c0392b", "#e74c3c", "#2980b9", "#5dade2"] + ["#7d3c98"] * 2 + ["#138d75"] * 2 + ["#b9770e"] * 2
        ax.bar(range(len(labs)), vals, color=cols); ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=55, ha="right", fontsize=8)
        ax.set_ylabel("accuracy drop (full - ablated)")
        ax.set_title(f"{label} full={full:.2f}\nleave-ONE (red/blue) vs leave-TWO (purple=bilateral, green=hemi, orange=crossed)", fontsize=9)
    fig.suptitle("Unilateral SENSORY/MOTOR ablation: leave-one vs leave-two-out (necessity hidden by redundancy)", fontsize=12)
    fig.tight_layout(); p = out / "locanmf_ablation_grouped_unilateral.png"; fig.savefig(p, dpi=130); plt.close(fig)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--date", default="0603")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    labs = [s["label"] for s in SESSIONS if s["label"].endswith(args.date)]
    for lab in labs:
        print("wrote", fig_region_ablation(lab, args.output).name, flush=True)
    print("wrote", fig_grouped_ablation(labs, args.output, args.date).name, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
