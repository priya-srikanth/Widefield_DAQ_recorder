"""Robustness analysis for the position decoder (SEPARATE from the canonical pipeline; does not
change it). Two checks on whether 'junk' LocaNMF components or the |weight| importance metric
distort the picture:

 1. Component QC filter -- drop FOV-edge (center-of-mass within 8% of the brain bbox, e.g. MOB)
    and tiny-footprint (mass < 0.3x median) components, then compare decoding all-vs-filtered.
 2. Permutation-based region importance (block-CV held-out) -- the suppressor-robust answer to
    'which regions carry the decode', vs ranking by |multivariate LR weight| (which surfaces
    suppressors / z-scored noise; see F12 and the top-component discussion).

    python -m wfield_local.locanmf_decoder_qc --output "<dir>" --date 0603
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
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

from wfield_local.locanmf_cue_lick_analysis import SESSIONS
from wfield_local.locanmf_position_decoder import _trial_features

FS = 31.23


def _args():
    return SimpleNamespace(source="locanmf", align="lick", baseline="none", pre_s=1.0, post_s=2.0, fs=FS, max_rt=2.0)


def _pipe():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))


def _bcv(X, y, g):
    ng = min(5, int(np.unique(g).size))
    return accuracy_score(y, cross_val_predict(_pipe(), X, y, cv=GroupKFold(ng), groups=g))


def qc_flags(A, mask, edge_frac=0.08, tiny_frac=0.3):
    """keep-mask + per-component reason. Drop FOV-edge (CoM near bbox edge) and tiny footprints."""
    ys, xs = np.where(mask); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    my = edge_frac * (y1 - y0); mx = edge_frac * (x1 - x0)
    ncomp = A.shape[2]; mass = np.array([np.nansum(A[:, :, c]) for c in range(ncomp)]); med = np.median(mass)
    keep = np.ones(ncomp, bool); reason = [""] * ncomp
    for c in range(ncomp):
        fp = A[:, :, c]; tot = fp.sum()
        if tot <= 0:
            keep[c] = False; reason[c] = "empty"; continue
        cy = (fp.sum(1) @ np.arange(fp.shape[0])) / tot; cx = (fp.sum(0) @ np.arange(fp.shape[1])) / tot
        if cy < y0 + my or cy > y1 - my or cx < x0 + mx or cx > x1 - mx:
            keep[c] = False; reason[c] = "edge"
        elif mass[c] < tiny_frac * med:
            keep[c] = False; reason[c] = "tiny"
    return keep, reason, mass


def _load(label):
    s = next(x for x in SESSIONS if x["label"] == label); mc = s["mc"]
    A = np.load(f"{mc}/locanmf_affine8v1_final/{label}_locanmf_A.npy")
    ad = glob.glob(f"{mc}/wfield_local_results/allen_aligned_affine8v1")[0]
    mask = np.load(f"{ad}/allen_brain_mask_native_grid.npy").astype(bool)
    names = {int(k): v for k, v in json.load(open(f"{ad}/allen_area_names.json"))}
    X, y, g, _, _, reg = _trial_features(s, _args())
    return A, mask, names, X, y, g, reg


def run(labels, out):
    fig1, axes1 = plt.subplots(1, len(labels), figsize=(5.4 * len(labels), 5), squeeze=False)
    accs_all, accs_qc, drop = [], [], []
    for ai, label in enumerate(labels):
        A, mask, names, X, y, g, reg = _load(label)
        keep, reason, mass = qc_flags(A, mask)
        a_all = _bcv(X, y, g); a_qc = _bcv(X[:, keep], y, g)
        accs_all.append(a_all); accs_qc.append(a_qc); drop.append((int((~keep).sum()), len(keep)))
        print(f"{label}: dropped {int((~keep).sum())}/{len(keep)} (edge={sum(r=='edge' for r in reason)}, "
              f"tiny={sum(r=='tiny' for r in reason)})  acc {a_all:.2f} -> QC {a_qc:.2f}", flush=True)
        imp = np.zeros(X.shape[1]); gkf = GroupKFold(min(5, int(np.unique(g).size)))
        for tr, te in gkf.split(X, y, groups=g):
            pipe = _pipe().fit(X[tr], y[tr])
            imp += permutation_importance(pipe, X[te], y[te], n_repeats=8, random_state=0, scoring="accuracy").importances_mean
        imp /= gkf.get_n_splits()
        regs = np.array([names.get(int(reg[c]), "?") for c in range(X.shape[1])])
        reg_imp = {rr: imp[regs == rr].sum() for rr in set(regs)}
        topr = sorted(reg_imp, key=reg_imp.get, reverse=True)[:14]
        ax = axes1[0][ai]
        cols = ["#c0392b" if r.startswith(("SSp", "SSs")) else "#2980b9" if r.startswith("MO") else "#888" for r in topr]
        ax.barh(range(len(topr))[::-1], [reg_imp[r] for r in topr], color=cols)
        ax.set_yticks(range(len(topr))[::-1]); ax.set_yticklabels(topr, fontsize=7)
        ax.set_xlabel("permutation importance (acc drop, summed)"); ax.set_title(label, fontsize=11); ax.axvline(0, color="k", lw=0.6)
    fig1.suptitle("Region importance by PERMUTATION (block-CV held-out) — suppressor-robust. "
                  "Red=SSp/SSs, blue=MO, grey=other", fontsize=12)
    fig1.tight_layout(); p1 = out / "locanmf_region_permutation_importance.png"; fig1.savefig(p1, dpi=130); plt.close(fig1)

    fig2, ax = plt.subplots(figsize=(8, 5)); x = np.arange(len(labels)); w = 0.35
    ax.bar(x - w / 2, accs_all, w, label="all components", color="#888")
    ax.bar(x + w / 2, accs_qc, w, label="QC-filtered", color="#27ae60")
    for i, (nd, nt) in enumerate(drop):
        ax.text(i, max(accs_all[i], accs_qc[i]) + 0.02, f"-{nd}/{nt}", ha="center", fontsize=9)
    ax.axhline(1 / 6, color="k", ls="--", lw=0.8); ax.set_xticks(x); ax.set_xticklabels([l[:4] for l in labels])
    ax.set_ylim(0, 1); ax.set_ylabel("decoding accuracy (block-CV)"); ax.legend(fontsize=9)
    ax.set_title("QC filter (drop edge/tiny components): does removing 'junk' change decoding?")
    fig2.tight_layout(); p2 = out / "locanmf_qc_filter_comparison.png"; fig2.savefig(p2, dpi=130); plt.close(fig2)
    return p1, p2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--date", default="0603")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    labels = [s["label"] for s in SESSIONS if s["label"].endswith(args.date)]
    p1, p2 = run(labels, args.output)
    print("wrote", p1, "+", p2, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
