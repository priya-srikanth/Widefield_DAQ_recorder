"""Cross-mouse comparison of position decoding + encoding (all sessions pooled per mouse), to look for
SYSTEMATIC differences in the cortical representation of movement / motor planning across mice.

Motivated by PS93's RIGHT orofacial deficit (tongue deviates right, minimal right whisking). Orofacial
movement is represented CONTRALATERALLY, so a right-side deficit predicts altered LEFT-hemisphere
representation and/or abnormal decoding of RIGHT-spout licks. We therefore quantify, per mouse:
  - overall + per-position decoding (first-lick 2s, block-CV) and laterality (L/center/R),
  - LEFT-spout vs RIGHT-spout decodability (movement-side asymmetry),
  - SSp-left-only vs SSp-right-only region decoding (cortical-hemisphere asymmetry),
  - encoding explained variance per position.
Per-session metrics are averaged within mouse (animal = unit of replication).

    python -m wfield_local.locanmf_cross_mouse --output "<dir>"
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix

from wfield_local.locanmf_cue_lick_analysis import SESSIONS
from wfield_local.locanmf_position_decoder import _trial_features
from wfield_local.plot_lick_aligned_averages import POSITION_NAMES, DISPLAY_ORDER

FS = 31.23
POSN = [POSITION_NAMES[c] for c in DISPLAY_ORDER]
LAT = {c: (0 if POSITION_NAMES[c].endswith("_L") else 2 if POSITION_NAMES[c].endswith("_R") else 1) for c in DISPLAY_ORDER}
LSPOUT = [c for c in DISPLAY_ORDER if POSITION_NAMES[c].endswith("_L")]
RSPOUT = [c for c in DISPLAY_ORDER if POSITION_NAMES[c].endswith("_R")]


def _args():
    return SimpleNamespace(source="locanmf", align="lick", baseline="none", pre_s=1.0, post_s=2.0, fs=FS, max_rt=2.0)


def _clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))


def _cvpred(X, y, g):
    ng = min(5, int(np.unique(g).size))
    return cross_val_predict(_clf(), X, y, cv=GroupKFold(ng), groups=g)


def _acc(X, y, g):
    if X.shape[1] == 0 or len(y) < 12:
        return np.nan
    return accuracy_score(y, _cvpred(X, y, g))


def per_session(label):
    s = next(x for x in SESSIONS if x["label"] == label)
    X, y, g, _, _, reg = _trial_features(s, _args())
    names = {int(k): v for k, v in json.load(open(glob.glob(
        f"{s['mc']}/wfield_local_results/allen_aligned_affine8v1/allen_area_names.json")[0]))}
    rn = np.array([names.get(int(reg[i]), "?") for i in range(X.shape[1])])
    pred = _cvpred(X, y, g); cm = confusion_matrix(y, pred, labels=DISPLAY_ORDER)
    recall = np.diag(cm) / np.maximum(cm.sum(1), 1)
    # laterality 3-way recall
    ylat = np.array([LAT[c] for c in y]); platp = _cvpred(X, ylat, g)
    clat = confusion_matrix(ylat, platp, labels=[0, 1, 2]); lat_recall = np.diag(clat) / np.maximum(clat.sum(1), 1)
    # hemisphere SSp-only
    sspL = np.array([i for i in range(X.shape[1]) if rn[i].startswith("SSp") and rn[i].endswith("_left")])
    sspR = np.array([i for i in range(X.shape[1]) if rn[i].startswith("SSp") and rn[i].endswith("_right")])
    accL = _acc(X[:, sspL], y, g) if sspL.size else np.nan
    accR = _acc(X[:, sspR], y, g) if sspR.size else np.nan
    # encoding EV per position (held-out R^2 restricted to each position's trials)
    P = np.stack([(y == p).astype(float) for p in DISPLAY_ORDER], 1); prd = np.zeros_like(X); ng = min(5, int(np.unique(g).size))
    for tr, te in GroupKFold(ng).split(X, y, g):
        prd[te] = Ridge(alpha=1.0).fit(P[tr], X[tr]).predict(P[te])
    xbar = X.mean(0); ev = []
    for p in DISPLAY_ORDER:
        m = y == p; tot = ((X[m] - xbar) ** 2).sum()
        ev.append(1 - ((X[m] - prd[m]) ** 2).sum() / max(tot, 1e-12))
    return dict(acc=accuracy_score(y, pred), recall=recall, lat_recall=lat_recall,
                ssp_left=accL, ssp_right=accR, ev=np.array(ev),
                Lrecall=np.nanmean([recall[DISPLAY_ORDER.index(c)] for c in LSPOUT]),
                Rrecall=np.nanmean([recall[DISPLAY_ORDER.index(c)] for c in RSPOUT]))


def fig_cross_mouse(out):
    by_mouse = defaultdict(list)
    for s in SESSIONS:
        by_mouse[s["label"][:4]].append(s["label"])
    mice = sorted(by_mouse)
    M = {}
    for mouse in mice:
        rows = []
        for lab in by_mouse[mouse]:
            try:
                rows.append(per_session(lab))
            except Exception as ex:
                print(f"  {lab} skip: {str(ex)[:50]}", flush=True)
        if rows:
            M[mouse] = rows
    mice = [m for m in mice if m in M]

    def agg(mouse, key):
        return np.nanmean([r[key] for r in M[mouse]], axis=0)

    def _ps(mouse, fn):
        """Per-session scalar values (finite only) for a mouse."""
        v = np.array([fn(r) for r in M[mouse]], float)
        return v[np.isfinite(v)]

    def barpts(ax, xs, fn, color, label):
        """Bar = mean across that mouse's sessions, error bar = SEM, + individual session points
        (deterministic jitter, alpha for transparency). xs = per-mouse x positions (aligned with `mice`)."""
        means, sems = [], []
        for xi, m in zip(xs, mice):
            v = _ps(m, fn)
            means.append(v.mean() if v.size else np.nan)
            sems.append(v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0)
            if v.size:
                off = np.linspace(-0.11, 0.11, v.size) if v.size > 1 else np.array([0.0])
                ax.scatter(xi + off, v, s=13, color=color, edgecolor="white", linewidth=0.4, alpha=0.5, zorder=4)
        ax.bar(xs, means, 0.4, yerr=sems, capsize=3, color=color, label=label, alpha=0.85,
               error_kw=dict(lw=1.0, ecolor="#222", zorder=3))

    fig = plt.figure(figsize=(17, 10)); x = np.arange(len(mice))
    # (A) overall + laterality: mean +- SEM across sessions + session points
    ax = fig.add_subplot(2, 3, 1)
    barpts(ax, x - 0.2, lambda r: r["acc"], "#34495e", "6-way")
    barpts(ax, x + 0.2, lambda r: np.nanmean(r["lat_recall"]), "#16a085", "laterality (3-way)")
    ax.axhline(1 / 6, color="grey", ls="--", lw=0.8); ax.axhline(1 / 3, color="purple", ls=":", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(mice); ax.set_ylim(0, 1); ax.set_ylabel("decoding accuracy"); ax.legend(fontsize=8)
    ax.set_title("Overall decoding per mouse (mean +- SEM, points = sessions)")
    # (B) per-position recall heatmap
    ax = fig.add_subplot(2, 3, 2)
    R = np.array([agg(m, "recall") for m in mice])
    im = ax.imshow(R, cmap="magma", vmin=0, vmax=R.max(), aspect="auto")
    ax.set_xticks(range(6)); ax.set_xticklabels(POSN, rotation=45, ha="right", fontsize=7); ax.set_yticks(x); ax.set_yticklabels(mice)
    ax.set_title("Per-position recall (mouse x position)"); fig.colorbar(im, ax=ax, shrink=0.8)
    # (C) LEFT vs RIGHT spout decodability -- movement-side asymmetry
    ax = fig.add_subplot(2, 3, 3)
    barpts(ax, x - 0.2, lambda r: r["Lrecall"], "#2980b9", "LEFT spouts")
    barpts(ax, x + 0.2, lambda r: r["Rrecall"], "#c0392b", "RIGHT spouts")
    ax.set_xticks(x); ax.set_xticklabels(mice); ax.set_ylim(0, 1); ax.set_ylabel("mean per-position recall"); ax.legend(fontsize=8)
    ax.set_title("Left vs Right spout decodability (mean +- SEM, points = sessions)")
    # (D) cortical hemisphere: SSp-left-only vs SSp-right-only
    ax = fig.add_subplot(2, 3, 4)
    barpts(ax, x - 0.2, lambda r: r["ssp_left"], "#8e44ad", "SSp-LEFT only")
    barpts(ax, x + 0.2, lambda r: r["ssp_right"], "#e67e22", "SSp-RIGHT only")
    ax.axhline(1 / 6, color="grey", ls="--", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(mice); ax.set_ylim(0, 1)
    ax.set_ylabel("region-only decoding accuracy"); ax.legend(fontsize=8)
    ax.set_title("Cortical hemisphere: SSp-left vs SSp-right (mean +- SEM, points = sessions)")
    # (E) encoding EV per position heatmap
    ax = fig.add_subplot(2, 3, 5)
    E = np.array([agg(m, "ev") for m in mice]); vmax = np.nanpercentile(np.abs(E), 98)
    im = ax.imshow(E, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(6)); ax.set_xticklabels(POSN, rotation=45, ha="right", fontsize=7); ax.set_yticks(x); ax.set_yticklabels(mice)
    ax.set_title("Encoding explained variance per position"); fig.colorbar(im, ax=ax, shrink=0.8)
    # (F) L/R asymmetry index summary
    ax = fig.add_subplot(2, 3, 6)
    barpts(ax, x - 0.2, lambda r: r["Lrecall"] - r["Rrecall"], "#c0392b", "L-spout minus R-spout recall")
    barpts(ax, x + 0.2, lambda r: r["ssp_left"] - r["ssp_right"], "#8e44ad", "SSp-left minus SSp-right acc")
    ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x); ax.set_xticklabels(mice); ax.set_ylabel("asymmetry (L - R)")
    ax.legend(fontsize=7); ax.set_title("L/R asymmetry indices, mean +- SEM + sessions (PS93 = right orofacial deficit)")
    nsess = {m: len(M[m]) for m in mice}
    fig.suptitle(f"Cross-mouse cortical representation of spout position (all sessions; n/mouse={nsess})", fontsize=13)
    fig.tight_layout(); p = out / "locanmf_cross_mouse_comparison.png"; fig.savefig(p, dpi=130); plt.close(fig)
    for m in mice:
        print(f"  {m}: acc={agg(m,'acc'):.2f} Lrec={agg(m,'Lrecall'):.2f} Rrec={agg(m,'Rrecall'):.2f} "
              f"SSpL={agg(m,'ssp_left'):.2f} SSpR={agg(m,'ssp_right'):.2f}", flush=True)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    print("wrote", fig_cross_mouse(args.output).name, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
