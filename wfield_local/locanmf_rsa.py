"""Representational Similarity Analysis (RSA) of spout-position coding, within & across animals.

Per session, each of the 6 spout positions gets a population activity pattern = mean LocaNMF component
activity over the 2 s post-first-lick window (no baseline; the same features the decoder uses). The
session RDM = 1 - Pearson correlation between the 6 position patterns (6x6, symmetric, diag 0). The RDM
is *basis-specific* (built in that session's own component space) but RSA is SECOND-ORDER and basis-FREE:
we compare RDMs (not raw patterns) across sessions/animals via Spearman correlation of their 15 unique
off-diagonal entries. This is exactly what makes cross-session / cross-animal comparison valid even
though LocaNMF returns a different component set per session.

Outputs:
  locanmf_rsa_sessions{tag}.png  -- session x session 2nd-order RSA matrix (ordered by animal) +
                                    within- vs across-animal RSA bar + animal x animal RDM-RSA.
  locanmf_rsa_rdms{tag}.png      -- per-animal mean RDM (6x6) heatmaps + grand-mean RDM.

    python -m wfield_local.locanmf_rsa --output "<dir>"
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wfield_local.locanmf_cue_lick_analysis import SESSIONS, ANIMAL_COLOR
from wfield_local.locanmf_position_decoder import _trial_features
from wfield_local.plot_lick_aligned_averages import POSITION_NAMES, DISPLAY_ORDER

FS = 31.23
POSN = [POSITION_NAMES[c] for c in DISPLAY_ORDER]


def _args():
    return SimpleNamespace(source="locanmf", align="lick", baseline="none", pre_s=1.0, post_s=2.0, fs=FS, max_rt=2.0)


def session_rdm(label):
    """6x6 representational dissimilarity matrix (1 - Pearson r between the 6 position activity patterns)."""
    s = next(x for x in SESSIONS if x["label"] == label)
    X, y, g, _, _, _ = _trial_features(s, _args())
    pos = np.array(DISPLAY_ORDER)
    P = np.vstack([X[y == p].mean(0) if (y == p).any() else np.full(X.shape[1], np.nan) for p in pos])
    R = np.corrcoef(P)
    return 1.0 - R


def _triu(m):
    iu = np.triu_indices(m.shape[0], 1)
    return m[iu]


def _rank(v):
    return np.argsort(np.argsort(v)).astype(float)


def _spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    return float(np.corrcoef(_rank(a[ok]), _rank(b[ok]))[0, 1])


def _collect(dates):
    labs = sorted([s["label"] for s in SESSIONS if dates is None or s["label"][-4:] in dates],
                  key=lambda l: (l[:4], l[-4:]))
    rdms = {}
    for l in labs:
        try:
            rdms[l] = session_rdm(l)
        except Exception as ex:
            print(f"  {l} skip: {str(ex)[:50]}", flush=True)
    return [l for l in labs if l in rdms], rdms


def fig_rsa_sessions(out, dates=None, tag=""):
    labs, rdms = _collect(dates)
    vecs = {l: _triu(rdms[l]) for l in labs}
    n = len(labs)
    S = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            S[i, j] = _spearman(vecs[labs[i]], vecs[labs[j]])
    animals = sorted({l[:4] for l in labs})
    aof = [l[:4] for l in labs]
    # within- vs across-animal second-order RSA
    wa, ac = {}, {}
    for a in animals:
        idx = [k for k in range(n) if aof[k] == a]
        oth = [k for k in range(n) if aof[k] != a]
        wpairs = [S[i, j] for ii, i in enumerate(idx) for j in idx[ii + 1:]]
        apairs = [S[i, j] for i in idx for j in oth]
        wa[a] = np.nanmean(wpairs) if wpairs else np.nan
        ac[a] = np.nanmean(apairs) if apairs else np.nan
    # animal-level RDM (mean of session RDMs) and animal x animal RSA
    ardm = {a: np.nanmean([rdms[l] for l in labs if l[:4] == a], axis=0) for a in animals}
    A = np.full((len(animals), len(animals)), np.nan)
    for i, ai in enumerate(animals):
        for j, aj in enumerate(animals):
            A[i, j] = _spearman(_triu(ardm[ai]), _triu(ardm[aj]))

    fig = plt.figure(figsize=(17, 6.4))
    # (1) session x session RSA matrix
    ax = fig.add_subplot(1, 3, 1)
    im = ax.imshow(S, cmap="viridis", vmin=np.nanpercentile(S, 5), vmax=1, aspect="auto")
    ax.set_xticks(range(n)); ax.set_xticklabels([f"{l[:4]} {l[-4:-2]}/{l[-2:]}" for l in labs], rotation=90, fontsize=6)
    ax.set_yticks(range(n)); ax.set_yticklabels([f"{l[:4]} {l[-4:-2]}/{l[-2:]}" for l in labs], fontsize=6)
    b = 0
    for a in animals[:-1]:
        b += sum(1 for x in aof if x == a)
        ax.axhline(b - 0.5, color="w", lw=1.4); ax.axvline(b - 0.5, color="w", lw=1.4)
    fig.colorbar(im, ax=ax, shrink=0.8); ax.set_title("Session x session RSA\n(Spearman of RDMs; blocks = animal)", fontsize=10)
    # (2) within vs across animal RSA
    ax = fig.add_subplot(1, 3, 2); x = np.arange(len(animals)); w = 0.38
    ax.bar(x - w / 2, [wa[a] for a in animals], w, label="within-animal", color="#2980b9")
    ax.bar(x + w / 2, [ac[a] for a in animals], w, label="across-animal", color="#c0392b")
    ax.set_xticks(x); ax.set_xticklabels(animals); ax.set_ylim(0, 1); ax.set_ylabel("mean 2nd-order RSA (Spearman)")
    ax.legend(fontsize=8); ax.set_title("Representational geometry:\nwithin- vs across-animal stability", fontsize=10)
    for i, a in enumerate(animals):
        ax.text(i, max(wa[a], ac[a]) + 0.02, f"{wa[a]-ac[a]:+.2f}", ha="center", fontsize=7)
    # (3) animal x animal RDM-RSA
    ax = fig.add_subplot(1, 3, 3)
    im = ax.imshow(A, cmap="viridis", vmin=np.nanpercentile(A, 5), vmax=1)
    ax.set_xticks(range(len(animals))); ax.set_xticklabels(animals); ax.set_yticks(range(len(animals))); ax.set_yticklabels(animals)
    for i in range(len(animals)):
        for j in range(len(animals)):
            ax.text(j, i, f"{A[i,j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if A[i, j] < 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8); ax.set_title("Animal x animal RDM similarity\n(mean RDM per animal)", fontsize=10)
    ttl = f"RSA of spout-position representational geometry{(' [' + tag + ']') if tag else ''} (n_sessions={n})"
    fig.suptitle(ttl, fontsize=13); fig.tight_layout()
    p = out / f"locanmf_rsa_sessions{('_' + tag) if tag else ''}.png"; fig.savefig(p, dpi=130); plt.close(fig)
    print(f"RSA within vs across animal (Spearman of RDMs){(' [' + tag + ']') if tag else ''}:", flush=True)
    for a in animals:
        print(f"  {a}: within={wa[a]:.2f}  across={ac[a]:.2f}  (within-across={wa[a]-ac[a]:+.2f})", flush=True)
    return p


def fig_rsa_rdms(out, dates=None, tag=""):
    labs, rdms = _collect(dates)
    animals = sorted({l[:4] for l in labs})
    ardm = {a: np.nanmean([rdms[l] for l in labs if l[:4] == a], axis=0) for a in animals}
    grand = np.nanmean([rdms[l] for l in labs], axis=0)
    mats = [(a, ardm[a]) for a in animals] + [("GRAND", grand)]
    vmax = np.nanpercentile([m for _, M in mats for m in _triu(M)], 98)
    fig, axes = plt.subplots(1, len(mats), figsize=(3.0 * len(mats), 3.4))
    for ax, (name, M) in zip(axes, mats):
        im = ax.imshow(M, cmap="magma", vmin=0, vmax=vmax)
        ax.set_xticks(range(6)); ax.set_xticklabels(POSN, rotation=90, fontsize=6)
        ax.set_yticks(range(6)); ax.set_yticklabels(POSN if name == animals[0] else [], fontsize=6)
        ax.set_title(name, fontsize=10)
    fig.colorbar(im, ax=axes.tolist(), shrink=0.7, label="dissimilarity (1 - r)")
    fig.suptitle(f"Mean representational dissimilarity matrix per animal{(' [' + tag + ']') if tag else ''} "
                 "(spout-position geometry; first-lick 2s)", fontsize=12)
    p = out / f"locanmf_rsa_rdms{('_' + tag) if tag else ''}.png"; fig.savefig(p, dpi=130); plt.close(fig)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--dates", default="", help="comma-separated MMDD subset (default all)")
    ap.add_argument("--tag", default="", help="filename tag")
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    dates = set(args.dates.split(",")) if args.dates else None
    print("wrote", fig_rsa_sessions(args.output, dates, args.tag).name, flush=True)
    print("wrote", fig_rsa_rdms(args.output, dates, args.tag).name, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
