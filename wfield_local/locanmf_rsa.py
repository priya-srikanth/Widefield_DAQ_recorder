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
import glob
import json
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


def _rdm_from(X, y, mask=None):
    pos = np.array(DISPLAY_ORDER)
    sel = np.ones(len(y), bool) if mask is None else mask
    P = np.vstack([X[sel & (y == p)].mean(0) if (sel & (y == p)).sum() >= 2 else np.full(X.shape[1], np.nan)
                   for p in pos])
    return 1.0 - np.corrcoef(P)


def session_rdm(label):
    """6x6 representational dissimilarity matrix (1 - Pearson r between the 6 position activity patterns)."""
    s = next(x for x in SESSIONS if x["label"] == label)
    X, y, _, _, _, _ = _trial_features(s, _args())
    return _rdm_from(X, y)


def _rdm_and_reliability(label):
    """Full-data RDM + split-half reliability: split BLOCKS by parity, build an RDM from each half, and
    Spearman-correlate them. That reliability is the per-session NOISE CEILING -- the maximum 2nd-order RSA
    any other RDM (another session of the same animal) could reach given this session's estimation noise."""
    s = next(x for x in SESSIONS if x["label"] == label)
    X, y, g, _, _, _ = _trial_features(s, _args())
    full = _rdm_from(X, y)
    ub = np.unique(g); half = set(ub[::2].tolist())
    mA = np.array([gi in half for gi in g]); mB = ~mA
    rel = _spearman(_triu(_rdm_from(X, y, mA)), _triu(_rdm_from(X, y, mB)))
    return full, rel


def _crossnobis_rdm(X, y, g):
    """Cross-validated (crossnobis) RDM: noise-UNBIASED dissimilarities. Diagonal noise-whitening, then
    for every ordered pair of block-folds (a,b): D[p,q] += <(mu_p^a - mu_q^a), (mu_p^b - mu_q^b)>_prec.
    Independent-fold noise cross-multiplies to ~0 -> identical positions give ~0 (can go negative), unlike
    the positively-biased 1-corr RDM. 6x6, symmetric, diag 0."""
    pos = np.array(DISPLAY_ORDER); P = len(pos)
    num = np.zeros(X.shape[1]); den = 0
    for p in pos:
        m = y == p
        if m.sum() >= 2:
            num += ((X[m] - X[m].mean(0)) ** 2).sum(0); den += int(m.sum()) - 1
    var = num / max(den, 1); prec = 1.0 / np.maximum(var, np.nanmedian(var) * 1e-3)
    ub = np.unique(g); k = min(4, len(ub))
    if k < 2:
        return np.full((P, P), np.nan)
    foldid = np.array([int(np.where(ub == gi)[0][0]) % k for gi in g])
    mean = {(f, ip): (X[(foldid == f) & (y == p)].mean(0) if ((foldid == f) & (y == p)).sum() >= 1 else None)
            for f in range(k) for ip, p in enumerate(pos)}
    D = np.zeros((P, P)); C = np.zeros((P, P))
    for fa in range(k):
        for fb in range(k):
            if fa == fb:
                continue
            for ip in range(P):
                for iq in range(ip + 1, P):
                    a, b, c, d = mean[(fa, ip)], mean[(fa, iq)], mean[(fb, ip)], mean[(fb, iq)]
                    if a is None or b is None or c is None or d is None:
                        continue
                    v = float(np.sum((a - b) * prec * (c - d)))
                    D[ip, iq] += v; D[iq, ip] += v; C[ip, iq] += 1; C[iq, ip] += 1
    out = np.full((P, P), np.nan); nz = C > 0; out[nz] = D[nz] / C[nz]; np.fill_diagonal(out, 0.0)
    return out


def _crossnobis_and_reliability(label):
    s = next(x for x in SESSIONS if x["label"] == label)
    X, y, g, _, _, _ = _trial_features(s, _args())
    full = _crossnobis_rdm(X, y, g)
    ub = np.unique(g); A = set(ub[::2].tolist()); mA = np.array([gi in A for gi in g])
    rel = _spearman(_triu(_crossnobis_rdm(X[mA], y[mA], g[mA])), _triu(_crossnobis_rdm(X[~mA], y[~mA], g[~mA])))
    return full, rel


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
    rdms, rels = {}, {}
    for l in labs:
        try:
            rdms[l], rels[l] = _rdm_and_reliability(l)
        except Exception as ex:
            print(f"  {l} skip: {str(ex)[:50]}", flush=True)
    return [l for l in labs if l in rdms], rdms, rels


def fig_rsa_sessions(out, dates=None, tag=""):
    labs, rdms, rels = _collect(dates)
    vecs = {l: _triu(rdms[l]) for l in labs}
    n = len(labs)
    S = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            S[i, j] = _spearman(vecs[labs[i]], vecs[labs[j]])
    animals = sorted({l[:4] for l in labs})
    aof = [l[:4] for l in labs]
    # within- vs across-animal second-order RSA, and split-half noise ceiling per animal
    wa, ac, nc = {}, {}, {}
    for a in animals:
        idx = [k for k in range(n) if aof[k] == a]
        oth = [k for k in range(n) if aof[k] != a]
        wpairs = [S[i, j] for ii, i in enumerate(idx) for j in idx[ii + 1:]]
        apairs = [S[i, j] for i in idx for j in oth]
        wa[a] = np.nanmean(wpairs) if wpairs else np.nan
        ac[a] = np.nanmean(apairs) if apairs else np.nan
        nc[a] = np.nanmean([rels[labs[k]] for k in idx])
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
    for i, a in enumerate(animals):                       # split-half noise ceiling (grey caps)
        ax.plot([i - w, i + w / 2 + w / 2], [nc[a], nc[a]], color="#555", ls="--", lw=1.2,
                label="noise ceiling (split-half)" if i == 0 else None)
        frac = wa[a] / nc[a] if nc[a] and np.isfinite(nc[a]) else np.nan
        ax.text(i - w / 2, wa[a] + 0.02, f"{100*frac:.0f}%" if np.isfinite(frac) else "", ha="center", fontsize=7, color="#2980b9")
    ax.set_xticks(x); ax.set_xticklabels(animals); ax.set_ylim(0, 1); ax.set_ylabel("mean 2nd-order RSA (Spearman)")
    ax.legend(fontsize=7); ax.set_title("Within- vs across-animal stability\n(% = within / noise ceiling)", fontsize=10)
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
        frac = wa[a] / nc[a] if nc[a] else np.nan
        print(f"  {a}: within={wa[a]:.2f} across={ac[a]:.2f} ceiling={nc[a]:.2f} "
              f"within/ceiling={frac:.0%} (within-across={wa[a]-ac[a]:+.2f})", flush=True)
    return p


def fig_rsa_rdms(out, dates=None, tag=""):
    labs, rdms, _ = _collect(dates)
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


def _rel_within(Xh, y, g):
    """Split-half (block-parity) RDM reliability within a column subset (one hemisphere)."""
    ub = np.unique(g); half = set(ub[::2].tolist())
    mA = np.array([gi in half for gi in g])
    return _spearman(_triu(_rdm_from(Xh, y, mA)), _triu(_rdm_from(Xh, y, ~mA)))


def _session_hemi(label):
    """RDM_L, RDM_R (from left- vs right-hemisphere LocaNMF components), per-hemisphere split-half
    reliability, and component counts. Hemisphere from the Allen region name suffix (_left/_right)."""
    s = next(x for x in SESSIONS if x["label"] == label)
    X, y, g, _, _, reg = _trial_features(s, _args())
    names = {int(k): v for k, v in json.load(open(glob.glob(
        f"{s['mc']}/wfield_local_results/allen_aligned_affine8v1/allen_area_names.json")[0]))}
    rn = np.array([names.get(int(reg[i]), "?") for i in range(X.shape[1])])
    L = np.array([n.endswith("_left") for n in rn]); R = np.array([n.endswith("_right") for n in rn])
    out = {"nL": int(L.sum()), "nR": int(R.sum())}
    out["L"] = _rdm_from(X[:, L], y) if L.sum() >= 3 else None
    out["R"] = _rdm_from(X[:, R], y) if R.sum() >= 3 else None
    out["relL"] = _rel_within(X[:, L], y, g) if L.sum() >= 3 else np.nan
    out["relR"] = _rel_within(X[:, R], y, g) if R.sum() >= 3 else np.nan
    out["LR"] = _spearman(_triu(out["L"]), _triu(out["R"])) if out["L"] is not None and out["R"] is not None else np.nan
    # disattenuated L-vs-R agreement: divide by sqrt(relL*relR) to remove each hemisphere's estimation
    # noise -> the true between-hemisphere geometry similarity (~1 = hemispheres agree). Unstable (NaN) when
    # either reliability is too low to correct against.
    rl, rr = out["relL"], out["relR"]
    out["LRnorm"] = (out["LR"] / np.sqrt(rl * rr)) if (np.isfinite(out["LR"]) and rl > 0.1 and rr > 0.1) else np.nan
    return out


def _collect_hemi(dates):
    labs = sorted([s["label"] for s in SESSIONS if dates is None or s["label"][-4:] in dates],
                  key=lambda l: (l[:4], l[-4:]))
    H = {}
    for l in labs:
        try:
            H[l] = _session_hemi(l)
        except Exception as ex:
            print(f"  {l} hemi skip: {str(ex)[:50]}", flush=True)
    return [l for l in labs if l in H], H


def fig_rsa_hemisphere(out, dates=None, tag=""):
    """Hemisphere-resolved position RDMs. (1) per-animal mean RDM_L (top) and RDM_R (bottom). (2) summary:
    within-session L-vs-R RDM agreement per animal (do the two hemispheres encode position the same way?),
    per-hemisphere split-half reliability, and animal x animal RDM similarity computed WITHIN the left and
    WITHIN the right hemisphere separately (is PS93's LEFT hemisphere -- contralateral to its right
    orofacial deficit -- an outlier where its right is not?)."""
    labs, H = _collect_hemi(dates)
    animals = sorted({l[:4] for l in labs})

    def amean(a, key):
        vals = [H[l][key] for l in labs if l[:4] == a and H[l].get(key) is not None and np.ndim(H[l][key]) == 2]
        return np.nanmean(vals, axis=0) if vals else np.full((6, 6), np.nan)
    rdmL = {a: amean(a, "L") for a in animals}; rdmR = {a: amean(a, "R") for a in animals}

    # ---- figure 1: per-animal hemisphere RDMs ----
    cols = animals + ["GRAND"]
    gl = np.nanmean([rdmL[a] for a in animals], axis=0); gr = np.nanmean([rdmR[a] for a in animals], axis=0)
    allv = [m for M in list(rdmL.values()) + list(rdmR.values()) + [gl, gr] for m in _triu(M) if np.isfinite(m)]
    vmax = np.nanpercentile(allv, 98)
    fig, axes = plt.subplots(2, len(cols), figsize=(2.7 * len(cols), 5.6))
    for j, name in enumerate(cols):
        ML = gl if name == "GRAND" else rdmL[name]; MR = gr if name == "GRAND" else rdmR[name]
        for row, M, hh in [(0, ML, "LEFT hem"), (1, MR, "RIGHT hem")]:
            ax = axes[row][j]; im = ax.imshow(M, cmap="magma", vmin=0, vmax=vmax)
            ax.set_xticks(range(6)); ax.set_xticklabels(POSN if row == 1 else [], rotation=90, fontsize=5.5)
            ax.set_yticks(range(6)); ax.set_yticklabels(POSN if j == 0 else [], fontsize=5.5)
            if row == 0:
                ax.set_title(name, fontsize=10)
            if j == 0:
                ax.set_ylabel(hh, fontsize=9)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="dissimilarity (1 - r)")
    fig.suptitle(f"Hemisphere-resolved position RDMs per animal{(' [' + tag + ']') if tag else ''} "
                 "(top = left-hem components, bottom = right-hem)", fontsize=12)
    p1 = out / f"locanmf_rsa_hemisphere_rdms{('_' + tag) if tag else ''}.png"; fig.savefig(p1, dpi=130); plt.close(fig)

    # ---- figure 2: summary ----
    def sess_vals(a, key):
        return [H[l][key] for l in labs if l[:4] == a and np.isfinite(H[l].get(key, np.nan))]
    fig = plt.figure(figsize=(16, 8)); x = np.arange(len(animals)); w = 0.38
    # (A) within-session L-vs-R RDM agreement
    ax = fig.add_subplot(2, 2, 1)
    for i, a in enumerate(animals):
        v = sess_vals(a, "LR")
        ax.bar(i, np.nanmean(v) if v else np.nan, 0.6, color=ANIMAL_COLOR.get(a, "k"), alpha=0.85)
        if v:
            ax.scatter([i] * len(v), v, s=16, color="k", alpha=0.5, zorder=3)
        vn = sess_vals(a, "LRnorm")
        if vn:
            ax.text(i, (np.nanmean(v) if v else 0) + 0.04, f"adj\n{np.nanmean(vn):.2f}", ha="center",
                    fontsize=8, fontweight="bold", color="#b30000")
    ax.set_xticks(x); ax.set_xticklabels(animals); ax.set_ylim(0, 1.15)
    ax.set_ylabel("L-vs-R RDM Spearman (within session)")
    ax.set_title("Do the two hemispheres encode position the SAME way?\n"
                 "bars = raw; red 'adj' = disattenuated by reliability (PS93 lowest = hemispheres diverge)", fontsize=9)
    # (B) per-hemisphere split-half reliability
    ax = fig.add_subplot(2, 2, 2)
    ax.bar(x - w / 2, [np.nanmean(sess_vals(a, "relL")) for a in animals], w, label="LEFT hem", color="#8e44ad")
    ax.bar(x + w / 2, [np.nanmean(sess_vals(a, "relR")) for a in animals], w, label="RIGHT hem", color="#e67e22")
    ax.set_xticks(x); ax.set_xticklabels(animals); ax.set_ylim(0, 1)
    ax.set_ylabel("split-half RDM reliability"); ax.legend(fontsize=8)
    ax.set_title("Per-hemisphere RDM reliability\n(low LEFT for PS93 = degraded contralateral geometry?)", fontsize=10)
    # (C,D) animal x animal RDM similarity within LEFT and within RIGHT hemisphere
    for sp, key, hh in [(3, "L", "LEFT"), (4, "R", "RIGHT")]:
        ax = fig.add_subplot(2, 2, sp)
        A = np.full((len(animals), len(animals)), np.nan)
        for i, ai in enumerate(animals):
            for jj, aj in enumerate(animals):
                A[i, jj] = _spearman(_triu(rdmL[ai] if key == "L" else rdmR[ai]),
                                     _triu(rdmL[aj] if key == "L" else rdmR[aj]))
        im = ax.imshow(A, cmap="viridis", vmin=np.nanpercentile(A, 5), vmax=1)
        ax.set_xticks(range(len(animals))); ax.set_xticklabels(animals); ax.set_yticks(range(len(animals))); ax.set_yticklabels(animals)
        for i in range(len(animals)):
            for jj in range(len(animals)):
                ax.text(jj, i, f"{A[i,jj]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if A[i, jj] < 0.6 else "black")
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"Animal x animal RDM similarity -- {hh} hemisphere", fontsize=10)
    fig.suptitle(f"Hemisphere-resolved RSA{(' [' + tag + ']') if tag else ''} "
                 "(PS93 = right orofacial deficit -> probe its LEFT/contralateral hemisphere)", fontsize=12)
    fig.tight_layout(); p2 = out / f"locanmf_rsa_hemisphere_summary{('_' + tag) if tag else ''}.png"; fig.savefig(p2, dpi=130); plt.close(fig)

    print(f"Hemisphere-resolved RSA{(' [' + tag + ']') if tag else ''} (per animal, mean over sessions):", flush=True)
    for a in animals:
        print(f"  {a}: L-vs-R={np.nanmean(sess_vals(a,'LR')):.2f} (adj {np.nanmean(sess_vals(a,'LRnorm')):.2f})  "
              f"relL={np.nanmean(sess_vals(a,'relL')):.2f} relR={np.nanmean(sess_vals(a,'relR')):.2f}  "
              f"nL/nR~{int(np.mean([H[l]['nL'] for l in labs if l[:4]==a]))}/"
              f"{int(np.mean([H[l]['nR'] for l in labs if l[:4]==a]))}", flush=True)
    return p1, p2


def fig_rsa_crossnobis(out, dates=None, tag=""):
    """Crossnobis (noise-unbiased) RDM version of fig_rsa_sessions: tests whether the apparent cross-day
    'drift' in the 1-corr RDM is real geometry change vs estimation-noise inflation. Same 3 panels."""
    labs = sorted([s["label"] for s in SESSIONS if dates is None or s["label"][-4:] in dates],
                  key=lambda l: (l[:4], l[-4:]))
    rdms, rels = {}, {}
    for l in labs:
        try:
            rdms[l], rels[l] = _crossnobis_and_reliability(l)
        except Exception as ex:
            print(f"  {l} cn skip: {str(ex)[:50]}", flush=True)
    labs = [l for l in labs if l in rdms]; vecs = {l: _triu(rdms[l]) for l in labs}
    n = len(labs); S = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            S[i, j] = _spearman(vecs[labs[i]], vecs[labs[j]])
    animals = sorted({l[:4] for l in labs}); aof = [l[:4] for l in labs]
    wa, ac, nc = {}, {}, {}
    for a in animals:
        idx = [k for k in range(n) if aof[k] == a]; oth = [k for k in range(n) if aof[k] != a]
        wp = [S[i, j] for ii, i in enumerate(idx) for j in idx[ii + 1:]]; ap2 = [S[i, j] for i in idx for j in oth]
        wa[a] = np.nanmean(wp) if wp else np.nan; ac[a] = np.nanmean(ap2) if ap2 else np.nan
        nc[a] = np.nanmean([rels[l] for l in labs if l[:4] == a])
    ardm = {a: np.nanmean([rdms[l] for l in labs if l[:4] == a], axis=0) for a in animals}
    A = np.full((len(animals), len(animals)), np.nan)
    for i, ai in enumerate(animals):
        for j, aj in enumerate(animals):
            A[i, j] = _spearman(_triu(ardm[ai]), _triu(ardm[aj]))
    fig = plt.figure(figsize=(18, 6))
    ax = fig.add_subplot(1, 3, 1)
    im = ax.imshow(S, cmap="viridis", vmin=np.nanpercentile(S, 5), vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels([f"{l[:4]} {l[-4:-2]}/{l[-2:]}" for l in labs], rotation=90, fontsize=6)
    ax.set_yticks(range(n)); ax.set_yticklabels([f"{l[:4]} {l[-4:-2]}/{l[-2:]}" for l in labs], fontsize=6)
    ax.set_title("Session x session RSA (CROSSNOBIS RDMs; blocks=animal)", fontsize=10); fig.colorbar(im, ax=ax, shrink=0.7)
    ax = fig.add_subplot(1, 3, 2); x = np.arange(len(animals)); w = 0.35
    ax.bar(x - w / 2, [wa[a] for a in animals], w, label="within-animal", color="#2980b9")
    ax.bar(x + w / 2, [ac[a] for a in animals], w, label="across-animal", color="#c0392b")
    for i, a in enumerate(animals):
        ax.plot([i - w, i + w], [nc[a], nc[a]], color="#555", ls="--", lw=1.2,
                label="noise ceiling (split-half)" if i == 0 else None)
        frac = wa[a] / nc[a] if nc[a] and np.isfinite(nc[a]) else np.nan
        ax.text(i, max(wa[a], 0) + 0.02, f"{100*frac:.0f}%" if np.isfinite(frac) else "", ha="center", fontsize=7, color="#2980b9")
    ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x); ax.set_xticklabels(animals); ax.set_ylabel("mean 2nd-order RSA (Spearman)")
    ax.legend(fontsize=7); ax.set_title("CROSSNOBIS within vs across\n(% = within / noise ceiling)", fontsize=10)
    ax = fig.add_subplot(1, 3, 3)
    im = ax.imshow(A, cmap="viridis", vmin=np.nanpercentile(A, 5), vmax=1)
    ax.set_xticks(range(len(animals))); ax.set_xticklabels(animals); ax.set_yticks(range(len(animals))); ax.set_yticklabels(animals)
    for i in range(len(animals)):
        for j in range(len(animals)):
            ax.text(j, i, f"{A[i,j]:.2f}", ha="center", va="center", fontsize=8, color="white" if A[i, j] < 0.6 else "black")
    ax.set_title("animal x animal crossnobis-RDM similarity", fontsize=10)
    fig.suptitle(f"Crossnobis (noise-unbiased) RDM RSA [{tag or 'all'}] (n={n}) — vs 1-corr RDM, tests if drift is real", fontsize=12)
    fig.tight_layout(); p = out / f"locanmf_rsa_crossnobis{('_' + tag) if tag else ''}.png"; fig.savefig(p, dpi=130); plt.close(fig)
    for a in animals:
        frac = wa[a] / nc[a] if nc[a] else np.nan
        print(f"  CN {a}: within={wa[a]:.2f} across={ac[a]:.2f} ceiling={nc[a]:.2f} within/ceiling={frac:.0%}", flush=True)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--dates", default="", help="comma-separated MMDD subset (default all)")
    ap.add_argument("--tag", default="", help="filename tag")
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    dates = set(args.dates.split(",")) if args.dates else None
    print("wrote", fig_rsa_sessions(args.output, dates, args.tag).name, flush=True)
    print("wrote", fig_rsa_crossnobis(args.output, dates, args.tag).name, flush=True)
    print("wrote", fig_rsa_rdms(args.output, dates, args.tag).name, flush=True)
    for nm in fig_rsa_hemisphere(args.output, dates, args.tag):
        print("wrote", nm.name, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
