"""Reproducible decoder reporting: per-position weights-by-region, SSp-vs-MO region groups,
window-length sweep, and pre-stroke baseline variability — plus the summary PPT. Everything is
computed from data (no hardcoded accuracies), reusing the canonical decoder feature extraction
(no per-trial baseline, block-aware CV; see locanmf_position_decoder.py and F10-F14 in
LOCANMF_LICK_CUE_ANALYSIS.md).

    python -m wfield_local.locanmf_decoder_weights --output "<dir>" --ppt

Consolidates the former ~/source one-off scripts (decoder_weights_fig, decoder_region_groups_fig,
window_sweep_fig, baseline_variability_fig, build_decoder_ppt).
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
from wfield_local.locanmf_position_decoder import _trial_features, _build_signal
from wfield_local.plot_lick_aligned_averages import POSITION_NAMES, DISPLAY_ORDER, _load_daq_events
from wfield_local.plot_spout_trial_averages import _load_daq_events as _load_cue, _classify_cues
from wfield_local.locanmf_crossanimal_dff import _frames

FS = 31.23
POSNAMES = [POSITION_NAMES[c] for c in DISPLAY_ORDER]


def _sess(label):
    return next(s for s in SESSIONS if s["label"] == label)


def _avail(date):
    return [s["label"] for s in SESSIONS if s["label"].endswith(date)]


def _args(align="lick", post_s=2.0, baseline="none"):
    return SimpleNamespace(source="locanmf", align=align, baseline=baseline,
                           pre_s=1.0, post_s=post_s, fs=FS, max_rt=2.0)


def _names(s):
    p = glob.glob(f"{s['mc']}/wfield_local_results/allen_aligned_affine8v1/allen_area_names.json")[0]
    return {int(k): v for k, v in json.load(open(p))}


def _clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))


def _bcv_acc(X, y, g):
    ng = min(5, int(np.unique(g).size))
    if ng < 2 or len(y) < 10:
        return float("nan")
    return accuracy_score(y, cross_val_predict(_clf(), X, y, cv=GroupKFold(ng), groups=g))


def _region_group(rn):
    if rn.startswith(("SSp", "SSs")):
        return "SSp/SSs"
    if rn.startswith("MO"):
        return "MOp/MOs"
    if rn.startswith("VIS"):
        return "Visual"
    if rn.startswith("AUD"):
        return "Auditory"
    if rn.startswith(("PL", "FRP", "ORB", "ACA", "ILA")):
        return "Frontal/Cing"
    return "Other"


# --------------------------------------------------------------------------- figures
def fig_weights_by_region(labels, out, tag=""):
    fig, axes = plt.subplots(1, len(labels), figsize=(5.7 * len(labels), 5.2), squeeze=False)
    for ax, lab in zip(axes[0], labels):
        s = _sess(lab); names = _names(s)
        X, y, g, _, _, reg = _trial_features(s, _args("lick", 2.0))
        lr = _clf().fit(X, y).named_steps["logisticregression"]; ci = {int(c): i for i, c in enumerate(lr.classes_)}
        comp_region = np.array([names.get(int(reg[i]), "?") for i in range(X.shape[1])])
        regions = sorted(set(comp_region)); M = np.zeros((6, len(regions)))
        for pi, pos in enumerate(DISPLAY_ORDER):
            if pos not in ci:
                continue
            w = lr.coef_[ci[pos]]
            for ri, rg in enumerate(regions):
                M[pi, ri] = np.clip(w[comp_region == rg], 0, None).sum()
        forced = [i for i, rg in enumerate(regions) if rg in ("MOp_left", "MOp_right", "MOs_left", "MOs_right")]
        sel = list(dict.fromkeys(list(np.argsort(M.sum(0))[::-1][:12]) + forced))
        top = np.array(sorted(sel, key=lambda t: regions[t]))
        im = ax.imshow(M[:, top], aspect="auto", cmap="magma", vmin=0)
        ax.set_xticks(range(len(top))); ax.set_xticklabels([regions[t] for t in top], rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(6)); ax.set_yticklabels(POSNAMES, fontsize=8); ax.set_title(lab, fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.7)
    axes[0][0].set_ylabel("intended spout position")
    fig.suptitle("Per-position decoder weight by Allen region (top-12 + MOp/MOs L/R) [no-baseline, first-lick 2s]\n"
                 "Contralateral SSp: left spouts->right-hemi SSp, right->left", fontsize=11)
    fig.tight_layout()
    p = out / f"locanmf_decoder_weights_by_region{('_' + tag) if tag else ''}.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_region_groups(labels, out):
    GROUPS = ["SSp/SSs", "MOp/MOs", "Frontal/Cing", "Visual", "Auditory", "Other"]
    COLORS = ["#c0392b", "#2980b9", "#8e44ad", "#27ae60", "#e67e22", "#95a5a6"]
    acc = {}; share = {}
    for lab in labels:
        s = _sess(lab); names = _names(s)
        X, y, g, _, _, reg = _trial_features(s, _args("lick", 2.0))
        grp_all = np.arange(X.shape[1])
        rn = np.array([names.get(int(reg[i]), "?") for i in range(X.shape[1])])
        ssp = np.array([i for i in range(X.shape[1]) if rn[i].startswith("SSp")])
        mo = np.array([i for i in range(X.shape[1]) if rn[i].startswith(("MOp", "MOs"))])
        acc[lab] = (_bcv_acc(X[:, grp_all], y, g),
                    _bcv_acc(X[:, ssp], y, g) if ssp.size else float("nan"),
                    _bcv_acc(X[:, mo], y, g) if mo.size else float("nan"))
        lr = _clf().fit(X, y).named_steps["logisticregression"]; ci = {int(c): i for i, c in enumerate(lr.classes_)}
        grp = np.array([_region_group(rn[i]) for i in range(X.shape[1])]); M = np.zeros((6, len(GROUPS)))
        for pi, pos in enumerate(DISPLAY_ORDER):
            if pos not in ci:
                continue
            w = np.clip(lr.coef_[ci[pos]], 0, None)
            for gi, gg in enumerate(GROUPS):
                M[pi, gi] = w[grp == gg].sum()
            if M[pi].sum() > 0:
                M[pi] /= M[pi].sum()
        share[lab] = M
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    ax = axes[0]; x = np.arange(len(labels)); w = 0.25
    for i, (gname, c) in enumerate(zip(["all", "SSp only", "MO only"], ["#34495e", "#c0392b", "#2980b9"])):
        ax.bar(x + (i - 1) * w, [acc[l][i] for l in labels], w, label=gname, color=c)
    ax.axhline(1 / 6, color="grey", ls="--", lw=1, label="chance")
    ax.set_xticks(x); ax.set_xticklabels([l[:4] + " " + l[-4:-2] + "/" + l[-2:] for l in labels], fontsize=8, rotation=20)
    ax.set_ylim(0, 1); ax.set_ylabel("decoding accuracy"); ax.set_title("Accuracy by feature set (first-lick 2s, block-CV)")
    ax.legend(fontsize=9)
    ax = axes[1]; Mmean = np.mean([share[l] for l in labels], 0); x = np.arange(6); bottom = np.zeros(6)
    for gi, (gg, c) in enumerate(zip(GROUPS, COLORS)):
        ax.bar(x, Mmean[:, gi], 0.7, bottom=bottom, label=gg, color=c); bottom += Mmean[:, gi]
    ax.set_xticks(x); ax.set_xticklabels(POSNAMES, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1); ax.set_ylabel("share of positive decoder weight")
    ax.set_title(f"Predictive-weight share by region group (mean of {len(labels)})"); ax.legend(fontsize=8, ncol=2)
    fig.suptitle("SSp dominates; MOp/MOs contribute (above chance) but secondary, strongest for FAR positions", fontsize=12)
    fig.tight_layout(); p = out / "locanmf_decoder_region_groups.png"; fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_window_sweep(labels, out, wins=(0.5, 1.0, 2.0, 3.5, 5.0)):
    fig, ax = plt.subplots(figsize=(8, 5.4)); colors = {"PS92": "#1f77b4", "PS94": "#d62728", "PS95": "#2ca02c"}
    for lab in labels:
        s = _sess(lab); row = []
        for w in wins:
            X, y, g, _, _, _ = _trial_features(s, _args("cue", w))
            row.append(_bcv_acc(X, y, g))
        an = lab[:4]; ls = "-" if lab.endswith("0603") else "--"
        ax.plot(wins, row, ls, marker="o", color=colors.get(an, "k"), label=f"{an} {lab[-4:-2]}/{lab[-2:]}")
    ax.axvline(2.0, color="grey", ls=":", lw=1.2); ax.axhline(1 / 6, color="k", ls="--", lw=0.8)
    ax.set_xlabel("post-cue window length (s)"); ax.set_ylabel("decoding accuracy (block-CV)")
    ax.set_xticks(wins); ax.set_ylim(0.15, 0.9); ax.legend(fontsize=9, ncol=2)
    ax.set_title("Decoding vs window: ~2s optimum (lick-bout); longer dilutes (ITI ~7s, not cross-trial bleed)", fontsize=10.5)
    fig.tight_layout(); p = out / "locanmf_decoder_window_sweep.png"; fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_baseline_variability(out):
    by_animal = {}
    for an in ("PS92", "PS94", "PS95"):
        days = {}
        for s in [x for x in SESSIONS if x["label"].startswith(an) and x["label"][-4:] in ("0601", "0602", "0603", "0604")]:
            Xl, yl, gl, _, _, _ = _trial_features(s, _args("lick", 2.0))
            Xp, yp, gp, Xnl, ynl, _ = _trial_features(s, _args("precue", 1.0))
            postlick = _bcv_acc(Xl, yl, gl); pre_eng = _bcv_acc(Xp, yp, gp)
            pre_nl = (accuracy_score(ynl, _clf().fit(Xp, yp).predict(Xnl)) if len(ynl) >= 6 else np.nan)
            days[f"{s['label'][-4:-2]}/{s['label'][-2:]}"] = (postlick, pre_eng, pre_nl)
        by_animal[an] = days
    titles = ["Post-lick 2 s (engaged)", "Pre-cue 1 s (engaged)", "Pre-cue (no-lick)"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    animals = list(by_animal); xpos = {a: i for i, a in enumerate(animals)}
    for idx, (ax, title) in enumerate(zip(axes, titles)):
        for a in animals:
            vals = [(d, v[idx]) for d, v in by_animal[a].items() if not np.isnan(v[idx])]
            x = xpos[a]
            if vals:
                vv = [v for _, v in vals]; ax.plot([x, x], [min(vv), max(vv)], color="grey", lw=2, zorder=1)
            for d, v in vals:
                ax.scatter(x, v, s=90, zorder=3); ax.annotate(d, (x, v), textcoords="offset points", xytext=(8, -3), fontsize=8)
        ax.axhline(1 / 6, color="k", ls="--", lw=0.8); ax.set_xticks(range(len(animals)))
        ax.set_xticklabels(animals); ax.set_xlim(-0.5, len(animals) - 0.3); ax.set_ylim(0, 1); ax.set_title(title, fontsize=11)
    axes[0].set_ylabel("decoding accuracy")
    fig.suptitle("Pre-stroke baseline variability (3 days/animal). Within-animal swing is large; a post-stroke "
                 "effect must clear this spread.", fontsize=11)
    fig.tight_layout(); p = out / "locanmf_decoder_baseline_variability.png"; fig.savefig(p, dpi=130); plt.close(fig)
    return p, by_animal


def fig_top_components(label, out, topn=10):
    """Spatial footprints (Allen-overlaid) of the top-N LocaNMF components by decoder weight."""
    s = _sess(label); mc = s["mc"]
    A = np.load(f"{mc}/locanmf_affine8v1_final/{label}_locanmf_A.npy")          # (H,W,ncomp)
    ad = glob.glob(f"{mc}/wfield_local_results/allen_aligned_affine8v1")[0]
    atlas = np.load(f"{ad}/allen_area_atlas_native_grid.npy"); mask = np.load(f"{ad}/allen_brain_mask_native_grid.npy").astype(bool)
    names = _names(s)
    X, y, g, _, _, reg = _trial_features(s, _args("lick", 2.0))
    lr = _clf().fit(X, y).named_steps["logisticregression"]; ci = {int(c): i for i, c in enumerate(lr.classes_)}
    coef = lr.coef_
    # Rank by UNIVARIATE block-CV accuracy ("does this component alone decode position?"), NOT by the
    # multivariate |weight| -- L2 + correlated features make weights surface suppressors/noise, not signal.
    uacc = np.array([_bcv_acc(X[:, [c]], y, g) for c in range(X.shape[1])])
    # Significance filter (only components above the ~95th-pct null) + fragment dedup (drop a
    # component whose trial-feature correlation with an already-selected, higher-acc component
    # exceeds 0.8 -- LocaNMF over-splitting produces tiny correlated slivers that are not new signal).
    n = len(y); sigthr = 1 / 6 + 1.645 * np.sqrt((1 / 6) * (5 / 6) / n)
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
    top = []
    for c in np.argsort(uacc)[::-1]:
        if uacc[c] <= sigthr:
            break
        if any(abs(Xz[:, c] @ Xz[:, sc]) / n > 0.8 for sc in top):
            continue
        top.append(int(c))
        if len(top) >= topn:
            break
    b = np.zeros_like(atlas, bool)
    b[:-1, :] |= atlas[:-1, :] != atlas[1:, :]; b[:, :-1] |= atlas[:, :-1] != atlas[:, 1:]
    ys, xs = np.where(mask); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    fig, axes = plt.subplots(2, (topn + 1) // 2, figsize=(1.8 * topn, 8))
    for ax, comp in zip(axes.ravel(), top):
        fp = A[:, :, comp].astype(float).copy(); fp[~mask] = np.nan
        ax.imshow(fp[y0:y1, x0:x1], cmap="magma")
        bb = np.where(b[y0:y1, x0:x1]); ax.scatter(bb[1], bb[0], s=0.2, c="white", alpha=0.35, marker=".")
        pos = DISPLAY_ORDER[int(np.argmax([coef[ci[p], comp] if p in ci else -9 for p in DISPLAY_ORDER]))]
        ax.set_title(f"comp#{comp} {names.get(int(reg[comp]),'?')}\n->{POSITION_NAMES[pos]} acc={uacc[comp]:.2f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[len(top):]:
        ax.axis("off")
    if not top:
        axes.ravel()[0].text(0.5, 0.5, "no components individually\nsignificant above chance\n(code is fully distributed)",
                             ha="center", va="center", fontsize=12, transform=axes.ravel()[0].transAxes)
    fig.suptitle(f"{label}: significant + fragment-deduped components by univariate block-CV accuracy "
                 f"({len(top)} shown, sig-thr={sigthr:.2f}; footprints Allen-overlaid, '->pos'=position most predicted)", fontsize=11)
    fig.tight_layout(); p = out / f"locanmf_top_components_{label}.png"; fig.savefig(p, dpi=120); plt.close(fig)
    return p


def _lick_trials(label):
    s = _sess(label); sig, _ = _build_signal(s, "locanmf"); T = sig.shape[1]
    cue = _load_cue(s["h5"]); lk = _load_daq_events(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    cf, lf, _ = _frames(s, cue, lk); codes = _classify_cues(cue["cue_samples"], cue["strobe_samples"], cue["strobe_codes"])
    blk = np.zeros(len(codes), int); b = 0
    for i in range(1, len(codes)):
        if codes[i] != codes[i - 1]:
            b += 1
        blk[i] = b
    ls = np.sort(lf); j = np.searchsorted(ls, cf, side="right")
    first = np.where(j < ls.size, ls[np.clip(j, 0, ls.size - 1)], -1); rt = first - cf
    keep = [k for k in range(cf.size) if codes[k] >= 0 and first[k] > 0 and 0 < rt[k] <= 2 * FS]
    return sig, T, np.array([int(first[k]) for k in keep]), np.array([int(codes[k]) for k in keep]), np.array([int(blk[k]) for k in keep])


def fig_temporal_dynamics(labels, out):
    """(A) sliding-window decoding vs time; (B) multi-bin temporal profile vs single window-mean."""
    win = int(round(0.5 * FS)); offs = np.arange(int(-1.5 * FS), int(3.0 * FS), int(0.25 * FS))
    nbin = 8; binlen = int(round(0.25 * FS)); colors = {"PS92": "#1f77b4", "PS93": "#d62728", "PS94": "#ff7f0e", "PS95": "#2ca02c"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4)); ax = axes[0]; mb = {}; allv = []
    for label in labels:
        sig, T, fl, y, g = _lick_trials(label)
        accs = []
        for o in offs:
            ok = (fl + o >= 0) & (fl + o + win <= T)
            accs.append(_bcv_acc(np.array([sig[:, f + o:f + o + win].mean(1) for f in fl[ok]]), y[ok], g[ok]))
        ax.plot(offs / FS, accs, marker="o", ms=3, color=colors.get(label[:4], "k"), label=label[:4])
        allv += [a for a in accs if a == a]
        ok = (fl >= 0) & (fl + nbin * binlen <= T)
        Xmb = np.array([np.concatenate([sig[:, f + bi * binlen:f + (bi + 1) * binlen].mean(1) for bi in range(nbin)]) for f in fl[ok]])
        Xmean = np.array([sig[:, f:f + nbin * binlen].mean(1) for f in fl[ok]])
        mb[label[:4]] = (_bcv_acc(Xmean, y[ok], g[ok]), _bcv_acc(Xmb, y[ok], g[ok]))
    ax.axvline(0, color="k", lw=1); ax.axvline(-0.16, color="grey", ls=":", lw=1); ax.axhline(1 / 6, color="k", ls="--", lw=0.8)
    ax.set_xlabel("time from first lick (s)"); ax.set_ylabel("accuracy (0.5s window, block-CV)")
    ax.set_ylim(max(0.0, min([1 / 6] + allv) - 0.05), min(1.0, max([1 / 6] + allv) + 0.05))
    ax.legend(fontsize=9); ax.set_title("Sliding-window decoding: when is position information present?")
    ax = axes[1]; x = np.arange(len(labels)); w = 0.35
    ax.bar(x - w / 2, [mb[l[:4]][0] for l in labels], w, label="single 2s mean", color="#888")
    ax.bar(x + w / 2, [mb[l[:4]][1] for l in labels], w, label="8x0.25s temporal bins", color="#d62728")
    ax.axhline(1 / 6, color="k", ls="--", lw=0.8); ax.set_xticks(x); ax.set_xticklabels([l[:4] for l in labels])
    ax.set_ylim(0, 1); ax.set_ylabel("accuracy (block-CV)"); ax.legend(fontsize=9); ax.set_title("Does temporal profile beat the window-mean?")
    dtag = sorted({l[-4:] for l in labels})[0]
    dlab = "/".join(sorted({l[-4:-2] + "/" + l[-2:] for l in labels}))
    fig.suptitle(f"Rolling temporal dynamics of spout-position coding (first-lick aligned, {dlab}; one line per animal)", fontsize=12)
    fig.tight_layout(); p = out / f"locanmf_decoder_temporal_dynamics_{dtag}.png"; fig.savefig(p, dpi=130); plt.close(fig)
    return p


def _blocks(codes):
    blk = np.zeros(len(codes), int); b = 0
    for i in range(1, len(codes)):
        if codes[i] != codes[i - 1]:
            b += 1
        blk[i] = b
    return blk


def _engaged(s):
    """sig, T, first-lick frames, position codes, block ids for engaged (cue+lick) trials."""
    sig, _ = _build_signal(s, "locanmf"); T = sig.shape[1]
    cue = _load_cue(s["h5"]); lk = _load_daq_events(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    cf, lf, _ = _frames(s, cue, lk); codes = _classify_cues(cue["cue_samples"], cue["strobe_samples"], cue["strobe_codes"])
    blk = _blocks(codes); ls = np.sort(lf); j = np.searchsorted(ls, cf, side="right")
    first = np.where(j < ls.size, ls[np.clip(j, 0, ls.size - 1)], -1); rt = first - cf
    keep = [k for k in range(cf.size) if codes[k] >= 0 and first[k] > 0 and 0 < rt[k] <= 2 * FS]
    return (sig, T, cf, np.array([int(first[k]) for k in keep]), np.array([int(codes[k]) for k in keep]),
            np.array([int(blk[k]) for k in keep]), np.array([int(cf[k]) for k in keep]))


def fig_rolling_cue(labels, out, tag):
    """Cue-aligned sliding-window decoder spanning pre-cue (ENL) -> post-cue, one line per session."""
    win = int(round(0.5 * FS)); offs = np.arange(int(-3.5 * FS), int(2.5 * FS), int(0.25 * FS))
    cols = {"PS92": "#1f77b4", "PS93": "#d62728", "PS94": "#ff7f0e", "PS95": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(10, 5.6)); allv = []
    for lab in labels:
        sig, T, _, _, y, g, cfk = _engaged(_sess(lab))
        accs = []
        for o in offs:
            ok = (cfk + o >= 0) & (cfk + o + win <= T)
            accs.append(_bcv_acc(np.array([sig[:, c + o:c + o + win].mean(1) for c in cfk[ok]]), y[ok], g[ok]))
        ax.plot(offs / FS, accs, marker="o", ms=3, color=cols.get(lab[:4], "k"), label=f"{lab[:4]} (n={len(y)})")
        allv += [a for a in accs if a == a]
    ax.axvspan(-3.0, 0, color="grey", alpha=0.08); ax.axvline(0, color="k", lw=1.2); ax.axhline(1 / 6, color="k", ls="--", lw=0.8)
    ax.set_xlabel("time from cue (s)"); ax.set_ylabel("decoding accuracy (0.5s window, block-CV)")
    ax.set_ylim(max(0.0, min([1 / 6] + allv) - 0.05), min(1.0, max([1 / 6] + allv) + 0.05))
    ax.legend(fontsize=9, loc="upper left"); ax.set_title(f"{tag} cue-aligned rolling decoder: pre-cue (ENL) -> post-cue", fontsize=11)
    fig.tight_layout(); p = out / f"locanmf_decoder_rolling_cue_{tag}.png"; fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_rolling_cue_by_animal(animal, labels, out):
    """Per-ANIMAL cue-aligned sliding-window decoder, one line per DATE (the rolling pre-cue->post-cue
    decoder for Section A). Shows how that animal's position-decoding time-course evolves across sessions.
    Reuses the fig_rolling_cue offsets/window."""
    labels = sorted(labels, key=lambda l: l[-4:])
    win = int(round(0.5 * FS)); offs = np.arange(int(-3.5 * FS), int(2.5 * FS), int(0.25 * FS))
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(10, 5.6)); allv = []
    for i, lab in enumerate(labels):
        sig, T, _, _, y, g, cfk = _engaged(_sess(lab))
        accs = []
        for o in offs:
            ok = (cfk + o >= 0) & (cfk + o + win <= T)
            accs.append(_bcv_acc(np.array([sig[:, c + o:c + o + win].mean(1) for c in cfk[ok]]), y[ok], g[ok]))
        c = cmap(i / max(1, len(labels) - 1))
        ax.plot(offs / FS, accs, marker="o", ms=3, color=c, label=f"{lab[-4:-2]}/{lab[-2:]} (n={len(y)})")
        allv += [a for a in accs if a == a]
    ax.axvspan(-3.0, 0, color="grey", alpha=0.08); ax.axvline(0, color="k", lw=1.2); ax.axhline(1 / 6, color="k", ls="--", lw=0.8)
    ax.set_xlabel("time from cue (s)  [grey = pre-cue ENL]"); ax.set_ylabel("decoding accuracy (0.5s window, block-CV)")
    ax.set_ylim(max(0.0, min([1 / 6] + allv) - 0.05), min(1.0, max([1 / 6] + allv) + 0.05))
    ax.legend(fontsize=9, loc="upper left", title="session")
    ax.set_title(f"{animal} cue-aligned rolling decoder across sessions (pre-cue ENL -> post-cue)", fontsize=11)
    fig.tight_layout(); p = out / f"locanmf_decoder_rolling_by_animal_{animal}.png"; fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_rolling_laterality(labels, out, tag):
    """Rolling cue-aligned LATERALITY (L/center/R, chance .33, solid) vs full 6-way (dashed)."""
    win = int(round(0.5 * FS)); offs = np.arange(int(-3.0 * FS), int(2.5 * FS), int(0.25 * FS))
    cols = {"PS92": "#1f77b4", "PS93": "#d62728", "PS94": "#ff7f0e", "PS95": "#2ca02c"}
    lat = {c: (0 if POSITION_NAMES[c].endswith("_L") else 2 if POSITION_NAMES[c].endswith("_R") else 1) for c in DISPLAY_ORDER}
    fig, ax = plt.subplots(figsize=(10, 5.8)); allv = []
    for lab in labels:
        sig, T, _, _, y6, g, cfk = _engaged(_sess(lab)); ylat = np.array([lat[c] for c in y6])
        al, a6 = [], []
        for o in offs:
            ok = (cfk + o >= 0) & (cfk + o + win <= T)
            X = np.array([sig[:, c + o:c + o + win].mean(1) for c in cfk[ok]])
            al.append(_bcv_acc(X, ylat[ok], g[ok])); a6.append(_bcv_acc(X, y6[ok], g[ok]))
        ax.plot(offs / FS, al, marker="o", ms=3, color=cols.get(lab[:4], "k"), label=f"{lab[:4]} laterality")
        ax.plot(offs / FS, a6, ls="--", lw=1, color=cols.get(lab[:4], "k"), alpha=0.7)
        allv += [a for a in al + a6 if a == a]
    ax.axvline(0, color="k", lw=1.2); ax.axhline(1 / 3, color="purple", ls=":", lw=1); ax.axhline(1 / 6, color="k", ls="--", lw=0.8)
    ax.set_xlabel("time from cue (s)"); ax.set_ylabel("decoding accuracy (0.5s window, block-CV)")
    ax.set_ylim(max(0.0, min([1 / 6] + allv) - 0.05), min(1.0, max([1 / 3] + allv) + 0.05))
    ax.legend(fontsize=8, loc="upper left", ncol=2); ax.set_title(f"{tag} rolling: LATERALITY (3-way, solid) vs 6-way (dashed)", fontsize=11)
    fig.tight_layout(); p = out / f"locanmf_decoder_rolling_laterality_{tag}.png"; fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_first40(out, date="0604", control="0603", minutes=40):
    """Decoding on the first N min (high-engagement window) vs full session; control date for comparison."""
    T40 = int(minutes * 60 * FS); post2 = int(round(2 * FS)); AN = ("PS92", "PS94", "PS95")
    acc = {d: {} for d in (date, control)}; eng = {}
    for d in (date, control):
        for an in AN:
            lab = f"{an}_{d}"
            try:
                s = _sess(lab)
            except StopIteration:
                continue
            sig, T, cf, first, y, g, cfk = _engaged(s)
            Xf, X40 = [], []; yf, y40, gf, g40 = [], [], [], []
            for i in range(len(first)):
                fl = first[i]
                if fl + post2 > T:
                    continue
                feat = sig[:, fl:fl + post2].mean(1)
                Xf.append(feat); yf.append(y[i]); gf.append(g[i])
                if cfk[i] < T40:
                    X40.append(feat); y40.append(y[i]); g40.append(g[i])
            acc[d][an] = (_bcv_acc(np.array(Xf), np.array(yf), np.array(gf)),
                          _bcv_acc(np.array(X40), np.array(y40), np.array(g40)))
            if d == date:
                # engaged fraction early vs late (cue followed by lick within 2s, over all cues)
                cue = _load_cue(s["h5"]); lk = _load_daq_events(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
                cf2, lf2, _ = _frames(s, cue, lk); codes = _classify_cues(cue["cue_samples"], cue["strobe_samples"], cue["strobe_codes"])
                ls = np.sort(lf2); j = np.searchsorted(ls, cf2, side="right")
                fst = np.where(j < ls.size, ls[np.clip(j, 0, ls.size - 1)], -1); rt = fst - cf2
                engf = (fst > 0) & (rt > 0) & (rt <= 2 * FS); early = cf2 < T40
                eng[an] = (engf[early & (codes >= 0)].mean(), engf[(~early) & (codes >= 0)].mean())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2)); x = np.arange(len(AN)); w = 0.2
    ax = axes[0]
    ax.bar(x - 1.5 * w, [acc[date][a][0] for a in AN], w, label=f"{date} full", color="#c0392b", alpha=0.5)
    ax.bar(x - 0.5 * w, [acc[date][a][1] for a in AN], w, label=f"{date} first {minutes}min", color="#c0392b")
    ax.bar(x + 0.5 * w, [acc[control][a][0] for a in AN], w, label=f"{control} full", color="#888", alpha=0.5)
    ax.bar(x + 1.5 * w, [acc[control][a][1] for a in AN], w, label=f"{control} first {minutes}min", color="#888")
    ax.axhline(1 / 6, color="k", ls="--", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(AN); ax.set_ylim(0, 1)
    ax.set_ylabel("decoding accuracy (block-CV)"); ax.legend(fontsize=8); ax.set_title(f"First-{minutes}min decoding: helps {date} (disengaged late), not {control}")
    ax = axes[1]
    ax.bar(x - w / 2, [eng[a][0] for a in AN], w, label=f"first {minutes}min", color="#27ae60")
    ax.bar(x + w / 2, [eng[a][1] for a in AN], w, label=f"after {minutes}min", color="#999")
    ax.set_xticks(x); ax.set_xticklabels(AN); ax.set_ylim(0, 1); ax.set_ylabel("engaged fraction"); ax.legend(fontsize=9)
    ax.set_title(f"{date} engagement collapses late in the session")
    fig.suptitle(f"{date} low decoding is partly late-session disengagement (only partly recovered)", fontsize=12)
    fig.tight_layout(); p = out / f"locanmf_decoder_first40min_{date}.png"; fig.savefig(p, dpi=130); plt.close(fig)
    return p


def _roi_sig_v(s, V):
    mc = s["mc"]; ad = f"{mc}/wfield_local_results/allen_aligned_affine8v{V}"
    U = np.load(f"{ad}/U_atlas.npy"); SVT = np.load(f"{mc}/wfield_local_results/SVTcorr.npy")
    atlas = np.load(f"{ad}/allen_area_atlas_native_grid.npy"); mask = np.load(f"{ad}/allen_brain_mask_native_grid.npy").astype(bool)
    Uf = U.reshape(-1, U.shape[2]); at = atlas.reshape(-1); mk = mask.reshape(-1)
    rois = [np.nanmean(Uf[(at == l) & mk], 0) @ SVT for l in np.unique(at) if l != 0 and ((at == l) & mk).sum() >= 20]
    return np.array(rois)


def fig_v1_vs_v2_alignment(labels, out, tag):
    """ROI decoder (alignment-sensitive) v1 vs v2 Allen registration, identical trials (first-lick 2s)."""
    post2 = int(round(2 * FS)); v1a, v2a = [], []
    for lab in labels:
        s = _sess(lab); cue = _load_cue(s["h5"]); lk = _load_daq_events(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
        cf, lf, _ = _frames(s, cue, lk); codes = _classify_cues(cue["cue_samples"], cue["strobe_samples"], cue["strobe_codes"])
        blk = _blocks(codes); ls = np.sort(lf); j = np.searchsorted(ls, cf, side="right")
        first = np.where(j < ls.size, ls[np.clip(j, 0, ls.size - 1)], -1); rt = first - cf
        accs = []
        for V in (1, 2):
            sig = _roi_sig_v(s, V); T = sig.shape[1]; X, y, g = [], [], []
            for k in range(cf.size):
                if codes[k] < 0 or not (first[k] > 0 and 0 < rt[k] <= 2 * FS):
                    continue
                fl = int(first[k])
                if fl + post2 > T:
                    continue
                X.append(sig[:, fl:fl + post2].mean(1)); y.append(int(codes[k])); g.append(int(blk[k]))
            accs.append(_bcv_acc(np.array(X), np.array(y), np.array(g)))
        v1a.append(accs[0]); v2a.append(accs[1])
    fig, ax = plt.subplots(figsize=(9, 5)); x = np.arange(len(labels)); w = 0.38
    ax.bar(x - w / 2, v1a, w, label="ROI v1", color="#888"); ax.bar(x + w / 2, v2a, w, label="ROI v2", color="#2980b9")
    ax.axhline(1 / 6, color="k", ls="--", lw=0.8); ax.set_xticks(x); ax.set_xticklabels([l[:4] for l in labels])
    ax.set_ylim(0, 0.8); ax.set_ylabel("decoding accuracy (block-CV)"); ax.legend(fontsize=9)
    ax.set_title(f"{tag} ROI decoder: v1 vs v2 Allen alignment (first-lick 2s, identical trials)")
    fig.tight_layout(); p = out / f"locanmf_decoder_v1_vs_v2_alignment_{tag}.png"; fig.savefig(p, dpi=130); plt.close(fig)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--weights-day", default="0603", help="date for the per-region weight/group figures")
    ap.add_argument("--ppt", action="store_true", help="also assemble the summary PPT from decoder + analysis figures")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    wl = _avail(args.weights_day)
    print("weights/region-groups on", wl, flush=True)
    print("wrote", fig_weights_by_region(wl, args.output), flush=True)
    print("wrote", fig_region_groups(wl, args.output), flush=True)
    print("wrote", fig_window_sweep(_avail("0603") + _avail("0604"), args.output), flush=True)
    p, by_animal = fig_baseline_variability(args.output)
    print("wrote", p, flush=True)
    print("baseline variability:", {a: {d: round(v[0], 2) for d, v in dd.items()} for a, dd in by_animal.items()}, flush=True)
    dates = ("0601", "0602", "0603", "0604", "0605")
    for d in dates:
        if _avail(d):
            print("wrote", fig_temporal_dynamics(_avail(d), args.output), flush=True)
    for lab in [s["label"] for s in SESSIONS if s["label"][-4:] in dates]:
        print("wrote", fig_top_components(lab, args.output), flush=True)
    # cue-aligned rolling + laterality for the continuous cohort; engagement + alignment checks for 6/4
    for fn, dd in ((fig_rolling_cue, "0605"), (fig_rolling_laterality, "0605"), (fig_v1_vs_v2_alignment, "0604")):
        if _avail(dd):
            try:
                print("wrote", fn(_avail(dd), args.output, dd).name, flush=True)
            except Exception as ex:
                print(f"{fn.__name__} skip: {str(ex)[:60]}", flush=True)
    if _avail("0604"):
        try:
            print("wrote", fig_first40(args.output, "0604", "0603").name, flush=True)
        except Exception as ex:
            print(f"fig_first40 skip: {str(ex)[:60]}", flush=True)
    if args.ppt:
        from wfield_local.locanmf_decoder_ppt import build_ppt
        print("wrote", build_ppt(args.output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
