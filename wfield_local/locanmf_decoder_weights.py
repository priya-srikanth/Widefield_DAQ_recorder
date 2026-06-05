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
def fig_weights_by_region(labels, out):
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
    fig.tight_layout(); p = out / "locanmf_decoder_weights_by_region.png"; fig.savefig(p, dpi=130); plt.close(fig)
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
    coef = lr.coef_; importance = np.abs(coef).sum(0); top = np.argsort(importance)[::-1][:topn]
    b = np.zeros_like(atlas, bool)
    b[:-1, :] |= atlas[:-1, :] != atlas[1:, :]; b[:, :-1] |= atlas[:, :-1] != atlas[:, 1:]
    ys, xs = np.where(mask); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    fig, axes = plt.subplots(2, (topn + 1) // 2, figsize=(1.8 * topn, 8))
    for ax, comp in zip(axes.ravel(), top):
        fp = A[:, :, comp].astype(float).copy(); fp[~mask] = np.nan
        ax.imshow(fp[y0:y1, x0:x1], cmap="magma")
        bb = np.where(b[y0:y1, x0:x1]); ax.scatter(bb[1], bb[0], s=0.2, c="white", alpha=0.35, marker=".")
        pos = DISPLAY_ORDER[int(np.argmax([coef[ci[p], comp] if p in ci else -9 for p in DISPLAY_ORDER]))]
        ax.set_title(f"comp#{comp} {names.get(int(reg[comp]),'?')}\n->{POSITION_NAMES[pos]} |w|={importance[comp]:.2f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[len(top):]:
        ax.axis("off")
    fig.suptitle(f"{label}: top-{topn} LocaNMF components by decoder weight (footprints, Allen-overlaid)", fontsize=13)
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
    nbin = 8; binlen = int(round(0.25 * FS)); colors = {"PS92": "#1f77b4", "PS94": "#d62728", "PS95": "#2ca02c"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4)); ax = axes[0]; mb = {}
    for label in labels:
        sig, T, fl, y, g = _lick_trials(label)
        accs = []
        for o in offs:
            ok = (fl + o >= 0) & (fl + o + win <= T)
            accs.append(_bcv_acc(np.array([sig[:, f + o:f + o + win].mean(1) for f in fl[ok]]), y[ok], g[ok]))
        ax.plot(offs / FS, accs, marker="o", ms=3, color=colors.get(label[:4], "k"), label=label[:4])
        ok = (fl >= 0) & (fl + nbin * binlen <= T)
        Xmb = np.array([np.concatenate([sig[:, f + bi * binlen:f + (bi + 1) * binlen].mean(1) for bi in range(nbin)]) for f in fl[ok]])
        Xmean = np.array([sig[:, f:f + nbin * binlen].mean(1) for f in fl[ok]])
        mb[label[:4]] = (_bcv_acc(Xmean, y[ok], g[ok]), _bcv_acc(Xmb, y[ok], g[ok]))
    ax.axvline(0, color="k", lw=1); ax.axvline(-0.16, color="grey", ls=":", lw=1); ax.axhline(1 / 6, color="k", ls="--", lw=0.8)
    ax.set_xlabel("time from first lick (s)"); ax.set_ylabel("accuracy (0.5s window, block-CV)"); ax.set_ylim(0.1, 0.9)
    ax.legend(fontsize=9); ax.set_title("Sliding-window decoding: when is position information present?")
    ax = axes[1]; x = np.arange(len(labels)); w = 0.35
    ax.bar(x - w / 2, [mb[l[:4]][0] for l in labels], w, label="single 2s mean", color="#888")
    ax.bar(x + w / 2, [mb[l[:4]][1] for l in labels], w, label="8x0.25s temporal bins", color="#d62728")
    ax.axhline(1 / 6, color="k", ls="--", lw=0.8); ax.set_xticks(x); ax.set_xticklabels([l[:4] for l in labels])
    ax.set_ylim(0, 1); ax.set_ylabel("accuracy (block-CV)"); ax.legend(fontsize=9); ax.set_title("Does temporal profile beat the window-mean?")
    dlab = "/".join(sorted({l[-4:-2] + "/" + l[-2:] for l in labels}))
    fig.suptitle(f"Rolling temporal dynamics of spout-position coding (first-lick aligned, {dlab}; one line per animal)", fontsize=12)
    fig.tight_layout(); p = out / "locanmf_decoder_temporal_dynamics.png"; fig.savefig(p, dpi=130); plt.close(fig)
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
    print("wrote", fig_temporal_dynamics(_avail("0603"), args.output), flush=True)
    for lab in [s["label"] for s in SESSIONS if s["label"][-4:] in ("0601", "0602", "0603", "0604")]:
        print("wrote", fig_top_components(lab, args.output), flush=True)
    if args.ppt:
        from wfield_local.locanmf_decoder_ppt import build_ppt
        print("wrote", build_ppt(args.output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
