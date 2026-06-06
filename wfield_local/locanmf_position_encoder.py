"""Position ENCODER (reverse of the decoder): predict expected neural activity from intended spout
position. The pre/post-stroke tool -- fit pre-stroke, then post-stroke residual (observed - predicted)
per intended position = the lesion's effect, computable even on no-lick/failed trials.

  spatial: ridge  one-hot(position) -> LocaNMF component activity (2s post-lick, no-baseline), block-CV.
    -> predicted per-position cortical map (footprint-reconstructed) + cross-validated encoding R^2 by region.
  temporal: per-position expected time-course of pooled SSp / MO activity (lick-aligned) -- "expected
    dynamics" per position; the linear encoder's prediction at each time bin.

    python -m wfield_local.locanmf_position_encoder --date 0605 --output "<dir>"
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
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from wfield_local.locanmf_cue_lick_analysis import SESSIONS
from wfield_local.locanmf_position_decoder import _trial_features, _build_signal
from wfield_local.plot_lick_aligned_averages import POSITION_NAMES, DISPLAY_ORDER, _load_daq_events
from wfield_local.plot_spout_trial_averages import _load_daq_events as _load_cue, _classify_cues
from wfield_local.locanmf_crossanimal_dff import _frames

FS = 31.23


def _sess(label):
    return next(s for s in SESSIONS if s["label"] == label)


def _args(post_s=2.0, baseline="none"):
    return SimpleNamespace(source="locanmf", align="lick", baseline=baseline, pre_s=1.0, post_s=post_s, fs=FS, max_rt=2.0)


def _names(s):
    return {int(k): v for k, v in json.load(open(glob.glob(
        f"{s['mc']}/wfield_local_results/allen_aligned_affine8v1/allen_area_names.json")[0]))}


def encode_spatial(label):
    """Ridge encoder position->activity; returns CV R^2 per component, per-position predicted activity (B)."""
    s = _sess(label)
    X, y, g, _, _, reg = _trial_features(s, _args(2.0))
    pos = np.array(DISPLAY_ORDER); P = np.stack([(y == p).astype(float) for p in pos], 1)
    pred = np.zeros_like(X); ng = min(5, int(np.unique(g).size))
    for tr, te in GroupKFold(ng).split(X, y, g):
        pred[te] = Ridge(alpha=1.0).fit(P[tr], X[tr]).predict(P[te])
    r2 = 1 - ((X - pred) ** 2).sum(0) / np.maximum(((X - X.mean(0)) ** 2).sum(0), 1e-12)
    B = Ridge(alpha=1.0).fit(P, X).coef_.T                       # (6, ncomp) predicted activity per position
    return dict(label=label, r2=r2, B=B, reg=reg, pos=pos, cv_r2=float(np.mean(r2)))


def fig_predicted_maps(label, out):
    """Predicted per-position EVOKED map (pre-lick-subtracted delta -> directly comparable to the SVD
    spout_positions delta maps; the no-baseline absolute version is dominated by common activity)."""
    s = _sess(label); e = encode_spatial(label)
    X, y, _, _, _, _ = _trial_features(s, _args(2.0, "precue"))     # delta features
    A = np.load(f"{s['mc']}/locanmf_affine8v1_final/{label}_locanmf_A.npy")
    mask = np.load(glob.glob(f"{s['mc']}/wfield_local_results/allen_aligned_affine8v1/allen_brain_mask_native_grid.npy")[0]).astype(bool)
    ys, xs = np.where(mask); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    B = np.stack([X[y == p].mean(0) for p in DISPLAY_ORDER])        # 6 x ncomp delta per-position means
    maps = [(B[p][None, None, :] * A).sum(2) for p in range(6)]
    vmax = np.nanpercentile([np.abs(m[mask]) for m in maps], 99)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, p in zip(axes.ravel(), range(6)):
        m = maps[p].astype(float); m[~mask] = np.nan
        im = ax.imshow(m[y0:y1, x0:x1], cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(POSITION_NAMES[DISPLAY_ORDER[p]], fontsize=11); ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(f"{label}: ENCODER expected EVOKED activity per intended position (pre-lick delta; "
                 f"matches SVD; single-trial CV R^2={e['cv_r2']:.3f})", fontsize=12)
    fig.tight_layout(); p = out / f"locanmf_encoder_predicted_maps_{label}.png"; fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_encoding_r2(label, out):
    s = _sess(label); e = encode_spatial(label); names = _names(s)
    regs = np.array([names.get(int(e["reg"][i]), "?") for i in range(len(e["r2"]))])
    reg_r2 = {r: float(np.mean(e["r2"][regs == r])) for r in set(regs)}
    top = sorted(reg_r2, key=reg_r2.get, reverse=True)[:16]
    cols = ["#c0392b" if r.startswith(("SSp", "SSs")) else "#2980b9" if r.startswith("MO") else "#888" for r in top]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(top))[::-1], [reg_r2[r] for r in top], color=cols)
    ax.set_yticks(range(len(top))[::-1]); ax.set_yticklabels(top, fontsize=8); ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("cross-validated encoding R^2 (activity explained by position)")
    ax.set_title(f"{label}: encoding R^2 by region (red=SSp, blue=MO)")
    fig.tight_layout(); p = out / f"locanmf_encoder_r2_by_region_{label}.png"; fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_temporal_encoder(label, out, pre_s=1.0, post_s=1.5):
    """Per-position expected time-course of pooled SSp / MO activity (lick-aligned)."""
    s = _sess(label); names = _names(s)
    sig, reg = _build_signal(s, "locanmf"); T = sig.shape[1]
    rn = np.array([names.get(int(reg[i]), "?") for i in range(sig.shape[0])])
    ssp = np.where(np.char.startswith(rn.astype(str), "SSp"))[0]
    mo = np.array([i for i in range(len(rn)) if rn[i].startswith(("MOp", "MOs"))])
    cue = _load_cue(s["h5"]); lk = _load_daq_events(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    cf, lf, _ = _frames(s, cue, lk); codes = _classify_cues(cue["cue_samples"], cue["strobe_samples"], cue["strobe_codes"])
    ls = np.sort(lf); j = np.searchsorted(ls, cf, side="right"); first = np.where(j < ls.size, ls[np.clip(j, 0, ls.size - 1)], -1); rt = first - cf
    pre = int(pre_s * FS); post = int(post_s * FS); tax = np.arange(-pre, post) / FS
    keep = [k for k in range(cf.size) if codes[k] >= 0 and first[k] > 0 and 0 < rt[k] <= 2 * FS and first[k] - pre >= 0 and first[k] + post <= T]
    flk = np.array([int(first[k]) for k in keep]); yk = np.array([int(codes[k]) for k in keep])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, idx, tit in [(axes[0], ssp, "SSp (pooled)"), (axes[1], mo, "MO (pooled)")]:
        if len(idx) == 0:
            ax.set_title(f"{tit}: none"); continue
        pooled = sig[idx][:, :].mean(0)                      # 1 x T pooled activity
        for p in DISPLAY_ORDER:
            tr = np.stack([pooled[f - pre:f + post] for f in flk[yk == p]])
            if len(tr) < 5:
                continue
            m = tr.mean(0) - tr[:, :pre].mean()
            ax.plot(tax, m, label=POSITION_NAMES[p])
        ax.axvline(0, color="k", lw=1); ax.axhline(0, color="grey", lw=0.6)
        ax.set_xlabel("time from first lick (s)"); ax.set_title(f"{tit} expected activity by position"); ax.legend(fontsize=7)
    axes[0].set_ylabel("pooled dF/F (baseline-sub)")
    fig.suptitle(f"{label}: TEMPORAL encoder -- expected activity time-course per intended position", fontsize=12)
    fig.tight_layout(); p = out / f"locanmf_encoder_temporal_{label}.png"; fig.savefig(p, dpi=130); plt.close(fig)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--date", default="0605")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for s in [x for x in SESSIONS if x["label"].endswith(args.date)]:
        lab = s["label"]
        try:
            e = encode_spatial(lab)
            print(f"{lab}: CV encoding R^2 mean={e['cv_r2']:.3f}", flush=True)
            for f in (fig_predicted_maps, fig_encoding_r2, fig_temporal_encoder):
                print("  wrote", f(lab, args.output).name, flush=True)
        except Exception as ex:
            print(f"{lab}: FAILED {type(ex).__name__}: {str(ex)[:80]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
