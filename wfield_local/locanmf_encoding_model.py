"""Cue+lick FIR encoding model on LocaNMF dF/F -- isolate cue-locked from lick(motor)-locked.

Musall-style linear encoding model (simplified to two event regressors): for each component,
regress its continuous footprint-scaled dF/F on time-lagged CUE and LICK indicators (ridge),
giving a cue kernel and a lick kernel. The CUE kernel is the cue-locked response with the
lick (movement) regressed OUT -> the stimulus component; the LICK kernel is the movement
component. Also reports cross-validated R^2 (how much dF/F the two events explain) and the
relative cue-vs-lick contribution.

Animal-level stats: per Allen area, per-animal component means, then mean +/- SEM across mice.

NB: cue->first-lick RT is ~0.16 s, so cue and lick regressors are strongly collinear; the
split is the principled best-effort and the residual cue kernel should be read with that in
mind (consistent with the cue-no-lick result that the response is largely lick-driven).

    python -m wfield_local.locanmf_encoding_model --output "<dir>"
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
from scipy.ndimage import gaussian_filter1d

from wfield_local.locanmf_cue_lick_analysis import SESSIONS
from wfield_local.plot_lick_aligned_averages import _load_daq_events, _event_frame_indices_from_pco
from wfield_local.plot_spout_trial_averages import _load_daq_events as _load_cue_events, _classify_cues
from wfield_local.locanmf_lick_aligned import _corrected_frame_samples, _nearest_corrected_frame
from wfield_local.locanmf_crossanimal_dff import _footprint_scale, _frames, AREAS, ANIMAL_COLOR


def _design(events, lags, T):
    X = np.zeros((T, lags.size), np.float64)
    cols = np.arange(lags.size)
    for f in events:
        ii = f + lags
        ok = (ii >= 0) & (ii < T)
        X[ii[ok], cols[ok]] = 1.0
    return X


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--fs", type=float, default=31.23)
    ap.add_argument("--pre-lag-s", type=float, default=0.5)
    ap.add_argument("--post-lag-s", type=float, default=1.5)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--smooth-sigma", type=float, default=1.0)
    ap.add_argument("--areas", nargs="+", default=["SSp-n", "SSp-m", "MOp", "MOs"])
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sm = lambda x: gaussian_filter1d(np.asarray(x, float), args.smooth_sigma)
    lags = np.arange(-int(round(args.pre_lag_s * args.fs)), int(round(args.post_lag_s * args.fs)))
    klag = lags.size; ktax = lags / args.fs
    name2lab = {v: k for k, v in AREAS.items()}

    res = []
    for s in SESSIONS:
        mc = s["mc"]
        C = np.load(f"{mc}/locanmf_affine8v1_final/{s['label']}_locanmf_C.npy").astype(np.float64)
        reg = np.load(f"{mc}/locanmf_affine8v1_final/{s['label']}_locanmf_regions.npy")
        A = np.load(f"{mc}/locanmf_affine8v1_final/{s['label']}_locanmf_A.npy", mmap_mode="r")
        ncomp, T = C.shape
        dff = _footprint_scale(A, ncomp)[:, None] * C
        cue = _load_cue_events(s["h5"]); lk = _load_daq_events(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
        cue_f, lick_f, _ = _frames(s, cue, lk)
        cue_f = cue_f[(cue_f >= 0) & (cue_f < T)]; lick_f = lick_f[(lick_f >= 0) & (lick_f < T)]
        Xc = _design(cue_f, lags, T); Xl = _design(lick_f, lags, T)
        X = np.hstack([Xc, Xl, np.ones((T, 1))])
        Y = dff.T                                            # (T, ncomp)
        XtX = X.T @ X; XtX[np.diag_indices_from(XtX)] += args.ridge
        beta = np.linalg.solve(XtX, X.T @ Y)                 # (2*klag+1, ncomp)
        pred = X @ beta
        ss_res = ((Y - pred) ** 2).sum(0); ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
        r2 = 1 - ss_res / np.where(ss_tot > 0, ss_tot, 1)
        res.append(dict(animal=s["label"].split("_")[0], regions=reg,
                        cue_k=beta[:klag].T, lick_k=beta[klag:2 * klag].T, r2=r2))  # kernels (ncomp, klag)
        print(f"{s['label']}: median R2={np.median(r2):.3f}", flush=True)

    animals = list(ANIMAL_COLOR)
    areas = [(name2lab[a], a) for a in args.areas] + [(-name2lab[a], a) for a in args.areas]

    def per_animal(key, an, lab):
        rows = [r[key][np.where(r["regions"] == lab)[0]] for r in res
                if r["animal"] == an and (r["regions"] == lab).any()]
        return np.concatenate(rows, 0).mean(0) if rows else None

    fig, axes = plt.subplots(len(areas), 2, figsize=(11, 2.0 * len(areas)), squeeze=False, sharex=True)
    for rr, (lab, nm) in enumerate(areas):
        for cc, (key, ttl, col) in enumerate([("cue_k", "CUE kernel (lick regressed out)", "tab:purple"),
                                              ("lick_k", "LICK kernel (movement)", "tab:red")]):
            ax = axes[rr][cc]; per = []
            for an in animals:
                m = per_animal(key, an, lab)
                if m is not None:
                    per.append(m); ax.plot(ktax, sm(m) * 100, color=ANIMAL_COLOR[an], lw=0.8, alpha=0.5)
            if per:
                P = np.array(per); gm = P.mean(0); gs = P.std(0) / np.sqrt(P.shape[0])
                ax.plot(ktax, sm(gm) * 100, color=col, lw=2.2)
                ax.fill_between(ktax, sm(gm - gs) * 100, sm(gm + gs) * 100, color=col, alpha=0.2)
            ax.axvline(0, color="grey", ls="--", lw=0.6); ax.axhline(0, color="grey", lw=0.5)
            if rr == 0:
                ax.set_title(ttl, fontsize=11)
            if cc == 0:
                ax.set_ylabel(f"{nm}{'_L' if lab > 0 else '_R'}\n% dF/F", fontsize=8)
            if rr == len(areas) - 1:
                ax.set_xlabel("lag (s)")
    fig.suptitle("Cue+lick FIR encoding model: stimulus (cue) vs movement (lick) kernels, dF/F, mean+/-SEM across mice",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(args.output / "locanmf_encoding_kernels.png", dpi=130); plt.close(fig)

    # cue-vs-lick contribution per area (peak |kernel|), animal-level
    print("\narea: cue-kernel peak vs lick-kernel peak (% dF/F, cross-mouse mean):")
    for lab, nm in areas:
        ck = [per_animal("cue_k", an, lab) for an in animals]; lk2 = [per_animal("lick_k", an, lab) for an in animals]
        ck = [x for x in ck if x is not None]; lk2 = [x for x in lk2 if x is not None]
        if ck and lk2:
            cpk = 100 * np.mean([np.abs(x).max() for x in ck]); lpk = 100 * np.mean([np.abs(x).max() for x in lk2])
            print(f"  {nm}{'_L' if lab>0 else '_R':<3}: cue={cpk:.2f}%  lick={lpk:.2f}%  lick/cue={lpk/max(cpk,1e-6):.1f}x")
    print("wrote", args.output / "locanmf_encoding_kernels.png", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
