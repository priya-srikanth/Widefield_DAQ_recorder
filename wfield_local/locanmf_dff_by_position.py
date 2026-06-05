"""LocaNMF lick-triggered DeltaF/F by spout position, per Allen area, animal-level stats.

Physical footprint-scaled dF/F (s_i*C_i; Churchland-convention units), lick-triggered and
split by the 6 spout positions, per-trial pre-lick baseline subtracted. Per area: the 6
position traces as the cross-animal mean (animal = unit of replication), contralateral-spout
positions drawn solid/bold and ipsilateral dashed, so the contralateral somatosensory tuning
is visible in % dF/F (not z). Keeps components individual (averaged only at the animal level).

    python -m wfield_local.locanmf_dff_by_position --output "<dir>" [--event lick|cue]
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
from wfield_local.plot_lick_aligned_averages import (
    _load_daq_events, _classify_events, _event_frame_indices_from_pco, POSITION_NAMES, DISPLAY_ORDER,
)
from wfield_local.plot_spout_trial_averages import _load_daq_events as _load_cue_events, _classify_cues
from wfield_local.locanmf_lick_aligned import _corrected_frame_samples, _nearest_corrected_frame
from wfield_local.locanmf_crossanimal_dff import _footprint_scale, _frames, AREAS, ANIMAL_COLOR

POS = [POSITION_NAMES[c] for c in DISPLAY_ORDER]
POS_COLOR = {"close_L": "#1f77b4", "close_center": "#2ca02c", "close_R": "#d62728",
             "far_L": "#17becf", "far_center": "#7f7f7f", "far_R": "#9467bd"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--event", choices=("lick", "cue"), default="lick")
    ap.add_argument("--fs", type=float, default=31.23)
    ap.add_argument("--pre-s", type=float, default=1.0)
    ap.add_argument("--post-s", type=float, default=1.5)
    ap.add_argument("--smooth-sigma", type=float, default=1.0)
    ap.add_argument("--areas", nargs="+", default=["SSp-n", "SSp-m", "MOp", "MOs"])
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sm = lambda x: gaussian_filter1d(np.asarray(x, float), args.smooth_sigma)
    pre_n = int(round(args.pre_s * args.fs)); post_n = int(round(args.post_s * args.fs))
    tax = np.arange(-pre_n, post_n) / args.fs
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
        fsd = cue["sample_rate_hz"]; cue_f, lick_f, csmp = _frames(s, cue, lk)
        if args.event == "lick":
            ev_f = lick_f; codes = _classify_events(lk["lick_samples"], lk["strobe_samples"], lk["strobe_codes"])
        else:
            ev_f = cue_f; codes = _classify_cues(cue["cue_samples"], cue["strobe_samples"], cue["strobe_codes"])

        def ok(fr):
            a, b = fr - pre_n, fr + post_n
            if a < 0 or b > T:
                return False
            return csmp is None or (csmp[b - 1] - csmp[a]) / fsd <= (args.pre_s + args.post_s + 1.0)

        valid = (codes >= 0) & np.array([ok(int(f)) for f in ev_f])
        pos_avg = {}
        for code in DISPLAY_ORDER:
            fr = ev_f[valid & (codes == code)]
            if fr.size == 0:
                pos_avg[POSITION_NAMES[code]] = None; continue
            segs = [dff[:, f - pre_n:f + post_n] - dff[:, f - pre_n:f].mean(1, keepdims=True) for f in fr]
            pos_avg[POSITION_NAMES[code]] = np.mean(segs, 0)   # (ncomp, win)
        res.append(dict(animal=s["label"].split("_")[0], regions=reg, pos=pos_avg))
        print(f"{s['label']} done", flush=True)

    animals = list(ANIMAL_COLOR)
    areas = [(name2lab[a], a) for a in args.areas] + [(-name2lab[a], a) for a in args.areas]

    def cross_mouse(pos, lab):
        per = []
        for an in animals:
            rows = [r["pos"][pos][np.where(r["regions"] == lab)[0]] for r in res
                    if r["animal"] == an and r["pos"][pos] is not None and (r["regions"] == lab).any()]
            if rows:
                per.append(np.concatenate(rows, 0).mean(0))
        return np.array(per).mean(0) if per else None

    fig, axes = plt.subplots(2, 4, figsize=(22, 9), squeeze=False)
    for ax, (lab, nm) in zip(axes.ravel(), areas):
        for pos in POS:
            m = cross_mouse(pos, lab)
            if m is None:
                continue
            is_contra = (("R" in pos and lab > 0) or ("L" in pos and lab < 0))
            ax.plot(tax, sm(m) * 100, color=POS_COLOR[pos], lw=2.0 if is_contra else 1.1,
                    ls="-" if is_contra else "--", label=pos + (" (contra)" if is_contra else ""))
        ax.axvline(0, color="grey", ls=":", lw=0.6); ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(f"{nm}{'_L' if lab > 0 else '_R'} (reg {lab})", fontsize=10)
        ax.set_xlabel(f"time from {args.event} (s)"); ax.set_ylabel("% dF/F", fontsize=8)
        ax.legend(fontsize=6, ncol=2)
    fig.suptitle(f"{args.event}-triggered LocaNMF dF/F by spout position, cross-mouse mean "
                 f"(solid/bold = contralateral spout)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    png = args.output / f"locanmf_dff_by_position_{args.event}.png"
    fig.savefig(png, dpi=130); plt.close(fig)
    print("wrote", png, flush=True)

    # ---- per-animal: rows = areas, cols = animals, 6 position lines (dF/F) ----
    def per_animal_pos(an, pos, lab):
        rows = [r["pos"][pos][np.where(r["regions"] == lab)[0]] for r in res
                if r["animal"] == an and r["pos"][pos] is not None and (r["regions"] == lab).any()]
        return np.concatenate(rows, 0).mean(0) if rows else None

    fig, axes = plt.subplots(len(areas), len(animals), figsize=(5.0 * len(animals), 1.9 * len(areas)),
                             squeeze=False, sharex=True)
    for rr, (lab, nm) in enumerate(areas):
        for cc, an in enumerate(animals):
            ax = axes[rr][cc]
            for pos in POS:
                m = per_animal_pos(an, pos, lab)
                if m is None:
                    continue
                is_contra = (("R" in pos and lab > 0) or ("L" in pos and lab < 0))
                ax.plot(tax, sm(m) * 100, color=POS_COLOR[pos], lw=1.8 if is_contra else 1.0,
                        ls="-" if is_contra else "--", label=pos + (" (contra)" if is_contra else ""))
            ax.axvline(0, color="grey", ls=":", lw=0.6); ax.axhline(0, color="grey", lw=0.5)
            if rr == 0:
                ax.set_title(an, fontsize=11)
            if cc == 0:
                ax.set_ylabel(f"{nm}{'_L' if lab > 0 else '_R'}\n% dF/F", fontsize=8)
            if rr == len(areas) - 1:
                ax.set_xlabel(f"time from {args.event} (s)")
            if rr == 0 and cc == len(animals) - 1:
                ax.legend(fontsize=5, ncol=2)
    fig.suptitle(f"{args.event}-triggered LocaNMF dF/F by spout position, PER ANIMAL "
                 f"(solid/bold = contralateral spout)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    png2 = args.output / f"locanmf_dff_by_position_{args.event}_per_animal.png"
    fig.savefig(png2, dpi=120); plt.close(fig)
    print("wrote", png2, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
