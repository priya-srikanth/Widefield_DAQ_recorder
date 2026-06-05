"""Cross-animal LocaNMF event-triggered activity in DeltaF/F, with animal-level stats.

Matches the Churchland-lab conventions (Musall 2019; Saxena 2020 LocaNMF; Couto 2021):
  * UNITS = DeltaF/F (the data are already mean-centered dF/F; LocaNMF's per-component scale
    ambiguity is removed with the scale-invariant footprint weight s_i = sum Ai^2 / sum Ai,
    so dff_i(t) = s_i * C_i(t) is the footprint-mean dF/F). No quiet-SD z-scoring.
  * STATS = animal is the unit of replication: average each animal's components (across its
    sessions) per Allen area, then plot mean +/- SEM ACROSS MICE (n = animals), not across
    components/trials.

Figure: rows = Allen areas, cols = {lick, cue}; bold = cross-animal mean +/- SEM (n mice),
thin = per-animal means. Also writes a cue+lick FIR encoding model (separate module step).

    python -m wfield_local.locanmf_crossanimal_dff --output "<dir>"
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

AREAS = {3: "MOp", 4: "MOs", 5: "SSp-n", 6: "SSp-m"}
ANIMAL_COLOR = {"PS92": "tab:blue", "PS94": "tab:orange", "PS95": "tab:green"}


def _footprint_scale(A, ncomp):
    s = np.empty(ncomp)
    for i in range(ncomp):
        Ai = np.asarray(A[:, :, i], dtype=np.float64)
        d = np.nansum(Ai)
        s[i] = (np.nansum(Ai ** 2) / d) if d > 1e-12 else 0.0
    return s


def _frames(s, cue, lk):
    mc = s["mc"]
    if s["regime"] == "B":
        fmdir = s["fmdir"] or mc
        fm = glob.glob(f"{fmdir}/*cleanpairs_frame_map.npz")[0]
        off = json.loads(Path(glob.glob(f"{fmdir}/*cleanpairs_summary.json")[0]).read_text())["chosen_exposure_offset"]
        csmp = _corrected_frame_samples(fm, cue["pco_samples"], int(off))
        return _nearest_corrected_frame(cue["cue_samples"], csmp), _nearest_corrected_frame(lk["lick_samples"], csmp), csmp
    return (_event_frame_indices_from_pco(cue["cue_samples"], cue["pco_samples"]) // 2,
            _event_frame_indices_from_pco(lk["lick_samples"], lk["pco_samples"]) // 2, None)


def _trig_avg(dff, frames, pre_n, post_n, T, csmp, fsd, pre_s, post_s):
    """Per-component event-triggered dF/F, per-trial pre-event baseline subtracted."""
    segs = []
    for f in frames:
        a, b = f - pre_n, f + post_n
        if a < 0 or b > T:
            continue
        if csmp is not None and (csmp[b - 1] - csmp[a]) / fsd > (pre_s + post_s + 1.0):
            continue
        w = dff[:, a:b]
        segs.append(w - w[:, :pre_n].mean(1, keepdims=True))
    if not segs:
        return None
    return np.mean(segs, 0)  # (ncomp, win)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, type=Path)
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
        dff = _footprint_scale(A, ncomp)[:, None] * C       # (ncomp, T) footprint-mean dF/F
        cue = _load_cue_events(s["h5"]); lk = _load_daq_events(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
        fsd = cue["sample_rate_hz"]; cue_f, lick_f, csmp = _frames(s, cue, lk)
        lick_avg = _trig_avg(dff, lick_f, pre_n, post_n, T, csmp, fsd, args.pre_s, args.post_s)
        cue_avg = _trig_avg(dff, cue_f, pre_n, post_n, T, csmp, fsd, args.pre_s, args.post_s)
        res.append(dict(animal=s["label"].split("_")[0], regions=reg, lick=lick_avg, cue=cue_avg))
        print(f"{s['label']} done", flush=True)

    animals = list(ANIMAL_COLOR)
    areas = [(k, AREAS[abs(k)]) for k in [name2lab[a] for a in args.areas]] \
        + [(-k, AREAS[abs(k)]) for k in [name2lab[a] for a in args.areas]]

    def per_animal_area(ev, an, lab):
        rows = [r[ev][np.where(r["regions"] == lab)[0]] for r in res
                if r["animal"] == an and r[ev] is not None and (r["regions"] == lab).any()]
        return np.concatenate(rows, 0).mean(0) if rows else None  # mean over this animal's components

    fig, axes = plt.subplots(len(areas), 2, figsize=(11, 2.0 * len(areas)), squeeze=False, sharex=True)
    for rr, (lab, nm) in enumerate(areas):
        for cc, ev in enumerate(["lick", "cue"]):
            ax = axes[rr][cc]
            per = []
            for an in animals:
                m = per_animal_area(ev, an, lab)
                if m is not None:
                    per.append(m)
                    ax.plot(tax, sm(m) * 100, color=ANIMAL_COLOR[an], lw=0.8, alpha=0.5)
            if per:
                P = np.array(per); gm = P.mean(0); gs = P.std(0) / np.sqrt(P.shape[0])
                ax.plot(tax, sm(gm) * 100, color="k", lw=2.2, label=f"mean of {P.shape[0]} mice")
                ax.fill_between(tax, sm(gm - gs) * 100, sm(gm + gs) * 100, color="k", alpha=0.2)
            ax.axvline(0, color="grey", ls="--", lw=0.6); ax.axhline(0, color="grey", lw=0.5)
            if rr == 0:
                ax.set_title(f"{ev}-triggered", fontsize=12)
            if cc == 0:
                ax.set_ylabel(f"{nm}{'_L' if lab > 0 else '_R'}\n% dF/F", fontsize=8)
            if rr == len(areas) - 1:
                ax.set_xlabel("time from event (s)")
            if rr == 0 and cc == 1:
                ax.legend(fontsize=7)
    fig.suptitle("Cross-animal LocaNMF event-triggered DeltaF/F (per-animal thin, mean+/-SEM across mice bold)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    png = args.output / "locanmf_crossanimal_dff.png"
    fig.savefig(png, dpi=130); plt.close(fig)
    print("wrote", png, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
