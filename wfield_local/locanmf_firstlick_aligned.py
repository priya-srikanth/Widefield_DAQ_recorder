"""LocaNMF activity aligned to the FIRST LICK after each cue, normalized to pre-cue 1 s.

For each trial: take the first lick that follows the cue (within --max-rt s), align the
component traces to that first lick, but baseline-normalize to the trial's PRE-CUE 1 s window
(subtract pre-cue mean, z by the per-component pooled pre-cue SD). This puts the response on
the motor-onset (first lick) clock while keeping a clean, task-independent baseline -- so the
cue->lick anticipatory ramp and the post-lick response are both visible above pre-cue.

Per orofacial area: one mean +/- SEM trace per animal (averaging that animal's components),
plus a printed reaction-time (cue->first-lick) summary.

    python -m wfield_local.locanmf_firstlick_aligned --output "<dir>" [--max-rt 3.0]
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

ORO = {3: "MOp", 4: "MOs", 5: "SSp-n", 6: "SSp-m"}
ANIMAL_COLOR = {"PS92": "tab:blue", "PS94": "tab:orange", "PS95": "tab:green"}


def _process(s, args):
    mc = s["mc"]
    C = np.load(f"{mc}/locanmf_affine8v1_final/{s['label']}_locanmf_C.npy").astype(np.float64)
    regions = np.load(f"{mc}/locanmf_affine8v1_final/{s['label']}_locanmf_regions.npy")
    ncomp, T = C.shape
    cue = _load_cue_events(s["h5"])
    lk = _load_daq_events(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    fsd = cue["sample_rate_hz"]
    if s["regime"] == "B":
        fmdir = s["fmdir"] or mc
        fm = glob.glob(f"{fmdir}/*cleanpairs_frame_map.npz")[0]
        off = json.loads(Path(glob.glob(f"{fmdir}/*cleanpairs_summary.json")[0]).read_text())["chosen_exposure_offset"]
        cs = _corrected_frame_samples(fm, cue["pco_samples"], int(off))
        cue_f = _nearest_corrected_frame(cue["cue_samples"], cs)
        lick_f = _nearest_corrected_frame(lk["lick_samples"], cs)
    else:
        cs = None
        cue_f = _event_frame_indices_from_pco(cue["cue_samples"], cue["pco_samples"]) // 2
        lick_f = _event_frame_indices_from_pco(lk["lick_samples"], lk["pco_samples"]) // 2
    codes = _classify_cues(cue["cue_samples"], cue["strobe_samples"], cue["strobe_codes"])

    pre_n = int(round(args.pre_s * args.fs)); post_n = int(round(args.post_s * args.fs)); win = pre_n + post_n
    tax = np.arange(-pre_n, post_n) / args.fs
    maxrt_n = int(round(args.max_rt * args.fs))

    lick_sorted = np.sort(lick_f)
    j = np.searchsorted(lick_sorted, cue_f, side="right")
    first_lick = np.full(cue_f.size, -1, np.int64)
    have = j < lick_sorted.size
    first_lick[have] = lick_sorted[j[have]]
    rt = first_lick - cue_f  # frames cue->first lick

    def win_ok(fr):  # lick-aligned window in bounds + contiguous
        a, b = fr - pre_n, fr + post_n
        if a < 0 or b > T:
            return False
        return cs is None or (cs[b - 1] - cs[a]) / fsd <= (args.pre_s + args.post_s + 1.0)

    valid = (codes >= 0) & (first_lick > 0) & (rt > 0) & (rt <= maxrt_n) \
        & (cue_f - pre_n >= 0) & np.array([win_ok(int(f)) for f in first_lick])

    # per-component pooled pre-cue SD over valid trials (baseline scale)
    cf = cue_f[valid]
    precue = np.stack([C[:, c - pre_n:c] for c in cf], 0)        # (ntrial, ncomp, pre_n)
    precue_mean = precue.mean(2)                                 # (ntrial, ncomp) per-trial baseline
    presd = (precue - precue_mean[:, :, None]).reshape(precue.shape[1], -1).std(1) if False else \
        (precue - precue.mean(2, keepdims=True)).transpose(1, 0, 2).reshape(ncomp, -1).std(1)
    presd = np.where(presd > 0, presd, 1.0)

    fl = first_lick[valid]
    lwin = np.stack([C[:, f - pre_n:f + post_n] for f in fl], 0)  # (ntrial, ncomp, win) around first lick
    norm = (lwin - precue_mean[:, :, None]) / presd[None, :, None]  # subtract pre-cue mean, z by pre-cue SD
    return dict(animal=s["label"].split("_")[0], label=s["label"], regions=regions, tax=tax,
                norm=norm.astype(np.float32), rt_s=(rt[valid] / args.fs), n=int(valid.sum()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--fs", type=float, default=31.23)
    ap.add_argument("--pre-s", type=float, default=1.0)
    ap.add_argument("--post-s", type=float, default=1.5)
    ap.add_argument("--max-rt", type=float, default=3.0, help="max cue->first-lick (s) to include a trial")
    ap.add_argument("--smooth-sigma", type=float, default=1.0)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sm = lambda x: gaussian_filter1d(np.asarray(x, float), args.smooth_sigma)

    res = [_process(s, args) for s in SESSIONS]
    tax = res[0]["tax"]
    for r in res:
        print(f"{r['label']}: trials={r['n']}  median RT={np.median(r['rt_s']):.2f}s", flush=True)
    areas = [(k, ORO[abs(k)]) for k in ORO] + [(-k, ORO[abs(k)]) for k in ORO]
    animals = list(ANIMAL_COLOR)

    fig, axes = plt.subplots(2, 4, figsize=(22, 9), squeeze=False)
    for ax, (lab, nm) in zip(axes.ravel(), areas):
        for an in animals:
            comp_means = []
            for r in res:
                if r["animal"] != an:
                    continue
                idx = np.where(r["regions"] == lab)[0]
                if idx.size and r["n"] > 0:
                    comp_means.append(r["norm"][:, idx, :].mean(0))   # (ncomp_in_area, win) trial-averaged
            if not comp_means:
                continue
            stk = np.concatenate(comp_means, 0)                       # components across this animal's sessions
            m = stk.mean(0); sem = stk.std(0) / np.sqrt(stk.shape[0])
            ax.plot(tax, sm(m), color=ANIMAL_COLOR[an], lw=1.9, label=f"{an} (n={stk.shape[0]})")
            ax.fill_between(tax, sm(m - sem), sm(m + sem), color=ANIMAL_COLOR[an], alpha=0.15)
        ax.axvline(0, color="k", ls="--", lw=0.7)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(f"{nm}{'_L' if lab > 0 else '_R'} (reg {lab})", fontsize=10)
        ax.set_xlabel("time from first lick (s)"); ax.set_ylabel("pre-cue z")
        ax.legend(fontsize=7)
    fig.suptitle("Pre-cue-normalized LocaNMF activity aligned to first lick after cue, per animal", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    png = args.output / "locanmf_firstlick_aligned_per_animal.png"
    fig.savefig(png, dpi=130); plt.close(fig)
    (args.output / "locanmf_firstlick_summary.json").write_text(json.dumps(
        {r["label"]: {"trials": r["n"], "median_rt_s": round(float(np.median(r["rt_s"])), 3)} for r in res}, indent=2))
    print("wrote", png, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
