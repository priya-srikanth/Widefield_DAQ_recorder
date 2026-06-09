"""Position-resolved cue alignment per animal, and cue-with-lick vs cue-without-lick.

Two cue analyses on the LocaNMF component traces (quiet-z), pooled per animal:

  (A) POSITION-SPECIFIC cue alignment: cue-triggered traces split by the 6 spout positions,
      one figure with rows = orofacial areas, cols = animals, 6 position lines per panel.

  (B) CUE x LICK dissociation: split each cue by whether a lick occurs within a response
      window after it (default 0-1 s). "cue+lick" trials carry sensory + the motor response;
      "cue, no-lick" trials isolate the stimulus-locked response without the movement.
      Figure: rows = areas, cols = animals, two lines (cue+lick vs cue-no-lick) per panel.

Recomputes from the DAQ h5 (cue + lick events) + LocaNMF C + quiet mask, per session, then
averages each animal's components. Reuses the lick-aligned helpers.

    python -m wfield_local.locanmf_cue_lick_analysis --output "<dir>" [--resp-s 1.0]
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

from wfield_local.plot_lick_aligned_averages import (
    _load_daq_events, _classify_events, _event_frame_indices_from_pco, POSITION_NAMES, DISPLAY_ORDER,
)
from wfield_local.plot_spout_trial_averages import (
    _load_daq_events as _load_cue_events, _classify_cues,
)
from wfield_local.locanmf_lick_aligned import (
    _corrected_frame_samples, _nearest_corrected_frame, _quiet_zscore,
)

OROFACIAL = {3: "MOp", 4: "MOs", 5: "SSp-n", 6: "SSp-m"}
ANIMAL_COLOR = {"PS92": "tab:blue", "PS94": "tab:orange", "PS95": "tab:green", "PS93": "tab:red"}
L = "M:/MICROSCOPE/Priya/Widefield/labcams"
D = "M:/MICROSCOPE/Priya/Widefield/DAQ_recorder_output"
SESSIONS = [
    dict(label="PS94_0601", mc=f"{L}/20260601/PS94_20260601_141614/motion_corrected",
         h5=f"{D}/PS94_baseline_20260601_141642.h5", regime="A", fmdir=None),
    dict(label="PS95_0601", mc=f"{L}/20260601/PS95_20260601_153653/motion_corrected",
         h5=f"{D}/PS95_baseline_20260601_153627.h5", regime="A", fmdir=None),
    dict(label="PS92_0602", mc=f"{L}/20260602/PS92_20260602_151820/illuminated_rescue/motion_corrected",
         h5=f"{D}/20250602/PS92_20260602_152607.h5", regime="B",
         fmdir=f"{L}/20260602/PS92_20260602_151820/illuminated_rescue"),
    dict(label="PS92_0603", mc=f"{L}/20260603/PS92_20260603_104008/motion_corrected",
         h5=f"{D}/20250603/PS92_20260603_104607.h5", regime="B", fmdir=None),
    dict(label="PS94_0603", mc=f"{L}/20260603/PS94_20260603/motion_corrected",
         h5=f"{D}/20250603/PS94_20260603_175946.h5", regime="B", fmdir=None),
    dict(label="PS95_0603", mc=f"{L}/20260603/PS95_20260603_194442/motion_corrected",
         h5=f"{D}/20250603/PS95_20260603_194902.h5", regime="B", fmdir=None),
    # 6/4 (full-FOV, regime A). PS92 DAQ was split into two recordings -> use the _concat h5
    # (QC: pco/2T = 1.000 for all three, channels + cue/lick classification OK).
    dict(label="PS92_0604", mc=f"{L}/20260604/PS92_20260604_132934/motion_corrected",
         h5=f"{D}/20250604/PS92_20260604_concat.h5", regime="A", fmdir=None),
    dict(label="PS94_0604", mc=f"{L}/20260604/PS94_20260604_151516/motion_corrected",
         h5=f"{D}/20250604/PS94_20260604_152103.h5", regime="A", fmdir=None),
    dict(label="PS95_0604", mc=f"{L}/20260604/PS95_20260604_165712/motion_corrected",
         h5=f"{D}/20250604/PS95_20260604_170729.h5", regime="A", fmdir=None),
    # 6/5. PS92 trial-triggered; PS93 (NEW animal), PS94, PS95 CONTINUOUS imaging (longer 2-3s ENL).
    # Same DAQ-recorder h5 format as prior days. All have a cleanpairs frame_map in motion_corrected/
    # -> regime B (frame_map), NOT pco//2. (Production SVD maps used the frame_map; regime A mis-indexed
    # frames -> degraded decoding, worst on the longest/continuous sessions. fmdir=None: found in mc.)
    dict(label="PS92_0605", mc=f"{L}/20260605/PS92_20260605_125023/motion_corrected",
         h5=f"{D}/20260605/PS92_20260605_125301.h5", regime="B", fmdir=None),
    dict(label="PS93_0605", mc=f"{L}/20260605/PS93_20260605_174659/motion_corrected",
         h5=f"{D}/20260605/PS93_20260605_175452.h5", regime="B", fmdir=None),
    dict(label="PS94_0605", mc=f"{L}/20260605/PS94_20260605_142009/motion_corrected",
         h5=f"{D}/20260605/PS94_20260605_142249.h5", regime="B", fmdir=None),
    dict(label="PS95_0605", mc=f"{L}/20260605/PS95_20260605_163102/motion_corrected",
         h5=f"{D}/20260605/PS95_20260605_163405.h5", regime="B", fmdir=None),
    # 6/6. All four sessions have *cleanpairs_frame_map.npz in mc -> regime B (fmdir=None), same as 6/5.
    dict(label="PS92_0606", mc=f"{L}/20260606/PS92_20260606_122451/motion_corrected",
         h5=f"{D}/20260606/PS92_20260606_122508.h5", regime="B", fmdir=None),
    dict(label="PS93_0606", mc=f"{L}/20260606/PS93_20260606_180117/motion_corrected",
         h5=f"{D}/20260606/PS93_20260606_180219.h5", regime="B", fmdir=None),
    dict(label="PS94_0606", mc=f"{L}/20260606/PS94_20260606_140854/motion_corrected",
         h5=f"{D}/20260606/PS94_20260606_140912.h5", regime="B", fmdir=None),
    dict(label="PS95_0606", mc=f"{L}/20260606/PS95_20260606_160806/motion_corrected",
         h5=f"{D}/20260606/PS95_20260606_160825.h5", regime="B", fmdir=None),
    # 6/7. All four have *cleanpairs_frame_map.npz in mc -> regime B (fmdir=None), same as 6/5, 6/6.
    dict(label="PS92_0607", mc=f"{L}/20260607/PS92_20260607_121538/motion_corrected",
         h5=f"{D}/20260607/PS92_20260607_121551.h5", regime="B", fmdir=None),
    dict(label="PS93_0607", mc=f"{L}/20260607/PS93_20260607_174844/motion_corrected",
         h5=f"{D}/20260607/PS93_20260607_174854.h5", regime="B", fmdir=None),
    dict(label="PS94_0607", mc=f"{L}/20260607/PS94_20260607_140731/motion_corrected",
         h5=f"{D}/20260607/PS94_20260607_140813.h5", regime="B", fmdir=None),
    dict(label="PS95_0607", mc=f"{L}/20260607/PS95_20260607_155000/motion_corrected",
         h5=f"{D}/20260607/PS95_20260607_155400.h5", regime="B", fmdir=None),
    # 6/8. cleanpairs_frame_map.npz present -> regime B (fmdir=None). PS95 uploaded later.
    dict(label="PS92_0608", mc=f"{L}/20260608/PS92_20260608_133759/motion_corrected",
         h5=f"{D}/20260608/PS92_20260608_133847.h5", regime="B", fmdir=None),
    dict(label="PS93_0608", mc=f"{L}/20260608/PS93_20260608_195203/motion_corrected",
         h5=f"{D}/20260608/PS93_20260608_195350.h5", regime="B", fmdir=None),
    dict(label="PS94_0608", mc=f"{L}/20260608/PS94_20260608_153651/motion_corrected",
         h5=f"{D}/20260608/PS94_20260608_153702.h5", regime="B", fmdir=None),
    dict(label="PS95_0608", mc=f"{L}/20260608/PS95_20260608_180943/motion_corrected",
         h5=f"{D}/20260608/PS95_20260608_180950.h5", regime="B", fmdir=None),
]


def _process(s, args):
    mc = s["mc"]
    quiet = glob.glob(f"{mc}/quiet_affine8v1/*quiet_frame.npy")[0]
    C = np.load(f"{mc}/locanmf_affine8v1_final/{s['label']}_locanmf_C.npy")
    regions = np.load(f"{mc}/locanmf_affine8v1_final/{s['label']}_locanmf_regions.npy")
    ncomp, T = C.shape
    Cn, _ = _quiet_zscore(C, np.load(quiet))

    cue = _load_cue_events(s["h5"])
    lk = _load_daq_events(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    fsd = cue["sample_rate_hz"]
    if s["regime"] == "B":
        fmdir = s["fmdir"] or mc
        fm = glob.glob(f"{fmdir}/*cleanpairs_frame_map.npz")[0]
        off = json.loads(Path(glob.glob(f"{fmdir}/*cleanpairs_summary.json")[0]).read_text())["chosen_exposure_offset"]
        csample = _corrected_frame_samples(fm, cue["pco_samples"], int(off))
        cue_f = _nearest_corrected_frame(cue["cue_samples"], csample)
        lick_f = _nearest_corrected_frame(lk["lick_samples"], csample)
    else:
        csample = None
        cue_f = _event_frame_indices_from_pco(cue["cue_samples"], cue["pco_samples"]) // 2
        lick_f = _event_frame_indices_from_pco(lk["lick_samples"], lk["pco_samples"]) // 2
    codes = _classify_cues(cue["cue_samples"], cue["strobe_samples"], cue["strobe_codes"])

    pre_n = int(round(args.pre_s * args.fs)); post_n = int(round(args.post_s * args.fs))
    resp_n = int(round(args.resp_s * args.fs))
    win = pre_n + post_n
    tax = np.arange(-pre_n, post_n) / args.fs
    lick_sorted = np.sort(lick_f)

    def ok(fr):
        a, b = fr - pre_n, fr + post_n
        if a < 0 or b > T:
            return False
        if csample is not None:
            return (csample[b - 1] - csample[a]) / fsd <= (args.pre_s + args.post_s + 1.0)
        return True

    # lick within (cue, cue+resp_n] ?
    j = np.searchsorted(lick_sorted, cue_f, side="right")
    has_lick = np.array([j[k] < lick_sorted.size and lick_sorted[j[k]] <= cue_f[k] + resp_n
                         for k in range(cue_f.size)])
    valid = (codes >= 0) & np.array([ok(int(f)) for f in cue_f])

    def avg(mask):
        fr = cue_f[mask]
        segs = [Cn[:, f - pre_n:f + post_n] for f in fr]
        return (np.mean(segs, 0).astype(np.float32) if segs else np.zeros((ncomp, win), np.float32)), len(segs)

    per_pos = {}
    for code in DISPLAY_ORDER:
        per_pos[POSITION_NAMES[code]] = avg(valid & (codes == code))
    withlick = avg(valid & has_lick)
    nolick = avg(valid & ~has_lick)
    return dict(label=s["label"], animal=s["label"].split("_")[0], time=tax, regions=regions,
                per_pos=per_pos, withlick=withlick, nolick=nolick,
                n_withlick=int((valid & has_lick).sum()), n_nolick=int((valid & ~has_lick).sum()))


def _animal_area_mean(results, animal, lab, key, sub=None):
    """Mean trace over all components of `animal` in area `lab` for result-array `key`."""
    rows = []
    for r in results:
        if r["animal"] != animal:
            continue
        idx = np.where(r["regions"] == lab)[0]
        if idx.size == 0:
            continue
        arr = r[key][0] if sub is None else r["per_pos"][sub][0]
        rows.append(arr[idx])
    if not rows:
        return None
    stk = np.concatenate(rows, 0)
    return stk.mean(0), stk.std(0) / np.sqrt(stk.shape[0]), stk.shape[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--fs", type=float, default=31.23)
    ap.add_argument("--pre-s", type=float, default=1.0)
    ap.add_argument("--post-s", type=float, default=1.5)
    ap.add_argument("--resp-s", type=float, default=1.0, help="post-cue window to count a lick")
    ap.add_argument("--smooth-sigma", type=float, default=1.0)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sm = lambda x: gaussian_filter1d(np.asarray(x, float), args.smooth_sigma)

    results = []
    for s in SESSIONS:
        r = _process(s, args)
        results.append(r)
        print(f"{r['label']}: cue+lick={r['n_withlick']} cue-no-lick={r['n_nolick']}", flush=True)
    time = results[0]["time"]
    areas = [(k, OROFACIAL[abs(k)]) for k in OROFACIAL] + [(-k, OROFACIAL[abs(k)]) for k in OROFACIAL]
    animals = list(ANIMAL_COLOR)

    # ---- (A) position-specific cue, per animal: rows=areas, cols=animals ----
    fig, axes = plt.subplots(len(areas), len(animals), figsize=(5.0 * len(animals), 1.9 * len(areas)),
                             squeeze=False, sharex=True)
    for rr, (lab, nm) in enumerate(areas):
        for cc, an in enumerate(animals):
            ax = axes[rr][cc]
            for code in DISPLAY_ORDER:
                pn = POSITION_NAMES[code]
                out = _animal_area_mean(results, an, lab, "per_pos", sub=pn)
                if out is None:
                    continue
                ax.plot(time, sm(out[0]), lw=1.2, label=pn)
            ax.axvline(0, color="grey", ls="--", lw=0.6); ax.axhline(0, color="grey", lw=0.5)
            if rr == 0:
                ax.set_title(an, fontsize=11)
            if cc == 0:
                ax.set_ylabel(f"{nm}{'_L' if lab > 0 else '_R'}\nquiet z", fontsize=8)
            if rr == len(areas) - 1:
                ax.set_xlabel("time from cue (s)")
            if rr == 0 and cc == len(animals) - 1:
                ax.legend(fontsize=5, ncol=2)
    fig.suptitle("Position-specific cue-triggered LocaNMF traces, per animal", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(args.output / "locanmf_cue_by_position_per_animal.png", dpi=120); plt.close(fig)

    # ---- (B) cue+lick vs cue-no-lick, per animal: rows=areas, cols=animals ----
    fig, axes = plt.subplots(len(areas), len(animals), figsize=(5.0 * len(animals), 1.9 * len(areas)),
                             squeeze=False, sharex=True)
    for rr, (lab, nm) in enumerate(areas):
        for cc, an in enumerate(animals):
            ax = axes[rr][cc]
            for key, col, lbl in [("withlick", "tab:red", "cue + lick"), ("nolick", "k", "cue, no lick")]:
                out = _animal_area_mean(results, an, lab, key)
                if out is None:
                    continue
                m, sem, n = out
                ax.plot(time, sm(m), color=col, lw=1.8, label=f"{lbl}")
                ax.fill_between(time, sm(m - sem), sm(m + sem), color=col, alpha=0.18)
            ax.axvline(0, color="grey", ls="--", lw=0.6); ax.axhline(0, color="grey", lw=0.5)
            if rr == 0:
                ax.set_title(an, fontsize=11)
            if cc == 0:
                ax.set_ylabel(f"{nm}{'_L' if lab > 0 else '_R'}\nquiet z", fontsize=8)
            if rr == len(areas) - 1:
                ax.set_xlabel("time from cue (s)")
            if rr == 0 and cc == 0:
                ax.legend(fontsize=6)
    fig.suptitle(f"Cue-triggered LocaNMF: cue+lick vs cue-no-lick (lick within {args.resp_s}s), per animal",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(args.output / "locanmf_cue_withlick_vs_nolick_per_animal.png", dpi=120); plt.close(fig)

    counts = {r["label"]: {"cue+lick": r["n_withlick"], "cue-no-lick": r["n_nolick"]} for r in results}
    (args.output / "locanmf_cue_lick_analysis_summary.json").write_text(
        json.dumps({"resp_s": args.resp_s, "counts": counts}, indent=2))
    print("wrote figures to", args.output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
