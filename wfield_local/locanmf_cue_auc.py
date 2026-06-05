"""Cue-evoked AUC (0-2 s) per spout position, per animal, for LocaNMF components.

For each session: cue-triggered LocaNMF traces by spout position, normalized to each trial's
pre-cue 1 s baseline (subtract pre-cue mean, z by the per-component pooled pre-cue SD -- the
quiet-mask z was unusable for PS92, and a data-driven quietest-frames SD over-corrects, so
pre-cue is the comparable baseline). Integrate 0-2 s post-cue -> AUC (z*s) per component, then
average each animal's components.

Output: grouped-bar figure (position x animal) per orofacial area + an AUC table (npz/json).

    python -m wfield_local.locanmf_cue_auc --output "<dir>"
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

from wfield_local.locanmf_cue_lick_analysis import SESSIONS
from wfield_local.plot_spout_trial_averages import _load_daq_events as _load_cue_events, _classify_cues
from wfield_local.plot_lick_aligned_averages import POSITION_NAMES, DISPLAY_ORDER
from wfield_local.plot_lick_aligned_averages import _event_frame_indices_from_pco
from wfield_local.locanmf_lick_aligned import _corrected_frame_samples, _nearest_corrected_frame, _quiet_zscore

ORO = {3: "MOp", 4: "MOs", 5: "SSp-n", 6: "SSp-m"}
ANIMAL_COLOR = {"PS92": "tab:blue", "PS94": "tab:orange", "PS95": "tab:green"}


def _process(s, args):
    mc = s["mc"]
    C = np.load(f"{mc}/locanmf_affine8v1_final/{s['label']}_locanmf_C.npy").astype(np.float64)
    regions = np.load(f"{mc}/locanmf_affine8v1_final/{s['label']}_locanmf_regions.npy")
    ncomp, T = C.shape
    cue = _load_cue_events(s["h5"])
    fsd = cue["sample_rate_hz"]
    if s["regime"] == "B":
        fmdir = s["fmdir"] or mc
        fm = glob.glob(f"{fmdir}/*cleanpairs_frame_map.npz")[0]
        off = json.loads(Path(glob.glob(f"{fmdir}/*cleanpairs_summary.json")[0]).read_text())["chosen_exposure_offset"]
        cs = _corrected_frame_samples(fm, cue["pco_samples"], int(off))
        cue_f = _nearest_corrected_frame(cue["cue_samples"], cs)
    else:
        cs = None
        cue_f = _event_frame_indices_from_pco(cue["cue_samples"], cue["pco_samples"]) // 2
    codes = _classify_cues(cue["cue_samples"], cue["strobe_samples"], cue["strobe_codes"])

    pre_n = int(round(args.pre_s * args.fs)); post_n = int(round(args.post_s * args.fs)); win = pre_n + post_n
    tax = np.arange(-pre_n, post_n) / args.fs
    aucm = (tax >= 0) & (tax <= args.auc_s)

    def ok(fr):
        a, b = fr - pre_n, fr + post_n
        if a < 0 or b > T:
            return False
        return cs is None or (cs[b - 1] - cs[a]) / fsd <= (args.pre_s + args.post_s + 1.0)

    valid = (codes >= 0) & np.array([ok(int(f)) for f in cue_f])
    # per-component pooled pre-cue SD across all valid cues (stable baseline scale)
    allfr = cue_f[valid]
    allwin = np.stack([C[:, f - pre_n:f + post_n] for f in allfr], 0)
    allsub = allwin - allwin[:, :, :pre_n].mean(2, keepdims=True)
    presd = allsub[:, :, :pre_n].reshape(allwin.shape[1], -1).std(1)
    presd = np.where(presd > 0, presd, 1.0)

    auc = {}  # position -> (ncomp,) AUC
    for code in DISPLAY_ORDER:
        m = valid & (codes == code)
        fr = cue_f[m]
        if fr.size == 0:
            auc[POSITION_NAMES[code]] = np.full(ncomp, np.nan); continue
        w = np.stack([C[:, f - pre_n:f + post_n] for f in fr], 0)
        w = w - w[:, :, :pre_n].mean(2, keepdims=True)        # per-trial pre-cue subtract
        avg = w.mean(0) / presd[:, None]                      # (ncomp,win) pre-cue z
        auc[POSITION_NAMES[code]] = np.trapezoid(avg[:, aucm], tax[aucm], axis=1)  # z*s, 0-auc_s
    return dict(animal=s["label"].split("_")[0], label=s["label"], regions=regions, auc=auc)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--fs", type=float, default=31.23)
    ap.add_argument("--pre-s", type=float, default=1.0)
    ap.add_argument("--post-s", type=float, default=2.2)
    ap.add_argument("--auc-s", type=float, default=2.0, help="integrate 0..auc_s post-cue")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    res = [_process(s, args) for s in SESSIONS]
    for r in res:
        print(r["label"], "done", flush=True)
    animals = list(ANIMAL_COLOR)
    positions = [POSITION_NAMES[c] for c in DISPLAY_ORDER]
    areas = [(k, ORO[abs(k)]) for k in ORO] + [(-k, ORO[abs(k)]) for k in ORO]

    # aggregate: area -> position -> animal -> (mean, sem, n) over that animal's components
    table = {}
    fig, axes = plt.subplots(2, 4, figsize=(22, 9), squeeze=False)
    for ax, (lab, nm) in zip(axes.ravel(), areas):
        x = np.arange(len(positions)); w = 0.26
        table[f"{nm}{'_L' if lab>0 else '_R'}"] = {}
        for ai, an in enumerate(animals):
            means, sems = [], []
            for pn in positions:
                vals = []
                for r in res:
                    if r["animal"] != an:
                        continue
                    idx = np.where(r["regions"] == lab)[0]
                    if idx.size:
                        vals.extend(r["auc"][pn][idx].tolist())
                vals = np.array([v for v in vals if np.isfinite(v)])
                means.append(np.nanmean(vals) if vals.size else np.nan)
                sems.append(np.nanstd(vals) / np.sqrt(vals.size) if vals.size else np.nan)
            ax.bar(x + (ai - 1) * w, means, w, yerr=sems, color=ANIMAL_COLOR[an], label=an, capsize=2)
            table[f"{nm}{'_L' if lab>0 else '_R'}"][an] = {p: (None if np.isnan(m) else round(float(m), 3))
                                                           for p, m in zip(positions, means)}
        ax.set_xticks(x); ax.set_xticklabels(positions, rotation=45, ha="right", fontsize=7)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(f"{nm}{'_L' if lab > 0 else '_R'} (reg {lab})", fontsize=10)
        ax.set_ylabel("cue AUC 0-2s (z*s)", fontsize=8)
        ax.legend(fontsize=7)
    fig.suptitle(f"Cue-evoked AUC (0-{args.auc_s:g}s, pre-cue z) by spout position, per animal", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    png = args.output / "locanmf_cue_auc_by_position_animal.png"
    fig.savefig(png, dpi=130); plt.close(fig)
    (args.output / "locanmf_cue_auc_table.json").write_text(json.dumps(table, indent=2))
    print("wrote", png, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
