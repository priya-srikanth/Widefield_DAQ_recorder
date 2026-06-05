"""Contralateral tuning of LocaNMF components, PER COMPONENT and per animal.

The pixel/SVD post-lick maps show more somatosensory activation for CONTRALATERAL-spout
licks. This quantifies the same per LocaNMF component: post-lick 150 ms response by spout
position, split by hemisphere, and a contralateral index (contra-minus-ipsi). It is a
WITHIN-component contrast across positions, so it is immune to the per-component
normalization (the quiet-SD problem that biased cross-animal magnitudes).

Outputs (components are NOT pooled within a region):
  * locanmf_contralateral_index_per_component.png -- strip plot: one dot per component,
    colored by animal, grouped by area x hemisphere, with the group mean.
  * locanmf_contralateral_position_profiles.png -- one small panel per component (its 6
    spout-position post-lick responses, contralateral spouts in red).
Prints per-animal contralateral index per area.

    python -m wfield_local.locanmf_contralateral --root "M:/.../labcams" --output "<dir>"
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

POS = ["close_L", "close_center", "close_R", "far_L", "far_center", "far_R"]
AREAS = {3: "MOp", 4: "MOs", 5: "SSp-n", 6: "SSp-m"}  # +label=left hemi, -label=right hemi
ANIMAL_COLOR = {"PS92": "tab:blue", "PS94": "tab:orange", "PS95": "tab:green"}


def _contra_index(hemi, v):
    """Scale-free contralateral modulation index (contra-ipsi)/(|contra|+|ipsi|), in [-1,1],
    so it is comparable across events (lick z vs cue dF/F) and across components."""
    lsp = np.nanmean([v["close_L"], v["far_L"]]); rsp = np.nanmean([v["close_R"], v["far_R"]])
    contra, ipsi = (rsp, lsp) if hemi == "L" else (lsp, rsp)
    denom = abs(contra) + abs(ipsi)
    return (contra - ipsi) / denom if denom > 1e-9 else np.nan


def _gather_lick(args, name2lab):
    """Per-component per-position post-lick (post_ms) means, read from the lick-aligned npz."""
    comps = []
    npzs = sorted(glob.glob(f"{args.root}/**/locanmf_lick_aligned_affine8v1/*_locanmf_lick_aligned.npz",
                            recursive=True))
    for f in npzs:
        z = np.load(f); t = z["time"]; reg = z["regions"]
        cnt = json.loads(Path(f.replace(".npz", "_summary.json")).read_text())["counts_by_position"]
        pm = (t >= 0) & (t <= args.post_ms / 1000.0)
        label = Path(f).name.replace("_locanmf_lick_aligned.npz", ""); animal = label.split("_")[0]
        for nm in args.areas:
            k = name2lab[nm]
            for lab, hemi in [(k, "L"), (-k, "R")]:
                for i in np.where(reg == lab)[0]:
                    v = {p: (z[f"{p}_mean"][i][pm].mean() if cnt.get(p, 0) > 0 else np.nan) for p in POS}
                    comps.append(dict(area=nm, hemi=hemi, animal=animal, label=label, idx=i, v=v,
                                      ci=_contra_index(hemi, v)))
    return comps


def _gather_cue(args, name2lab):
    """Per-component per-position cue-triggered response, mean over 0..cue_window_s post-cue,
    computed fresh from C (pre-cue 1 s baseline-subtracted). Parallel to the lick version but
    cue-aligned over a longer (2 s) window."""
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS
    from wfield_local.plot_spout_trial_averages import _load_daq_events as _load_cue, _classify_cues
    from wfield_local.plot_lick_aligned_averages import _event_frame_indices_from_pco
    from wfield_local.locanmf_lick_aligned import _corrected_frame_samples, _nearest_corrected_frame
    fs = 31.23; pre_n = int(round(fs)); post_n = int(round(args.cue_post_s * fs))
    comps = []
    for s in SESSIONS:
        mc = s["mc"]
        C = np.load(f"{mc}/locanmf_affine8v1_final/{s['label']}_locanmf_C.npy").astype(np.float64)
        reg = np.load(f"{mc}/locanmf_affine8v1_final/{s['label']}_locanmf_regions.npy")
        T = C.shape[1]; cue = _load_cue(s["h5"]); fsd = cue["sample_rate_hz"]
        if s["regime"] == "B":
            fmdir = s["fmdir"] or mc; fm = glob.glob(f"{fmdir}/*cleanpairs_frame_map.npz")[0]
            off = json.loads(Path(glob.glob(f"{fmdir}/*cleanpairs_summary.json")[0]).read_text())["chosen_exposure_offset"]
            cs = _corrected_frame_samples(fm, cue["pco_samples"], int(off))
            cue_f = _nearest_corrected_frame(cue["cue_samples"], cs)
        else:
            cs = None; cue_f = _event_frame_indices_from_pco(cue["cue_samples"], cue["pco_samples"]) // 2
        codes = _classify_cues(cue["cue_samples"], cue["strobe_samples"], cue["strobe_codes"])
        tax = np.arange(-pre_n, post_n) / fs
        am = (tax >= 0) & (tax <= args.cue_window_s)

        def ok(fr):
            a, b = fr - pre_n, fr + post_n
            if a < 0 or b > T:
                return False
            return cs is None or (cs[b - 1] - cs[a]) / fsd <= (1.0 + args.cue_post_s + 1.0)

        valid = (codes >= 0) & np.array([ok(int(f)) for f in cue_f])
        animal = s["label"].split("_")[0]
        for nm in args.areas:
            k = name2lab[nm]
            for lab, hemi in [(k, "L"), (-k, "R")]:
                for i in np.where(reg == lab)[0]:
                    v = {}
                    from wfield_local.plot_lick_aligned_averages import POSITION_NAMES, DISPLAY_ORDER
                    code_of = {POSITION_NAMES[c]: c for c in DISPLAY_ORDER}
                    for p in POS:
                        m = valid & (codes == code_of[p]); fr = cue_f[m]
                        if fr.size == 0:
                            v[p] = np.nan; continue
                        w = np.stack([C[i, f - pre_n:f + post_n] for f in fr], 0)
                        w = w - w[:, :pre_n].mean(1, keepdims=True)   # per-trial pre-cue subtract
                        v[p] = float(w.mean(0)[am].mean())            # mean over 0..cue_window_s
                    comps.append(dict(area=nm, hemi=hemi, animal=animal, label=s["label"], idx=int(i), v=v,
                                      ci=_contra_index(hemi, v)))
    return comps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="M:/MICROSCOPE/Priya/Widefield/labcams")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--event", choices=("lick", "cue"), default="lick",
                    help="lick: post-lick window from lick npz; cue: 0..cue_window_s post-cue, fresh from C")
    ap.add_argument("--post-ms", type=float, default=150.0, help="(lick) post-lick window")
    ap.add_argument("--cue-window-s", type=float, default=2.0, help="(cue) integrate 0..this s post-cue")
    ap.add_argument("--cue-post-s", type=float, default=2.2, help="(cue) extracted window length post-cue")
    ap.add_argument("--areas", nargs="+", default=["SSp-n", "SSp-m", "MOp", "MOs"])
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    name2lab = {v: k for k, v in AREAS.items()}

    comps = _gather_lick(args, name2lab) if args.event == "lick" else _gather_cue(args, name2lab)
    win_label = f"{args.post_ms:.0f}ms post-lick" if args.event == "lick" else f"0-{args.cue_window_s:g}s post-cue"

    groups = [f"{nm}_{h}" for nm in args.areas for h in ("L", "R")]
    # ---- per-animal index per area (printed) ----
    print(f"contralateral index (contra-ipsi, {win_label}), per animal:")
    for g in groups:
        nm, h = g.rsplit("_", 1)
        line = f"  {g:<9}"
        for an in ANIMAL_COLOR:
            ci = np.array([c["ci"] for c in comps if c["area"] == nm and c["hemi"] == h and c["animal"] == an
                           and np.isfinite(c["ci"])])
            line += f"  {an}={ci.mean():+.2f}(n{ci.size})" if ci.size else f"  {an}=--"
        print(line)

    # ---- Figure 1: strip plot, one dot per component, colored by animal ----
    fig, ax = plt.subplots(figsize=(1.4 * len(groups) + 2, 6))
    rng = np.random.RandomState(0)
    for gi, g in enumerate(groups):
        nm, h = g.rsplit("_", 1)
        gc = [c for c in comps if c["area"] == nm and c["hemi"] == h and np.isfinite(c["ci"])]
        for c in gc:
            ax.scatter(gi + rng.uniform(-0.18, 0.18), c["ci"], s=34,
                       color=ANIMAL_COLOR.get(c["animal"], "grey"), edgecolor="k", lw=0.3, zorder=3)
        if gc:
            m = np.mean([c["ci"] for c in gc])
            ax.plot([gi - 0.3, gi + 0.3], [m, m], color="k", lw=2.2, zorder=4)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.set_ylabel(f"contralateral index (contra - ipsi), {win_label}")
    ax.set_title(f"Per-component contralateral tuning [{args.event}] (dot=component, color=animal, bar=group mean)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=a) for a, c in ANIMAL_COLOR.items()]
    ax.legend(handles=handles, fontsize=9)
    fig.tight_layout()
    fig.savefig(args.output / f"locanmf_contralateral_index_per_component_{args.event}.png", dpi=140); plt.close(fig)

    # ---- Figure 2: per-component position profiles (not pooled) ----
    comps_sorted = sorted(comps, key=lambda c: (args.areas.index(c["area"]), c["hemi"], c["animal"]))
    n = len(comps_sorted); ncol = 6; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.0 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    for j, c in enumerate(comps_sorted):
        ax = axes[j // ncol][j % ncol]; ax.set_axis_on()
        vals = [c["v"][p] for p in POS]
        colors = ["tab:red" if (("R" in p and c["hemi"] == "L") or ("L" in p and c["hemi"] == "R")) else "tab:grey"
                  for p in POS]
        ax.bar(range(6), vals, color=colors)
        ax.set_xticks(range(6)); ax.set_xticklabels(POS, rotation=90, fontsize=5)
        ax.axhline(0, color="k", lw=0.4)
        ax.set_title(f"{c['area']}_{c['hemi']} {c['animal']} #{c['idx']}", fontsize=6,
                     color=ANIMAL_COLOR.get(c["animal"], "k"))
    fig.suptitle(f"Per-component {win_label} by spout position (red = contralateral spout)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(args.output / f"locanmf_contralateral_position_profiles_{args.event}.png", dpi=120); plt.close(fig)
    print("wrote per-component figures to", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
