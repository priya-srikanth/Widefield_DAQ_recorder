"""One master figure: per-animal lick- AND cue-triggered LocaNMF traces, all orofacial areas.

Compiles the whole lick/cue LocaNMF result into a single plot: rows = Allen areas (orofacial,
both hemispheres), columns = {lick, cue}. Each panel shows one mean +/- SEM trace PER ANIMAL
(averaging that animal's components, quiet-z units) with the individual components faint
behind -- so cross-animal differences and the lick-vs-cue contrast are visible at a glance.

Reads every session's ``*_lick_aligned.npz`` and ``*_cue_aligned.npz`` (from
locanmf_lick_aligned.py with --event lick/cue).

    python -m wfield_local.locanmf_event_master \
        --root "M:/MICROSCOPE/Priya/Widefield/labcams" --output "<dir>"
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

POSNAMES = ["close_center", "close_L", "close_R", "far_center", "far_L", "far_R"]
OROFACIAL = {3: "MOp", 4: "MOs", 5: "SSp-n", 6: "SSp-m"}
ANIMAL_COLOR = {"PS92": "tab:blue", "PS94": "tab:orange", "PS95": "tab:green"}


def _sm(x, s):
    return gaussian_filter1d(np.asarray(x, float), s) if s and s > 0 else np.asarray(x, float)


def _load_event(root, event):
    """Per-component count-weighted pooled-across-positions trace for every session."""
    npzs = sorted(glob.glob(f"{root}/**/locanmf_lick_aligned_affine8v1/*_locanmf_{event}_aligned.npz",
                            recursive=True))
    traces, region, animal, time = [], [], [], None
    for f in npzs:
        z = np.load(f)
        summ = json.loads(Path(f.replace(".npz", "_summary.json")).read_text())
        counts = summ["counts_by_position"]
        label = summ["label"]
        time = z["time"]
        reg = z["regions"]
        num = np.zeros((reg.shape[0], time.shape[0])); den = 0
        for n in POSNAMES:
            if f"{n}_mean" in z.files and counts.get(n, 0) > 0:
                num += counts[n] * z[f"{n}_mean"]; den += counts[n]
        pooled = num / den if den else num
        for i in range(reg.shape[0]):
            traces.append(pooled[i]); region.append(int(reg[i])); animal.append(label.split("_")[0])
    return np.array(traces), np.array(region), np.array(animal), time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="M:/MICROSCOPE/Priya/Widefield/labcams")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--events", nargs="+", default=["lick", "cue"])
    ap.add_argument("--smooth-sigma", type=float, default=1.0)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    data = {ev: _load_event(args.root, ev) for ev in args.events}
    areas = [(k, OROFACIAL[abs(k)]) for k in OROFACIAL] + [(-k, OROFACIAL[abs(k)]) for k in OROFACIAL]
    nrow, ncol = len(areas), len(args.events)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 2.0 * nrow), squeeze=False, sharex=True)
    for r, (lab, nm) in enumerate(areas):
        for c, ev in enumerate(args.events):
            ax = axes[r][c]
            traces, region, animal, time = data[ev]
            if time is None:
                continue
            sel = region == lab
            for tr, an in zip(traces[sel], animal[sel]):
                ax.plot(time, _sm(tr, args.smooth_sigma), color=ANIMAL_COLOR.get(an, "grey"), alpha=0.15, lw=0.6)
            for an in [a for a in ANIMAL_COLOR if (sel & (animal == a)).any()]:
                am = sel & (animal == an); ta = traces[am]
                ma = ta.mean(0); sa = ta.std(0) / np.sqrt(am.sum())
                ax.plot(time, _sm(ma, args.smooth_sigma), color=ANIMAL_COLOR[an], lw=2.0,
                        label=f"{an} (n={int(am.sum())})")
                ax.fill_between(time, _sm(ma - sa, args.smooth_sigma), _sm(ma + sa, args.smooth_sigma),
                                color=ANIMAL_COLOR[an], alpha=0.15)
            ax.axvline(0, color="grey", ls="--", lw=0.6); ax.axhline(0, color="grey", lw=0.5)
            if c == 0:
                ax.set_ylabel(f"{nm}{'_L' if lab > 0 else '_R'}\nquiet z", fontsize=8)
            if r == 0:
                ax.set_title(f"{ev}-triggered", fontsize=12)
            if r == nrow - 1:
                ax.set_xlabel(f"time from event (s)")
            ax.legend(fontsize=5, loc="upper right")
    fig.suptitle("LocaNMF event-triggered traces by Allen area, per animal (lick vs cue; individual components faint)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    png = args.output / "locanmf_lick_cue_master.png"
    fig.savefig(png, dpi=120); plt.close(fig)
    print("wrote", png, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
