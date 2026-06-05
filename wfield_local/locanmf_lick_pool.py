"""Pool lick-triggered LocaNMF traces across sessions/animals, grouped by Allen area.

Reads every session's ``*_locanmf_lick_aligned.npz`` (+ ``_summary.json``) from
``locanmf_lick_aligned.py`` and, keeping INDIVIDUAL components (tagged by animal/session and
their Allen seed region), produces the cross-animal view of lick-evoked activity:

  * per component, a count-weighted pooled-across-positions lick-triggered trace (all licks),
    in quiet-z-score units;
  * cross-animal OVERLAY figures for the orofacial areas (one line per component-session,
    colored by animal; bold grand mean +/- SEM) -- shows reproducibility across animals;
  * an area x session responsiveness heatmap (peak post-lick quiet-z, averaged over the
    area's components) -- which areas are lick-modulated and how consistently;
  * a flat pooled npz (traces N x T + parallel region/animal/session/comp arrays) for
    downstream filtering (e.g. position-resolved or single-area analyses).

    python -m wfield_local.locanmf_lick_pool \
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


def _sm(x, sigma):
    return gaussian_filter1d(np.asarray(x, float), sigma) if sigma and sigma > 0 else x

POSNAMES = ["close_center", "close_L", "close_R", "far_center", "far_L", "far_R"]
OROFACIAL = {3: "MOp", 4: "MOs", 5: "SSp-n", 6: "SSp-m"}
ANIMAL_COLOR = {"PS92": "tab:blue", "PS94": "tab:orange", "PS95": "tab:green"}


def _load_session(npz_path: str, names: dict) -> dict:
    npz = np.load(npz_path)
    summ = json.loads(Path(npz_path.replace(".npz", "_summary.json")).read_text())
    counts = summ["counts_by_position"]
    label = summ["label"]
    time = npz["time"]; regions = npz["regions"]; ncomp = regions.shape[0]
    num = np.zeros((ncomp, time.shape[0]), np.float64); den = 0
    per = {}
    for name in POSNAMES:
        if f"{name}_mean" in npz.files:
            per[name] = npz[f"{name}_mean"]
            c = int(counts.get(name, 0))
            if c > 0:
                num += c * npz[f"{name}_mean"]; den += c
    pooled = (num / den) if den > 0 else num
    return dict(label=label, animal=label.split("_")[0], time=time, regions=regions,
                pooled=pooled.astype(np.float32), per=per, counts=counts, total=den)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="M:/MICROSCOPE/Priya/Widefield/labcams")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--names-json", default=None,
                    help="allen_area_names.json for region labels (any session's; default: first found)")
    ap.add_argument("--post-window", type=float, nargs=2, default=(0.0, 0.5),
                    help="post-lick window (s) for the responsiveness peak")
    ap.add_argument("--smooth-sigma", type=float, default=1.0, help="Gaussian frames for display smoothing")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    npzs = sorted(glob.glob(f"{args.root}/**/locanmf_lick_aligned_affine8v1/*_locanmf_lick_aligned.npz",
                            recursive=True))
    if not npzs:
        print("no lick-aligned npz found under", args.root); return 1
    names_json = args.names_json or glob.glob(f"{args.root}/**/allen_aligned_affine8v1/allen_area_names.json",
                                              recursive=True)[0]
    names = {int(k): v for k, v in json.loads(Path(names_json).read_text())}
    sessions = [_load_session(p, names) for p in npzs]
    time = sessions[0]["time"]
    print(f"pooled {len(sessions)} sessions: {[s['label'] for s in sessions]}", flush=True)

    # ---- flat pooled table (one row per component per session) ----
    rows_trace, row_region, row_animal, row_session, row_comp = [], [], [], [], []
    for s in sessions:
        for i in range(s["regions"].shape[0]):
            rows_trace.append(s["pooled"][i]); row_region.append(int(s["regions"][i]))
            row_animal.append(s["animal"]); row_session.append(s["label"]); row_comp.append(i)
    traces = np.array(rows_trace, np.float32)
    region = np.array(row_region); animal = np.array(row_animal); session = np.array(row_session)
    np.savez_compressed(args.output / "locanmf_lick_pooled.npz", time=time, traces=traces,
                        region=region, animal=animal, session=session, comp=np.array(row_comp))

    # ---- cross-animal orofacial overlay (8 panels: 4 areas x 2 hemispheres) ----
    pmask = (time >= args.post_window[0]) & (time <= args.post_window[1])
    areas = [(lab, nm) for lab, nm in OROFACIAL.items()] + [(-lab, nm) for lab, nm in OROFACIAL.items()]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), squeeze=False)
    for ax, (lab, nm) in zip(axes.ravel(), areas):
        sel = region == lab
        ax.set_title(f"{nm}{'_L' if lab > 0 else '_R'} (reg {lab}, n={int(sel.sum())} comp)", fontsize=9)
        if sel.sum() == 0:
            ax.text(0.5, 0.5, "none", ha="center", va="center", transform=ax.transAxes); continue
        for tr, an in zip(traces[sel], animal[sel]):
            ax.plot(time, _sm(tr, args.smooth_sigma), color=ANIMAL_COLOR.get(an, "grey"), alpha=0.5, lw=0.9)
        gm = traces[sel].mean(0); gs = traces[sel].std(0) / np.sqrt(sel.sum())
        ax.plot(time, _sm(gm, args.smooth_sigma), color="k", lw=2.2, label="grand mean")
        ax.fill_between(time, _sm(gm - gs, args.smooth_sigma), _sm(gm + gs, args.smooth_sigma), color="k", alpha=0.2)
        ax.axvline(0, color="grey", lw=0.6, ls="--"); ax.axhline(0, color="grey", lw=0.5)
        ax.set_xlabel("time from lick (s)"); ax.set_ylabel("quiet z-score")
    handles = [plt.Line2D([], [], color=c, label=a) for a, c in ANIMAL_COLOR.items()]
    fig.legend(handles=handles, loc="upper right", fontsize=9)
    fig.suptitle("Lick-evoked LocaNMF traces by Allen area, across animals (one line per component-session)")
    fig.tight_layout()
    fig.savefig(args.output / "locanmf_lick_orofacial_across_animals.png", dpi=130); plt.close(fig)

    # ---- area x session responsiveness heatmap (peak post-lick z, mean over area's comps) ----
    sess_labels = [s["label"] for s in sessions]
    uniq_regions = sorted(set(int(r) for r in region), key=lambda r: (abs(r), r < 0))
    H = np.full((len(uniq_regions), len(sessions)), np.nan)
    for ri, lab in enumerate(uniq_regions):
        for ci, s in enumerate(sessions):
            m = (region == lab) & (session == s["label"])
            if m.any():
                H[ri, ci] = np.nanmean(traces[m][:, pmask].mean(1))
    fig, ax = plt.subplots(figsize=(max(6, len(sessions) * 1.2), max(8, len(uniq_regions) * 0.22)))
    vlim = np.nanpercentile(np.abs(H), 98) or 1.0
    im = ax.imshow(H, aspect="auto", cmap="magma", vmin=0, vmax=vlim)
    ax.set_xticks(range(len(sessions))); ax.set_xticklabels(sess_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(uniq_regions)))
    ax.set_yticklabels([f"{names.get(l, '?')} ({l})" for l in uniq_regions], fontsize=5)
    ax.set_title(f"Lick responsiveness: peak post-lick quiet-z ({args.post_window[0]}-{args.post_window[1]} s)")
    fig.colorbar(im, ax=ax, shrink=0.6, label="peak post-lick z")
    fig.tight_layout()
    fig.savefig(args.output / "locanmf_lick_responsiveness_area_x_session.png", dpi=140); plt.close(fig)

    summary = {
        "sessions": sess_labels, "n_components_total": int(traces.shape[0]),
        "n_areas": len(uniq_regions), "post_window_s": list(args.post_window),
        "outputs": ["locanmf_lick_pooled.npz", "locanmf_lick_orofacial_across_animals.png",
                    "locanmf_lick_responsiveness_area_x_session.png"],
    }
    (args.output / "locanmf_lick_pool_summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote pooled outputs to", args.output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
