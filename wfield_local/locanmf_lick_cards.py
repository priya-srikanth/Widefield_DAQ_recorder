"""Per-component cards: LocaNMF spatial map (Allen outline) + its lick-triggered trace.

For each focus Allen area, one ROW per component across all sessions/animals: the
component's spatial footprint ``A_i`` (with the atlas region outline) beside its
lick-triggered trace by spout position (lightly smoothed for display, quiet-z units).
Individual components, ordered by area so the same areas line up across animals -- the
"which component is this, and how does it respond to licking" view.

    python -m wfield_local.locanmf_lick_cards \
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

from wfield_local.atlas_overlay import region_edges

POSNAMES = ["close_center", "close_L", "close_R", "far_center", "far_L", "far_R"]
OROFACIAL = {3: "MOp", 4: "MOs", 5: "SSp-n", 6: "SSp-m"}


def _discover(root):
    out = []
    for npz in sorted(glob.glob(f"{root}/**/locanmf_lick_aligned_affine8v1/*_locanmf_lick_aligned.npz",
                                recursive=True)):
        p = Path(npz)
        label = p.name.replace("_locanmf_lick_aligned.npz", "")
        mc = p.parent.parent
        A = mc / "locanmf_affine8v1_final" / f"{label}_locanmf_A.npy"
        atlas = mc / "wfield_local_results" / "allen_aligned_affine8v1" / "allen_area_atlas_native_grid.npy"
        if A.exists() and atlas.exists():
            out.append(dict(label=label, animal=label.split("_")[0], npz=npz, A=A, atlas=atlas))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="M:/MICROSCOPE/Priya/Widefield/labcams")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--areas", type=int, nargs="+", default=None,
                    help="Allen seed labels (default: orofacial MOp/MOs/SSp-n/SSp-m both hemispheres)")
    ap.add_argument("--smooth-sigma", type=float, default=1.0, help="Gaussian frames for display smoothing")
    ap.add_argument("--max-per-area", type=int, default=14)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.areas is None:
        args.areas = [k for k in OROFACIAL] + [-k for k in OROFACIAL]
    sessions = _discover(args.root)
    if not sessions:
        print("no sessions found"); return 1
    print(f"{len(sessions)} sessions: {[s['label'] for s in sessions]}", flush=True)

    # cache per-session loads lazily
    cache = {}

    def load(s):
        if s["label"] not in cache:
            npz = np.load(s["npz"])
            cache[s["label"]] = dict(
                A=np.load(s["A"], mmap_mode="r"),
                atlas=np.load(s["atlas"]),
                regions=npz["regions"], time=npz["time"],
                per={n: npz[f"{n}_mean"] for n in POSNAMES if f"{n}_mean" in npz.files},
                counts=json.loads(Path(s["npz"].replace(".npz", "_summary.json")).read_text())["counts_by_position"],
            )
        return cache[s["label"]]

    for lab in args.areas:
        # gather (session, comp_idx) for this area
        items = []
        for s in sessions:
            d = load(s)
            for i in np.where(d["regions"] == lab)[0]:
                items.append((s, int(i)))
        nm = OROFACIAL.get(abs(lab), str(lab)) + ("_L" if lab > 0 else "_R")
        if not items:
            print(f"  {nm} (reg {lab}): no components"); continue
        items = items[:args.max_per_area]
        nrow = len(items)
        fig, axes = plt.subplots(nrow, 2, figsize=(9, 2.1 * nrow), squeeze=False,
                                 gridspec_kw={"width_ratios": [1, 2.4]})
        for r, (s, i) in enumerate(items):
            d = load(s)
            # spatial map + Allen outline
            axm = axes[r][0]; axm.set_axis_off()
            m = np.asarray(d["A"][:, :, i], dtype=np.float64)
            lim = np.nanpercentile(np.abs(m), 99) or 1.0
            axm.imshow(m, cmap="magma", vmin=0, vmax=lim)
            edg = region_edges(d["atlas"])
            axm.imshow(np.ma.masked_where(~edg, edg), cmap="cool", alpha=0.5, vmin=0, vmax=1)
            axm.set_title(f"{s['label']}  comp#{i}", fontsize=8)
            # lick trace by position
            axt = axes[r][1]
            t = d["time"]
            for n in POSNAMES:
                if d["counts"].get(n, 0) == 0 or n not in d["per"]:
                    continue
                tr = gaussian_filter1d(d["per"][n][i].astype(float), args.smooth_sigma) if args.smooth_sigma > 0 else d["per"][n][i]
                axt.plot(t, tr, lw=1.2, label=f"{n} ({d['counts'][n]})")
            axt.axvline(0, color="grey", ls="--", lw=0.6); axt.axhline(0, color="grey", lw=0.5)
            axt.set_ylabel("quiet z"); axt.set_xlabel("time from lick (s)")
            axt.legend(fontsize=5, ncol=3, loc="upper right")
        fig.suptitle(f"{nm} (reg {lab}): LocaNMF components across animals — map + lick-triggered traces")
        fig.tight_layout()
        png = args.output / f"lick_cards_{nm}_reg{lab}.png"
        fig.savefig(png, dpi=130); plt.close(fig)
        print(f"  wrote {png}  ({nrow} components)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
