"""Energy-ranked component montage for LocaNMF outputs.

Re-renders the LocaNMF A/C components ordered by DESCENDING component energy
(``||A_i||_F * ||C_i||_2`` = the magnitude of the rank-1 ``A_i C_i`` contribution),
i.e. the natural "importance rank" -- strongest components first. The stock montage
in ``run_locanmf.py`` is ordered by Allen *region* (seed label), which is good for
anatomy but not for spotting the dominant components; this gives the complementary view.

Pure post-processing of the saved ``A``/``C``/``regions`` arrays -- it does NOT recompute
LocaNMF, so it is ~seconds and can backfill existing output dirs:

    python -m wfield_local.montage_by_energy <output_dir> <tag>
    # tag = file stem before _A.npy, e.g. PS94_0601_locanmf

``run_locanmf.py`` calls ``save_energy_montage`` automatically, writing
``<tag>_components_byenergy.png`` next to the region-ordered montage.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def component_energy(A, C):
    """Per-component energy ``||A_i||_F * ||C_i||_2`` (NaNs in A treated as 0)."""
    A = np.asarray(A, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    aF = np.sqrt(np.nansum(A ** 2, axis=(0, 1)))   # spatial Frobenius norm per component
    cN = np.sqrt(np.sum(C ** 2, axis=1))           # temporal L2 norm per component
    return aF * cN


def save_energy_montage(A, C, regions, png_path, title=""):
    """Write an 8-col montage of A[:,:,i] ordered by descending component energy.

    Returns (png_path, order, energy) where ``order`` indexes the original component
    axis from highest to lowest energy.
    """
    A = np.asarray(A)
    C = np.asarray(C)
    regions = np.asarray(regions)
    energy = component_energy(A, C)
    order = np.argsort(energy)[::-1]
    ncomp = A.shape[2]
    ncol = 8
    nrow = int(np.ceil(max(ncomp, 1) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, 2.0 * nrow), squeeze=False)
    for r in range(nrow * ncol):
        ax = axes[r // ncol, r % ncol]
        ax.set_axis_off()
        if r < ncomp:
            i = order[r]
            m = A[:, :, i]
            lim = np.nanpercentile(np.abs(m), 99) or 1.0
            ax.imshow(m, cmap="magma", vmin=0, vmax=lim)
            ax.set_title(f"#{r} reg {int(regions[i])} E={energy[i]:.0f}", fontsize=7)
    fig.suptitle(title or f"components by energy rank (n={ncomp}, desc ||A||*||C||)")
    fig.tight_layout()
    png_path = Path(png_path)
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    return png_path, order, energy


def _main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("tag", help="file stem before _A.npy, e.g. PS94_0601_locanmf")
    args = ap.parse_args()
    A = np.load(args.output_dir / f"{args.tag}_A.npy")
    C = np.load(args.output_dir / f"{args.tag}_C.npy")
    regions = np.load(args.output_dir / f"{args.tag}_regions.npy")
    png = args.output_dir / f"{args.tag}_components_byenergy.png"
    png, _order, _energy = save_energy_montage(A, C, regions, png, title=f"{args.tag} components by energy rank")
    print("wrote", png)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
