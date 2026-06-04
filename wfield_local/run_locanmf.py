"""Run LocaNMF on an Allen-aligned wfield session (localized semi-NMF components).

LocaNMF (Saxena, Kinsella et al., PLoS Comput Biol 2020, pcbi.1007791) re-factorizes
the denoised low-rank widefield data into NON-NEGATIVE, anatomically-LOCALIZED spatial
components anchored to Allen atlas regions, with signed temporal traces. Unlike raw
SVD components (delocalized, not reproducible), LocaNMF components are interpretable
and comparable ACROSS sessions and animals -- the right basis for cross-animal /
functional-subnetwork analysis.

It consumes exactly what our alignment step already produces, so no re-preprocessing:
  --allen-dir   <...\allen_aligned_*>   with  U_atlas.npy,
                allen_area_atlas_native_grid.npy, allen_brain_mask_native_grid.npy
  --svt         SVTcorr.npy             (defaults to <allen-dir>\..\..\SVTcorr.npy)

Calls wfield.local_nmf.compute_locaNMF(U, V, atlas, brain_mask, ...) and saves
A (H,W,ncomp), C (ncomp,T), regions, a summary, and a component montage.

==============================================================================
GPU / ENV SETUP  -- DO THIS ON THE NVIDIA MACHINE (CHECK / INSTALL HERE)
==============================================================================
LocaNMF needs the `locanmf` package (PyTorch + a compiled C++/CUDA extension) and
`torch`. It runs on CPU but is GPU-oriented; use the NVIDIA box.

1) CHECK the GPU + driver:
     nvidia-smi                         # driver must be > 418.x; note the CUDA version
2) CHECK torch sees CUDA (in the target env):
     python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
3) INSTALL options (pick one; the prebuilt conda build is easiest):
   (a) Prebuilt (reference, Python 3.6):
         conda create -n locanmf python=3.6 locanmf -c ss5513 -c pytorch
       Then this script needs `wfield` importable in that env too (the tutorial
       commit / churchlandlab fork exposes compute_locaNMF). If wfield won't install
       on 3.6, copy wfield/local_nmf.py's compute_locaNMF body here (it only needs
       numpy + torch + locanmf).
   (b) Newer Python (3.10/3.11): there is NO maintained prebuilt locanmf; build it
       from source against your installed torch+CUDA:
         git clone https://github.com/ikinsella/locaNMF
         # match a torch build to your CUDA (see nvidia-smi), then:
         pip install torch --index-url https://download.pytorch.org/whl/cu<XXX>
         cd locaNMF && pip install .        # compiles the CUDA/C++ extension
       Our wfield (0.4.2, py3.11) already has wfield.local_nmf.compute_locaNMF, so
       this path keeps one env with wfield + locanmf + torch.
   ! There is currently no newer-Python prebuilt package; (b) compiles from source.
==============================================================================

Run (after data is on N:\MICROSCOPE\Priya\Widefield\labcams\...):
  python -m wfield_local.run_locanmf \
    --allen-dir "N:\...\motion_corrected\wfield_local_results\allen_aligned_affine8v1" \
    --label PS94_0603 --output "N:\...\motion_corrected\locanmf_affine8v1" \
    --maxrank 20 --loc-thresh 70 --r2-thresh 0.99 --device auto
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allen-dir", type=Path, required=True,
                    help="allen_aligned_* folder (U_atlas.npy, allen_area_atlas_native_grid.npy, allen_brain_mask_native_grid.npy)")
    ap.add_argument("--svt", type=Path, default=None, help="SVTcorr.npy (default: <allen-dir>/../SVTcorr.npy then /../../)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--mode", choices=("locanmf", "snmf", "both"), default="locanmf",
                    help="locanmf = atlas-seeded localized components (reproducible, region-anchored); "
                         "snmf = whole-brain seed (global correlation-network components); both = run each")
    ap.add_argument("--minrank", type=int, default=1)
    ap.add_argument("--maxrank", type=int, default=20, help="LocaNMF max components per region (~10-20)")
    ap.add_argument("--snmf-maxrank", type=int, default=200, help="sNMF total components (whole-brain seed)")
    ap.add_argument("--snmf-loc-thresh", type=float, default=1.0, help="sNMF localization (~1 = global)")
    ap.add_argument("--min-pixels", type=int, default=100)
    ap.add_argument("--loc-thresh", type=float, default=70.0, help="%% of component area kept inside the atlas region")
    ap.add_argument("--r2-thresh", type=float, default=0.99, help="fraction of variance to capture")
    ap.add_argument("--nonnegative-temporal", action="store_true")
    ap.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    ad = args.allen_dir

    # locate SVTcorr (lives in the wfield_local_results dir, usually allen-dir's parent)
    svt_path = args.svt
    if svt_path is None:
        for cand in (ad.parent / "SVTcorr.npy", ad.parent.parent / "SVTcorr.npy"):
            if cand.exists():
                svt_path = cand
                break
    if svt_path is None or not Path(svt_path).exists():
        raise FileNotFoundError("Could not find SVTcorr.npy; pass --svt explicitly.")

    U = np.load(ad / "U_atlas.npy")                                  # (H, W, K)
    atlas = np.load(ad / "allen_area_atlas_native_grid.npy")        # (H, W) region labels
    brain_mask = np.load(ad / "allen_brain_mask_native_grid.npy").astype(bool)  # (H, W)
    V = np.load(svt_path)                                            # (K, T) temporal
    print(f"[{args.label}] U{U.shape} atlas{atlas.shape} mask{brain_mask.shape} SVT{V.shape}", flush=True)
    print(f"[{args.label}] SVTcorr: {svt_path}", flush=True)

    # GPU sanity print (does not fail if torch missing -> compute_locaNMF will error clearly)
    try:
        import torch
        print(f"[{args.label}] torch {torch.__version__} cuda_available={torch.cuda.is_available()} "
              f"cuda={getattr(torch.version,'cuda',None)} "
              f"device={'cuda:'+torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (slow)'}", flush=True)
    except Exception as e:
        print(f"[{args.label}] torch not importable yet: {e}", flush=True)

    from wfield.local_nmf import compute_locaNMF  # needs locanmf + torch installed

    # sNMF = whole-brain seed (no localization); LocaNMF = atlas-seeded localized.
    runs = []
    if args.mode in ("locanmf", "both"):
        runs.append(("locanmf", atlas.astype(np.int32), args.maxrank, args.loc_thresh))
    if args.mode in ("snmf", "both"):
        runs.append(("snmf", brain_mask.astype(np.int32), args.snmf_maxrank, args.snmf_loc_thresh))

    def _save(mode, A, C, regions, maxrank, loc_thresh):
        A = np.asarray(A); C = np.asarray(C); regions = np.asarray(regions); ncomp = A.shape[2]
        tag = f"{args.label}_{mode}"
        print(f"[{args.label}] {mode} -> A{A.shape} C{C.shape} ({ncomp} components)", flush=True)
        np.save(args.output / f"{tag}_A.npy", A.astype(np.float32))
        np.save(args.output / f"{tag}_C.npy", C.astype(np.float32))
        np.save(args.output / f"{tag}_regions.npy", regions)
        ncol = 8; nrow = int(np.ceil(max(ncomp, 1) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, 2.0 * nrow), squeeze=False)
        for i in range(nrow * ncol):
            ax = axes[i // ncol, i % ncol]; ax.set_axis_off()
            if i < ncomp:
                m = A[:, :, i]; lim = np.nanpercentile(np.abs(m), 99) or 1.0
                ax.imshow(m, cmap="magma", vmin=0, vmax=lim)
                ax.set_title(f"#{i} reg {int(regions[i])}", fontsize=7)
        fig.suptitle(f"{tag} components (n={ncomp}, maxrank={maxrank}, loc_thresh={loc_thresh})")
        fig.tight_layout()
        png = args.output / f"{tag}_components.png"
        fig.savefig(png, dpi=120); plt.close(fig)
        uniq, counts = np.unique(regions, return_counts=True)
        (args.output / f"{tag}_summary.json").write_text(json.dumps({
            "label": args.label, "mode": mode, "allen_dir": str(ad), "svt": str(svt_path),
            "n_components": int(ncomp), "components_per_region": {int(r): int(c) for r, c in zip(uniq, counts)},
            "params": {"minrank": args.minrank, "maxrank": maxrank, "min_pixels": args.min_pixels,
                       "loc_thresh": loc_thresh, "r2_thresh": args.r2_thresh,
                       "nonnegative_temporal": args.nonnegative_temporal, "device": args.device},
            "outputs": [f"{tag}_A.npy", f"{tag}_C.npy", f"{tag}_regions.npy", png.name],
            "note": "A: (H,W,ncomp) spatial maps (NaN outside brain); C: (ncomp,T) temporal; "
                    "regions: seed label per component. snmf = whole-brain seed (global networks); "
                    "locanmf = atlas-seeded localized.",
        }, indent=2))
        print("wrote", png, flush=True)

    for mode, atlas_arg, maxrank, loc_thresh in runs:
        print(f"[{args.label}] running {mode}: maxrank={maxrank} loc_thresh={loc_thresh}", flush=True)
        A, C, regions = compute_locaNMF(
            U.astype(np.float32), V.astype(np.float32), atlas_arg, brain_mask,
            minrank=args.minrank, maxrank=maxrank, min_pixels=args.min_pixels,
            loc_thresh=loc_thresh, r2_thresh=args.r2_thresh,
            nonnegative_temporal=args.nonnegative_temporal, device=args.device,
        )
        _save(mode, A, C, regions, maxrank, loc_thresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
