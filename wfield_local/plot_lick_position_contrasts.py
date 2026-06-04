"""Plot pairwise spout-position contrasts from post-lick maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


PAIRWISE = [
    ("close_L", "close_R"),
    ("close_center", "close_R"),
    ("close_center", "close_L"),
    ("far_L", "far_R"),
    ("far_center", "far_R"),
    ("far_center", "far_L"),
    ("close_L", "far_L"),
    ("close_R", "far_R"),
    ("close_center", "far_center"),
]


# region_edges is centralized in wfield_local.atlas_overlay (symmetric edge fix)
from wfield_local.atlas_overlay import region_edges as _region_edges


def _overlay_edges(ax, edges: np.ndarray, alpha: float = 0.7) -> None:
    rgba = np.zeros((*edges.shape, 4), dtype=np.float32)
    rgba[edges] = (0, 0, 0, alpha)
    ax.imshow(rgba, interpolation="nearest")


def _robust_lim(arrays, pct=99.0) -> float:
    vals = np.concatenate([np.asarray(a).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0
    return max(float(np.percentile(np.abs(vals), pct)), 1e-9)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot post-lick spout-position contrast maps.")
    parser.add_argument("--label", required=True)
    parser.add_argument("--lick-maps", type=Path, required=True)
    parser.add_argument("--allen-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--percentile", type=float, default=99.0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    src = np.load(args.lick_maps)
    maps = {}
    for key in src.files:
        if key.endswith("_post"):
            maps[key[: -len("_post")]] = src[key]

    contrasts = {}
    for a, b in PAIRWISE:
        if a in maps and b in maps:
            contrasts[f"{a} - {b}"] = maps[a] - maps[b]

    atlas = np.load(args.allen_dir / "allen_area_atlas_native_grid.npy")
    edges = _region_edges(atlas)
    lim = _robust_lim(contrasts.values(), pct=args.percentile) if contrasts else 1.0

    fig, axes = plt.subplots(3, 3, figsize=(12, 11), constrained_layout=True)
    im = None
    for ax, (name, arr) in zip(axes.ravel(), contrasts.items()):
        im = ax.imshow(arr, cmap="RdBu_r", vmin=-lim, vmax=lim)
        _overlay_edges(ax, edges)
        ax.set_title(name)
        ax.set_axis_off()
    for ax in axes.ravel()[len(contrasts) :]:
        ax.set_axis_off()
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, label="Difference in post-lick map")
    fig.suptitle(f"{args.label} pairwise spout-position contrasts from post-lick maps")
    png = args.output / f"{args.label}_lick_aligned_pairwise_spout_position_contrasts.png"
    fig.savefig(png, dpi=180)
    plt.close(fig)

    np.savez_compressed(
        args.output / f"{args.label}_lick_aligned_pairwise_spout_position_contrasts.npz",
        **{name.replace(" ", "_").replace("-", "minus"): arr for name, arr in contrasts.items()},
    )
    summary = {
        "label": args.label,
        "lick_maps": str(args.lick_maps),
        "allen_dir": str(args.allen_dir),
        "output_png": str(png),
        "display_limit": lim,
        "pairwise_definition": "Each contrast is first condition's post-lick map minus second condition's post-lick map.",
        "contrasts": list(contrasts.keys()),
    }
    (args.output / f"{args.label}_lick_aligned_pairwise_spout_position_contrasts_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
