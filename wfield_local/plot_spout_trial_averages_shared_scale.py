"""Replot saved spout-position pre/post/delta maps with one shared color scale."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


DISPLAY_ORDER = [1, 0, 2, 4, 3, 5]
POSITION_NAMES = {
    0: "close_center",
    1: "close_L",
    2: "close_R",
    3: "far_center",
    4: "far_L",
    5: "far_R",
}


def _region_edges(atlas: np.ndarray) -> np.ndarray:
    valid = np.isfinite(atlas) & (atlas != 0)
    edges = np.zeros_like(valid, dtype=bool)
    edges[:-1, :] |= atlas[:-1, :] != atlas[1:, :]
    edges[:, :-1] |= atlas[:, :-1] != atlas[:, 1:]
    return ndimage.binary_dilation(edges & valid, iterations=1)


def _overlay_regions(ax, edges: np.ndarray) -> None:
    overlay = np.zeros((*edges.shape, 4), dtype=np.float32)
    overlay[edges] = (0, 0, 0, 0.65)
    ax.imshow(overlay, interpolation="nearest")


def _parse_counts(summary_path: Path) -> dict[str, int]:
    if not summary_path.exists():
        return {}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return {str(k): int(v) for k, v in payload.get("counts_by_position", {}).items()}


def _shared_limit(trial, percentile: float) -> float:
    arrays = []
    for pos in POSITION_NAMES.values():
        for key in ("pre", "post", "delta"):
            name = f"{pos}_{key}"
            if name in trial.files:
                arrays.append(trial[name].ravel())
    vals = np.concatenate(arrays)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1e-6
    return max(float(np.nanpercentile(np.abs(vals), percentile)), 1e-6)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot spout maps with one shared color scale.")
    parser.add_argument("--label", required=True)
    parser.add_argument("--trial-maps", type=Path, required=True)
    parser.add_argument("--allen-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--percentile", type=float, default=99.0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    trial = np.load(args.trial_maps)
    atlas = np.load(args.allen_dir / "allen_area_atlas_native_grid.npy")
    edges = _region_edges(atlas)
    counts = _parse_counts(args.summary) if args.summary else {}
    lim = _shared_limit(trial, args.percentile)

    fig, axes = plt.subplots(6, 3, figsize=(13, 18), constrained_layout=True)
    im = None
    for row, code in enumerate(DISPLAY_ORDER):
        pos = POSITION_NAMES[code]
        for col, key in enumerate(("pre", "post", "delta")):
            ax = axes[row, col]
            ax.set_axis_off()
            arr_name = f"{pos}_{key}"
            if arr_name not in trial.files:
                ax.set_title(f"{pos}: no {key}")
                continue
            im = ax.imshow(trial[arr_name], cmap="RdBu_r", vmin=-lim, vmax=lim)
            _overlay_regions(ax, edges)
            label = {"pre": "1 s pre-cue", "post": "1 s post-cue", "delta": "post - pre"}[key]
            n = counts.get(pos, "?")
            ax.set_title(f"{pos} n={n} | {label}", fontsize=10)

    if im is not None:
        fig.colorbar(
            im,
            ax=axes.ravel().tolist(),
            shrink=0.78,
            pad=0.01,
            label=f"Shared scale across all pre/post/delta panels (±{lim:.4g})",
        )
    fig.suptitle(
        f"{args.label} cue averages with Allen outlines - shared color scale",
        fontsize=14,
    )
    out = args.output / f"{args.label}_spout_positions_1s_pre_post_delta_shared_scale.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)

    summary = {
        "label": args.label,
        "trial_maps": str(args.trial_maps),
        "allen_dir": str(args.allen_dir),
        "output": str(out),
        "shared_display_limit": lim,
        "percentile": args.percentile,
        "note": "Pre, post, and post-pre panels all use the same diverging color scale.",
    }
    (args.output / f"{args.label}_spout_positions_shared_scale_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
