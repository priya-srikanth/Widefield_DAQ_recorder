"""Compare cue-aligned and lick-aligned maps by spout position."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


DISPLAY_ORDER = ["close_L", "close_center", "close_R", "far_L", "far_center", "far_R"]


# region_edges is centralized in wfield_local.atlas_overlay (symmetric edge fix)
from wfield_local.atlas_overlay import region_edges as _region_edges


def _overlay_regions(ax, edges: np.ndarray) -> None:
    rgba = np.zeros((*edges.shape, 4), dtype=np.float32)
    rgba[edges] = (0, 0, 0, 0.65)
    ax.imshow(rgba, interpolation="nearest")


def _robust_limit(arrays, percentile: float) -> float:
    vals = np.concatenate([np.asarray(a).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1e-6
    return max(float(np.nanpercentile(np.abs(vals), percentile)), 1e-6)


def _load_maps(path: Path, suffix: str) -> dict[str, np.ndarray]:
    src = np.load(path)
    out = {}
    for pos in DISPLAY_ORDER:
        key = f"{pos}_{suffix}"
        if key in src.files:
            out[pos] = src[key]
    return out


def _load_counts(path: Path | None, field: str = "counts_by_position") -> dict[str, int]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): int(v) for k, v in payload.get(field, {}).items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare cue and lick aligned maps by spout position.")
    parser.add_argument("--label", required=True)
    parser.add_argument("--cue-maps", type=Path, required=True)
    parser.add_argument("--lick-maps", type=Path, required=True)
    parser.add_argument("--allen-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cue-summary", type=Path, default=None)
    parser.add_argument("--lick-summary", type=Path, default=None)
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--cue-label", default="1 s post-cue")
    parser.add_argument("--lick-label", default="150 ms post-lick")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    cue = _load_maps(args.cue_maps, "post")
    lick = _load_maps(args.lick_maps, "post")
    positions = [pos for pos in DISPLAY_ORDER if pos in cue and pos in lick]
    if not positions:
        raise ValueError("No matching spout-position post maps found in cue and lick NPZ files.")

    delta = {pos: lick[pos] - cue[pos] for pos in positions}
    atlas = np.load(args.allen_dir / "allen_area_atlas_native_grid.npy")
    edges = _region_edges(atlas)
    lim = _robust_limit(
        [cue[pos] for pos in positions] + [lick[pos] for pos in positions] + [delta[pos] for pos in positions],
        args.percentile,
    )
    cue_counts = _load_counts(args.cue_summary)
    lick_counts = _load_counts(args.lick_summary)

    fig, axes = plt.subplots(len(positions), 3, figsize=(12, 3.0 * len(positions)), constrained_layout=True)
    if len(positions) == 1:
        axes = axes[None, :]
    im = None
    for row, pos in enumerate(positions):
        panels = [
            (cue[pos], f"{args.cue_label}\nn={cue_counts.get(pos, '?')}"),
            (lick[pos], f"{args.lick_label}\nn={lick_counts.get(pos, '?')}"),
            (delta[pos], "lick - cue"),
        ]
        for col, (arr, title) in enumerate(panels):
            ax = axes[row, col]
            ax.set_axis_off()
            im = ax.imshow(arr, cmap="RdBu_r", vmin=-lim, vmax=lim)
            _overlay_regions(ax, edges)
            prefix = pos if col == 0 else ""
            ax.set_title(f"{prefix} {title}".strip(), fontsize=10)
    if im is not None:
        fig.colorbar(
            im,
            ax=axes.ravel().tolist(),
            shrink=0.78,
            pad=0.01,
            label=f"Shared scale across cue, lick, and lick-cue panels (±{lim:.4g})",
        )
    fig.suptitle(f"{args.label}: cue-aligned vs lick-aligned spout-position maps", fontsize=14)
    png = args.output / f"{args.label}_cue_vs_lick_spout_position_maps.png"
    fig.savefig(png, dpi=180)
    plt.close(fig)

    np.savez_compressed(
        args.output / f"{args.label}_cue_vs_lick_spout_position_maps.npz",
        **{f"{pos}_cue_post": cue[pos] for pos in positions},
        **{f"{pos}_lick_post": lick[pos] for pos in positions},
        **{f"{pos}_lick_minus_cue": delta[pos] for pos in positions},
    )
    summary = {
        "label": args.label,
        "cue_maps": str(args.cue_maps),
        "lick_maps": str(args.lick_maps),
        "allen_dir": str(args.allen_dir),
        "output_png": str(png),
        "display_limit": lim,
        "positions": positions,
        "cue_label": args.cue_label,
        "lick_label": args.lick_label,
        "delta_definition": "lick post map minus cue post map; note that default windows differ unless labels/inputs were changed.",
    }
    (args.output / f"{args.label}_cue_vs_lick_spout_position_maps_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
