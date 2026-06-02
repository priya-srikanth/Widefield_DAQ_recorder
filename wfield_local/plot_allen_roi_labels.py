"""Plot a color-coded Allen/wfield ROI atlas with region labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy import ndimage


def _load_name_map(path: Path) -> dict[int, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for code, name in raw:
        code = int(code)
        if code <= 0:
            continue
        label = str(name)
        label = label.removesuffix("_left").removesuffix("_right")
        out[code] = label
    return out


def _region_edges(atlas: np.ndarray) -> np.ndarray:
    valid = np.isfinite(atlas) & (atlas != 0)
    edges = np.zeros_like(valid, dtype=bool)
    edges[:-1, :] |= atlas[:-1, :] != atlas[1:, :]
    edges[:, :-1] |= atlas[:, :-1] != atlas[:, 1:]
    return ndimage.binary_dilation(edges & valid, iterations=1)


def _label_position(mask: np.ndarray) -> tuple[float, float]:
    yx = ndimage.center_of_mass(mask)
    if not np.all(np.isfinite(yx)):
        ys, xs = np.nonzero(mask)
        return float(np.median(xs)), float(np.median(ys))
    return float(yx[1]), float(yx[0])


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot color-coded Allen ROI labels.")
    parser.add_argument("--allen-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Allen/wfield ROI labels")
    args = parser.parse_args()

    atlas = np.load(args.allen_dir / "allen_area_atlas_native_grid.npy")
    name_map = _load_name_map(args.allen_dir / "allen_area_names.json")
    ids = sorted(int(v) for v in np.unique(atlas[np.isfinite(atlas) & (atlas > 0)]))
    id_to_idx = {code: idx + 1 for idx, code in enumerate(ids)}
    idx_image = np.zeros_like(atlas, dtype=np.int16)
    for code, idx in id_to_idx.items():
        idx_image[atlas == code] = idx

    colors = [(1, 1, 1, 1)]
    colors.extend(plt.cm.hsv(np.linspace(0, 1, len(ids), endpoint=False)))
    cmap = ListedColormap(colors)
    edges = _region_edges(atlas)

    fig, ax = plt.subplots(figsize=(8.5, 7.2), constrained_layout=True)
    ax.imshow(idx_image, cmap=cmap, interpolation="nearest", vmin=0, vmax=len(ids))
    rgba = np.zeros((*edges.shape, 4), dtype=np.float32)
    rgba[edges] = (0, 0, 0, 0.7)
    ax.imshow(rgba, interpolation="nearest")

    for code in ids:
        mask = atlas == code
        if int(mask.sum()) < 30:
            continue
        x, y = _label_position(mask)
        label = name_map.get(code, str(code))
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=6.8,
            color="black",
            weight="bold",
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "alpha": 0.55, "edgecolor": "none"},
        )

    ax.set_title(args.title)
    ax.set_axis_off()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    plt.close(fig)

    summary = {
        "allen_dir": str(args.allen_dir),
        "output": str(args.output),
        "roi_count": len(ids),
        "roi_labels": {str(code): name_map.get(code, str(code)) for code in ids},
        "note": "Atlas grid contains positive ROI IDs shared across hemispheres; labels use the bilateral acronym without _left/_right suffix.",
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
