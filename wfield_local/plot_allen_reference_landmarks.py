"""Plot the fixed Allen/wfield reference landmark positions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from wfield.allen import atlas_from_landmarks_file


def _edges(atlas: np.ndarray) -> np.ndarray:
    valid = np.isfinite(atlas) & (atlas != 0)
    edge = np.zeros_like(valid, dtype=bool)
    edge[:-1, :] |= atlas[:-1, :] != atlas[1:, :]
    edge[:, :-1] |= atlas[:, :-1] != atlas[:, 1:]
    return ndimage.binary_dilation(edge & valid, iterations=1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot fixed reference landmark positions from a wfield landmark JSON.")
    parser.add_argument("--landmarks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="Allen/wfield reference landmarks")
    args = parser.parse_args()

    raw = json.loads(args.landmarks.read_text(encoding="utf-8"))
    target = raw["landmarks_im"]
    names = target["name"]
    xy = np.column_stack([target["x"], target["y"]]).astype(float)

    atlas, _, brain_mask = atlas_from_landmarks_file(str(args.landmarks), do_transform=False)
    edge_map = _edges(atlas)

    fig, ax = plt.subplots(figsize=(6.5, 6.0), constrained_layout=True)
    ax.imshow(brain_mask, cmap="gray", alpha=0.25, interpolation="nearest")
    rgba = np.zeros((*edge_map.shape, 4), dtype=np.float32)
    rgba[edge_map] = (0, 0, 0, 0.85)
    ax.imshow(rgba, interpolation="nearest")
    ax.scatter(xy[:, 0], xy[:, 1], s=110, facecolors="white", edgecolors="black", linewidths=2.0)
    for name, (x, y) in zip(names, xy):
        ax.text(x + 6, y - 6, name, color="crimson", fontsize=9, weight="bold")
    ax.set_title(args.label)
    ax.set_axis_off()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(str(args.output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
