"""Plot before/after Allen transform landmark diagnostics.

Left panel:
    Native mean 470 nm image. Red X marks the points clicked in the image
    (``landmarks_match``). White open circles mark the atlas target coordinates
    from ``landmarks_im`` on the same pixel grid for visual comparison.

Right panel:
    Allen-transformed mean 470 nm image. Red X marks the clicked points after
    applying the saved transform. White open circles mark the target coordinates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from wfield.allen import load_allen_landmarks


def _edges(atlas: np.ndarray) -> np.ndarray:
    valid = np.isfinite(atlas) & (atlas != 0)
    edges = np.zeros_like(valid, dtype=bool)
    edges[:-1, :] |= atlas[:-1, :] != atlas[1:, :]
    edges[:, :-1] |= atlas[:, :-1] != atlas[:, 1:]
    return ndimage.binary_dilation(edges & valid, iterations=1)


def _overlay_edges(ax, edges: np.ndarray) -> None:
    rgba = np.zeros((*edges.shape, 4), dtype=np.float32)
    rgba[edges] = (0, 0, 0, 0.7)
    ax.imshow(rgba, interpolation="nearest")


def _version(path: Path, current_version: str | None) -> str:
    if path.stem == "dorsal_cortex_landmarks":
        return current_version or "current"
    return path.stem.removeprefix("dorsal_cortex_landmarks_")


def _aligned_version(path: Path) -> str:
    if path.name == "allen_aligned":
        return "unversioned"
    return path.name.removeprefix("allen_aligned_")


def _load_points(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    clicked = raw["landmarks_match"]
    target = raw["landmarks_im"]
    names = clicked.get("name") or [f"pt{i + 1}" for i in range(len(clicked["x"]))]
    clicked_xy = np.column_stack([clicked["x"], clicked["y"]]).astype(float)
    target_xy = np.column_stack([target["x"], target["y"]]).astype(float)
    return names, clicked_xy, target_xy


def _load_aligned_transforms(results: Path):
    transforms = {}
    for folder in sorted(results.glob("allen_aligned*")):
        summary = folder / "allen_transform_summary.json"
        if not summary.exists():
            continue
        payload = json.loads(summary.read_text(encoding="utf-8"))
        transforms[folder] = np.asarray(payload["transform"], dtype=float)
    return transforms


def _match_jsons(json_dir: Path, results: Path, current_version: str | None):
    aligned = _load_aligned_transforms(results)
    matches = []
    for json_path in sorted(json_dir.glob("dorsal_cortex_landmarks*.json")):
        transform = load_allen_landmarks(str(json_path))["transform"]
        params = np.asarray(transform.params, dtype=float)
        best_folder = None
        best_err = np.inf
        for folder, aligned_params in aligned.items():
            err = float(np.max(np.abs(params - aligned_params)))
            if err < best_err:
                best_err = err
                best_folder = folder
        if best_folder is not None and best_err < 1e-6:
            matches.append(
                {
                    "json": json_path,
                    "json_version": _version(json_path, current_version),
                    "aligned": best_folder,
                    "aligned_version": _aligned_version(best_folder),
                    "transform": transform,
                }
            )
    return matches


def _sort_key(item):
    label = item["json_version"]
    if label.startswith("v"):
        try:
            return int(label[1:])
        except ValueError:
            pass
    return 999


def _scatter_landmarks(ax, names, xy, marker, color, label, size=70):
    if marker == "o":
        ax.scatter(xy[:, 0], xy[:, 1], s=size, facecolors="none", edgecolors=color, linewidths=2.0, label=label)
    else:
        ax.scatter(xy[:, 0], xy[:, 1], s=size, marker=marker, c=color, linewidths=2.0, label=label)
    for name, (x, y) in zip(names, xy):
        ax.text(x + 5, y - 5, str(name), color="yellow", fontsize=7.5, weight="bold")


def _plot_one(label: str, results: Path, item, output: Path) -> Path:
    names, clicked_xy, target_xy = _load_points(item["json"])
    # skimage.warp treats the transform passed by wfield as an inverse map:
    # output coordinates -> input coordinates. Therefore an input/native point
    # lands in output/aligned space via transform.inverse(point).
    transformed_xy = item["transform"].inverse(clicked_xy)
    native_frames = np.load(results / "frames_average.npy")
    aligned_frames = np.load(item["aligned"] / "frames_average_atlas.npy")
    atlas = np.load(item["aligned"] / "allen_area_atlas_native_grid.npy")
    edge_map = _edges(atlas)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)

    native = native_frames[1]
    axes[0].imshow(native, cmap="gray", vmin=np.percentile(native, 1), vmax=np.percentile(native, 99.5))
    _scatter_landmarks(axes[0], names, clicked_xy, "x", "#ff2c2c", "clicked points")
    _scatter_landmarks(axes[0], names, target_xy, "o", "white", "target locations")
    axes[0].set_title("Before transform: native mean 470")
    axes[0].set_axis_off()

    aligned = aligned_frames[1]
    axes[1].imshow(aligned, cmap="gray", vmin=np.percentile(aligned, 1), vmax=np.percentile(aligned, 99.5))
    _overlay_edges(axes[1], edge_map)
    _scatter_landmarks(axes[1], names, transformed_xy, "x", "#ff2c2c", "clicked points after transform")
    _scatter_landmarks(axes[1], names, target_xy, "o", "white", "target locations")
    for (x0, y0), (x1, y1) in zip(transformed_xy, target_xy):
        axes[1].plot([x0, x1], [y0, y1], color="#ffcc00", linewidth=1.0, alpha=0.7)
    axes[1].set_title("After transform: aligned mean 470")
    axes[1].set_axis_off()

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], loc="lower center", ncol=2)
    fig.suptitle(f"{label} {item['json_version']} / {item['aligned_version']} alignment before vs after")
    out = output / f"{label}_{item['json_version']}_alignment_before_after.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot before/after transform landmark diagnostics.")
    parser.add_argument("--label", required=True)
    parser.add_argument("--json-dir", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--current-version", default=None)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    matches = sorted(_match_jsons(args.json_dir, args.results, args.current_version), key=_sort_key)
    outputs = [_plot_one(args.label, args.results, item, args.output) for item in matches]
    summary = {
        "label": args.label,
        "outputs": [str(p) for p in outputs],
        "note": "Before panel: red X = clicked landmarks in native image; white circles = target atlas/reference coordinates. After panel: red X = clicked landmarks after applying transform; white circles = same target coordinates. Yellow lines show residual offset after transform.",
        "matched_versions": [
            {
                "json": str(item["json"]),
                "json_version": item["json_version"],
                "aligned": str(item["aligned"]),
                "aligned_version": item["aligned_version"],
            }
            for item in matches
        ],
    }
    (args.output / f"{args.label}_alignment_before_after_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
