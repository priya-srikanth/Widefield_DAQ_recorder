"""Annotate Allen-aligned mean 470 nm images with chosen landmark points."""

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


def _region_edges(atlas: np.ndarray) -> np.ndarray:
    valid = np.isfinite(atlas) & (atlas != 0)
    edges = np.zeros_like(valid, dtype=bool)
    edges[:-1, :] |= atlas[:-1, :] != atlas[1:, :]
    edges[:, :-1] |= atlas[:, :-1] != atlas[:, 1:]
    return ndimage.binary_dilation(edges & valid, iterations=1)


def _overlay_edges(ax, edges: np.ndarray, alpha: float = 0.75) -> None:
    rgba = np.zeros((*edges.shape, 4), dtype=np.float32)
    rgba[edges] = (0, 0, 0, alpha)
    ax.imshow(rgba, interpolation="nearest")


def _json_version(path: Path, current_version: str | None) -> str:
    stem = path.stem
    if stem == "dorsal_cortex_landmarks":
        return current_version or "current"
    return stem.removeprefix("dorsal_cortex_landmarks_")


def _aligned_version(path: Path) -> str:
    name = path.name
    if name == "allen_aligned":
        return "unversioned"
    return name.removeprefix("allen_aligned_")


def _load_json_transform(path: Path):
    landmarks = load_allen_landmarks(str(path))
    return landmarks["transform"]


def _load_clicked_points(path: Path) -> dict[str, np.ndarray | list[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    match = raw.get("landmarks_match") or raw.get("landmarks")
    target = raw.get("landmarks_im")
    if not match:
        raise ValueError(f"{path} has no landmarks_match/landmarks coordinates")
    names = match.get("name") or [f"point_{i + 1}" for i in range(len(match["x"]))]
    clicked = np.column_stack([match["x"], match["y"]]).astype(float)
    targets = None
    if target and "x" in target and "y" in target:
        targets = np.column_stack([target["x"], target["y"]]).astype(float)
    return {"names": names, "clicked": clicked, "targets": targets}


def _load_aligned_transforms(results: Path) -> dict[Path, np.ndarray]:
    transforms = {}
    for folder in sorted(results.glob("allen_aligned*")):
        summary = folder / "allen_transform_summary.json"
        frames = folder / "frames_average_atlas.npy"
        atlas = folder / "allen_area_atlas_native_grid.npy"
        if not (summary.exists() and frames.exists() and atlas.exists()):
            continue
        payload = json.loads(summary.read_text(encoding="utf-8"))
        transforms[folder] = np.asarray(payload["transform"], dtype=float)
    return transforms


def _match_jsons_to_aligned(json_paths: list[Path], aligned: dict[Path, np.ndarray], current_version: str | None):
    matches = []
    for json_path in json_paths:
        transform = _load_json_transform(json_path)
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
                    "json_version": _json_version(json_path, current_version),
                    "aligned": best_folder,
                    "aligned_version": _aligned_version(best_folder),
                    "transform": transform,
                    "match_error": best_err,
                }
            )
    return matches


def _plot_one(label: str, item: dict, output: Path) -> Path:
    aligned = item["aligned"]
    frames = np.load(aligned / "frames_average_atlas.npy")
    atlas = np.load(aligned / "allen_area_atlas_native_grid.npy")
    edges = _region_edges(atlas)
    points = _load_clicked_points(item["json"])
    # wfield applies this transform through skimage.warp, where it is used as
    # an inverse map from output/aligned pixels back to input/native pixels.
    # Therefore native clicked points land in aligned space via inverse().
    transformed_clicked = item["transform"].inverse(points["clicked"])
    targets = points["targets"]

    image = frames[1]
    fig, ax = plt.subplots(figsize=(6.5, 6.2), constrained_layout=True)
    ax.imshow(image, cmap="gray", vmin=np.percentile(image, 1), vmax=np.percentile(image, 99.5))
    _overlay_edges(ax, edges)
    if targets is not None:
        ax.scatter(
            targets[:, 0],
            targets[:, 1],
            s=95,
            facecolors="none",
            edgecolors="white",
            linewidths=2.4,
            label="atlas target",
        )
        ax.scatter(
            targets[:, 0],
            targets[:, 1],
            s=55,
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
        )
    ax.scatter(
        transformed_clicked[:, 0],
        transformed_clicked[:, 1],
        s=70,
        marker="x",
        c="#ff2c2c",
        linewidths=2.0,
        label="clicked point, transformed",
    )
    for name, (x, y) in zip(points["names"], transformed_clicked):
        ax.text(
            x + 5,
            y - 5,
            str(name),
            color="yellow",
            fontsize=8,
            weight="bold",
            path_effects=[],
        )
    ax.set_title(f"{label} {item['aligned_version']} mean 470 + Allen outlines")
    ax.set_axis_off()
    ax.legend(loc="lower right", fontsize=8, framealpha=0.75)
    out = output / f"{label}_{item['aligned_version']}_mean470_allen_landmark_points.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _plot_grid(label: str, items: list[dict], output: Path) -> Path | None:
    if not items:
        return None
    n = len(items)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.8 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, item in zip(axes, items):
        aligned = item["aligned"]
        frames = np.load(aligned / "frames_average_atlas.npy")
        atlas = np.load(aligned / "allen_area_atlas_native_grid.npy")
        edges = _region_edges(atlas)
        points = _load_clicked_points(item["json"])
        transformed_clicked = item["transform"].inverse(points["clicked"])
        image = frames[1]
        ax.imshow(image, cmap="gray", vmin=np.percentile(image, 1), vmax=np.percentile(image, 99.5))
        _overlay_edges(ax, edges, alpha=0.7)
        ax.scatter(transformed_clicked[:, 0], transformed_clicked[:, 1], s=52, marker="x", c="#ff2c2c", linewidths=1.8)
        for name, (x, y) in zip(points["names"], transformed_clicked):
            ax.text(x + 4, y - 4, str(name), color="yellow", fontsize=7, weight="bold")
        ax.set_title(f"{item['aligned_version']} from {item['json_version']}")
        ax.set_axis_off()
    for ax in axes[len(items) :]:
        ax.set_axis_off()
    fig.suptitle(f"{label}: clicked alignment points on mean 470 + Allen outlines")
    out = output / f"{label}_alignment_landmark_points_comparison_grid.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot landmark points on aligned mean 470 nm atlas overlays.")
    parser.add_argument("--label", required=True)
    parser.add_argument("--json-dir", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--current-version", default=None)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    json_paths = sorted(args.json_dir.glob("dorsal_cortex_landmarks*.json"))
    aligned = _load_aligned_transforms(args.results)
    matches = _match_jsons_to_aligned(json_paths, aligned, args.current_version)
    matches.sort(key=lambda x: x["aligned_version"])

    outputs = [_plot_one(args.label, item, args.output) for item in matches]
    grid = _plot_grid(args.label, matches, args.output)
    if grid:
        outputs.append(grid)

    summary = {
        "label": args.label,
        "json_dir": str(args.json_dir),
        "results": str(args.results),
        "output": str(args.output),
        "matched_versions": [
            {
                "json": str(item["json"]),
                "json_version": item["json_version"],
                "aligned": str(item["aligned"]),
                "aligned_version": item["aligned_version"],
                "match_error": item["match_error"],
            }
            for item in matches
        ],
        "outputs": [str(path) for path in outputs],
        "point_note": "Red x marks the clicked image landmark after applying that version's Allen transform. White open circles mark atlas target landmarks when present in the JSON.",
    }
    (args.output / f"{args.label}_alignment_landmark_points_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
