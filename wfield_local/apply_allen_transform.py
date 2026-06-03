"""Apply a saved wfield Allen landmark transform to spatial outputs.

The transform in ``dorsal_cortex_landmarks.json`` is used the same way wfield's
viewer uses it: native camera images are resampled into the landmark reference
space. Native-space files are left untouched; warped outputs get ``_atlas`` in
their names.

Images are warped onto the Allen atlas grid (``--dims``, default 540x640), which
is the grid the atlas/region masks are built on. This matters for ROI-cropped
recordings: their native frames are smaller than the atlas grid, so warping to
the input size (the old behaviour) left the warped image and the atlas on
different grids and the overlay came out misaligned/cut-off. Full-FOV (540x640)
recordings are unaffected since input size already equals the atlas grid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from skimage.transform import warp as _sk_warp

from wfield.allen import atlas_from_landmarks_file, load_allen_landmarks


def _warp_image(image: np.ndarray, transform, dims) -> np.ndarray:
    # Warp into the Allen atlas grid (``dims``), NOT the input image size. For a
    # full-FOV recording these are equal, but for an ROI-cropped recording the
    # input is smaller than the atlas grid; warping to the input size leaves the
    # warped image and the atlas on different grids (misaligned/cut-off overlay).
    return _sk_warp(
        np.asarray(image, dtype=np.float64), transform, output_shape=tuple(dims),
        order=1, mode="constant", cval=0, clip=True, preserve_range=True,
    ).astype(np.float32)


def _warp_stack_last_axis(stack: np.ndarray, transform, dims) -> np.ndarray:
    warped = np.empty((dims[0], dims[1], stack.shape[-1]), dtype=np.float32)
    for idx in range(stack.shape[-1]):
        print(f"warping component {idx + 1}/{stack.shape[-1]}", flush=True)
        warped[..., idx] = _warp_image(stack[..., idx], transform, dims)
    return warped


def _maybe_warp_npy(path: Path, transform, output: Path, dims) -> str | None:
    arr = np.load(path)
    out_path = output / f"{path.stem}_atlas.npy"
    if arr.ndim == 2:
        np.save(out_path, _warp_image(arr, transform, dims))
    elif arr.ndim == 3 and arr.shape[0] in (1, 2):
        warped = np.stack([_warp_image(arr[i], transform, dims) for i in range(arr.shape[0])])
        np.save(out_path, warped.astype(np.float32))
    elif arr.ndim == 3:
        np.save(out_path, _warp_stack_last_axis(arr, transform, dims))
    else:
        return None
    return str(out_path)


def _warp_npz_images(path: Path, transform, output: Path, dims) -> str:
    src = np.load(path)
    warped = {}
    for key in src.files:
        value = src[key]
        if isinstance(value, np.ndarray) and value.shape == src[src.files[0]].shape and value.ndim == 2:
            warped[key] = _warp_image(value, transform, dims)
        elif isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[-2:] == src[src.files[0]].shape[-2:]:
            warped[key] = _warp_image(value, transform, dims)
        else:
            warped[key] = value
    out_path = output / f"{path.stem}_atlas.npz"
    np.savez_compressed(out_path, **warped)
    return str(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Warp wfield outputs into Allen landmark space.")
    parser.add_argument("results", type=Path, help="Folder containing wfield outputs")
    parser.add_argument(
        "--landmarks",
        type=Path,
        required=True,
        help="dorsal_cortex_landmarks.json from the session",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--dims",
        type=int,
        nargs=2,
        default=(540, 640),
        metavar=("H", "W"),
        help="Allen atlas grid size to warp into (default 540 640). Images are warped "
        "to this grid, not their input size, so ROI-cropped recordings stay aligned "
        "with the atlas.",
    )
    parser.add_argument(
        "--include-npz",
        action="store_true",
        help="Also warp 2D image arrays inside NPZ summary files.",
    )
    args = parser.parse_args()

    output = args.output or (args.results / "allen_aligned")
    output.mkdir(parents=True, exist_ok=True)
    dims = tuple(args.dims)

    landmarks = load_allen_landmarks(str(args.landmarks))
    transform = landmarks["transform"]
    copied = []

    for name in ("U.npy", "frames_average.npy", "rcoeffs.npy", "mask.npy"):
        path = args.results / name
        if path.exists():
            warped = _maybe_warp_npy(path, transform, output, dims)
            if warped:
                copied.append(warped)

    if args.include_npz:
        for path in sorted(args.results.glob("*.npz")):
            copied.append(_warp_npz_images(path, transform, output, dims))

    try:
        # do_transform=False keeps the atlas in REFERENCE space (bregma at
        # bregma_offset on the dims grid), matching the warped images above
        # (skimage.warp(image, T) yields reference-space output). Using
        # do_transform=True would place the atlas in IMAGE space and only line up
        # when T is near-identity (e.g. a full-FOV recording already at the
        # reference pose); for a non-identity transform it shifts the overlay.
        atlas, area_names, brain_mask = atlas_from_landmarks_file(str(args.landmarks), dims=list(dims), do_transform=False)
        np.save(output / "allen_area_atlas_native_grid.npy", atlas.astype(np.float32))
        np.save(output / "allen_brain_mask_native_grid.npy", brain_mask.astype(np.uint8))
        (output / "allen_area_names.json").write_text(
            json.dumps(list(area_names), indent=2), encoding="utf-8"
        )
    except Exception as exc:
        print(f"Could not generate Allen atlas masks: {exc}", flush=True)

    summary = {
        "results": str(args.results),
        "landmarks": str(args.landmarks),
        "output": str(output),
        "dims": list(dims),
        "transform": np.asarray(transform.params).tolist(),
        "outputs": copied,
        "note": "Spatial arrays were warped into the Allen atlas grid (dims) with "
        "skimage.transform.warp, so ROI-cropped recordings stay aligned with the atlas. "
        "Native-space files were not modified.",
    }
    (output / "allen_transform_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Wrote Allen-aligned outputs to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
