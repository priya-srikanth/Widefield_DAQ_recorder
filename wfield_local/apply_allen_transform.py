"""Apply a saved wfield Allen landmark transform to spatial outputs.

The transform in ``dorsal_cortex_landmarks.json`` is used the same way wfield's
viewer uses it: native camera images are resampled into the landmark reference
space. Native-space files are left untouched; warped outputs get ``_atlas`` in
their names.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wfield.allen import atlas_from_landmarks_file, load_allen_landmarks
from wfield.utils import im_apply_transform


def _warp_image(image: np.ndarray, transform) -> np.ndarray:
    return im_apply_transform(np.asarray(image), transform).astype(np.float32)


def _warp_stack_last_axis(stack: np.ndarray, transform) -> np.ndarray:
    warped = np.empty_like(stack, dtype=np.float32)
    for idx in range(stack.shape[-1]):
        print(f"warping component {idx + 1}/{stack.shape[-1]}", flush=True)
        warped[..., idx] = _warp_image(stack[..., idx], transform)
    return warped


def _maybe_warp_npy(path: Path, transform, output: Path) -> str | None:
    arr = np.load(path)
    out_path = output / f"{path.stem}_atlas.npy"
    if arr.ndim == 2:
        np.save(out_path, _warp_image(arr, transform))
    elif arr.ndim == 3 and arr.shape[0] in (1, 2):
        warped = np.stack([_warp_image(arr[i], transform) for i in range(arr.shape[0])])
        np.save(out_path, warped.astype(np.float32))
    elif arr.ndim == 3:
        np.save(out_path, _warp_stack_last_axis(arr, transform))
    else:
        return None
    return str(out_path)


def _warp_npz_images(path: Path, transform, output: Path) -> str:
    src = np.load(path)
    warped = {}
    for key in src.files:
        value = src[key]
        if isinstance(value, np.ndarray) and value.shape == src[src.files[0]].shape and value.ndim == 2:
            warped[key] = _warp_image(value, transform)
        elif isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[-2:] == src[src.files[0]].shape[-2:]:
            warped[key] = _warp_image(value, transform)
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
        "--include-npz",
        action="store_true",
        help="Also warp 2D image arrays inside NPZ summary files.",
    )
    args = parser.parse_args()

    output = args.output or (args.results / "allen_aligned")
    output.mkdir(parents=True, exist_ok=True)

    landmarks = load_allen_landmarks(str(args.landmarks))
    transform = landmarks["transform"]
    copied = []

    for name in ("U.npy", "frames_average.npy", "rcoeffs.npy", "mask.npy"):
        path = args.results / name
        if path.exists():
            warped = _maybe_warp_npy(path, transform, output)
            if warped:
                copied.append(warped)

    if args.include_npz:
        for path in sorted(args.results.glob("*.npz")):
            copied.append(_warp_npz_images(path, transform, output))

    try:
        atlas, area_names, brain_mask = atlas_from_landmarks_file(str(args.landmarks), do_transform=True)
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
        "transform": np.asarray(transform.params).tolist(),
        "outputs": copied,
        "note": "Spatial arrays were resampled with wfield.utils.im_apply_transform. Native-space files were not modified.",
    }
    (output / "allen_transform_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Wrote Allen-aligned outputs to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
