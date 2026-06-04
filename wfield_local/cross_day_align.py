"""Cross-day alignment of the same animal via the motion-corrected mean 470 nm image.

Strategy (reference-based, vasculature-driven):
  1. Pick ONE session as the reference and CCF-align it with its landmark JSON
     (native -> 540x640 Allen reference space). Its warped mean 470 nm image is the
     vasculature template.
  2. For every other session, register its mean 470 nm image to that template:
       - initialize from the session's own landmark transform if available (brings
         it roughly into CCF), then
       - refine with an intensity-based AFFINE registration on the vasculature
         (OpenCV ECC). If a session has no landmarks, initialize from scratch with
         SIFT + RANSAC.
  3. The composed transform (native -> reference/CCF frame) can then be applied to
     that session's U components (or any map) so all days share one frame AND carry
     Allen labels (from the reference).

Vasculature (the dark branching pattern in the mean image) is a far denser, more
repeatable fiducial than the handful of landmarks, so this removes the day-to-day
wobble that independent per-session landmark fits leave behind. Across ANIMALS this
approach does NOT apply (different vessels) -- use CCF/landmarks + ROI-level
comparison there.

Config JSON:
  {
    "animal": "PS94",
    "func_channel": 1,                 # index of 470 nm in frames_average.npy
    "reference": "PS94_0603",
    "output": "M:/Widefield/xday/PS94",
    "warp_u": false,                   # also warp U.npy into the reference frame
    "sessions": {
      "PS94_0601": {"results": ".../wfield_local_results", "landmarks": ".../dorsal_cortex_landmarks_v1.json"},
      "PS94_0603": {"results": ".../wfield_local_results", "landmarks": ".../dorsal_cortex_landmarks_v1.json"},
      "PS94_0604": {"results": ".../wfield_local_results", "landmarks": ".../dorsal_cortex_landmarks_v1.json"}
    }
  }

Run:  python -m wfield_local.cross_day_align config.json
Imports cv2 + skimage + wfield (for the landmark transform); run as its own process.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import warp as sk_warp, AffineTransform
from skimage.feature import SIFT, match_descriptors
from skimage.measure import ransac

from wfield.allen import load_allen_landmarks

DIMS = (540, 640)


def _prep(img: np.ndarray) -> np.ndarray:
    """High-pass + normalize so registration keys on vasculature, not illumination."""
    g = img.astype(np.float32)
    g = np.nan_to_num(g)
    blur = cv2.GaussianBlur(g, (0, 0), sigmaX=25)
    hp = g - blur
    s = hp.std() + 1e-6
    return (hp - hp.mean()) / s


def _warp_native_to_ccf(img: np.ndarray, transform, dims=DIMS) -> np.ndarray:
    return sk_warp(np.asarray(img, np.float64), transform, output_shape=dims,
                   order=1, mode="constant", cval=0, preserve_range=True).astype(np.float32)


def _refine_ecc(moving_ccf: np.ndarray, ref_p: np.ndarray, mask: np.ndarray):
    """Greedy residual registration of moving_ccf onto the reference (prepped).

    Tries rigid then affine ECC and keeps a residual ONLY if it raises the masked
    NCC vs the landmark-init alignment, so refinement can never make it worse.
    Returns (residual_2x3, ncc_after, method).
    """
    mbool = mask.astype(bool)
    mov_p = _prep(moving_ccf)
    best_w = np.eye(2, 3, dtype=np.float32)
    best_ncc = _ncc(mov_p, ref_p, mbool)
    best_name = "init-only"
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-7)
    for name, motion in (("euclidean", cv2.MOTION_EUCLIDEAN), ("affine", cv2.MOTION_AFFINE)):
        try:
            w = np.eye(2, 3, dtype=np.float32)
            _, w = cv2.findTransformECC(ref_p, mov_p, w, motion, crit, mask, 5)
            cand = cv2.warpAffine(moving_ccf, w, (DIMS[1], DIMS[0]), flags=cv2.INTER_LINEAR)
            nc = _ncc(_prep(cand), ref_p, mbool)
            if nc > best_ncc + 1e-4:
                best_w, best_ncc, best_name = w, nc, "ecc-" + name
        except cv2.error:
            pass
    # SIFT + RANSAC feature residual (handles large offset/scale ECC can't bridge)
    try:
        model, ninl = _sift_ransac_affine(mov_p, ref_p)            # maps moving -> ref
        w = np.asarray(model.inverse.params[:2], np.float32)        # cv2 wants ref -> moving
        cand = cv2.warpAffine(moving_ccf, w, (DIMS[1], DIMS[0]), flags=cv2.INTER_LINEAR)
        nc = _ncc(_prep(cand), ref_p, mbool)
        if nc > best_ncc + 1e-4:
            best_w, best_ncc, best_name = w, nc, f"sift({ninl})"
    except Exception:
        pass
    return best_w, best_ncc, best_name


def _sift_ransac_affine(moving: np.ndarray, ref: np.ndarray):
    """From-scratch affine (moving->ref) when there is no landmark initialization."""
    def feats(im):
        s = SIFT(); s.detect_and_extract(im); return s.keypoints, s.descriptors
    kp_m, de_m = feats(moving); kp_r, de_r = feats(ref)
    matches = match_descriptors(de_m, de_r, cross_check=True, max_ratio=0.8)
    src = kp_m[matches[:, 0]][:, ::-1]; dst = kp_r[matches[:, 1]][:, ::-1]  # (x,y)
    model, inliers = ransac((src, dst), AffineTransform, min_samples=3,
                            residual_threshold=3, max_trials=2000)
    return model, int(inliers.sum())


def _ncc(a, b, mask):
    a = a[mask]; b = b[mask]
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    func = int(cfg.get("func_channel", 1))
    out = Path(cfg["output"]); out.mkdir(parents=True, exist_ok=True)
    ref_id = cfg["reference"]
    sessions = cfg["sessions"]
    warp_u = bool(cfg.get("warp_u", False))

    def mean470(results, fc):
        favg = np.load(Path(results) / "frames_average.npy")
        return favg[fc].astype(np.float32)

    # --- reference template in CCF ---
    ref = sessions[ref_id]
    ref_T = load_allen_landmarks(str(ref["landmarks"]))["transform"]
    ref_ccf = _warp_native_to_ccf(mean470(ref["results"], int(ref.get("func_channel", func))), ref_T)
    ref_p = _prep(ref_ccf)
    brain = (ref_ccf > 0).astype(np.uint8)
    brain = cv2.erode(brain, np.ones((9, 9), np.uint8))  # ignore warp-boundary ring

    results = {}
    warped = {}
    for sid, s in sessions.items():
        m470 = mean470(s["results"], int(s.get("func_channel", func)))
        init_T = None
        if s.get("landmarks") and Path(s["landmarks"]).exists():
            init_T = load_allen_landmarks(str(s["landmarks"]))["transform"]
        if sid == ref_id:
            init_ccf = ref_ccf; resid = np.eye(2, 3, dtype=np.float32); how = "reference"
        else:
            if init_T is not None:
                init_ccf = _warp_native_to_ccf(m470, init_T); initdesc = "landmark-init"
            else:
                init_T = ref_T  # no landmarks: rough placement on ref grid; SIFT/ECC handle the rest
                init_ccf = _warp_native_to_ccf(m470, ref_T); initdesc = "no-landmarks"
            resid, _nc, motion = _refine_ecc(init_ccf, ref_p, brain)
            how = f"{initdesc} + {motion}"
        # final warped mean in reference frame
        final = cv2.warpAffine(init_ccf, resid, (DIMS[1], DIMS[0]), flags=cv2.INTER_LINEAR)
        warped[sid] = final
        ncc_before = _ncc(_prep(init_ccf), ref_p, brain.astype(bool))
        ncc_after = _ncc(_prep(final), ref_p, brain.astype(bool))
        np.save(out / f"{sid}_mean470_in_ref.npy", final.astype(np.float32))
        results[sid] = {
            "results": s["results"], "method": how,
            "ncc_before": ncc_before, "ncc_after": ncc_after,
            "init_transform": np.asarray(init_T.params).tolist() if init_T is not None else None,
            "residual_affine_2x3": np.asarray(resid).tolist(),
        }
        print(f"[{sid}] {how}  NCC {ncc_before:.3f} -> {ncc_after:.3f}", flush=True)

        if warp_u and sid != ref_id:
            U = np.load(Path(s["results"]) / "U.npy")  # (H,W,K)
            Ux = np.empty((*DIMS, U.shape[2]), np.float32)
            for k in range(U.shape[2]):
                ccf = _warp_native_to_ccf(U[..., k], init_T)
                Ux[..., k] = cv2.warpAffine(ccf, resid, (DIMS[1], DIMS[0]), flags=cv2.INTER_LINEAR)
            np.save(out / f"{sid}_U_xday.npy", Ux)
            print(f"[{sid}] warped U -> {sid}_U_xday.npy {Ux.shape}", flush=True)
        elif warp_u and sid == ref_id:
            U = np.load(Path(s["results"]) / "U.npy")
            Ux = np.stack([_warp_native_to_ccf(U[..., k], ref_T) for k in range(U.shape[2])], -1).astype(np.float32)
            np.save(out / f"{sid}_U_xday.npy", Ux)

    # --- QC figure: ref (green) vs each session (magenta); overlap -> grey ---
    ids = list(sessions)
    n = len(ids)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 8.4))
    if n == 1:
        axes = axes.reshape(2, 1)

    def norm(im):
        v = im.copy(); lo, hi = np.percentile(v[brain.astype(bool)], [2, 98])
        return np.clip((v - lo) / (hi - lo + 1e-9), 0, 1)

    refn = norm(ref_ccf)
    for j, sid in enumerate(ids):
        sn = norm(warped[sid])
        rgb = np.zeros((*DIMS, 3), np.float32)
        rgb[..., 1] = refn          # reference -> green
        rgb[..., 0] = sn; rgb[..., 2] = sn  # session -> magenta
        axes[0, j].imshow(rgb); axes[0, j].set_axis_off()
        axes[0, j].set_title(f"{sid}\nNCC {results[sid]['ncc_after']:.3f}", fontsize=9)
        axes[1, j].imshow(sn, cmap="gray"); axes[1, j].set_axis_off()
        axes[1, j].set_title(f"{sid} mean470 -> ref frame", fontsize=8)
    fig.suptitle(f"{cfg.get('animal','')} cross-day alignment to {ref_id} (470 nm vasculature)\n"
                 "row1: reference=green, session=magenta (white/grey = aligned vessels)", fontsize=12)
    fig.tight_layout()
    qc = out / f"{cfg.get('animal','animal')}_cross_day_alignment_qc.png"
    fig.savefig(qc, dpi=140); plt.close(fig)

    (out / "cross_day_align_summary.json").write_text(json.dumps(
        {"animal": cfg.get("animal"), "reference": ref_id, "func_channel": func,
         "dims": list(DIMS), "qc_png": str(qc), "sessions": results}, indent=2))
    print(f"\nwrote {qc}\nNCC after (higher=better vessel alignment):")
    for sid in ids:
        print(f"  {sid}: {results[sid]['ncc_after']:.3f}  ({results[sid]['method']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
