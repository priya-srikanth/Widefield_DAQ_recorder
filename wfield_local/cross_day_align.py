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

Two modes (``--mode`` or config ``"mode"``):
  * reference-native (PREFERRED for day-to-day): register every session's mean 470
    to ONE reference session in its NATIVE frame via FFT phase-correlation (rigid;
    optional euclidean refine), then apply the reference's SINGLE Allen landmark
    transform to carry the whole stack into CCF. One landmark fit total; for fixed-
    FOV recordings this reaches NCC ~0.99 (vs ~0.6-0.9 for per-session landmarks).
  * ccf-per-session (default, legacy): CCF-align each session via its OWN landmarks,
    then intensity ECC refine. More landmark work; lower/variable NCC.

Config JSON:
  {
    "animal": "PS94",
    "mode": "reference-native",        # or "ccf-per-session"
    "func_channel": 1,                 # index of 470 nm in frames_average.npy
    "reference": "PS94_0603",          # the ONE session whose Allen landmarks -> CCF
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
import shutil
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import warp as sk_warp, AffineTransform
from skimage.feature import SIFT, match_descriptors
from skimage.measure import ransac
from skimage.registration import phase_cross_correlation

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


def _mean470(results, fc):
    return np.load(Path(results) / "frames_average.npy")[fc].astype(np.float32)


def _register_to_reference_native(mov, ref, mask):
    """Rigid registration of a session mean onto the REFERENCE session's native
    mean (no CCF in the loop): robust FFT phase-correlation translation, then an
    optional euclidean (rot+trans) ECC refine seeded from it. Keep-best by masked
    NCC on the high-pass vasculature. Returns (cv2 2x3 ref->mov, ncc, method).

    For same-FOV day-to-day recordings the offset is a small rigid translation, so
    phase correlation alone typically reaches NCC ~0.99 -- better than, and far more
    robust than, gradient ECC from identity.
    """
    h, w = ref.shape
    if mov.shape != ref.shape:
        mov = cv2.resize(mov, (w, h), interpolation=cv2.INTER_AREA)
    refp = _prep(ref); movp = _prep(mov); mb = mask.astype(bool)
    best_w = np.eye(2, 3, dtype=np.float32)
    best_ncc = _ncc(movp, refp, mb); best_name = "identity"
    try:
        sh, _, _ = phase_cross_correlation(refp, movp, upsample_factor=20, normalization=None)
        wt = np.array([[1, 0, sh[1]], [0, 1, sh[0]]], np.float32)   # cv2 ref->mov: tx=col, ty=row
        nc = _ncc(_prep(cv2.warpAffine(mov, wt, (w, h), flags=cv2.INTER_LINEAR)), refp, mb)
        if nc > best_ncc:
            best_w, best_ncc, best_name = wt, nc, "phasecorr"
    except Exception:
        pass
    try:   # add rotation if it helps, seeded from the translation
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 2000, 1e-7)
        _, we = cv2.findTransformECC(refp, movp, best_w.copy(), cv2.MOTION_EUCLIDEAN, crit, mask, 5)
        nc = _ncc(_prep(cv2.warpAffine(mov, we, (w, h), flags=cv2.INTER_LINEAR)), refp, mb)
        if nc > best_ncc + 1e-4:
            best_w, best_ncc, best_name = we, nc, "phasecorr+euclidean"
    except cv2.error:
        pass
    return best_w, best_ncc, best_name


def run_reference_native(cfg, out, func):
    """Reference-native cross-day mode: register every session's vasculature to ONE
    reference session in its native frame, then apply the reference's single Allen
    transform to carry the whole stack into CCF. One landmark fit total."""
    ref_id = cfg["reference"]; sessions = cfg["sessions"]; warp_u = bool(cfg.get("warp_u", False))
    ref = sessions[ref_id]
    ref_native = _mean470(ref["results"], int(ref.get("func_channel", func)))
    ref_T = load_allen_landmarks(str(ref["landmarks"]))["transform"]   # the ONE CCF transform
    h, w = ref_native.shape
    mask = cv2.erode((ref_native > np.percentile(ref_native, 30)).astype(np.uint8), np.ones((9, 9), np.uint8))
    refp = _prep(ref_native); mb = mask.astype(bool)

    results, warped = {}, {}
    for sid, s in sessions.items():
        mov = _mean470(s["results"], int(s.get("func_channel", func)))
        if sid == ref_id:
            w2x3 = np.eye(2, 3, dtype=np.float32); ncc = 1.0; method = "reference"; mov_in_ref = ref_native
        else:
            w2x3, ncc, method = _register_to_reference_native(mov, ref_native, mask)
            movr = cv2.resize(mov, (w, h), interpolation=cv2.INTER_AREA) if mov.shape != (h, w) else mov
            mov_in_ref = cv2.warpAffine(movr, w2x3, (w, h), flags=cv2.INTER_LINEAR)
        warped[sid] = mov_in_ref
        ccf = _warp_native_to_ccf(mov_in_ref, ref_T)
        np.save(out / f"{sid}_mean470_in_ref_native.npy", mov_in_ref.astype(np.float32))
        np.save(out / f"{sid}_mean470_in_ccf.npy", ccf.astype(np.float32))
        results[sid] = {"results": s["results"], "method": method, "ncc_to_ref": float(ncc),
                        "session_to_ref_2x3": np.asarray(w2x3).tolist()}
        print(f"[{sid}] reference-native: {method}  NCC->ref {ncc:.3f}", flush=True)
        if warp_u:
            U = np.load(Path(s["results"]) / "U.npy")
            Ux = np.empty((*DIMS, U.shape[2]), np.float32)
            for k in range(U.shape[2]):
                uk = U[..., k]
                if uk.shape != (h, w):
                    uk = cv2.resize(uk, (w, h), interpolation=cv2.INTER_AREA)
                Ux[..., k] = _warp_native_to_ccf(cv2.warpAffine(uk, w2x3, (w, h), flags=cv2.INTER_LINEAR), ref_T)
            np.save(out / f"{sid}_U_xday.npy", Ux)
            if sid != ref_id:
                # assemble a LocaNMF/maps-ready allen dir in the REFERENCE's CCF for
                # this session (U + both-channel mean warped to ref CCF; atlas/mask
                # copied from the reference's own Allen alignment).
                ref_allen = Path(ref["results"]) / "allen_aligned_affine8v1"
                ad = Path(s["results"]) / cfg.get("allen_dir_name", "allen_aligned_affine8v1")
                ad.mkdir(parents=True, exist_ok=True)
                np.save(ad / "U_atlas.npy", Ux)
                fa = np.load(Path(s["results"]) / "frames_average.npy")  # (2,H,W) raw mean
                fax = np.empty((fa.shape[0], *DIMS), np.float32)
                for c in range(fa.shape[0]):
                    ck = fa[c]
                    if ck.shape != (h, w):
                        ck = cv2.resize(ck, (w, h), interpolation=cv2.INTER_AREA)
                    fax[c] = _warp_native_to_ccf(cv2.warpAffine(ck, w2x3, (w, h), flags=cv2.INTER_LINEAR), ref_T)
                np.save(ad / "frames_average_atlas.npy", fax)
                for fn in ("allen_area_atlas_native_grid.npy", "allen_brain_mask_native_grid.npy", "allen_area_names.json"):
                    src = ref_allen / fn
                    if src.exists():
                        shutil.copy2(src, ad / fn)
                (ad / "cross_day_allen_summary.json").write_text(json.dumps(
                    {"session": sid, "reference": ref_id, "ncc_to_ref": float(ncc),
                     "note": "U + frames_average warped to reference CCF via session->ref "
                             "(native phase-corr) + reference Allen transform; atlas/mask from reference."},
                    indent=2))
                print(f"[{sid}] emitted allen dir -> {ad}", flush=True)

    # QC in the reference NATIVE frame (where the alignment is measured)
    ids = list(sessions); n = len(ids)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 8.4))
    if n == 1:
        axes = axes.reshape(2, 1)
    def norm(im):
        lo, hi = np.percentile(im[mb], [2, 98]); return np.clip((im - lo) / (hi - lo + 1e-9), 0, 1)
    refn = norm(ref_native)
    for j, sid in enumerate(ids):
        sn = norm(warped[sid]); rgb = np.zeros((h, w, 3), np.float32)
        rgb[..., 1] = refn; rgb[..., 0] = sn; rgb[..., 2] = sn
        axes[0, j].imshow(rgb); axes[0, j].set_axis_off()
        axes[0, j].set_title(f"{sid}\nNCC {results[sid]['ncc_to_ref']:.3f}", fontsize=9)
        axes[1, j].imshow(sn, cmap="gray"); axes[1, j].set_axis_off()
        axes[1, j].set_title(f"{sid} -> ref native", fontsize=8)
    fig.suptitle(f"{cfg.get('animal','')} cross-day (reference-native) -> {ref_id} (470 nm vasculature)\n"
                 "row1: reference=green, session=magenta (white/grey=aligned). One Allen fit (reference) -> CCF.",
                 fontsize=12)
    fig.tight_layout()
    qc = out / f"{cfg.get('animal','animal')}_cross_day_alignment_qc.png"
    fig.savefig(qc, dpi=140); plt.close(fig)
    (out / "cross_day_align_summary.json").write_text(json.dumps(
        {"animal": cfg.get("animal"), "mode": "reference-native", "reference": ref_id,
         "func_channel": func, "dims": list(DIMS), "qc_png": str(qc), "sessions": results}, indent=2))
    print(f"\nwrote {qc}\nNCC to reference (reference-native):")
    for sid in ids:
        print(f"  {sid}: {results[sid]['ncc_to_ref']:.3f}  ({results[sid]['method']})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path)
    ap.add_argument("--mode", choices=("ccf-per-session", "reference-native"), default=None,
                    help="ccf-per-session = CCF-align each session via its own landmarks then ECC "
                         "(default); reference-native = register every session to ONE reference in "
                         "native space (phase-corr) + the reference's single Allen transform to CCF")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    mode = args.mode or cfg.get("mode", "ccf-per-session")
    if mode == "reference-native":
        func0 = int(cfg.get("func_channel", 1))
        outp = Path(cfg["output"]); outp.mkdir(parents=True, exist_ok=True)
        return run_reference_native(cfg, outp, func0)
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
