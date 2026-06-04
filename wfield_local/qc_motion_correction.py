"""QC the quality of wfield motion correction for one session.

Reads what the motion step saved (motion_correction_shifts.npy / _rotation.npy) and
the pre/post movies (the cleanpairs .dat input and the motioncorrect_*.bin output)
and emits one QC figure + a pass/warn summary JSON:

  - per-frame shift traces (y, x) over the session + rotation
  - shift-magnitude histogram with thresholds and the fraction of frames beyond
  - mean image sharpness (variance-of-Laplacian focus) raw vs corrected
  - residual-motion temporal-std image of the corrected movie (vessel-edge halos
    = leftover motion); computed on channel 0 to limit neural-activity confound

Big shifts (a large fraction of the FOV) or a cluster of frames pinned at the
search-radius limit indicate frames the registration could not solve. Imports only
numpy/scipy/matplotlib (no wfield), so it runs as its own process.

Usage:
  python -m wfield_local.qc_motion_correction --motion-dir <motion_corrected> \
      --label PS94_0603 --output <dir> [--nsample 200] [--warn-px 5] [--big-px 20]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

DAT_RE = re.compile(r"_(\d+)_(\d+)_(\d+)_uint16")


def _open_stack(path):
    m = DAT_RE.search(os.path.basename(path))
    ch, H, W = int(m.group(1)), int(m.group(2)), int(m.group(3))
    n = os.path.getsize(path) // (ch * H * W * 2)
    return np.memmap(path, mode="r", dtype=np.uint16, shape=(n, ch, H, W)), (n, ch, H, W)


def _focus(img: np.ndarray) -> float:
    """Variance of the Laplacian = image sharpness (higher = crisper)."""
    return float(ndimage.laplace(img.astype(np.float64)).var())


def main() -> int:
    ap = argparse.ArgumentParser(description="QC motion correction for one session.")
    ap.add_argument("--motion-dir", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--nsample", type=int, default=200)
    ap.add_argument("--warn-px", type=float, default=5.0, help="shift magnitude flagged as notable")
    ap.add_argument("--big-px", type=float, default=20.0, help="shift magnitude flagged as failure")
    ap.add_argument("--fs", type=float, default=31.23)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    md = args.motion_dir

    shifts = np.load(md / "motion_correction_shifts.npy")
    rot = np.load(md / "motion_correction_rotation.npy")
    y = np.asarray(shifts["y"], np.float64)
    x = np.asarray(shifts["x"], np.float64)
    # shape (Npairs, Nchan); worst channel per pair for the magnitude summary
    mag = np.sqrt(y ** 2 + x ** 2)
    mag_pair = mag.max(axis=1) if mag.ndim == 2 else mag
    y_t = y.mean(axis=1) if y.ndim == 2 else y
    x_t = x.mean(axis=1) if x.ndim == 2 else x
    rot_t = (rot.mean(axis=1) if rot.ndim == 2 else rot)
    n = len(mag_pair)
    t = np.arange(n) / args.fs / 60.0  # minutes

    frac_warn = float(np.mean(mag_pair > args.warn_px))
    frac_big = float(np.mean(mag_pair > args.big_px))
    maxshift = float(np.nanmax(mag_pair))
    # frames pinned near the search-radius limit (registration failures)
    near_max = float(np.mean(mag_pair > 0.95 * maxshift)) if maxshift > args.big_px else 0.0

    # mean-image sharpness raw vs corrected (channel 0)
    focus = {"raw": None, "corrected": None}
    mean_raw = mean_cor = std_cor = None
    raw_dat = sorted(glob.glob(str(md / "*cleanpairs_*_uint16.dat")))
    cor_bin = sorted(glob.glob(str(md / "motioncorrect_*.bin")))
    good = np.flatnonzero(mag_pair <= args.big_px)  # exclude blown-up frames from imagery
    samp_src = good if good.size > 50 else np.arange(n)
    samp = samp_src[np.linspace(0, samp_src.size - 1, min(args.nsample, samp_src.size)).astype(int)]
    try:
        if raw_dat:
            mm, _ = _open_stack(raw_dat[0])
            mean_raw = np.asarray(mm[samp, 0], np.float32).mean(0)
            focus["raw"] = _focus(mean_raw)
        if cor_bin:
            mm, _ = _open_stack(cor_bin[0])
            block = np.asarray(mm[samp, 0], np.float32)
            mean_cor = block.mean(0)
            std_cor = block.std(0)
            focus["corrected"] = _focus(mean_cor)
    except Exception as exc:
        print("image QC skipped:", exc, flush=True)

    focus_ratio = (focus["corrected"] / focus["raw"]) if (focus["raw"] and focus["corrected"]) else None

    # ---- figure ----
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    step = max(1, n // 4000)
    ax[0, 0].plot(t[::step], y_t[::step], lw=0.5, label="y")
    ax[0, 0].plot(t[::step], x_t[::step], lw=0.5, label="x")
    ax[0, 0].set_title("Per-frame shift (px)"); ax[0, 0].set_xlabel("min"); ax[0, 0].set_ylabel("px"); ax[0, 0].legend()

    ax[0, 1].hist(mag_pair, bins=200, log=True, color="0.4")
    ax[0, 1].axvline(args.warn_px, color="orange", lw=1, label=f"warn {args.warn_px:g}px")
    ax[0, 1].axvline(args.big_px, color="red", lw=1, label=f"fail {args.big_px:g}px")
    ax[0, 1].set_title("Shift magnitude histogram"); ax[0, 1].set_xlabel("px"); ax[0, 1].set_ylabel("frames (log)"); ax[0, 1].legend()

    ax[0, 2].axis("off")
    # Verdict from the SHIFT distribution (the reliable signal). The focus ratio
    # only downgrades when there was real motion to sharpen out: with sub-pixel
    # motion the warp interpolation slightly softens the mean (ratio < 1) even on
    # a perfect correction, so it must not flag low-motion sessions.
    p95 = float(np.percentile(mag_pair, 95))
    med = float(np.median(mag_pair))
    focus_ok = (focus_ratio is None) or (p95 < 1.5) or (focus_ratio >= 0.9)
    if frac_big < 0.001 and med < 1.0 and focus_ok:
        verdict = "GOOD"
    elif frac_big < 0.02 and focus_ok:
        verdict = "CHECK"
    else:
        verdict = "POOR"
    lines = [
        f"{args.label}   motion-correction QC: {verdict}", "",
        f"frames (pairs): {n}",
        f"max shift: {maxshift:.1f} px",
        f"median shift: {np.median(mag_pair):.2f} px",
        f"frames > {args.warn_px:g}px: {frac_warn*100:.2f}%",
        f"frames > {args.big_px:g}px: {frac_big*100:.2f}%",
        f"frames near search-limit (~{maxshift:.0f}px): {near_max*100:.2f}%",
        f"rotation range: {float(np.nanmin(rot_t)):.2f}..{float(np.nanmax(rot_t)):.2f} deg",
        "",
        f"focus raw: {focus['raw']:.1f}" if focus['raw'] else "focus raw: n/a",
        f"focus corrected: {focus['corrected']:.1f}" if focus['corrected'] else "focus corrected: n/a",
        f"focus ratio (cor/raw): {focus_ratio:.2f}" if focus_ratio else "focus ratio: n/a",
        "(>1 = sharper after correction)",
    ]
    ax[0, 2].text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace")

    for a, img, ttl in [(ax[1, 0], mean_raw, "raw mean (ch0)"),
                        (ax[1, 1], mean_cor, "corrected mean (ch0)")]:
        a.axis("off"); a.set_title(ttl)
        if img is not None:
            a.imshow(img, cmap="gray", vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.5))
    ax[1, 2].axis("off"); ax[1, 2].set_title("corrected temporal std (residual motion)")
    if std_cor is not None:
        ax[1, 2].imshow(std_cor, cmap="magma", vmax=np.percentile(std_cor, 99))

    fig.suptitle(f"{args.label} motion-correction QC  [{verdict}]", fontsize=15)
    fig.tight_layout()
    png = args.output / f"{args.label}_motion_qc.png"
    fig.savefig(png, dpi=130); plt.close(fig)

    summary = {
        "label": args.label, "motion_dir": str(md), "verdict": verdict,
        "n_frames": n, "max_shift_px": maxshift, "median_shift_px": float(np.median(mag_pair)),
        "frac_over_warn": frac_warn, "frac_over_big": frac_big, "frac_near_search_limit": near_max,
        "warn_px": args.warn_px, "big_px": args.big_px,
        "focus_raw": focus["raw"], "focus_corrected": focus["corrected"], "focus_ratio": focus_ratio,
        "rotation_min_max": [float(np.nanmin(rot_t)), float(np.nanmax(rot_t))],
        "notes": "shift magnitude = max over channels per pair; imagery excludes frames > big_px.",
    }
    (args.output / f"{args.label}_motion_qc_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print("wrote", png, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
