"""Lick-triggered LocaNMF component traces, by spout position, quiet-z-scored.

For one session: detect licks on ``lick_analog``, assign each lick to a spout position,
map it to a corrected-frame index, and extract a peri-lick window of every LocaNMF
component's temporal trace ``C`` (the C time axis IS the corrected-frame axis). Traces are
quiet-normalized per session: for each component, subtract its mean over the quiet-period
frames and z-score by its quiet-period SD (so units are "sigma above the not-running/
not-licking baseline"), using the per-corrected-frame mask from ``quiet_periods.py``.

We KEEP the full (n_licks x time) single-trial tensor per component x position, and also
store the trial average (mean +/- SEM) -- the average is the cross-animal overlay view, the
trials retain single-trial temporal structure. Components stay individual but each is tagged
with its Allen seed region (``regions.npy``) so the same areas line up across animals.

Frame mapping (reuses the lick-map plotters' helpers):
  * regime A (full-FOV, no relabel): corrected frame = nearest pco_exposure pulse // 2.
  * regime B (relabeled cleanpairs): frame_map["original_frame_index_ch0"] + offset -> pco
    sample per corrected frame, nearest to the lick sample, with a contiguity guard so a
    peri-lick window never spans a dropped/boundary frame.

Run (regime B, e.g. PS94 6/3):
  python -m wfield_local.locanmf_lick_aligned \
    --daq-h5 ".../DAQ_recorder_output/20250603/PS94_20260603_175946.h5" \
    --locanmf-dir ".../locanmf_affine8v1_final" --label PS94_0603 \
    --quiet-frame ".../quiet_affine8v1/PS94_0603_affine8v1_quiet_frame.npy" \
    --frame-map ".../pco_..._cleanpairs_frame_map.npz" --offset 0 \
    --output ".../locanmf_lick_aligned_affine8v1"
Regime A (6/1): omit --frame-map/--offset (uses pco//2).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# h5py/numpy-only helpers (these modules do NOT import wfield)
from wfield_local.plot_lick_aligned_averages import (
    _load_daq_events,
    _classify_events,
    _event_frame_indices_from_pco,
    POSITION_NAMES,
    DISPLAY_ORDER,
)

# Allen seed labels for the orofacial network (both hemispheres) used by the quick-look fig.
OROFACIAL = {3: "MOp", 4: "MOs", 5: "SSp-n", 6: "SSp-m"}


def _corrected_frame_samples(frame_map: Path, pco_samples: np.ndarray, offset: int) -> np.ndarray:
    fm = np.load(frame_map)
    if "original_frame_index_ch0" not in fm.files:
        raise KeyError(f"{frame_map} has no 'original_frame_index_ch0' (keys: {fm.files})")
    idx = np.clip(fm["original_frame_index_ch0"] + offset, 0, len(pco_samples) - 1)
    return pco_samples[idx]


def _nearest_corrected_frame(event_samples: np.ndarray, csample: np.ndarray) -> np.ndarray:
    ins = np.clip(np.searchsorted(csample, event_samples), 1, len(csample) - 1)
    prev = np.abs(event_samples - csample[ins - 1])
    nxt = np.abs(csample[ins] - event_samples)
    return np.where(prev <= nxt, ins - 1, ins).astype(np.int64)


def _quiet_zscore(C: np.ndarray, quiet: np.ndarray) -> tuple[np.ndarray, dict]:
    """Per-component: subtract quiet-frame mean, divide by quiet-frame SD.

    The quiet mask is used ONLY to estimate each component's baseline mean/SD over the
    (thousands of) quiet frames, so when its length differs slightly from T (some regime-A
    masks were built on a marginally different frame count than SVTcorr) we align to the
    common prefix -- the baseline statistics are unaffected by dropping a few tail frames.
    """
    quiet = np.asarray(quiet).astype(bool)
    T = C.shape[1]
    length_mismatch = int(quiet.shape[0] - T)
    if quiet.shape[0] != T:
        L = min(quiet.shape[0], T)
        qmask = np.zeros(T, bool)
        qmask[:L] = quiet[:L]
        quiet = qmask
    nq = int(quiet.sum())
    if nq < 2:
        raise ValueError(f"only {nq} quiet frames; cannot z-score")
    mu = C[:, quiet].mean(axis=1, keepdims=True)
    sd = C[:, quiet].std(axis=1, keepdims=True)
    sd_safe = np.where(sd > 0, sd, 1.0)
    Cn = (C - mu) / sd_safe
    return Cn.astype(np.float32), {"quiet_frames": nq, "zero_sd_components": int((sd <= 0).sum()),
                                   "quiet_mask_length_minus_T": length_mismatch}


def analyze(args) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    ev = _load_daq_events(
        args.daq_h5, args.lick_channel, args.lick_thresh_upper_v, args.lick_thresh_lower_v,
        tuple(args.lockout_s), args.refractory_s,
    )
    fsd = ev["sample_rate_hz"]
    C = np.load(args.locanmf_dir / f"{args.label}_locanmf_C.npy")          # (ncomp, T)
    regions = np.load(args.locanmf_dir / f"{args.label}_locanmf_regions.npy")
    ncomp, T = C.shape
    Cn, qinfo = _quiet_zscore(C, np.load(args.quiet_frame))

    regime = "B" if args.frame_map is not None else "A"
    if regime == "B":
        offset = args.offset if args.offset is not None else json.loads(
            Path(args.cleanpairs_summary).read_text())["chosen_exposure_offset"]
        csample = _corrected_frame_samples(args.frame_map, ev["pco_samples"], int(offset))
        lick_frames = _nearest_corrected_frame(ev["lick_samples"], csample)
    else:
        offset = None
        csample = None
        lick_frames = _event_frame_indices_from_pco(ev["lick_samples"], ev["pco_samples"]) // 2
    codes = _classify_events(ev["lick_samples"], ev["strobe_samples"], ev["strobe_codes"])

    pre_n = int(round(args.pre_s * args.fs))
    post_n = int(round(args.post_s * args.fs))
    tax = np.arange(-pre_n, post_n) / args.fs
    win = pre_n + post_n

    def window_ok(fr: int) -> bool:
        a, b = fr - pre_n, fr + post_n
        if a < 0 or b > T:
            return False
        if csample is not None:  # regime B contiguity guard
            return (csample[b - 1] - csample[a]) / fsd <= (args.pre_s + args.post_s + 1.0)
        return True

    valid = (codes >= 0) & np.array([window_ok(int(f)) for f in lick_frames])
    print(f"[{args.label}] regime {regime}  licks={lick_frames.size} valid_with_window={int(valid.sum())} "
          f"quiet_frames={qinfo['quiet_frames']}", flush=True)

    out = {"time": tax.astype(np.float32), "regions": regions.astype(np.int64)}
    counts = {}
    trials_store = {}
    for code in DISPLAY_ORDER:
        name = POSITION_NAMES[code]
        fr = lick_frames[valid & (codes == code)]
        counts[name] = int(fr.size)
        if fr.size == 0:
            out[f"{name}_mean"] = np.zeros((ncomp, win), np.float32)
            out[f"{name}_sem"] = np.zeros((ncomp, win), np.float32)
            continue
        trials = np.stack([Cn[:, f - pre_n:f + post_n] for f in fr], axis=0)  # (n_licks, ncomp, win)
        out[f"{name}_mean"] = trials.mean(0).astype(np.float32)
        out[f"{name}_sem"] = (trials.std(0) / np.sqrt(fr.size)).astype(np.float32)
        if args.save_trials:
            trials_store[f"{name}_trials"] = trials.astype(np.float32)

    npz = args.output / f"{args.label}_locanmf_lick_aligned.npz"
    np.savez_compressed(npz, **out, **trials_store)

    summary = {
        "label": args.label, "daq_h5": str(args.daq_h5), "locanmf_dir": str(args.locanmf_dir),
        "regime": regime, "offset": offset, "fs": args.fs, "pre_s": args.pre_s, "post_s": args.post_s,
        "n_components": int(ncomp), "T": int(T), "lick_channel": args.lick_channel,
        "detected_licks": int(ev["lick_samples"].size), "valid_licks_with_window": int(valid.sum()),
        "counts_by_position": counts, "quiet": qinfo,
        "normalization": "per-component quiet-frame z-score (subtract quiet mean, divide quiet SD)",
        "saved_trials": bool(args.save_trials),
        "outputs": [npz.name],
    }
    (args.output / f"{args.label}_locanmf_lick_aligned_summary.json").write_text(json.dumps(summary, indent=2))

    _quicklook(args, out, regions, counts)
    print(f"[{args.label}] wrote {npz}", flush=True)
    return 0


def _quicklook(args, out, regions, counts):
    """Overlay the 6 spout-position trial-averaged traces for the most lick-responsive
    component in each orofacial area (post-lick |mean| peak), as a sanity figure."""
    tax = out["time"]
    pre_mask = tax < 0
    post_mask = (tax >= 0) & (tax <= 0.5)
    areas = [(lab, nm) for lab, nm in OROFACIAL.items()] + [(-lab, nm) for lab, nm in OROFACIAL.items()]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), squeeze=False)
    for ax, (lab, nm) in zip(axes.ravel(), areas):
        ax.set_title(f"{nm}{'_L' if lab>0 else '_R'} (reg {lab})", fontsize=9)
        idx = np.where(regions == lab)[0]
        if idx.size == 0:
            ax.text(0.5, 0.5, "no component", ha="center", va="center", transform=ax.transAxes); continue
        # pick the component in this area with the largest post-lick response (pooled positions)
        best, bestresp = None, -1
        for i in idx:
            resp = max(abs(float(out[f"{POSITION_NAMES[c]}_mean"][i][post_mask].mean())) for c in DISPLAY_ORDER)
            if resp > bestresp:
                bestresp, best = resp, i
        for c in DISPLAY_ORDER:
            name = POSITION_NAMES[c]
            if counts.get(name, 0) == 0:
                continue
            m = out[f"{name}_mean"][best]; s = out[f"{name}_sem"][best]
            ax.plot(tax, m, label=f"{name} (n={counts[name]})", lw=1.3)
            ax.fill_between(tax, m - s, m + s, alpha=0.15)
        ax.axvline(0, color="k", lw=0.6, ls="--"); ax.axhline(0, color="grey", lw=0.5)
        ax.set_xlabel("time from lick (s)"); ax.set_ylabel("quiet z-score")
        ax.legend(fontsize=6, ncol=2)
    fig.suptitle(f"{args.label}: lick-triggered LocaNMF traces by spout position (most responsive comp/area)")
    fig.tight_layout()
    png = args.output / f"{args.label}_locanmf_lick_aligned_orofacial.png"
    fig.savefig(png, dpi=130); plt.close(fig)
    print(f"[{args.label}] wrote {png}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--daq-h5", type=Path, required=True)
    p.add_argument("--locanmf-dir", type=Path, required=True)
    p.add_argument("--label", required=True, help="e.g. PS94_0603 (file stem of *_locanmf_C.npy)")
    p.add_argument("--quiet-frame", type=Path, required=True, help="*_quiet_frame.npy (per-corrected-frame mask)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frame-map", type=Path, default=None, help="regime B: cleanpairs frame_map.npz (omit for regime A)")
    p.add_argument("--offset", type=int, default=None, help="regime B chosen_exposure_offset (else --cleanpairs-summary)")
    p.add_argument("--cleanpairs-summary", type=Path, default=None)
    p.add_argument("--fs", type=float, default=31.23)
    p.add_argument("--pre-s", type=float, default=1.0)
    p.add_argument("--post-s", type=float, default=1.5)
    p.add_argument("--save-trials", action="store_true", default=True)
    p.add_argument("--no-save-trials", dest="save_trials", action="store_false")
    p.add_argument("--lick-channel", default="lick_analog")
    p.add_argument("--lick-thresh-upper-v", type=float, default=2.5)
    p.add_argument("--lick-thresh-lower-v", type=float, default=1.0)
    p.add_argument("--lockout-s", type=float, nargs=2, default=(0.001, 0.020))
    p.add_argument("--refractory-s", type=float, default=0.10)
    args = p.parse_args()
    if args.frame_map is not None and args.offset is None and args.cleanpairs_summary is None:
        p.error("regime B needs --offset or --cleanpairs-summary")
    return analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
