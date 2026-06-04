"""Cue- and lick-aligned spout-position maps for RELABELED (cleanpairs) movies.

The stock plotters (plot_spout_trial_averages / plot_lick_aligned_averages) map a
DAQ event to a corrected-frame index as (nearest pco pulse)//2. That is only valid
for a contiguous raw movie. A relabeled "cleanpairs" movie (--relabel-mode rescue)
is a non-contiguous subset of kept 415/470 pairs, so each corrected frame t maps to
DAQ sample pco_samples[frame_map["original_frame_index_ch0"][t] + offset]; events
map to the nearest such sample, with a contiguity guard so an averaging window never
crosses a dropped/boundary frame.

This module generalizes the one-off _ps92_spout.py / _ps92_lick.py to any session
and writes the SAME output filenames / npz keys as the stock plotters, so the
downstream regime-independent steps (plot_spout_position_contrasts,
plot_lick_position_contrasts, plot_lick_vs_cue_spout_maps) work unchanged.

Imports h5py (via the stock plotter helpers) but NOT wfield, so it is safe to run
in the wfield env as its own process (avoids the wfield+h5py DLL clash).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wfield_local.plot_spout_trial_averages import (
    _load_daq_events as _load_cue_events,
    _classify_cues,
    _weighted_map,
    _region_edges,
    _overlay_regions,
    POSITION_NAMES,
    DISPLAY_ORDER,
)
from wfield_local.plot_lick_aligned_averages import (
    _load_daq_events as _load_lick_events,
    _classify_events,
    _shared_limit,
)


def _corrected_frame_samples(frame_map: Path, pco_samples: np.ndarray, offset: int) -> np.ndarray:
    """DAQ sample for each corrected (paired) frame via the cleanpairs frame map."""
    fm = np.load(frame_map)
    if "original_frame_index_ch0" not in fm.files:
        raise KeyError(
            f"{frame_map} has no 'original_frame_index_ch0' (keys: {fm.files})"
        )
    idx = fm["original_frame_index_ch0"] + offset
    idx = np.clip(idx, 0, len(pco_samples) - 1)
    return pco_samples[idx]


def _nearest_corrected_frame(event_samples: np.ndarray, csample: np.ndarray) -> np.ndarray:
    ins = np.clip(np.searchsorted(csample, event_samples), 1, len(csample) - 1)
    prev = np.abs(event_samples - csample[ins - 1])
    nxt = np.abs(csample[ins] - event_samples)
    return np.where(prev <= nxt, ins - 1, ins).astype(np.int64)


def _offset_from_summary(summary_path: Path) -> int:
    d = json.loads(Path(summary_path).read_text())
    return int(d["chosen_exposure_offset"])


def _load_common(args):
    U = np.load(args.allen_dir / "U_atlas.npy", mmap_mode="r")
    SVTcorr = np.load(args.wfield_results / "SVTcorr.npy", mmap_mode="r")
    atlas = np.load(args.allen_dir / "allen_area_atlas_native_grid.npy")
    edges = _region_edges(atlas)
    offset = args.offset if args.offset is not None else _offset_from_summary(args.cleanpairs_summary)
    return U, SVTcorr, edges, offset


def run_cue(args) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    ev = _load_cue_events(args.daq_h5)
    U, SVTcorr, edges, offset = _load_common(args)
    T = SVTcorr.shape[1]
    csample = _corrected_frame_samples(args.frame_map, ev["pco_samples"], offset)
    fsd = ev["sample_rate_hz"]
    cue_frames = _nearest_corrected_frame(ev["cue_samples"], csample)
    cue_codes = _classify_cues(ev["cue_samples"], ev["strobe_samples"], ev["strobe_codes"])

    pre_n = int(round(args.pre_s * args.fs))
    post_n = int(round(args.post_s * args.fs))

    def contiguous(ci: int) -> bool:
        a, b = ci - pre_n, ci + post_n
        if a < 0 or b > T:
            return False
        return (csample[b - 1] - csample[a]) / fsd <= (args.pre_s + args.post_s + 1.0)

    valid = (cue_codes >= 0) & np.array([contiguous(int(c)) for c in cue_frames])
    print(f"cues={len(cue_frames)} valid={int(valid.sum())}", flush=True)

    maps, counts = {}, {}
    for code in DISPLAY_ORDER:
        tf = cue_frames[valid & (cue_codes == code)]
        name = POSITION_NAMES[code]
        counts[name] = int(len(tf))
        if len(tf) == 0:
            continue
        pre_sum = np.zeros(SVTcorr.shape[0]); post_sum = np.zeros(SVTcorr.shape[0])
        for fr in tf:
            pre_sum += np.asarray(SVTcorr[:, fr - pre_n:fr]).mean(1)
            post_sum += np.asarray(SVTcorr[:, fr:fr + post_n]).mean(1)
        pre_map = _weighted_map(U, (pre_sum / len(tf)).astype(np.float32))
        post_map = _weighted_map(U, (post_sum / len(tf)).astype(np.float32))
        maps[name] = {"pre": pre_map, "post": post_map, "delta": (post_map - pre_map).astype(np.float32)}

    all_pp = np.concatenate([v["pre"].ravel() for v in maps.values()] + [v["post"].ravel() for v in maps.values()])
    activity_lim = max(float(np.nanpercentile(np.abs(all_pp), 99.0)), 1e-6)
    all_d = np.concatenate([v["delta"].ravel() for v in maps.values()])
    delta_lim = max(float(np.nanpercentile(np.abs(all_d), 99.0)), 1e-6)

    fig, axes = plt.subplots(6, 3, figsize=(13, 18), constrained_layout=True)
    for row, code in enumerate(DISPLAY_ORDER):
        name = POSITION_NAMES[code]
        for col, key in enumerate(("pre", "post", "delta")):
            ax = axes[row, col]; ax.set_axis_off()
            if name not in maps:
                ax.set_title(f"{name}: no trials"); continue
            lim = delta_lim if key == "delta" else activity_lim
            im = ax.imshow(maps[name][key], cmap="RdBu_r", vmin=-lim, vmax=lim)
            _overlay_regions(ax, edges)
            lab = {"pre": "1 s pre-cue", "post": "1 s post-cue", "delta": "post - pre"}[key]
            ax.set_title(f"{name} n={counts[name]} | {lab}", fontsize=10)
        fig.colorbar(im, ax=axes[row, :], shrink=0.7, pad=0.01)
    fig.suptitle(f"{args.label} cue averages, hemo-corrected, Allen outlines (frame-map mapping)", fontsize=14)
    png = args.output / f"{args.label}_spout_positions_1s_pre_post_delta_allen_overlay.png"
    fig.savefig(png, dpi=180); plt.close(fig)

    np.savez_compressed(
        args.output / f"{args.label}_spout_positions_1s_pre_post_delta_maps.npz",
        **{f"{name}_{key}": arr for name, vals in maps.items() for key, arr in vals.items()},
    )
    summary = {
        "label": args.label, "daq_h5": str(args.daq_h5), "wfield_results": str(args.wfield_results),
        "allen_dir": str(args.allen_dir), "frame_map": str(args.frame_map), "offset": offset,
        "pre_s": args.pre_s, "post_s": args.post_s, "fs": args.fs,
        "cue_count": int(len(ev["cue_samples"])), "valid_cues_with_windows": int(valid.sum()),
        "counts_by_position": counts, "activity_display_limit": activity_lim, "delta_display_limit": delta_lim,
        "frame_mapping": "relabeled cleanpairs: DAQ cue -> nearest kept corrected frame via frame_map+pco pulses",
    }
    (args.output / f"{args.label}_spout_positions_1s_pre_post_delta_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(counts, indent=2), flush=True)
    print("wrote", png, flush=True)
    return 0


def run_lick(args) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    ev = _load_lick_events(
        args.daq_h5, args.lick_channel, args.lick_thresh_upper_v, args.lick_thresh_lower_v,
        tuple(args.lockout_s), args.refractory_s,
    )
    U, SVTcorr, edges, offset = _load_common(args)
    T = SVTcorr.shape[1]
    csample = _corrected_frame_samples(args.frame_map, ev["pco_samples"], offset)
    fsd = ev["sample_rate_hz"]
    lick_frames = _nearest_corrected_frame(ev["lick_samples"], csample)
    codes = _classify_events(ev["lick_samples"], ev["strobe_samples"], ev["strobe_codes"])

    post_n = max(1, int(round(args.post_s * args.fs)))

    def ok(fr: int) -> bool:
        return 0 <= fr and fr + post_n <= T and (csample[fr + post_n - 1] - csample[fr]) / fsd <= (args.post_s + 1.0)

    valid = (codes >= 0) & np.array([ok(int(fr)) for fr in lick_frames])
    print(f"licks={len(lick_frames)} valid={int(valid.sum())}", flush=True)

    maps, counts = {}, {}
    for code in DISPLAY_ORDER:
        ef = lick_frames[valid & (codes == code)]
        name = POSITION_NAMES[code]; counts[name] = int(ef.size)
        if ef.size == 0:
            continue
        post_sum = np.zeros(SVTcorr.shape[0])
        for fr in ef:
            post_sum += np.asarray(SVTcorr[:, fr:fr + post_n]).mean(1)
        maps[name] = _weighted_map(U, (post_sum / ef.size).astype(np.float32))

    lim = _shared_limit(maps, 99.0)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    im = None
    for ax, code in zip(axes.ravel(), DISPLAY_ORDER):
        name = POSITION_NAMES[code]; ax.set_axis_off()
        if name not in maps:
            ax.set_title(f"{name}: no licks"); continue
        im = ax.imshow(maps[name], cmap="RdBu_r", vmin=-lim, vmax=lim)
        _overlay_regions(ax, edges)
        ax.set_title(f"{name} n={counts[name]} | {args.post_s*1000:.0f} ms post-lick", fontsize=10)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.78, pad=0.01, label=f"Shared scale (+/-{lim:.4g})")
    fig.suptitle(f"{args.label} post-lick hemo-corrected averages by spout position (frame-map mapping)", fontsize=14)
    ms = int(round(args.post_s * 1000))
    png = args.output / f"{args.label}_lick_aligned_{ms}ms_post_by_spout.png"
    fig.savefig(png, dpi=180); plt.close(fig)

    np.savez_compressed(
        args.output / f"{args.label}_lick_aligned_{ms}ms_post_by_spout_maps.npz",
        **{f"{name}_post": arr for name, arr in maps.items()},
    )

    # ---- optional: quiet-period-normalized (post-lick minus quiet baseline) ----
    quietnorm_png = None
    if getattr(args, "quiet_frame", None) is not None and maps:
        from wfield_local.quiet_periods import quiet_baseline_svt
        qmap = _weighted_map(U, quiet_baseline_svt(SVTcorr, np.load(args.quiet_frame)))
        norm_maps = {name: (arr - qmap).astype(np.float32) for name, arr in maps.items()}
        nlim = _shared_limit(norm_maps, 99.0)
        fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
        im = None
        for ax, code in zip(axes.ravel(), DISPLAY_ORDER):
            name = POSITION_NAMES[code]; ax.set_axis_off()
            if name not in norm_maps:
                ax.set_title(f"{name}: no licks"); continue
            im = ax.imshow(norm_maps[name], cmap="RdBu_r", vmin=-nlim, vmax=nlim)
            _overlay_regions(ax, edges)
            ax.set_title(f"{name} n={counts[name]} | {args.post_s*1000:.0f} ms post-lick (quiet-norm)", fontsize=10)
        if im is not None:
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.78, pad=0.01,
                         label=f"post-lick minus quiet baseline (+/-{nlim:.4g})")
        fig.suptitle(f"{args.label} post-lick, normalized to quiet-period baseline", fontsize=14)
        quietnorm_png = args.output / f"{args.label}_lick_aligned_{ms}ms_post_by_spout_quietnorm.png"
        fig.savefig(quietnorm_png, dpi=180); plt.close(fig)
        np.savez_compressed(
            args.output / f"{args.label}_lick_aligned_{ms}ms_post_by_spout_quietnorm_maps.npz",
            **{f"{name}_postnorm": arr for name, arr in norm_maps.items()},
        )

    summary = {
        "label": args.label, "daq_h5": str(args.daq_h5), "wfield_results": str(args.wfield_results),
        "allen_dir": str(args.allen_dir), "frame_map": str(args.frame_map), "offset": offset,
        "post_s": args.post_s, "fs": args.fs, "lick_channel": args.lick_channel,
        "detected_lick_count": int(ev["lick_samples"].size), "valid_licks_with_windows": int(valid.sum()),
        "counts_by_position": counts, "display_limit": lim,
        "quiet_frame": str(args.quiet_frame) if getattr(args, "quiet_frame", None) else None,
        "quietnorm_png": str(quietnorm_png) if quietnorm_png else None,
        "frame_mapping": "relabeled cleanpairs: DAQ lick -> nearest kept corrected frame via frame_map+pco pulses",
    }
    (args.output / f"{args.label}_lick_aligned_{ms}ms_post_by_spout_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(counts, indent=2), flush=True)
    print("wrote", png, flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--what", choices=("cue", "lick"), required=True)
    p.add_argument("--daq-h5", type=Path, required=True)
    p.add_argument("--wfield-results", type=Path, required=True)
    p.add_argument("--allen-dir", type=Path, required=True)
    p.add_argument("--frame-map", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--offset", type=int, default=None, help="cleanpairs chosen_exposure_offset (else read from --cleanpairs-summary)")
    p.add_argument("--cleanpairs-summary", type=Path, default=None)
    p.add_argument("--fs", type=float, default=31.23)
    p.add_argument("--pre-s", type=float, default=1.0)
    p.add_argument("--post-s", type=float, default=None, help="cue default 1.0; lick default 0.150")
    p.add_argument("--quiet-frame", type=Path, default=None,
                   help="(lick only) *_quiet_frame.npy from quiet_periods.py -> also emit a "
                        "quiet-normalized figure/npz (post-lick minus quiet baseline)")
    p.add_argument("--lick-channel", default="lick_analog")
    p.add_argument("--lick-thresh-upper-v", type=float, default=2.5)
    p.add_argument("--lick-thresh-lower-v", type=float, default=1.0)
    p.add_argument("--lockout-s", type=float, nargs=2, default=(0.001, 0.020))
    p.add_argument("--refractory-s", type=float, default=0.10)
    args = p.parse_args()
    if args.offset is None and args.cleanpairs_summary is None:
        p.error("need --offset or --cleanpairs-summary")
    if args.post_s is None:
        args.post_s = 1.0 if args.what == "cue" else 0.150
    return run_cue(args) if args.what == "cue" else run_lick(args)


if __name__ == "__main__":
    raise SystemExit(main())
