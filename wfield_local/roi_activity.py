"""ROI-based (Allen-area) activity extraction + optional event-aligned responses.

A lightweight CPU baseline alongside LocaNMF: average the Allen-aligned U over each
atlas region to get a region x time trace, then optionally align to cue/lick events
by spout position. Runs in the wfield CPU env (numpy + h5py; no torch/locanmf/GPU).

The region-averaged trace is cheap because averaging commutes with U @ SVTcorr:
    trace_r(t) = mean_{pixels in r}( U_atlas @ SVTcorr )[pixel, t]
               = ( mean of U_atlas over region r ) @ SVTcorr
so we never reconstruct the pixel movie. Units match the maps: hemo-corrected ΔF/F
relative to the session mean (high-pass filtered).

This is the simple "ROI signal" (one trace per area); LocaNMF gives a denoised,
region-anchored, multi-component alternative (run_locanmf.py). Use this as a fast,
robust baseline and cross-check.

Inputs (allen_aligned_* dir + its results SVTcorr):
  U_atlas.npy, allen_area_atlas_native_grid.npy, allen_brain_mask_native_grid.npy,
  allen_area_names.json (optional), SVTcorr.npy

Examples:
  # region traces only
  python -m wfield_local.roi_activity --allen-dir <...\allen_aligned_affine8v1> \
      --label PS94_0603 --output <...\roi_activity_affine8v1>
  # + cue/lick event-aligned per-region responses by spout position (regime B)
  python -m wfield_local.roi_activity --allen-dir <...> --label PS94_0603 --output <...> \
      --daq-h5 <session.h5> --what both \
      --frame-map <...\*_cleanpairs_frame_map.npz> --cleanpairs-summary <...\*_cleanpairs_summary.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

POSITION_NAMES = {0: "close_center", 1: "close_L", 2: "close_R", 3: "far_center", 4: "far_L", 5: "far_R"}
DISPLAY_ORDER = [1, 0, 2, 4, 3, 5]


def _build_regions(atlas, brain_mask, names_map):
    """Return [(key, label, flat_pixel_index)] per atlas region.

    The wfield Allen atlas is already lateralized: each signed label is one
    hemisphere of one area, and allen_area_names.json names include _left/_right
    (e.g. label 3 = 'MOp_left', -3 = 'MOp_right'). So one region per signed label.
    """
    valid = (brain_mask & np.isfinite(atlas) & (atlas != 0)).ravel()
    flat = atlas.ravel()
    out = []
    for lab in np.unique(flat[valid]):
        lab = int(lab)
        idx = np.flatnonzero((flat == lab) & valid)
        if idx.size:
            out.append((names_map.get(lab, f"region_{lab}"), lab, idx))
    return out


def _load_events(args, what):
    if what == "cue":
        from wfield_local.plot_spout_trial_averages import _load_daq_events, _classify_cues
        ev = _load_daq_events(args.daq_h5)
        codes = _classify_cues(ev["cue_samples"], ev["strobe_samples"], ev["strobe_codes"])
        return ev, ev["cue_samples"], codes
    from wfield_local.plot_lick_aligned_averages import _load_daq_events, _classify_events
    ev = _load_daq_events(args.daq_h5, args.lick_channel, args.lick_thresh_upper_v,
                          args.lick_thresh_lower_v, tuple(args.lockout_s), args.refractory_s)
    codes = _classify_events(ev["lick_samples"], ev["strobe_samples"], ev["strobe_codes"])
    return ev, ev["lick_samples"], codes


def _event_frames(args, ev, samples, T):
    """Map DAQ event samples -> corrected-frame indices (regime A raw//2, or B frame-map)."""
    if args.frame_map is None:                              # regime A
        from wfield_local.plot_spout_trial_averages import _event_frame_indices_from_pco
        return _event_frame_indices_from_pco(samples, ev["pco_samples"]) // 2
    from wfield_local.framemap_event_maps import _corrected_frame_samples, _nearest_corrected_frame, _offset_from_summary
    offset = args.offset if args.offset is not None else _offset_from_summary(args.cleanpairs_summary)
    csample = _corrected_frame_samples(args.frame_map, ev["pco_samples"], offset)
    return _nearest_corrected_frame(samples, csample)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allen-dir", type=Path, required=True)
    ap.add_argument("--svt", type=Path, default=None)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--fs", type=float, default=31.23)
    ap.add_argument("--overview-regions", default="MOp_left,MOp_right,MOs_left,MOs_right",
                    help="comma-separated region keys to plot (traces + locations) in the overview")
    # optional event alignment
    ap.add_argument("--daq-h5", type=Path, default=None)
    ap.add_argument("--quiet-frame", type=Path, default=None,
                    help="*_quiet_frame.npy from quiet_periods.py -> also emit quiet-normalized "
                         "per-region position maps (post minus quiet-period baseline)")
    ap.add_argument("--what", choices=("cue", "lick", "both"), default="both")
    ap.add_argument("--pre-s", type=float, default=1.0)
    ap.add_argument("--cue-post-s", type=float, default=1.0)
    ap.add_argument("--lick-post-s", type=float, default=0.150)
    ap.add_argument("--frame-map", type=Path, default=None)          # regime B
    ap.add_argument("--cleanpairs-summary", type=Path, default=None)
    ap.add_argument("--offset", type=int, default=None)
    ap.add_argument("--lick-channel", default="lick_analog")
    ap.add_argument("--lick-thresh-upper-v", type=float, default=2.5)
    ap.add_argument("--lick-thresh-lower-v", type=float, default=1.0)
    ap.add_argument("--lockout-s", type=float, nargs=2, default=(0.001, 0.020))
    ap.add_argument("--refractory-s", type=float, default=0.10)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ad = args.allen_dir

    svt_path = args.svt
    if svt_path is None:
        for c in (ad.parent / "SVTcorr.npy", ad.parent.parent / "SVTcorr.npy"):
            if c.exists():
                svt_path = c; break
    if svt_path is None or not Path(svt_path).exists():
        raise FileNotFoundError("Could not find SVTcorr.npy; pass --svt.")

    U = np.load(ad / "U_atlas.npy")                                   # (H, W, K)
    atlas = np.load(ad / "allen_area_atlas_native_grid.npy")
    brain_mask = np.load(ad / "allen_brain_mask_native_grid.npy").astype(bool)
    SVT = np.load(svt_path)                                           # (K, T)
    names_map = {}
    npath = ad / "allen_area_names.json"
    if npath.exists():
        try:
            nm = json.loads(npath.read_text())
            if isinstance(nm, dict):
                names_map = {int(k): str(v) for k, v in nm.items()}
            elif isinstance(nm, list) and nm and isinstance(nm[0], (list, tuple)) and len(nm[0]) == 2:
                names_map = {int(p[0]): str(p[1]) for p in nm}     # [[label, name], ...]
            elif isinstance(nm, list):
                names_map = {i + 1: str(n) for i, n in enumerate(nm)}
        except Exception:
            pass

    H, W, K = U.shape
    regions = _build_regions(atlas, brain_mask, names_map)
    Ubar = np.stack([U.reshape(-1, K)[idx].mean(0) for _, _, idx in regions])  # (R, K)
    traces = (Ubar @ SVT).astype(np.float32)                          # (R, T)
    keys = [k for k, _, _ in regions]
    qbase = None  # per-region quiet-period baseline (R,)
    if args.quiet_frame is not None:
        qf = np.load(args.quiet_frame).astype(bool)
        T = traces.shape[1]
        qf = qf[:T] if qf.size >= T else np.pad(qf, (0, T - qf.size))
        qidx = np.flatnonzero(qf)
        if qidx.size:
            qbase = traces[:, qidx].mean(1)
            print(f"[{args.label}] quiet baseline from {qidx.size} quiet frames", flush=True)
        else:
            print(f"[{args.label}] WARN: no quiet frames in mask; skipping quiet-norm", flush=True)
    print(f"[{args.label}] {len(keys)} regions x T={traces.shape[1]}", flush=True)

    np.save(args.output / f"{args.label}_roi_traces.npy", traces)
    meta = {"label": args.label, "allen_dir": str(ad), "svt": str(svt_path), "fs": args.fs,
            "n_regions": len(keys),
            "regions": [{"key": k, "label": int(l), "n_pixels": int(idx.size)}
                        for (k, l, idx) in regions]}
    (args.output / f"{args.label}_roi_traces_meta.json").write_text(json.dumps(meta, indent=2))

    # ---- optional event-aligned per-region responses by spout position ----
    if args.daq_h5 is not None:
        T = traces.shape[1]
        whats = ["cue", "lick"] if args.what == "both" else [args.what]
        for what in whats:
            ev, samples, codes = _load_events(args, what)
            frames = _event_frames(args, ev, samples, T)
            pre_n = int(round(args.pre_s * args.fs))
            post_n = max(1, int(round((args.cue_post_s if what == "cue" else args.lick_post_s) * args.fs)))
            pos_keys = [POSITION_NAMES[c] for c in DISPLAY_ORDER]
            post = {p: None for p in pos_keys}; pre = {p: None for p in pos_keys}; counts = {}
            for code in DISPLAY_ORDER:
                pname = POSITION_NAMES[code]
                fr = frames[(codes == code)]
                fr = fr[(fr - pre_n >= 0) & (fr + post_n <= T)]
                counts[pname] = int(fr.size)
                if fr.size == 0:
                    continue
                po = np.stack([traces[:, f:f + post_n].mean(1) for f in fr]).mean(0)
                post[pname] = po
                if what == "cue":
                    pr = np.stack([traces[:, f - pre_n:f].mean(1) for f in fr]).mean(0)
                    pre[pname] = pr
            # assemble matrices (regions x positions)
            R = len(keys)
            postM = np.full((R, len(pos_keys)), np.nan, np.float32)
            deltaM = np.full((R, len(pos_keys)), np.nan, np.float32)
            for j, p in enumerate(pos_keys):
                if post[p] is not None:
                    postM[:, j] = post[p]
                    if what == "cue" and pre[p] is not None:
                        deltaM[:, j] = post[p] - pre[p]
            np.savez_compressed(args.output / f"{args.label}_{what}_roi_by_position.npz",
                                keys=np.array(keys), positions=np.array(pos_keys),
                                post=postM, delta=deltaM)
            # heatmap (regions x positions); cue shows delta, lick shows post
            M = deltaM if what == "cue" else postM
            lim = np.nanpercentile(np.abs(M), 99) or 1e-6
            fig, axx = plt.subplots(figsize=(max(6, 0.5 * len(pos_keys) + 4), max(6, 0.16 * R)))
            im = axx.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
            axx.set_xticks(range(len(pos_keys))); axx.set_xticklabels(pos_keys, rotation=45, ha="right")
            axx.set_yticks(range(R)); axx.set_yticklabels(keys, fontsize=5)
            axx.set_title(f"{args.label} {what} per-region "
                          f"{'(post-pre delta)' if what=='cue' else f'({int(args.lick_post_s*1000)}ms post)'}")
            fig.colorbar(im, ax=axx, shrink=0.6, label="ΔF/F")
            fig.tight_layout(); fig.savefig(args.output / f"{args.label}_{what}_roi_by_position.png", dpi=130)
            plt.close(fig)

            # quiet-period-normalized: post minus per-region quiet baseline
            if qbase is not None:
                postN = postM - qbase[:, None]
                np.savez_compressed(args.output / f"{args.label}_{what}_roi_by_position_quietnorm.npz",
                                    keys=np.array(keys), positions=np.array(pos_keys), post=postN)
                nlim = np.nanpercentile(np.abs(postN), 99) or 1e-6
                fig, axx = plt.subplots(figsize=(max(6, 0.5 * len(pos_keys) + 4), max(6, 0.16 * R)))
                imn = axx.imshow(postN, aspect="auto", cmap="RdBu_r", vmin=-nlim, vmax=nlim)
                axx.set_xticks(range(len(pos_keys))); axx.set_xticklabels(pos_keys, rotation=45, ha="right")
                axx.set_yticks(range(R)); axx.set_yticklabels(keys, fontsize=5)
                axx.set_title(f"{args.label} {what} per-region post minus quiet baseline")
                fig.colorbar(imn, ax=axx, shrink=0.6, label="ΔF/F vs quiet")
                fig.tight_layout(); fig.savefig(args.output / f"{args.label}_{what}_roi_by_position_quietnorm.png", dpi=130)
                plt.close(fig)

            (args.output / f"{args.label}_{what}_roi_by_position_summary.json").write_text(json.dumps({
                "label": args.label, "what": what, "counts_by_position": counts,
                "pre_s": args.pre_s, "post_s": args.cue_post_s if what == "cue" else args.lick_post_s,
                "regime": "B(frame-map)" if args.frame_map else "A(raw//2)",
                "quiet_frame": str(args.quiet_frame) if args.quiet_frame else None,
                "quietnorm": qbase is not None,
            }, indent=2))
            print(f"[{args.label}] {what}: counts {counts}", flush=True)

    def _centroid(idx):
        return float((idx // W).mean()), float((idx % W).mean())

    # ---- reference: labeled Allen atlas (named regions at their centroids) ----
    figR, axR = plt.subplots(figsize=(9.5, 9.5))
    axR.imshow(np.where(brain_mask, atlas, np.nan), cmap="tab20", interpolation="nearest")
    for (k, _lab, idx) in regions:
        cy, cx = _centroid(idx)
        axR.text(cx, cy, k, fontsize=4.5, ha="center", va="center", color="black")
    axR.set_axis_off(); axR.set_title(f"{args.label} Allen reference regions ({len(keys)})")
    figR.tight_layout(); figR.savefig(args.output / f"{args.label}_allen_reference_labeled.png", dpi=170)
    plt.close(figR)

    # ---- ROI overview: selected regions' LOCATIONS + traces ----
    want = [s.strip() for s in args.overview_regions.split(",") if s.strip()]
    key2row = {k: i for i, (k, _, _) in enumerate(regions)}
    sel = [(k, idx) for (k, _l, idx) in regions if k in want]
    missing = [w for w in want if w not in key2row]
    if missing:
        print(f"[{args.label}] overview regions not found: {missing}", flush=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(sel), 1)))
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].imshow(np.where(brain_mask, atlas, np.nan), cmap="Greys", alpha=0.25, interpolation="nearest")
    overlay = np.zeros((H, W, 4), np.float32)
    for (k, idx), col in zip(sel, colors):
        overlay[idx // W, idx % W, :] = col
        cy, cx = _centroid(idx)
        ax[0].text(cx, cy, k, fontsize=7, ha="center", va="center", weight="bold", color="black")
    ax[0].imshow(overlay, interpolation="nearest"); ax[0].set_axis_off()
    ax[0].set_title("selected ROI locations")
    tt = np.arange(min(traces.shape[1], int(60 * args.fs))) / args.fs
    for i, ((k, idx), col) in enumerate(zip(sel, colors)):
        ax[1].plot(tt, traces[key2row[k], :len(tt)] + i * 0.04, lw=0.8, color=col, label=k)
    ax[1].set_xlabel("s"); ax[1].set_ylabel("ΔF/F (offset per ROI)"); ax[1].legend(fontsize=8)
    ax[1].set_title("selected ROI traces (first 60 s)")
    fig.suptitle(f"{args.label} ROI overview ({', '.join(k for k, _ in sel)})")
    fig.tight_layout(); fig.savefig(args.output / f"{args.label}_roi_overview.png", dpi=140); plt.close(fig)
    print(f"[{args.label}] wrote ROI traces + reference + overview to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
