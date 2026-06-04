"""Spout-position cue averages for the PS92 RESCUED recording (one-off).

Reuses helpers from wfield_local.plot_spout_trial_averages but maps DAQ events to
corrected-frame indices via the rescue frame_map (the rescued movie is a
non-contiguous subset of the original recording, so the standard raw//2 mapping
does not apply). Uses the locally-recorrected SVTcorr (functional channel fix).
Output format/naming matches the PS95 v6 reference.
"""
import os, json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from wfield_local.plot_spout_trial_averages import (
    _load_daq_events, _classify_cues, _weighted_map, _region_edges, _overlay_regions,
    POSITION_NAMES, DISPLAY_ORDER,
)

VERSION = sys.argv[1] if len(sys.argv) > 1 else "v2"
BASE = r"E:\labcams_data\20260602\PS92\PS92_20260602_151820\illuminated_rescue"
RES = os.path.join(BASE, "motion_corrected", "wfield_local_results")
ALLEN = os.path.join(RES, f"allen_aligned_{VERSION}")
DAQ = r"E:\DAQ_recorder_output\PS92_20260602_152607.h5"
FRAMEMAP = os.path.join(BASE, "pco_edge_run001_00000000_2_487_480_uint16_daq_led_cleanpairs_frame_map.npz")
OUT = os.path.join(BASE, "motion_corrected", f"spout_trial_averages_allen_{VERSION}")
LABEL = f"PS92_{VERSION}"
FS = 31.23
PRE_S = POST_S = 1.0
OFFSET = 1  # frame_map -> DAQ pco edge offset (cleanpairs chosen_exposure_offset)


def corrected_frame_for_events(event_samples, pco_samples, csample):
    ins = np.clip(np.searchsorted(csample, event_samples), 1, len(csample) - 1)
    prev = np.abs(event_samples - csample[ins - 1]); nxt = np.abs(csample[ins] - event_samples)
    return np.where(prev <= nxt, ins - 1, ins).astype(np.int64)


def main():
    os.makedirs(OUT, exist_ok=True)
    ev = _load_daq_events(DAQ)
    U = np.load(os.path.join(ALLEN, "U_atlas.npy"), mmap_mode="r")
    SVTcorr = np.load(os.path.join(RES, "SVTcorr.npy"), mmap_mode="r")
    atlas = np.load(os.path.join(ALLEN, "allen_area_atlas_native_grid.npy"))
    edges = _region_edges(atlas)
    T = SVTcorr.shape[1]

    fm = np.load(FRAMEMAP)
    csample = ev["pco_samples"][fm["original_frame_index_ch0"] + OFFSET]   # DAQ sample of each corrected frame
    fsd = ev["sample_rate_hz"]
    cue_frames = corrected_frame_for_events(ev["cue_samples"], ev["pco_samples"], csample)
    cue_codes = _classify_cues(ev["cue_samples"], ev["strobe_samples"], ev["strobe_codes"])

    pre_n = int(round(PRE_S * FS)); post_n = int(round(POST_S * FS))
    # contiguity guard: window must stay within continuous kept-frame time (no trial-boundary crossing)
    def contiguous(ci):
        a = ci - pre_n; b = ci + post_n
        if a < 0 or b > T:
            return False
        return (csample[b - 1] - csample[a]) / fsd <= (PRE_S + POST_S + 1.0)
    valid = (cue_codes >= 0) & np.array([contiguous(int(ci)) for ci in cue_frames])
    print(f"cues={len(cue_frames)} valid(with pre/post within-trial)={int(valid.sum())}", flush=True)

    maps, counts = {}, {}
    for code in DISPLAY_ORDER:
        tf = cue_frames[valid & (cue_codes == code)]
        counts[POSITION_NAMES[code]] = int(len(tf))
        if len(tf) == 0:
            continue
        pre_sum = np.zeros(SVTcorr.shape[0]); post_sum = np.zeros(SVTcorr.shape[0])
        for fr in tf:
            pre_sum += np.asarray(SVTcorr[:, fr - pre_n:fr]).mean(1)
            post_sum += np.asarray(SVTcorr[:, fr:fr + post_n]).mean(1)
        pre_mean = (pre_sum / len(tf)).astype(np.float32)
        post_mean = (post_sum / len(tf)).astype(np.float32)
        pre_map = _weighted_map(U, pre_mean); post_map = _weighted_map(U, post_mean)
        maps[POSITION_NAMES[code]] = {"pre": pre_map, "post": post_map, "delta": (post_map - pre_map).astype(np.float32)}

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
    fig.suptitle(f"{LABEL} (RESCUED) cue averages, hemo-corrected (func ch fixed), Allen outlines", fontsize=14)
    png = os.path.join(OUT, f"{LABEL}_spout_positions_1s_pre_post_delta_allen_overlay.png")
    fig.savefig(png, dpi=180); plt.close(fig)

    npz = {f"{name}_{key}": arr for name, vals in maps.items() for key, arr in vals.items()}
    np.savez_compressed(os.path.join(OUT, f"{LABEL}_spout_positions_1s_pre_post_delta_maps.npz"), **npz)

    summary = {
        "label": LABEL, "rescued": True, "daq_h5": DAQ, "wfield_results": RES, "allen_dir": ALLEN,
        "frame_map": FRAMEMAP, "frame_mapping": "rescued: DAQ cue -> nearest kept corrected frame via frame_map+pco pulses",
        "pre_s": PRE_S, "post_s": POST_S, "fs": FS,
        "cue_count": int(len(ev["cue_samples"])), "valid_cues_with_windows": int(valid.sum()),
        "counts_by_position": counts, "activity_display_limit": activity_lim, "delta_display_limit": delta_lim,
        "note": "Functional-channel swap fixed locally; SVTcorr recomputed with functional_channel=0.",
    }
    open(os.path.join(OUT, f"{LABEL}_spout_positions_1s_pre_post_delta_summary.json"), "w").write(json.dumps(summary, indent=2))
    print(json.dumps(counts, indent=2), flush=True)
    print("wrote", png, flush=True)


if __name__ == "__main__":
    main()
