"""Post-lick spout-position averages for the PS92 RESCUED recording (one-off).

Reuses helpers from wfield_local.plot_lick_aligned_averages but maps DAQ licks to
corrected-frame indices via the rescue frame_map. Uses locally-recorrected SVTcorr.
Output format/naming matches the PS95 v6 reference.
"""
import os, json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from wfield_local.plot_lick_aligned_averages import (
    _load_daq_events, _classify_events, _weighted_map, _region_edges, _overlay_regions,
    _shared_limit, POSITION_NAMES, DISPLAY_ORDER,
)

VERSION = sys.argv[1] if len(sys.argv) > 1 else "v2"
BASE = r"E:\labcams_data\20260602\PS92\PS92_20260602_151820\illuminated_rescue"
RES = os.path.join(BASE, "motion_corrected", "wfield_local_results")
ALLEN = os.path.join(RES, f"allen_aligned_{VERSION}")
DAQ = r"E:\DAQ_recorder_output\PS92_20260602_152607.h5"
FRAMEMAP = os.path.join(BASE, "pco_edge_run001_00000000_2_487_480_uint16_daq_led_cleanpairs_frame_map.npz")
OUT = os.path.join(BASE, "motion_corrected", f"lick_aligned_{VERSION}")
LABEL = f"PS92_{VERSION}"
FS = 31.23
POST_S = 0.150
OFFSET = 1


def main():
    os.makedirs(OUT, exist_ok=True)
    ev = _load_daq_events(DAQ, "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    U = np.load(os.path.join(ALLEN, "U_atlas.npy"), mmap_mode="r")
    SVTcorr = np.load(os.path.join(RES, "SVTcorr.npy"), mmap_mode="r")
    atlas = np.load(os.path.join(ALLEN, "allen_area_atlas_native_grid.npy"))
    edges = _region_edges(atlas)
    T = SVTcorr.shape[1]

    fm = np.load(FRAMEMAP)
    csample = ev["pco_samples"][fm["original_frame_index_ch0"] + OFFSET]
    fsd = ev["sample_rate_hz"]
    ins = np.clip(np.searchsorted(csample, ev["lick_samples"]), 1, len(csample) - 1)
    prev = np.abs(ev["lick_samples"] - csample[ins - 1]); nxt = np.abs(csample[ins] - ev["lick_samples"])
    lick_frames = np.where(prev <= nxt, ins - 1, ins).astype(np.int64)
    codes = _classify_events(ev["lick_samples"], ev["strobe_samples"], ev["strobe_codes"])

    post_n = max(1, int(round(POST_S * FS)))
    def ok(fr):
        return 0 <= fr and fr + post_n <= T and (csample[fr + post_n - 1] - csample[fr]) / fsd <= (POST_S + 1.0)
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
        ax.set_title(f"{name} n={counts[name]} | {POST_S*1000:.0f} ms post-lick", fontsize=10)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.78, pad=0.01, label=f"Shared scale (+/-{lim:.4g})")
    fig.suptitle(f"{LABEL} (RESCUED) post-lick hemo-corrected averages by spout position", fontsize=14)
    png = os.path.join(OUT, f"{LABEL}_lick_aligned_{int(round(POST_S*1000))}ms_post_by_spout.png")
    fig.savefig(png, dpi=180); plt.close(fig)

    np.savez_compressed(os.path.join(OUT, f"{LABEL}_lick_aligned_{int(round(POST_S*1000))}ms_post_by_spout_maps.npz"),
                        **{f"{name}_post": arr for name, arr in maps.items()})
    summary = {"label": LABEL, "rescued": True, "daq_h5": DAQ, "post_s": POST_S, "fs": FS,
               "detected_lick_count": int(ev["lick_samples"].size), "valid_licks_with_windows": int(valid.sum()),
               "counts_by_position": counts, "display_limit": lim, "lick_detection": ev["lick_detection"],
               "frame_mapping": "rescued: DAQ lick -> nearest kept corrected frame via frame_map+pco pulses",
               "note": "Functional-channel swap fixed locally; SVTcorr recomputed with functional_channel=0."}
    open(os.path.join(OUT, f"{LABEL}_lick_aligned_{int(round(POST_S*1000))}ms_post_by_spout_summary.json"), "w").write(json.dumps(summary, indent=2))
    print(json.dumps(counts, indent=2), flush=True)
    print("wrote", png, flush=True)


if __name__ == "__main__":
    main()
