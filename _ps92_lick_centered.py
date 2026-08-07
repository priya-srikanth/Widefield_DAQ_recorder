"""PS92 lick maps with a CENTERED 150 ms window (-75..+75 ms), for comparison
with the post-lick (0..150 ms) version. See analysis note: post is recommended
for GCaMP (indicator lag), centered shown only for visual comparison.
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from wfield_local.plot_lick_aligned_averages import (
    _load_daq_events, _classify_events, _weighted_map, _region_edges, _overlay_regions,
    _shared_limit, POSITION_NAMES, DISPLAY_ORDER,
)

BASE = r"E:\labcams_data\20260602\PS92\PS92_20260602_151820\illuminated_rescue"
RES = os.path.join(BASE, "motion_corrected", "wfield_local_results")
ALLEN = os.path.join(RES, "allen_aligned_v2")
DAQ = r"E:\DAQ_recorder_output\PS92_20260602_152607.h5"
FRAMEMAP = os.path.join(BASE, "pco_edge_run001_00000000_2_487_480_uint16_daq_led_cleanpairs_frame_map.npz")
OUT = os.path.join(BASE, "motion_corrected", "lick_aligned_v2")
FS = 31.23
HALF_S = 0.075   # +/- 75 ms = 150 ms centered
OFFSET = 1


def main():
    ev = _load_daq_events(DAQ, "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    U = np.load(os.path.join(ALLEN, "U_atlas.npy"), mmap_mode="r")
    SVTcorr = np.load(os.path.join(RES, "SVTcorr.npy"), mmap_mode="r")
    atlas = np.load(os.path.join(ALLEN, "allen_area_atlas_native_grid.npy"))
    edges = _region_edges(atlas); T = SVTcorr.shape[1]
    fm = np.load(FRAMEMAP)
    csample = ev["pco_samples"][fm["original_frame_index_ch0"] + OFFSET]; fsd = ev["sample_rate_hz"]
    ins = np.clip(np.searchsorted(csample, ev["lick_samples"]), 1, len(csample) - 1)
    prev = np.abs(ev["lick_samples"] - csample[ins - 1]); nxt = np.abs(csample[ins] - ev["lick_samples"])
    lf = np.where(prev <= nxt, ins - 1, ins).astype(np.int64)
    codes = _classify_events(ev["lick_samples"], ev["strobe_samples"], ev["strobe_codes"])
    half = max(1, int(round(HALF_S * FS)))   # frames each side

    def ok(fr):
        a, b = fr - half, fr + half + 1
        return a >= 0 and b <= T and (csample[b - 1] - csample[a]) / fsd <= (2 * HALF_S + 1.0)
    valid = (codes >= 0) & np.array([ok(int(fr)) for fr in lf])
    maps, counts = {}, {}
    for code in DISPLAY_ORDER:
        ef = lf[valid & (codes == code)]; name = POSITION_NAMES[code]; counts[name] = int(ef.size)
        if ef.size == 0:
            continue
        acc = np.zeros(SVTcorr.shape[0])
        for fr in ef:
            acc += np.asarray(SVTcorr[:, fr - half:fr + half + 1]).mean(1)
        maps[name] = _weighted_map(U, (acc / ef.size).astype(np.float32))
    lim = _shared_limit(maps, 99.0)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True); im = None
    for ax, code in zip(axes.ravel(), DISPLAY_ORDER):
        name = POSITION_NAMES[code]; ax.set_axis_off()
        if name not in maps:
            ax.set_title(f"{name}: no licks"); continue
        im = ax.imshow(maps[name], cmap="RdBu_r", vmin=-lim, vmax=lim); _overlay_regions(ax, edges)
        ax.set_title(f"{name} n={counts[name]} | +/-75 ms (centered)", fontsize=10)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.78, pad=0.01, label=f"+/-{lim:.4g}")
    fig.suptitle("PS92_v2 (RESCUED) CENTERED +/-75 ms lick averages (comparison; post-lick recommended)", fontsize=13)
    png = os.path.join(OUT, "PS92_v2_lick_aligned_150ms_CENTERED_by_spout.png")
    fig.savefig(png, dpi=180); plt.close(fig)
    json.dump({"window": "centered +/-75ms", "counts": counts, "note": "for comparison vs post-lick; post recommended for GCaMP lag"},
              open(png.replace(".png", "_summary.json"), "w"), indent=2)
    print("valid", int(valid.sum()), "wrote", png)


if __name__ == "__main__":
    main()
