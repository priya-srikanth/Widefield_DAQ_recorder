"""Compare cue- and lick-aligned activity maps for 415 nm (isosbestic) vs raw 470 nm
vs hemo-corrected 470 nm, for one example session. All three share the same U_atlas
spatial basis (SVD was on the interleaved 415/470 data):
  415        = U_atlas @ SVT[:, 0::2]      (isosbestic -> hemodynamics/artifact)
  raw 470    = U_atlas @ SVT[:, 1::2]      (neural + hemo)
  corr 470   = U_atlas @ SVTcorr           (neural, hemo regressed out)
Pools across spout positions (all valid trials). Shared color scale across the 3
signals per row so relative magnitude is honest. Reuses framemap_event_maps event logic.
"""
import os, types, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from wfield_local.framemap_event_maps import (
    _load_common, _corrected_frame_samples, _nearest_corrected_frame,
    _load_cue_events, _classify_cues, _load_lick_events, _classify_events,
    _weighted_map, _overlay_regions,
)
NL = r"N:\MICROSCOPE\Priya\Widefield\labcams"
OUT = os.path.join(NL, "channel_comparison"); os.makedirs(OUT, exist_ok=True)
FS = 31.23; CUE_PRE = 2.0; CUE_POST = 2.0; LICK_POST = 0.15
SESS = ("20260608", "PS92_20260608_133759", "PS92_0608")  # example session

date, sess, lab = SESS
mc = Path(rf"{NL}\{date}\{sess}\motion_corrected")
args = types.SimpleNamespace(
    daq_h5=Path(rf"N:\MICROSCOPE\Priya\Widefield\DAQ_recorder_output\{date}\PS92_20260608_133847.h5"),
    wfield_results=mc / "wfield_local_results",
    allen_dir=mc / "wfield_local_results" / "allen_aligned_affine8v1",
    frame_map=next(mc.glob("*cleanpairs_frame_map.npz")),
    cleanpairs_summary=next(mc.glob("*cleanpairs_summary.json")),
    offset=None, fs=FS,
    lick_channel="lick_analog", lick_thresh_upper_v=2.5, lick_thresh_lower_v=1.0,
    lockout_s=(0.001, 0.020), refractory_s=0.10,
)
U, SVTcorr, edges, offset = _load_common(args)
SVT = np.load(args.wfield_results / "SVT.npy", mmap_mode="r")
svt_470 = SVT[:, 1::2]; svt_415 = SVT[:, 0::2]                 # functional_channel=1
SIGNALS = [("415 nm (isosbestic)", svt_415), ("470 nm (raw)", svt_470), ("470 nm (hemo-corrected)", SVTcorr)]
T = SVTcorr.shape[1]

def win_avg(S, frames, a, b):
    acc = np.zeros(S.shape[0])
    for fr in frames: acc += np.asarray(S[:, fr + a:fr + b]).mean(1)
    return acc / len(frames)

def row_plot(axrow, sigmaps, titles, edges, title_prefix):
    lim = max(np.nanpercentile(np.abs(np.concatenate([m.ravel() for m in sigmaps])), 99.0), 1e-6)
    for ax, m, t in zip(axrow, sigmaps, titles):
        ax.set_axis_off(); im = ax.imshow(m, cmap="RdBu_r", vmin=-lim, vmax=lim)
        _overlay_regions(ax, edges); ax.set_title(t, fontsize=12)
    from matplotlib.cm import ScalarMappable
    cb = axrow[-1].figure.colorbar(im, ax=list(axrow), shrink=0.8, pad=0.01)
    cb.set_label(f"{title_prefix} (shared +/-{lim:.4g})", fontsize=11)

# ---- CUE ----
ev = _load_cue_events(args.daq_h5)
csample = _corrected_frame_samples(args.frame_map, ev["pco_samples"], offset); fsd = ev["sample_rate_hz"]
cue_frames = _nearest_corrected_frame(ev["cue_samples"], csample)
cue_codes = _classify_cues(ev["cue_samples"], ev["strobe_samples"], ev["strobe_codes"])
pre_n = int(round(CUE_PRE * FS)); post_n = int(round(CUE_POST * FS))
def cok(ci): a, b = ci - pre_n, ci + post_n; return a >= 0 and b <= T and (csample[b-1]-csample[a])/fsd <= CUE_PRE+CUE_POST+1.0
cvalid = (cue_codes >= 0) & np.array([cok(int(c)) for c in cue_frames])
cf = cue_frames[cvalid]; print(f"cue: pooled {len(cf)} valid trials")
fig, axes = plt.subplots(2, 3, figsize=(14, 9.5), constrained_layout=True)
post = [_weighted_map(U, win_avg(S, cf, 0, post_n).astype(np.float32)) for _, S in SIGNALS]
delta = [_weighted_map(U, (win_avg(S, cf, 0, post_n) - win_avg(S, cf, -pre_n, 0)).astype(np.float32)) for _, S in SIGNALS]
row_plot(axes[0], post, [f"{n}\n{CUE_POST:g}s post-cue" for n, _ in SIGNALS], edges, "post-cue mean")
row_plot(axes[1], delta, [f"{n}\npost - pre" for n, _ in SIGNALS], edges, "post - pre-cue")
fig.suptitle(f"{lab}: cue-aligned maps, pooled ({len(cf)} trials) - 415 vs raw 470 vs corrected 470", fontsize=15)
for ext in ("png", "svg"): fig.savefig(os.path.join(OUT, f"{lab}_cue_415_vs_470_vs_corr.{ext}"), dpi=160)
plt.close(fig)

# ---- LICK ----
evl = _load_lick_events(args.daq_h5, args.lick_channel, args.lick_thresh_upper_v, args.lick_thresh_lower_v, tuple(args.lockout_s), args.refractory_s)
csl = _corrected_frame_samples(args.frame_map, evl["pco_samples"], offset); fsl = evl["sample_rate_hz"]
lick_frames = _nearest_corrected_frame(evl["lick_samples"], csl)
lcodes = _classify_events(evl["lick_samples"], evl["strobe_samples"], evl["strobe_codes"])
lpost = max(1, int(round(LICK_POST * FS)))
def lok(fr): return 0 <= fr and fr + lpost <= T and (csl[fr+lpost-1]-csl[fr])/fsl <= LICK_POST+1.0
lvalid = (lcodes >= 0) & np.array([lok(int(fr)) for fr in lick_frames])
lf = lick_frames[lvalid]; print(f"lick: pooled {len(lf)} valid events")
fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), constrained_layout=True)
lmaps = [_weighted_map(U, win_avg(S, lf, 0, lpost).astype(np.float32)) for _, S in SIGNALS]
row_plot(axes, lmaps, [f"{n}\n{LICK_POST*1000:.0f}ms post-lick" for n, _ in SIGNALS], edges, "post-lick mean")
fig.suptitle(f"{lab}: lick-aligned maps, pooled ({len(lf)} events) - 415 vs raw 470 vs corrected 470", fontsize=15)
for ext in ("png", "svg"): fig.savefig(os.path.join(OUT, f"{lab}_lick_415_vs_470_vs_corr.{ext}"), dpi=160)
plt.close(fig)
print(f"\nwrote cue + lick comparisons -> {OUT}")
