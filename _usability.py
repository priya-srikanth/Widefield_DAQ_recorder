import numpy as np, h5py, os, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

base = r"E:\labcams_data\20260602\PS92\PS92_20260602_151820\illuminated_rescue"
mc = os.path.join(base, "motion_corrected")
res = os.path.join(mc, "wfield_local_results")
daq_path = r"E:\DAQ_recorder_output\PS92_20260602_152607.h5"

# --- 1. motion-correction shifts ---
sh = np.load(os.path.join(mc, "motion_correction_shifts.npy"))
print("=== motion shifts ===", sh.shape, sh.dtype)
if sh.dtype.names:
    comps = {name: np.asarray(sh[name], dtype=np.float64).ravel() for name in sh.dtype.names}
else:
    comps = {f"axis{i}": sh[..., i].astype(np.float64).ravel() for i in range(sh.shape[-1])}
for name, s in comps.items():
    print(f"  {name}: min={s.min():.1f} max={s.max():.1f} mean={s.mean():.2f} "
          f"|>10|={int(np.sum(np.abs(s)>10))} |>30|={int(np.sum(np.abs(s)>30))} |>50|={int(np.sum(np.abs(s)>50))} of {len(s)}")

# --- 2. frame-map pair validity ---
fm = np.load(os.path.join(base, "pco_edge_run001_00000000_2_487_480_uint16_daq_led_cleanpairs_frame_map.npz"))
print("=== frame_map ===", list(fm.keys()))
c0 = fm["original_frame_index_ch0"]; c1 = fm["original_frame_index_ch1"]
gap = c1 - c0
print(f"  pairs={len(c0)}  within-pair gap: ==1:{int(np.sum(gap==1))} ==2:{int(np.sum(gap==2))} >2:{int(np.sum(gap>2))} max={int(gap.max())}")
between = c0[1:] - c1[:-1]
print(f"  between consecutive pairs: >2 (likely trial boundaries):{int(np.sum(between>2))} max={int(between.max())}")
print(f"  cross-trial suspicious pairs (within-pair gap>2): {int(np.sum(gap>2))} ({100*np.mean(gap>2):.2f}%)")

# --- 3. SVD reconstruction sanity ---
U = np.load(os.path.join(res, "U.npy"), mmap_mode="r")
SVTcorr = np.load(os.path.join(res, "SVTcorr.npy"))
favg = np.load(os.path.join(res, "frames_average.npy"))
print("=== SVD ===  U", U.shape, "SVTcorr", SVTcorr.shape, "favg", favg.shape)
varc = SVTcorr.var(axis=1)
Uarr = np.asarray(U, dtype=np.float32)
varmap = np.tensordot(Uarr**2, varc, axes=([2], [0]))
print(f"  std-map finite={np.isfinite(varmap).all()} range=[{np.sqrt(np.nanmin(varmap)):.3g},{np.sqrt(np.nanmax(varmap)):.3g}]")
print(f"  SVTcorr per-frame value range mean abs={np.mean(np.abs(SVTcorr)):.3g}")

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
ax[0].imshow(favg[0], cmap="gray"); ax[0].set_title("mean 415")
ax[1].imshow(favg[1], cmap="gray"); ax[1].set_title("mean 470")
im = ax[2].imshow(np.sqrt(np.clip(varmap, 0, None)), cmap="magma"); ax[2].set_title("SVTcorr temporal-std map")
for a in ax: a.axis("off")
plt.colorbar(im, ax=ax[2], fraction=0.046)
plt.tight_layout(); plt.savefig(os.path.join(base, "_usability_check.png"), dpi=110)
print("  saved _usability_check.png")

# --- 4. DAQ event alignment via frame_map + pco pulses ---
with h5py.File(daq_path, "r") as f:
    fs = float(f.attrs["sample_rate_hz"])
    di = [s.decode() for s in f["digital/channel_names"][:]]
    an = [s.decode() for s in f["analog/channel_names"][:]]
    bits = np.unpackbits(f["digital/packed_samples"][:, 0][:, None], axis=1, bitorder="little")
    raw = f["analog/samples_int16"][:].astype(np.float32)
    volts = raw * f["analog/int16_scale_volts_per_count"][:] + f["analog/int16_offset_volts"][:]

def rises(sig, thr=0.5):
    b = (sig > thr).astype(np.int8); return np.flatnonzero(np.diff(b) == 1) + 1

pco = rises(bits[:, di.index("pco_exposure")].astype(float))
cue = rises(bits[:, di.index("cue")].astype(float))
print("=== DAQ ===", f"pco_pulses={len(pco)} cues={len(cue)} fs={fs}")

offset = 1  # from cleanpairs_summary chosen_exposure_offset
valid = (c0 + offset) < len(pco)
csample = pco[c0[valid] + offset]  # DAQ sample index of each kept corrected (415) frame
print(f"  corrected frames mapped to DAQ pco: {int(valid.sum())}/{len(c0)}")

# (a) STANDARD pipeline mapping (raw//2, contiguous assumption): which SVTcorr index does a cue get?
ins = np.clip(np.searchsorted(pco, cue), 1, len(pco) - 1)
prev = np.abs(cue - pco[ins - 1]); nxt = np.abs(pco[ins] - cue)
raw_cue_frame = np.where(prev <= nxt, ins - 1, ins)
std_corr_idx = raw_cue_frame // 2  # what the unmodified script would use
# (b) CORRECT mapping via frame_map: nearest kept corrected frame to each cue
ins2 = np.clip(np.searchsorted(csample, cue), 1, len(csample) - 1)
prev2 = np.abs(cue - csample[ins2 - 1]); nxt2 = np.abs(csample[ins2] - cue)
correct_corr_idx = np.where(prev2 <= nxt2, ins2 - 1, ins2)
dist_ms = np.minimum(prev2, nxt2) / fs * 1000
# does standard mapping land on the right frame?
agree = np.sum(std_corr_idx[valid_idx := np.arange(len(cue))] == correct_corr_idx)
print(f"  cues with a kept frame within 1s: {int(np.sum(dist_ms<1000))}/{len(cue)}  median dist={np.median(dist_ms):.1f}ms")
print(f"  cues where STANDARD raw//2 mapping == CORRECT frame_map mapping: {int(agree)}/{len(cue)}  "
      f"(low => standard pipeline misaligns rescued data)")
print(f"  std vs correct index abs diff: median={np.median(np.abs(std_corr_idx-correct_corr_idx)):.0f} "
      f"max={np.max(np.abs(std_corr_idx-correct_corr_idx))}")
