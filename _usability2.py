import numpy as np, h5py, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

base = r"E:\labcams_data\20260602\PS92\PS92_20260602_151820\illuminated_rescue"
res = os.path.join(base, "motion_corrected", "wfield_local_results")
daq_path = r"E:\DAQ_recorder_output\PS92_20260602_152607.h5"
fs_cam = 31.23

U = np.asarray(np.load(os.path.join(res, "U.npy")), dtype=np.float32)   # (H,W,K)
SVT = np.asarray(np.load(os.path.join(res, "SVTcorr.npy")), dtype=np.float32)  # (K,T)
H, W, K = U.shape
T = SVT.shape[1]
U2 = U.reshape(-1, K)

# proper temporal std map via covariance (exact, cheap)
Cov = np.cov(SVT)                      # (K,K)
var_pix = np.einsum("pi,ij,pj->p", U2, Cov, U2)
std_map = np.sqrt(np.clip(var_pix, 0, None)).reshape(H, W)

# --- correct cue -> corrected-frame mapping via frame_map ---
fm = np.load(os.path.join(base, "pco_edge_run001_00000000_2_487_480_uint16_daq_led_cleanpairs_frame_map.npz"))
c0 = fm["original_frame_index_ch0"]
with h5py.File(daq_path, "r") as f:
    fsd = float(f.attrs["sample_rate_hz"])
    di = [s.decode() for s in f["digital/channel_names"][:]]
    bits = np.unpackbits(f["digital/packed_samples"][:, 0][:, None], axis=1, bitorder="little")
def rises(sig, thr=0.5):
    b = (sig > thr).astype(np.int8); return np.flatnonzero(np.diff(b) == 1) + 1
pco = rises(bits[:, di.index("pco_exposure")].astype(float))
cue = rises(bits[:, di.index("cue")].astype(float))
offset = 1
csample = pco[c0 + offset]            # DAQ sample of each kept corrected frame (T,)
ins = np.clip(np.searchsorted(csample, cue), 1, len(csample) - 1)
prev = np.abs(cue - csample[ins - 1]); nxt = np.abs(csample[ins] - cue)
cue_idx = np.where(prev <= nxt, ins - 1, ins)   # corrected-frame index per cue

# cue-triggered mean dF/F delta (post 1s - pre 1s), in component space then project once
n = int(round(1.0 * fs_cam))
pre_acc = np.zeros(K); post_acc = np.zeros(K); used = 0
for ci in cue_idx:
    if ci - n < 0 or ci + n > T:
        continue
    # skip windows that span a big temporal gap (trial boundary) using csample continuity
    if (csample[ci + n - 1] - csample[ci - n]) / fsd > 4.0:   # >4s real time => spans gap
        continue
    pre_acc += SVT[:, ci - n:ci].mean(axis=1)
    post_acc += SVT[:, ci + 1:ci + 1 + n].mean(axis=1)
    used += 1
delta = (post_acc - pre_acc) / max(used, 1)
delta_map = (U2 @ delta).reshape(H, W)
print(f"cue-triggered average used {used}/{len(cue_idx)} trials")
print(f"delta_map range [{delta_map.min():.4f},{delta_map.max():.4f}]  std-map p99={np.percentile(std_map,99):.3f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
v = np.percentile(std_map, 99)
im0 = ax[0].imshow(std_map, cmap="magma", vmax=v); ax[0].set_title("SVTcorr temporal std (p99 scaled)")
plt.colorbar(im0, ax=ax[0], fraction=0.046)
m = np.percentile(np.abs(delta_map), 99)
im1 = ax[1].imshow(delta_map, cmap="seismic", vmin=-m, vmax=m)
ax[1].set_title(f"cue-triggered post-pre dF/F (n={used})")
plt.colorbar(im1, ax=ax[1], fraction=0.046)
for a in ax: a.axis("off")
plt.tight_layout(); plt.savefig(os.path.join(base, "_usability_check2.png"), dpi=120)
print("saved _usability_check2.png")
