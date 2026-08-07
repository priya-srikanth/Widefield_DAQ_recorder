import numpy as np, h5py, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

base = r"E:\labcams_data\20260602\PS92\PS92_20260602_151820\illuminated_rescue"
res = os.path.join(base, "motion_corrected", "wfield_local_results")
daq_path = r"E:\DAQ_recorder_output\PS92_20260602_152607.h5"
fs_cam = 31.23

U = np.asarray(np.load(os.path.join(res, "U.npy")), dtype=np.float32)        # (H,W,K)
SVT = np.asarray(np.load(os.path.join(res, "SVT.npy")), dtype=np.float32)    # (K, 2*npairs) interleaved
favg = np.asarray(np.load(os.path.join(res, "frames_average.npy")), dtype=np.float32)  # (2,H,W) [415,470]
H, W, K = U.shape
U2 = U.reshape(-1, K)
svtA = SVT[:, 0::2]   # labeled 415 (channel 0)
svtB = SVT[:, 1::2]   # labeled 470 (channel 1)
npairs = svtA.shape[1]
print("npairs", npairs, "SVT", SVT.shape)

# brain ROI mask from mean image (exclude dim edges)
meanimg = favg.mean(0)
mask = meanimg > (0.35 * meanimg.max())
# erode edges a little
from scipy.ndimage import binary_erosion
mask = binary_erosion(mask, iterations=8)
P = mask.ravel()
meanU_P = U2[P].mean(0)                       # (K,)
# SVT/SVTcorr are already dF/F-scaled (U embeds 1/F0), so reconstruction U@SVT is dF/F.
F0_A = 1.0
F0_B = 1.0
print(f"ROI pixels {P.sum()}  (reconstruction already dF/F)")

# events -> corrected frame index via frame_map + pco
fm = np.load(os.path.join(base, "pco_edge_run001_00000000_2_487_480_uint16_daq_led_cleanpairs_frame_map.npz"))
c0 = fm["original_frame_index_ch0"]
with h5py.File(daq_path, "r") as f:
    fsd = float(f.attrs["sample_rate_hz"])
    di = [s.decode() for s in f["digital/channel_names"][:]]
    an = [s.decode() for s in f["analog/channel_names"][:]]
    bits = np.unpackbits(f["digital/packed_samples"][:, 0][:, None], axis=1, bitorder="little")
    raw = f["analog/samples_int16"][:].astype(np.float32)
    volts = raw * f["analog/int16_scale_volts_per_count"][:] + f["analog/int16_offset_volts"][:]
def rises(sig, thr=0.5):
    b = (sig > thr).astype(np.int8); return np.flatnonzero(np.diff(b) == 1) + 1
pco = rises(bits[:, di.index("pco_exposure")].astype(float))
cue = rises(bits[:, di.index("cue")].astype(float))
csample = pco[c0 + 1]
def to_frames(ev):
    ins = np.clip(np.searchsorted(csample, ev), 1, len(csample) - 1)
    prev = np.abs(ev - csample[ins - 1]); nxt = np.abs(csample[ins] - ev)
    return np.where(prev <= nxt, ins - 1, ins)
cue_idx = to_frames(cue)

# lick onsets (hysteresis-lite on lick_analog)
lick = volts[:, an.index("lick_analog")]
# rest-high signal: onset = drop below 2.5 after being above
lb = (lick < 2.5).astype(np.int8)
lick_on = np.flatnonzero(np.diff(lb) == 1) + 1
# refractory 100ms
keep = [lick_on[0]] if len(lick_on) else []
for s in lick_on[1:]:
    if (s - keep[-1]) / fsd > 0.1:
        keep.append(s)
lick_on = np.array(keep)
lick_idx = to_frames(lick_on)
print(f"cues={len(cue_idx)} lick_onsets={len(lick_idx)}")

# ROI-mean dF/F time course, event-triggered, per channel
pre = int(round(1.0 * fs_cam)); post = int(round(2.0 * fs_cam))
lags = np.arange(-pre, post + 1)
def trig(svt, idx, F0):
    tc = []
    for li in lags:
        vals = []
        for ci in idx:
            j = ci + li
            if 0 <= j < npairs and abs((csample[min(ci+post,npairs-1)] - csample[max(ci-pre,0)]) / fsd) < 5.0:
                vals.append(meanU_P @ svt[:, j])
        tc.append(np.mean(vals) / F0)
    return np.array(tc)
tcA_cue = trig(svtA, cue_idx, F0_A)
tcB_cue = trig(svtB, cue_idx, F0_B)
tcA_lick = trig(svtA, lick_idx, F0_A)
tcB_lick = trig(svtB, lick_idx, F0_B)
# baseline subtract (pre-event mean)
def bsub(tc): return tc - tc[:pre].mean()
tA_c, tB_c, tA_l, tB_l = map(bsub, [tcA_cue, tcB_cue, tcA_lick, tcB_lick])
t = lags / fs_cam
print(f"cue peak dF/F  labeled415={tA_c.max():+.4f}  labeled470={tB_c.max():+.4f}")
print(f"lick peak dF/F labeled415={tA_l.max():+.4f}  labeled470={tB_l.max():+.4f}")

# spatial cue-triggered post-pre map per channel
n = pre
preA = postA = preB = postB = 0; used = 0
for ci in cue_idx:
    if ci - n < 0 or ci + 1 + n > npairs:
        continue
    if (csample[ci + n - 1] - csample[ci - n]) / fsd > 4.0:
        continue
    preA = preA + svtA[:, ci - n:ci].mean(1); postA = postA + svtA[:, ci + 1:ci + 1 + n].mean(1)
    preB = preB + svtB[:, ci - n:ci].mean(1); postB = postB + svtB[:, ci + 1:ci + 1 + n].mean(1)
    used += 1
mapA = (U2 @ ((postA - preA) / used)).reshape(H, W)
mapB = (U2 @ ((postB - preB) / used)).reshape(H, W)

fig, ax = plt.subplots(2, 2, figsize=(12, 9))
mA = np.percentile(np.abs(mapA[mask]), 99); mB = np.percentile(np.abs(mapB[mask]), 99)
ax[0,0].imshow(np.where(mask, mapA, np.nan), cmap="seismic", vmin=-mA, vmax=mA); ax[0,0].set_title(f"cue post-pre dF/F  LABELED 415 (n={used})"); ax[0,0].axis("off")
ax[0,1].imshow(np.where(mask, mapB, np.nan), cmap="seismic", vmin=-mB, vmax=mB); ax[0,1].set_title(f"cue post-pre dF/F  LABELED 470 (n={used})"); ax[0,1].axis("off")
ax[1,0].plot(t, tA_c, color="violet", label="labeled 415"); ax[1,0].plot(t, tB_c, color="royalblue", label="labeled 470")
ax[1,0].axvline(0, color="k", lw=0.6); ax[1,0].set_title("cue-triggered ROI dF/F"); ax[1,0].set_xlabel("s"); ax[1,0].legend()
ax[1,1].plot(t, tA_l, color="violet", label="labeled 415"); ax[1,1].plot(t, tB_l, color="royalblue", label="labeled 470")
ax[1,1].axvline(0, color="k", lw=0.6); ax[1,1].set_title("lick-triggered ROI dF/F"); ax[1,1].set_xlabel("s"); ax[1,1].legend()
plt.tight_layout(); plt.savefig(os.path.join(base, "_channel_identity.png"), dpi=120)
print("saved _channel_identity.png")
print("INTERPRETATION: the channel with the large POSITIVE GCaMP transient is the true 470 (functional).")
