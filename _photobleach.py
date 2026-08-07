"""Photobleaching check: 470/415 baseline fluorescence drift over a session.

Samples camera frames evenly, restricts to a brain ROI, labels each frame as
415/470 from the DAQ LED TTLs, then plots time-binned MEDIAN intensity per
channel (robust to activity/movement) and fits the slow trend.
"""
import h5py, numpy as np, os, sys, re, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion

DAQ = sys.argv[1] if len(sys.argv) > 1 else r"E:\DAQ_recorder_output\20250603\PS94_20260603_175946.h5"
DAT = sys.argv[2] if len(sys.argv) > 2 else r"E:\labcams_data\20260603\PS94\raw_widefield_data\pco_edge_run000_00000000_2_462_464_uint16.dat"
NSAMP = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
OUT = sys.argv[4] if len(sys.argv) > 4 else os.path.join(os.path.dirname(DAT), "..", "_photobleach.png")
m = re.search(r"_(\d+)_(\d+)_(\d+)_uint16", os.path.basename(DAT)); H, W = int(m.group(2)), int(m.group(3))

with h5py.File(DAQ, "r") as f:
    fs = float(f.attrs["sample_rate_hz"])
    an = [s.decode() for s in f["analog/channel_names"][:]]; di = [s.decode() for s in f["digital/channel_names"][:]]
    packed = f["digital/packed_samples"][:, 0]; sc = f["analog/int16_scale_volts_per_count"][:]; of = f["analog/int16_offset_volts"][:]
    def ac(n): i = an.index(n); return f["analog/samples_int16"][:, i].astype(np.float32) * sc[i] + of[i]
    led415 = ac("led415_ttl"); led470 = ac("led470_ttl")
def db(b): return (packed >> b) & 1
def rises(x, t=0.5): b = (np.asarray(x) > t).astype(np.int8); return np.flatnonzero(np.diff(b) == 1) + 1
pco = rises(db(di.index("pco_exposure")))
i2 = np.clip(pco + int(0.002 * fs), 0, len(packed) - 1)
code = np.where((led415[i2] > 1.5) & ~(led470[i2] > 1.5), 415,
        np.where((led470[i2] > 1.5) & ~(led415[i2] > 1.5), 470, 0))
pco_t = pco / fs

nphys = os.path.getsize(DAT) // (H * W * 2)
mm = np.memmap(DAT, mode="r", dtype=np.uint16, shape=(nphys, H, W))
n = min(nphys, len(pco))
samp = np.linspace(0, n - 1, NSAMP).astype(int)
print(f"dat frames={nphys} pco={len(pco)} sampling {NSAMP}")

# brain ROI from a robust average of sampled frames
avg = np.zeros((H, W), np.float64)
for k in samp[::8]:
    avg += mm[k]
avg /= len(samp[::8])
mask = binary_erosion(avg > (0.45 * avg.max()), iterations=6)
P = mask.ravel()
print(f"ROI pixels {int(P.sum())}/{H*W}")

roi_mean = np.array([mm[k].reshape(-1)[P].mean() for k in samp], dtype=np.float64)
lab = code[np.clip(samp, 0, len(code) - 1)]
t = pco_t[np.clip(samp, 0, len(pco) - 1)]

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
res = {}
NB = 50
edges = np.linspace(t.min(), t.max(), NB + 1)
ctr = 0.5 * (edges[:-1] + edges[1:])
for c, col, name in [(415, "violet", "415"), (470, "royalblue", "470")]:
    msk = lab == c
    if msk.sum() < 20: continue
    tt, vv = t[msk], roi_mean[msk]
    ax[0].plot(tt, vv, ".", ms=2, color=col, alpha=0.18)
    # binned medians
    bmed = np.array([np.median(vv[(tt >= edges[i]) & (tt < edges[i+1])]) if np.any((tt >= edges[i]) & (tt < edges[i+1])) else np.nan for i in range(NB)])
    good = np.isfinite(bmed)
    ax[0].plot(ctr[good], bmed[good], "-o", ms=4, color=col, lw=2, label=name + " (binned median)")
    # linear fit to binned medians
    p = np.polyfit(ctr[good], bmed[good], 1)
    start, end = np.polyval(p, ctr[good][0]), np.polyval(p, ctr[good][-1])
    pct = 100 * (end - start) / start
    res[name] = dict(start=float(start), end=float(end), pct=float(pct), per_min=float(p[0]*60), median_level=float(np.median(vv)))
    ax[1].plot(ctr[good], bmed[good] / bmed[good][0], "-o", ms=4, color=col, lw=2, label=name)
    print(f"{name}: ROI median {np.median(vv):.0f}; binned start={start:.0f} end={end:.0f} drift={pct:+.1f}% ({p[0]*60:+.1f}/min)")

ax[0].set_xlabel("session time (s)"); ax[0].set_ylabel("brain-ROI mean intensity"); ax[0].legend(); ax[0].set_title("ROI intensity (binned median + fit)")
ax[1].axhline(1.0, color="k", lw=0.6); ax[1].set_xlabel("session time (s)"); ax[1].set_ylabel("binned median / first bin"); ax[1].legend(); ax[1].set_title("Normalized bleaching trend")
plt.suptitle("Photobleaching: %s" % os.path.basename(DAT)); plt.tight_layout()
op = os.path.abspath(OUT); plt.savefig(op, dpi=120); print("saved", op)
print(json.dumps(res, indent=2))
