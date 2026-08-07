import numpy as np, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

base = r"E:\labcams_data\20260602\PS92\PS92_20260602_151820\illuminated_rescue"
dat = os.path.join(base, "pco_edge_run001_00000000_2_487_480_uint16_daq_led_cleanpairs_2_487_480_uint16.dat")
H, W = 487, 480
npairs = os.path.getsize(dat) // (2 * H * W * 2)
mm = np.memmap(dat, mode="r", dtype=np.uint16, shape=(npairs, 2, H, W))
print(f"pairs={npairs}  ch0=labeled 415, ch1=labeled 470")

# sample pairs evenly across the whole recording
nsamp = 1500
idx = np.linspace(0, npairs - 1, nsamp).astype(int)
ch0 = np.asarray(mm[idx, 0], dtype=np.float32)   # (nsamp,H,W)
ch1 = np.asarray(mm[idx, 1], dtype=np.float32)

# per-frame mean intensity
m0 = ch0.reshape(nsamp, -1).mean(1)
m1 = ch1.reshape(nsamp, -1).mean(1)
print(f"mean intensity  415: {m0.mean():.1f}+/-{m0.std():.1f}   470: {m1.mean():.1f}+/-{m1.std():.1f}")

# discriminant in the 470-minus-415 spatial direction (built from sample means)
t0 = ch0.mean(0); t1 = ch1.mean(0)
d = (t1 - t0).ravel()
d = d / (np.linalg.norm(d) + 1e-9)
s0 = ch0.reshape(nsamp, -1) @ d        # projection of each 415 frame
s1 = ch1.reshape(nsamp, -1) @ d        # projection of each 470 frame
# 470 should project higher than its paired 415 on the (470-415) axis
frac_correct = float(np.mean(s1 > s0))
# overlap of the two distributions
thr = 0.5 * (s0.mean() + s1.mean())
acc = 0.5 * (np.mean(s0 < thr) + np.mean(s1 > thr))
print(f"within-pair 470>415 on discriminant: {100*frac_correct:.2f}%")
print(f"single-frame label separability (threshold acc): {100*acc:.2f}%")
print(f"discriminant: 415 proj {s0.mean():.1f}+/-{s0.std():.1f}   470 proj {s1.mean():.1f}+/-{s1.std():.1f}")

# detect temporal swaps: sign of (s1-s0) over time
flips = int(np.sum((s1 - s0) <= 0))
print(f"pairs where 470 does NOT exceed 415 (possible swap/ambiguous): {flips}/{nsamp}")

# figure: example frames at 5 time points + projection over time + intensity hist
pos = np.linspace(0, npairs - 1, 5).astype(int)
fig = plt.figure(figsize=(15, 7))
for j, p in enumerate(pos):
    a = fig.add_subplot(3, 5, j + 1); a.imshow(mm[p, 0], cmap="gray"); a.axis("off")
    a.set_title(f"415  pair {p}", fontsize=8)
    b = fig.add_subplot(3, 5, j + 6); b.imshow(mm[p, 1], cmap="gray"); b.axis("off")
    b.set_title(f"470  pair {p}", fontsize=8)
axp = fig.add_subplot(3, 2, 5)
axp.plot(idx, s0, ".", ms=2, color="violet", label="415 frames")
axp.plot(idx, s1, ".", ms=2, color="royalblue", label="470 frames")
axp.set_xlabel("pair index (time)"); axp.set_ylabel("470-415 discriminant"); axp.legend(fontsize=7)
axp.set_title("channel separation over recording (no overlap = no swaps)", fontsize=9)
axh = fig.add_subplot(3, 2, 6)
axh.hist(m0, bins=40, alpha=0.6, color="violet", label="415 mean intensity")
axh.hist(m1, bins=40, alpha=0.6, color="royalblue", label="470 mean intensity")
axh.legend(fontsize=7); axh.set_title("per-frame mean intensity by label", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(base, "_channel_check.png"), dpi=110)
print("saved _channel_check.png")
