import numpy as np, os, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

base = r"E:\labcams_data\20260602\PS92\PS92_20260602_151820\illuminated_rescue"
res = os.path.join(base, "motion_corrected", "wfield_local_results")
allen = os.path.join(res, "allen_aligned_v6")
dat = os.path.join(base, "pco_edge_run001_00000000_2_487_480_uint16_daq_led_cleanpairs_2_487_480_uint16.dat")
lm = json.load(open(r"E:\labcams_data\20260602\PS92\PS92_20260602_151820\raw_widefield_data\dorsal_cortex_landmarks.json"))

H, W = 487, 480
favg = np.load(os.path.join(res, "frames_average.npy"))          # [415,470] after swap, native
favg_atlas = np.load(os.path.join(allen, "frames_average_atlas.npy"))  # warped
# raw (pre motion-correction) mean of real-470 = dat ch0
npairs = os.path.getsize(dat) // (2 * H * W * 2)
mm = np.memmap(dat, mode="r", dtype=np.uint16, shape=(npairs, 2, H, W))
idx = np.linspace(0, npairs - 1, 200).astype(int)
raw_mean = np.asarray(mm[idx, 0], dtype=np.float32).mean(0)        # real 470 functional

print("frames_average shape (native):", favg.shape, " -> H=%d W=%d" % (favg.shape[1], favg.shape[2]))
print("landmarks bregma_offset:", lm["bregma_offset"], "(=> expects %dx%d image)" % (lm["bregma_offset"][0]*2, lm["bregma_offset"][1]*2))
print("landmarks_im x range:", min(lm["landmarks_im"]["x"]), max(lm["landmarks_im"]["x"]),
      " y range:", min(lm["landmarks_im"]["y"]), max(lm["landmarks_im"]["y"]))

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(raw_mean, cmap="gray"); ax[0].set_title("RAW mean (pre motion-corr) 487x480");
ax[1].imshow(favg[1], cmap="gray"); ax[1].set_title("MOTION-CORRECTED mean (native) 487x480")
ax[2].imshow(favg_atlas[1], cmap="gray"); ax[2].set_title("WARPED to atlas (landmark transform)")
for a in ax: a.axis("off")
plt.tight_layout(); plt.savefig(os.path.join(base, "_align_diag.png"), dpi=120)
print("saved _align_diag.png")
