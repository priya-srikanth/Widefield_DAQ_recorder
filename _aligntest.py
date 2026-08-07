import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.transform import warp
from wfield.allen import load_allen_landmarks, atlas_from_landmarks_file

RES = r"E:\labcams_data\20260602\PS92\PS92_20260602_151820\illuminated_rescue\motion_corrected\wfield_local_results"
LM = r"E:\labcams_data\20260602\PS92\PS92_20260602_151820\illuminated_rescue\dorsal_cortex_landmarks.json"
favg = np.load(os.path.join(RES, "frames_average.npy"))   # (2,487,480) native [415,470]
H, W = favg.shape[1], favg.shape[2]
lm = load_allen_landmarks(LM)
T = lm["transform"]


def edges_of(a):
    a = np.asarray(a)
    e = np.zeros(a.shape, bool)
    e[:-1, :] |= a[:-1, :] != a[1:, :]; e[:, :-1] |= a[:, :-1] != a[:, 1:]
    e &= np.isfinite(a) & (a != 0)
    return ndimage.binary_dilation(e, iterations=1)


def overlay(ax, img, atlas, title):
    ax.imshow(img, cmap="gray")
    e = edges_of(atlas)
    ov = np.zeros((*e.shape, 4)); ov[e] = (1, 0, 0, 0.85)
    ax.imshow(ov); ax.set_title(title, fontsize=9); ax.axis("off")


# A) atlas at NATIVE dims, overlaid on NATIVE (un-warped) image
atlas_native, _, _ = atlas_from_landmarks_file(LM, dims=[H, W], do_transform=True)
# B) image warped to 540x640 ref grid, atlas at 540x640
warped = warp(favg[1].astype(float), T, output_shape=(540, 640), order=1, mode="constant", cval=0, preserve_range=True)
atlas_ref, _, _ = atlas_from_landmarks_file(LM, dims=[540, 640], do_transform=True)
# C) image warped with INVERSE transform to ref grid
warped_inv = warp(favg[1].astype(float), T.inverse, output_shape=(540, 640), order=1, mode="constant", cval=0, preserve_range=True)

fig, ax = plt.subplots(1, 3, figsize=(16, 5))
overlay(ax[0], favg[1], atlas_native, "A) NATIVE image + atlas@native dims (do_transform)")
overlay(ax[1], warped, atlas_ref, "B) warp(img,T)@540x640 + atlas@540x640")
overlay(ax[2], warped_inv, atlas_ref, "C) warp(img,T.inverse)@540x640 + atlas@540x640")
plt.tight_layout(); plt.savefig(os.path.join(RES, "..", "_aligntest.png"), dpi=120)
print("native dims", H, W, "saved _aligntest.png")
print("T params:\n", np.asarray(T.params))
