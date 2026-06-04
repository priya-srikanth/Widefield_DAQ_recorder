"""ROI-aware Allen alignment for the cropped PS92 recording.

apply_allen_transform warps images to their INPUT size; for a full-FOV (540x640)
recording that equals the atlas grid, but this recording is a 487x480 ROI crop,
so the warped image (487x480) and the Allen atlas (540x640) ended up on different
grids -> misaligned/cut-off overlay. Fix: warp the spatial maps onto the SAME
540x640 atlas grid the atlas is built on, using the (unchanged) landmark
transform. No re-clicking needed; the transform already maps image->atlas coords.

General lesson for future ROI recordings: warp to the atlas dims, not input dims.
"""
import os, json
import numpy as np
from skimage.transform import warp
from wfield.allen import load_allen_landmarks, atlas_from_landmarks_file

RES = r"E:\labcams_data\20260602\PS92\PS92_20260602_151820\illuminated_rescue\motion_corrected\wfield_local_results"
LM = r"E:\labcams_data\20260602\PS92\PS92_20260602_151820\illuminated_rescue\dorsal_cortex_landmarks.json"
OUT = os.path.join(RES, "allen_aligned_v2")
DIMS = (540, 640)


def warp_to(img):
    return warp(np.asarray(img, dtype=np.float64), T, output_shape=DIMS,
                order=1, mode="constant", cval=0, clip=True, preserve_range=True).astype(np.float32)


lm = load_allen_landmarks(LM)
T = lm["transform"]
os.makedirs(OUT, exist_ok=True)
print("transform params:\n", np.asarray(T.params))

U = np.load(os.path.join(RES, "U.npy"))           # (487,480,K)
print("U", U.shape)
U_atlas = np.zeros((DIMS[0], DIMS[1], U.shape[2]), np.float32)
for k in range(U.shape[2]):
    U_atlas[..., k] = warp_to(U[..., k])
np.save(os.path.join(OUT, "U_atlas.npy"), U_atlas)
print("U_atlas", U_atlas.shape)

favg = np.load(os.path.join(RES, "frames_average.npy"))   # (2,487,480) [415,470] (swapped)
favg_atlas = np.stack([warp_to(favg[i]) for i in range(favg.shape[0])])
np.save(os.path.join(OUT, "frames_average_atlas.npy"), favg_atlas)
print("frames_average_atlas", favg_atlas.shape)

rc = np.load(os.path.join(RES, "rcoeffs.npy"))
print("rcoeffs raw", rc.shape)
if rc.ndim == 1:
    rc = rc.reshape(487, 480)
if rc.ndim == 2:
    np.save(os.path.join(OUT, "rcoeffs_atlas.npy"), warp_to(rc))
else:
    np.save(os.path.join(OUT, "rcoeffs_atlas.npy"), np.stack([warp_to(rc[i]) for i in range(rc.shape[0])]))

atlas, names, mask = atlas_from_landmarks_file(LM, dims=list(DIMS), do_transform=True)
np.save(os.path.join(OUT, "allen_area_atlas_native_grid.npy"), np.asarray(atlas, np.float32))
np.save(os.path.join(OUT, "allen_brain_mask_native_grid.npy"), np.asarray(mask, np.uint8))
json.dump(list(names), open(os.path.join(OUT, "allen_area_names.json"), "w"), indent=2)
json.dump({"results": RES, "landmarks": LM, "dims": list(DIMS),
           "transform": np.asarray(T.params).tolist(),
           "note": "ROI-aware: spatial maps warped to atlas dims (540x640) to match Allen grid."},
          open(os.path.join(OUT, "allen_transform_summary.json"), "w"), indent=2)
print("atlas", np.asarray(atlas).shape, "wrote", OUT)

# quick overlay sanity check
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage
a = np.asarray(atlas)
edges = np.zeros(a.shape, bool)
edges[:-1, :] |= a[:-1, :] != a[1:, :]; edges[:, :-1] |= a[:, :-1] != a[:, 1:]
edges &= np.isfinite(a) & (a != 0)
edges = ndimage.binary_dilation(edges, iterations=1)
fig, ax = plt.subplots(1, 2, figsize=(11, 5))
ax[0].imshow(favg_atlas[1], cmap="gray"); ax[0].set_title("470 mean warped to 540x640")
ax[1].imshow(favg_atlas[1], cmap="gray")
ov = np.zeros((*edges.shape, 4)); ov[edges] = (1, 0, 0, 0.8)
ax[1].imshow(ov); ax[1].set_title("470 mean + Allen outlines (v2, ROI-aware)")
for x in ax: x.axis("off")
plt.tight_layout(); plt.savefig(os.path.join(RES, "..", "_allen_v2_check.png"), dpi=120)
print("saved _allen_v2_check.png")
