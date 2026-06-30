"""Export example SVG figures with enlarged fonts to N:\\...\\labcams\\example_figures_svg.
(1) quiet-period-normalized post-lick maps (per session, 6/5-6/8) - enlarged colorbar
    tick + label fonts. Replotted from the saved *_quietnorm_maps.npz (faithful: reuses
    the pipeline's region outlines + shared-limit helpers).
(2) Allen/wfield ROI-labels atlas (per animal, from its 6/6 CCF atlas) - enlarged region
    name labels.
No pipeline files are modified.
"""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from wfield_local.plot_spout_trial_averages import _region_edges, _overlay_regions, POSITION_NAMES, DISPLAY_ORDER
from wfield_local.plot_lick_aligned_averages import _shared_limit

NL = r"N:\MICROSCOPE\Priya\Widefield\labcams"
OUT = os.path.join(NL, "example_figures_svg"); os.makedirs(OUT, exist_ok=True)
CBAR_TICK = 20; CBAR_LABEL = 20; PANEL_TITLE = 13; SUPTITLE = 17; ROI_LABEL = 17

DAYS = {
 "PS92": {"0605":"PS92_20260605_125023","0606":"PS92_20260606_122451","0607":"PS92_20260607_121538","0608":"PS92_20260608_133759"},
 "PS93": {"0605":"PS93_20260605_174659","0606":"PS93_20260606_180117","0607":"PS93_20260607_174844","0608":"PS93_20260608_195203"},
 "PS94": {"0605":"PS94_20260605_142009","0606":"PS94_20260606_140854","0607":"PS94_20260607_140731","0608":"PS94_20260608_153651"},
 "PS95": {"0605":"PS95_20260605_163102","0606":"PS95_20260606_160806","0607":"PS95_20260607_155000","0608":"PS95_20260608_180943"},
}
def mc(an, d): return rf"{NL}\2026{d}\{DAYS[an][d]}\motion_corrected"
def allen(an, d): return rf"{mc(an,d)}\wfield_local_results\allen_aligned_affine8v1"

def lick_svg(an, d):
    lab = f"{an}_{d}_affine8v1"
    npz = rf"{mc(an,d)}\lick_aligned_affine8v1\{lab}_lick_aligned_150ms_post_by_spout_quietnorm_maps.npz"
    if not os.path.exists(npz): return f"  [skip] {lab}: no quietnorm npz"
    z = np.load(npz)
    norm = {k[:-len("_postnorm")]: z[k] for k in z.files if k.endswith("_postnorm")}
    atlas = np.load(rf"{allen(an,d)}\allen_area_atlas_native_grid.npy")
    edges = _region_edges(atlas); nlim = _shared_limit(norm, 99.0)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True); im = None
    for ax, code in zip(axes.ravel(), DISPLAY_ORDER):
        name = POSITION_NAMES[code]; ax.set_axis_off()
        if name not in norm: ax.set_title(f"{name}: no licks", fontsize=PANEL_TITLE); continue
        im = ax.imshow(norm[name], cmap="RdBu_r", vmin=-nlim, vmax=nlim)
        _overlay_regions(ax, edges)
        ax.set_title(f"{name} | 150 ms post-lick (quiet-norm)", fontsize=PANEL_TITLE)
    if im is not None:
        cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.78, pad=0.02)
        cb.ax.tick_params(labelsize=CBAR_TICK)
        cb.set_label(f"post-lick minus quiet baseline (+/-{nlim:.4g})", fontsize=CBAR_LABEL)
    fig.suptitle(f"{lab} post-lick, normalized to quiet-period baseline", fontsize=SUPTITLE)
    out = os.path.join(OUT, f"{lab}_lick_quietnorm.svg"); fig.savefig(out); plt.close(fig)
    return f"  wrote {os.path.basename(out)}"

def roi_svg(an):
    ad = allen(an, "0606")
    atlas = np.load(rf"{ad}\allen_area_atlas_native_grid.npy")
    names = {int(i): n for i, n in json.load(open(rf"{ad}\allen_area_names.json"))}
    edges = _region_edges(atlas)
    left = [i for i in np.unique(atlas) if i > 0]
    cmap = cm.get_cmap("hsv", len(left) + 1)
    rgb = np.ones((*atlas.shape, 3))
    for k, i in enumerate(left): rgb[atlas == i] = cmap(k)[:3]
    fig, ax = plt.subplots(figsize=(13, 11)); ax.set_axis_off()
    ax.imshow(rgb); _overlay_regions(ax, edges)
    for i in left:
        ys, xs = np.where(atlas == i)
        if len(xs) < 30: continue
        nm = names.get(int(i), str(int(i))).replace("_left", "")
        ax.text(xs.mean(), ys.mean(), nm, fontsize=ROI_LABEL, ha="center", va="center", weight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
    ax.set_title(f"Allen/wfield ROI labels ({an}, 6/6 CCF)", fontsize=SUPTITLE)
    out = os.path.join(OUT, f"{an}_allen_roi_labels.svg"); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return f"  wrote {os.path.basename(out)}"

if __name__ == "__main__":
    print("=== quiet-norm post-lick SVGs (6/5-6/8) ===")
    for an in DAYS:
        for d in DAYS[an]: print(lick_svg(an, d))
    print("=== Allen/wfield ROI-labels SVGs (per animal) ===")
    for an in DAYS: print(roi_svg(an))
    print(f"\nALL SVGs -> {OUT}")
