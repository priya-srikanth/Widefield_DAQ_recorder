"""Interactive Allen landmark registration for a labcams recording.

Pick a recording, scrub to a frame/time, choose the 415/470 channel, place
dorsal-cortex landmarks, preview how the Allen atlas morphs onto your image with
either a SIMILARITY or AFFINE transform (and compare them side by side), then save
a dorsal_cortex_landmarks_<label>.json that apply_allen_transform.py consumes
after motion correction + SVD.

Landmarks:
  medial  : OB_center, OB_left, OB_right, RSP_base   (minimal similarity set)
  lateral : MOp_left, MOp_right, SS_left, SS_right    (add these for affine)

Similarity (4 DOF: translate/rotate/uniform-scale) needs >=2-3 points; the medial
points are nearly collinear so they constrain little beyond similarity. Adding the
lateral points breaks that collinearity and lets an AFFINE fit (6 DOF: independent
AP/ML scale + shear) correct aspect-ratio/shear mismatch.

Usage:
  python -m wfield_local.allen_register <path-to-*_2_H_W_uint16.dat>
  python -m wfield_local.allen_register             # file picker
  python -m wfield_local.allen_register --selftest  # headless logic check

Controls: time/avg sliders; channel radio (415/470); landmark radio (which point
the next click sets); transform radio (similarity/affine); "compare" check (draw
both); "overlay" check (draw atlas); Save (writes JSON with the label in the box).
"""
from __future__ import annotations

import argparse
import os
import re

import numpy as np
from pandas import DataFrame
from skimage.transform import estimate_transform

from wfield.allen import (
    allen_load_reference,
    allen_landmarks_to_image_space,
    allen_transform_regions,
    save_allen_landmarks,
)

# wfield dorsal-cortex defaults + lateral border anchors (mm, Allen space)
LM_NAMES = ["OB_center", "OB_left", "OB_right", "RSP_base",
            "MOp_left", "MOp_right", "SS_left", "SS_right"]
LM_MM = {
    "OB_center": (0.0, -3.45), "OB_left": (-1.95, -3.45), "OB_right": (1.95, -3.45),
    "RSP_base": (0.0, 3.2),
    "MOp_left": (-3.63, -1.87), "MOp_right": (3.63, -1.87),   # lateral MOp / anterolateral
    "SS_left": (-5.08, 1.53), "SS_right": (5.08, 1.53),       # lateral SS (SSs edge)
}
LM_COLOR = {
    "OB_center": "#0367fc", "OB_left": "#fc9d03", "OB_right": "#fc9d03", "RSP_base": "#fc4103",
    "MOp_left": "#03fc6b", "MOp_right": "#03fc6b", "SS_left": "#fc03d2", "SS_right": "#fc03d2",
}
RESOLUTION = 0.0194
BREGMA_OFFSET = [320, 270]
DAT_RE = re.compile(r"_(\d+)_(\d+)_(\d+)_uint16\.dat$")


def _ref_image_xy(name):
    x, y = LM_MM[name]
    return x / RESOLUTION + BREGMA_OFFSET[0], y / RESOLUTION + BREGMA_OFFSET[1]


def compute_transform(points: dict, ttype: str = "affine"):
    """Fit reference->image transform from placed points.

    Returns (transform, names_used, effective_type) or (None, [], type) if too few.
    """
    names = [n for n in LM_NAMES if n in points]
    eff = ttype
    if eff == "affine" and len(names) < 3:
        eff = "similarity"
    if len(names) < 2:
        return None, [], eff
    ref = np.array([_ref_image_xy(n) for n in names], dtype=float)
    dst = np.array([points[n] for n in names], dtype=float)
    tform = estimate_transform("affine" if eff == "affine" else "similarity", ref, dst)
    return tform, names, eff


def save_landmarks_json(points: dict, filename: str, ttype: str = "affine"):
    tform, names, eff = compute_transform(points, ttype)
    if tform is None:
        raise ValueError("Need at least 3 points for affine (2 for similarity).")
    landmarks = DataFrame({
        "x": [LM_MM[n][0] for n in names], "y": [LM_MM[n][1] for n in names],
        "name": names, "color": [LM_COLOR[n] for n in names]})
    match = DataFrame({
        "x": [points[n][0] for n in names], "y": [points[n][1] for n in names],
        "name": names, "color": [LM_COLOR[n] for n in names]})
    save_allen_landmarks(
        landmarks, filename=filename, resolution=RESOLUTION,
        landmarks_match=match, bregma_offset=BREGMA_OFFSET,
        transform=tform, transform_type=eff)
    # wfield's save_allen_landmarks does NOT persist transform_type, so without
    # this an affine fit reloads as a SimilarityTransform downstream. Write the
    # type (and the inverse, used by apply_allen_transform) ourselves.
    import json
    with open(filename) as fh:
        d = json.load(fh)
    d["transform_type"] = eff
    try:
        d["transform_inverse"] = np.asarray(tform._inv_matrix).tolist()
    except Exception:
        d["transform_inverse"] = np.linalg.inv(np.asarray(tform.params)).tolist()
    with open(filename, "w") as fh:
        json.dump(d, fh, sort_keys=True, indent=4)
    return filename, tform, eff, names


class DatReader:
    def __init__(self, path):
        m = DAT_RE.search(os.path.basename(path))
        if not m:
            raise ValueError(f"Cannot parse H/W from {path}")
        self.nchan, self.H, self.W = int(m.group(1)), int(m.group(2)), int(m.group(3))
        self.nphys = os.path.getsize(path) // (self.H * self.W * 2)
        self.npairs = self.nphys // 2
        self.mm = np.memmap(path, mode="r", dtype=np.uint16, shape=(self.nphys, self.H, self.W))

    def image(self, pair_center, channel_offset, navg):
        half = max(1, navg // 2)
        p0 = max(0, pair_center - half); p1 = min(self.npairs, pair_center + half)
        idx = np.arange(p0, p1) * 2 + channel_offset
        idx = idx[(idx >= 0) & (idx < self.nphys)]
        if len(idx) == 0:
            return np.zeros((self.H, self.W), np.float32)
        return np.asarray(self.mm[idx], dtype=np.float32).mean(0)


def run_gui(dat_path):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, RadioButtons, Button, CheckButtons, TextBox

    reader = DatReader(dat_path)
    ccf_regions, _proj, _outline = allen_load_reference("dorsal_cortex")
    points = {}
    overlay_artists, marker_artists = [], []
    state = {"channel": 0, "navg": 60, "pair": reader.npairs // 2,
             "place": "OB_center", "overlay": True, "compare": False, "ttype": "affine"}
    sess_dir = os.path.dirname(dat_path)

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_axes([0.04, 0.18, 0.60, 0.78])
    ax.set_title("click to place the selected landmark")
    im = ax.imshow(reader.image(state["pair"], state["channel"], state["navg"]), cmap="gray")

    def overlay_for(tform, color):
        nccf = allen_transform_regions(tform, ccf_regions, RESOLUTION, BREGMA_OFFSET)
        for _, r in nccf.iterrows():
            for side in ("left", "right"):
                ln, = ax.plot(r[side + "_x"], r[side + "_y"], "-", color=color, lw=0.7, alpha=0.85)
                overlay_artists.append(ln)

    def draw():
        for a in overlay_artists + marker_artists:
            a.remove()
        overlay_artists.clear(); marker_artists.clear()
        # markers
        for n, (x, y) in points.items():
            d, = ax.plot(x, y, "o", ms=10, mfc="none", mec=LM_COLOR[n], mew=2)
            t = ax.text(x + 4, y, n, color=LM_COLOR[n], fontsize=7)
            marker_artists += [d, t]
        # overlays
        if state["overlay"]:
            try:
                if state["compare"]:
                    ts, ns, _ = compute_transform(points, "similarity")
                    ta, na, _ = compute_transform(points, "affine")
                    if ts is not None:
                        overlay_for(ts, "cyan")
                    if ta is not None and len(na) >= 3:
                        overlay_for(ta, "magenta")
                else:
                    tf, ns, eff = compute_transform(points, state["ttype"])
                    if tf is not None:
                        overlay_for(tf, "cyan")
            except Exception as e:
                print("overlay error:", e)
        fig.canvas.draw_idle()

    def refresh_image():
        img = reader.image(state["pair"], state["channel"], state["navg"])
        im.set_data(img)
        lo, hi = np.percentile(img, [1, 99.5]); im.set_clim(lo, hi)
        draw()

    def on_click(event):
        if event.inaxes is not ax or event.xdata is None or event.button != 1:
            return
        points[state["place"]] = (float(event.xdata), float(event.ydata))
        # auto-advance to next unplaced landmark
        draw()
    fig.canvas.mpl_connect("button_press_event", on_click)

    ax_t = fig.add_axes([0.04, 0.10, 0.60, 0.03])
    Slider(ax_t, "time (pair)", 0, max(1, reader.npairs - 1), valinit=state["pair"], valstep=1)\
        .on_changed(lambda v: (state.update(pair=int(v)), refresh_image()))
    ax_a = fig.add_axes([0.04, 0.06, 0.60, 0.03])
    Slider(ax_a, "avg frames", 1, 400, valinit=state["navg"], valstep=1)\
        .on_changed(lambda v: (state.update(navg=int(v)), refresh_image()))

    ax_c = fig.add_axes([0.68, 0.80, 0.12, 0.11]); ax_c.set_title("channel", fontsize=9)
    RadioButtons(ax_c, ("415", "470")).on_clicked(
        lambda l: (state.update(channel=0 if l == "415" else 1), refresh_image()))
    ax_tt = fig.add_axes([0.83, 0.80, 0.14, 0.11]); ax_tt.set_title("transform", fontsize=9)
    RadioButtons(ax_tt, ("affine", "similarity")).on_clicked(
        lambda l: (state.update(ttype=l), draw()))
    ax_l = fig.add_axes([0.68, 0.46, 0.29, 0.30]); ax_l.set_title("place landmark", fontsize=9)
    RadioButtons(ax_l, tuple(LM_NAMES)).on_clicked(lambda l: state.update(place=l))

    ax_ov = fig.add_axes([0.68, 0.39, 0.14, 0.05])
    co = CheckButtons(ax_ov, ["overlay"], [True]); co.on_clicked(
        lambda l: (state.update(overlay=not state["overlay"]), draw()))
    ax_cmp = fig.add_axes([0.83, 0.39, 0.14, 0.05])
    cc = CheckButtons(ax_cmp, ["compare"], [False]); cc.on_clicked(
        lambda l: (state.update(compare=not state["compare"]), draw()))

    ax_lab = fig.add_axes([0.74, 0.31, 0.23, 0.045])
    tb = TextBox(ax_lab, "label ", initial="v1")
    ax_save = fig.add_axes([0.68, 0.23, 0.13, 0.06]); b_save = Button(ax_save, "Save")
    ax_clear = fig.add_axes([0.84, 0.23, 0.13, 0.06]); b_clear = Button(ax_clear, "Clear")
    msg = fig.text(0.68, 0.04, "", fontsize=7, wrap=True)

    def do_save(_):
        lab = (tb.text or "v1").strip() or "v1"
        fn = os.path.join(sess_dir, f"dorsal_cortex_landmarks_{lab}.json")
        try:
            fn, tf, eff, ns = save_landmarks_json(points, fn, state["ttype"])
            msg.set_text("Saved (%s, %d pts): %s" % (eff, len(ns), fn)); print("[allen_register] saved", fn, eff, ns)
        except Exception as e:
            msg.set_text("Save failed: %s" % e)
        fig.canvas.draw_idle()
    b_save.on_clicked(do_save)
    b_clear.on_clicked(lambda _: (points.clear(), draw(), msg.set_text("cleared")))

    refresh_image()
    fig.text(0.68, 0.135, "compare = cyan(similarity) + magenta(affine)\nSave dir:\n" + sess_dir, fontsize=7)
    plt.show()


def selftest():
    ccf, _, _ = allen_load_reference("dorsal_cortex")
    print("regions:", len(ccf))
    pts = {n: (_ref_image_xy(n)[0] * 0.92 + 15, _ref_image_xy(n)[1] * 0.95 + 8) for n in LM_NAMES}
    for ttype in ("similarity", "affine"):
        tf, names, eff = compute_transform(pts, ttype)
        print(f"{ttype}: eff={eff} npts={len(names)} params=\n{np.asarray(tf.params)}")
        nccf = allen_transform_regions(tf, ccf, RESOLUTION, BREGMA_OFFSET)
        assert len(nccf) == len(ccf)
    import tempfile, json
    for ttype in ("similarity", "affine"):
        fn = os.path.join(tempfile.gettempdir(), f"_allen_register_{ttype}.json")
        save_landmarks_json(pts, fn, ttype)
        d = json.loads(open(fn).read())
        assert d["transform_type"] == ttype, d.get("transform_type")
        assert {"landmarks", "landmarks_im", "landmarks_match", "transform", "bregma_offset", "resolution"} <= set(d)
        print(f"saved {ttype} JSON OK keys={sorted(d)}")
    print("SELFTEST PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dat", nargs="?", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return 0
    dat = args.dat
    if dat is None:
        try:
            import tkinter as tk
            from tkinter import filedialog
            tk.Tk().withdraw()
            dat = filedialog.askopenfilename(title="Select labcams .dat", filetypes=[("labcams dat", "*.dat")])
        except Exception:
            pass
    if not dat:
        print("No recording selected."); return 1
    run_gui(dat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
