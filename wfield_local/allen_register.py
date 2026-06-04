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


def _draw_reference_guide(ax, ccf_regions):
    """Draw the Allen dorsal-cortex outline + suggested landmark positions."""
    from skimage.transform import SimilarityTransform
    nccf = allen_transform_regions(SimilarityTransform(), ccf_regions, RESOLUTION, BREGMA_OFFSET)
    for _, r in nccf.iterrows():
        for side in ("left", "right"):
            ax.plot(r[side + "_x"], r[side + "_y"], "-", color="0.65", lw=0.5)
    for n in LM_NAMES:
        x, y = _ref_image_xy(n)
        ax.plot(x, y, "o", ms=7, color=LM_COLOR[n])
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(3, 2),
                    fontsize=6, color=LM_COLOR[n])
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Allen reference: suggested placements", fontsize=8)


def run_gui(dat_path):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, RadioButtons, Button, CheckButtons, TextBox

    reader = DatReader(dat_path)
    ccf_regions, _proj, _outline = allen_load_reference("dorsal_cortex")
    points = {}
    overlay_artists, marker_artists = [], []
    keep = []  # retain widget refs so matplotlib does not garbage-collect them
    state = {"channel": 0, "navg": 60, "pair": reader.npairs // 2,
             "place": "OB_center", "overlay": True, "compare": False, "ttype": "affine",
             "swap": False, "drag": None}
    sess_dir = os.path.dirname(dat_path)

    def eff_off():
        # which physical frame parity to show as the chosen channel; "swap" flips
        # the 415/470 assignment (the parity->wavelength mapping is session-specific)
        return state["channel"] ^ (1 if state["swap"] else 0)

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0.03, 0.22, 0.50, 0.73])
    ax.set_title("click=place  drag pt=move  Del=delete  drag bg/right-drag=pan  scroll=zoom")
    im = ax.imshow(reader.image(state["pair"], eff_off(), state["navg"]), cmap="gray")

    # reference guide panel (static)
    ax_ref = fig.add_axes([0.56, 0.55, 0.20, 0.40])
    _draw_reference_guide(ax_ref, ccf_regions)

    def overlay_for(tform, color):
        nccf = allen_transform_regions(tform, ccf_regions, RESOLUTION, BREGMA_OFFSET)
        for _, r in nccf.iterrows():
            for side in ("left", "right"):
                ln, = ax.plot(r[side + "_x"], r[side + "_y"], "-", color=color, lw=0.7, alpha=0.85)
                overlay_artists.append(ln)

    def draw(fast=False):
        for a in overlay_artists + marker_artists:
            a.remove()
        overlay_artists.clear(); marker_artists.clear()
        for n, (x, y) in points.items():
            d, = ax.plot(x, y, "o", ms=10, mfc="none", mec=LM_COLOR[n], mew=2)
            t = ax.text(x + 4, y, n, color=LM_COLOR[n], fontsize=7)
            marker_artists.extend([d, t])
        if state["overlay"] and not fast:
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
        img = reader.image(state["pair"], eff_off(), state["navg"])
        im.set_data(img)
        lo, hi = np.percentile(img, [1, 99.5]); im.set_clim(lo, hi)
        draw()

    def _nearest_point(x, y):
        # threshold scales with current zoom so it stays grabbable
        x0, x1 = ax.get_xlim(); thr = 0.03 * abs(x1 - x0) + 6
        best, bd = None, thr
        for n, (px, py) in points.items():
            dd = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
            if dd < bd:
                bd, best = dd, n
        return best

    pan = {}  # captured at press: fixed transform + start limits, so pan is stable

    def _start_pan(event):
        pan["inv"] = ax.transData.inverted()
        pan["px"], pan["py"] = event.x, event.y
        pan["xlim"], pan["ylim"] = ax.get_xlim(), ax.get_ylim()

    def _do_pan(event):
        if event.x is None or "inv" not in pan:
            return
        d0 = pan["inv"].transform((pan["px"], pan["py"]))
        d1 = pan["inv"].transform((event.x, event.y))
        ddx, ddy = d1[0] - d0[0], d1[1] - d0[1]
        ax.set_xlim(pan["xlim"][0] - ddx, pan["xlim"][1] - ddx)
        ax.set_ylim(pan["ylim"][0] - ddy, pan["ylim"][1] - ddy)
        fig.canvas.draw_idle()

    def on_press(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        if event.button == 3:                      # right-drag = pan
            state["mode"] = "pan"; _start_pan(event); return
        if event.button == 1:
            near = _nearest_point(event.xdata, event.ydata)
            if near is not None:                   # left-drag on a label = move it
                state["mode"] = "drag_label"; state["drag"] = near; return
            # left on empty: click=place, drag=pan -> decide on motion/release
            state["mode"] = "pending"
            state["press_xy"] = (float(event.xdata), float(event.ydata))
            state["press_pix"] = (event.x, event.y)
            _start_pan(event)

    def on_motion(event):
        m = state.get("mode")
        if m == "drag_label" and event.inaxes is ax and event.xdata is not None:
            points[state["drag"]] = (float(event.xdata), float(event.ydata)); draw(fast=True)
        elif m == "pan":
            _do_pan(event)
        elif m == "pending" and event.x is not None:
            dx = event.x - state["press_pix"][0]; dy = event.y - state["press_pix"][1]
            if (dx * dx + dy * dy) ** 0.5 > 5:     # moved -> it's a pan
                state["mode"] = "pan"; _do_pan(event)

    def on_release(event):
        m = state.get("mode")
        if m == "drag_label":
            draw()
        elif m == "pending":                       # released without dragging = place
            points[state["place"]] = state["press_xy"]; draw()
        state["mode"] = None; state["drag"] = None

    def on_scroll(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        scale = 1 / 1.2 if event.step > 0 else 1.2
        x, y = event.xdata, event.ydata
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        ax.set_xlim(x - (x - x0) * scale, x + (x1 - x) * scale)
        ax.set_ylim(y - (y - y0) * scale, y + (y1 - y) * scale)
        fig.canvas.draw_idle()

    def reset_view(_=None):
        ax.set_xlim(-0.5, reader.W - 0.5); ax.set_ylim(reader.H - 0.5, -0.5)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("scroll_event", on_scroll)

    # sliders (left margin leaves room for labels)
    ax_t = fig.add_axes([0.12, 0.13, 0.41, 0.03])
    s_t = Slider(ax_t, "time (pair)", 0, max(1, reader.npairs - 1), valinit=state["pair"], valstep=1)
    s_t.on_changed(lambda v: (state.update(pair=int(v)), refresh_image()))
    ax_a = fig.add_axes([0.12, 0.08, 0.41, 0.03])
    s_a = Slider(ax_a, "avg frames", 1, 400, valinit=state["navg"], valstep=1)
    s_a.on_changed(lambda v: (state.update(navg=int(v)), refresh_image()))

    ax_c = fig.add_axes([0.78, 0.82, 0.09, 0.11]); ax_c.set_title("channel", fontsize=9)
    r_c = RadioButtons(ax_c, ("415", "470"))
    r_c.on_clicked(lambda l: (state.update(channel=0 if l == "415" else 1), refresh_image()))
    ax_sw = fig.add_axes([0.56, 0.46, 0.19, 0.05])
    c_sw = CheckButtons(ax_sw, ["swap 415/470"], [False])
    c_sw.on_clicked(lambda l: (state.update(swap=not state["swap"]), refresh_image()))
    ax_tt = fig.add_axes([0.88, 0.82, 0.11, 0.11]); ax_tt.set_title("transform", fontsize=9)
    r_tt = RadioButtons(ax_tt, ("affine", "similarity"))
    r_tt.on_clicked(lambda l: (state.update(ttype=l), draw()))
    ax_l = fig.add_axes([0.78, 0.43, 0.21, 0.35]); ax_l.set_title("place landmark", fontsize=9)
    r_l = RadioButtons(ax_l, tuple(LM_NAMES))
    r_l.on_clicked(lambda l: state.update(place=l))

    ax_ov = fig.add_axes([0.78, 0.36, 0.10, 0.05])
    co = CheckButtons(ax_ov, ["overlay"], [True])
    co.on_clicked(lambda l: (state.update(overlay=not state["overlay"]), draw()))
    ax_cmp = fig.add_axes([0.89, 0.36, 0.10, 0.05])
    cc = CheckButtons(ax_cmp, ["compare"], [False])
    cc.on_clicked(lambda l: (state.update(compare=not state["compare"]), draw()))

    ax_lab = fig.add_axes([0.83, 0.29, 0.16, 0.045])
    tb = TextBox(ax_lab, "label ", initial="v1")
    ax_save = fig.add_axes([0.78, 0.235, 0.10, 0.05]); b_save = Button(ax_save, "Save")
    ax_clear = fig.add_axes([0.89, 0.235, 0.10, 0.05]); b_clear = Button(ax_clear, "Clear")
    ax_load = fig.add_axes([0.78, 0.17, 0.10, 0.05]); b_load = Button(ax_load, "Load")
    ax_del = fig.add_axes([0.89, 0.17, 0.10, 0.05]); b_del = Button(ax_del, "Delete pt")
    msg = fig.text(0.78, 0.02, "", fontsize=7, wrap=True)

    def do_save(_):
        lab = (tb.text or "v1").strip() or "v1"
        fn = os.path.join(sess_dir, f"dorsal_cortex_landmarks_{lab}.json")
        try:
            fn, tf, eff, ns = save_landmarks_json(points, fn, state["ttype"])
            msg.set_text("Saved (%s, %d pts):\n%s" % (eff, len(ns), fn)); print("[allen_register] saved", fn, eff, ns)
        except Exception as e:
            msg.set_text("Save failed: %s" % e)
        fig.canvas.draw_idle()

    def load_into(fn):
        import json
        with open(fn) as fh:
            d = json.load(fh)
        lm = d.get("landmarks_match") or d.get("landmarks_im")  # clicked image coords
        if not lm or "name" not in lm or "x" not in lm:
            raise ValueError("file has no landmarks_match with names")
        points.clear()
        for nm, x, y in zip(lm["name"], lm["x"], lm["y"]):
            if nm in LM_MM:
                points[nm] = (float(x), float(y))
        tt = d.get("transform_type")
        if tt in ("affine", "similarity"):
            state["ttype"] = tt
            try:
                r_tt.set_active(("affine", "similarity").index(tt))
            except Exception:
                pass
        mlab = re.match(r"dorsal_cortex_landmarks_(.+)\.json$", os.path.basename(fn))
        if mlab:
            try:
                tb.set_val(mlab.group(1))
            except Exception:
                pass
        draw()
        msg.set_text("Loaded %d pts (%s)\n%s" % (len(points), tt or "?", os.path.basename(fn)))

    def do_load(_):
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            fn = filedialog.askopenfilename(
                initialdir=sess_dir, title="Load landmark JSON",
                filetypes=[("landmark json", "*.json"), ("all", "*.*")])
            root.destroy()
            if fn:
                load_into(fn)
        except Exception as e:
            msg.set_text("Load failed: %s" % e)
        fig.canvas.draw_idle()

    def delete_point(name):
        if name in points:
            del points[name]; draw(); msg.set_text("deleted " + name)
            fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ("delete", "backspace"):
            tgt = None
            if event.inaxes is ax and event.xdata is not None:
                tgt = _nearest_point(event.xdata, event.ydata)
            delete_point(tgt or state["place"])  # nearest under cursor, else selected

    b_save.on_clicked(do_save)
    b_clear.on_clicked(lambda _: (points.clear(), draw(), msg.set_text("cleared")))
    b_load.on_clicked(do_load)
    b_del.on_clicked(lambda _: delete_point(state["place"]))
    fig.canvas.mpl_connect("key_press_event", on_key)
    ax_reset = fig.add_axes([0.12, 0.02, 0.14, 0.035]); b_reset = Button(ax_reset, "Reset view")
    b_reset.on_clicked(reset_view)

    keep.extend([s_t, s_a, r_c, c_sw, r_tt, r_l, co, cc, tb,
                 b_save, b_clear, b_load, b_del, b_reset])
    fig._allen_widgets = keep  # extra strong ref on the figure
    refresh_image()
    fig.text(0.78, 0.10, "compare = cyan(similarity) + magenta(affine)\n"
             "Del key = remove pt under cursor\nSave dir:\n" + sess_dir, fontsize=7)
    plt.show()


def make_guide(path=None, show=False):
    """Save (and optionally show) a standalone Allen landmark placement guide."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ccf_regions, _, _ = allen_load_reference("dorsal_cortex")
    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_reference_guide(ax, ccf_regions)
    ax.set_title("Allen dorsal-cortex landmarks (suggested placements)")
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "allen_landmark_guide.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print("[allen_register] guide saved:", path)
    if show:
        plt.show()
    return path


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
        # load round-trip: landmarks_match -> points must match what we saved
        lm = d["landmarks_match"]
        reloaded = {nm: (float(x), float(y)) for nm, x, y in zip(lm["name"], lm["x"], lm["y"]) if nm in LM_MM}
        assert set(reloaded) == set(pts), (set(pts) - set(reloaded))
        for n in pts:
            assert abs(reloaded[n][0] - pts[n][0]) < 1e-6 and abs(reloaded[n][1] - pts[n][1]) < 1e-6
        print(f"saved+reloaded {ttype} JSON OK ({len(reloaded)} pts) keys={sorted(d)}")
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
