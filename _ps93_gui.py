"""Interactive review GUI for PS93 2026-08-05 recovered spout positions (dead strobe bit1 +
empty behavior log -> recovered from cam1 head-on video). Shows the AMBIGUOUS trials only
(deg0 -> close_center/close_R, deg1 -> close_L/far_center), sorted closest-to-threshold FIRST,
each with its cam1 frame + detected spout-x + the 6 reference positions. Accept or override
per trial; on save writes a reviewed trials.csv + a changes log so the maps can be re-run.

Run locally in the wfield env (needs a display):
    set PYTHONPATH=C:\\Github\\Widefield_DAQ_recorder
    C:\\ProgramData\\anaconda3\\envs\\wfield\\python.exe _ps93_gui.py
Smoke-test without opening a window (verifies data + frame extraction, writes a preview PNG):
    ... _ps93_gui.py --selftest

Keys:  1 = first candidate   2 = second candidate   Right/Enter = accept + next
       Left = previous        s = save                q = save + quit
"""
import os, sys, subprocess, json, csv
import numpy as np
import imageio_ffmpeg

OUT = r"C:\Github\Widefield_DAQ_recorder\_ps93dis"
AVI = r"N:\MICROSCOPE\Priya\Behavior_cameras\widefield\20260805\PS93\cam1_2026-08-05T20_20_03.avi"
FF = imageio_ffmpeg.get_ffmpeg_exe(); FPS = 250.0; W = H = 600
NM = {0: "close_center", 1: "close_L", 2: "close_R", 3: "far_center", 4: "far_L", 5: "far_R"}
CAND = {0: (0, 2), 1: (1, 3)}          # DAQ code -> (candidate A, candidate B) for the split
OFFSET_S = 0.6                          # sample the frame 0.6 s after the cue (spout down)

d = np.load(os.path.join(OUT, "ps93_autolabels.npz"))
XS, DC, TRUE, CONF, CF = d["xs"], d["dc"], d["true"], d["conf"], d["camframe"]
THR = {0: float(d["thr0"]), 1: float(d["thr1"])}
REVIEWED = TRUE.copy().astype(int)

_cache = {}
def grab(camframe):
    if camframe in _cache: return _cache[camframe]
    t = camframe / FPS + OFFSET_S
    p = subprocess.run([FF, "-nostdin", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", AVI,
                        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True)
    img = (np.frombuffer(p.stdout[:W*H], np.uint8).reshape(H, W) if len(p.stdout) >= W*H
           else np.zeros((H, W), np.uint8))
    _cache[camframe] = img; return img

# ambiguous trials (deg0/deg1), sorted closest-to-threshold first
AMB = np.where(np.isin(DC, [0, 1]) & (TRUE >= 0))[0]
AMB = AMB[np.argsort(CONF[AMB])]

def ref_frames():
    """PS93's own cleanest example per position (max-conf for split codes; median for far_L/R)."""
    refs = {}
    for k in range(6):
        idx = np.where(TRUE == k)[0]
        if len(idx) == 0: continue
        i = idx[np.argmax(CONF[idx])] if k in (0, 1, 2, 3) else idx[len(idx)//2]
        refs[k] = (grab(CF[i]), XS[i])
    return refs

def save_review():
    csvp = os.path.join(OUT, "ps93_reviewed_trials.csv")
    with open(csvp, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pos_idx", "pos_name"])
        for c in REVIEWED: w.writerow([int(c), NM[int(c)]])
    changes = [{"trial": int(i), "from": NM[int(TRUE[i])], "to": NM[int(REVIEWED[i])]}
               for i in range(len(TRUE)) if REVIEWED[i] != TRUE[i]]
    json.dump({"n_changes": len(changes), "changes": changes},
              open(os.path.join(OUT, "ps93_review_changes.json"), "w"), indent=2)
    print(f"[saved] {csvp}  ({len(changes)} change(s) vs auto-labels)")
    for c in changes: print(f"    trial {c['trial']}: {c['from']} -> {c['to']}")
    return len(changes)

def selftest():
    print(f"ambiguous trials: {len(AMB)} (closest-first). thr0={THR[0]:.0f} thr1={THR[1]:.0f}")
    refs = ref_frames()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 6, figsize=(19, 7))
    for j, k in enumerate(range(6)):
        a = ax[0, j]
        if k in refs:
            im, x = refs[k]; a.imshow(im, cmap="gray"); a.axvline(x, c="lime", lw=1)
            a.set_title(f"REF {NM[k]}", fontsize=9)
        a.axis("off")
    for j, i in enumerate(AMB[:6]):
        a = ax[1, j]; a.imshow(grab(CF[i]), cmap="gray")
        a.axvline(XS[i], c="orange", lw=1); a.axvline(THR[DC[i]], c="cyan", ls=":", lw=1)
        a.set_title(f"tr{i} deg{DC[i]}->{NM[TRUE[i]]}\nx={XS[i]:.0f} m={CONF[i]:.0f}", fontsize=8)
        a.axis("off")
    fig.suptitle("PS93 8/5 GUI self-test — top: references, bottom: 6 closest ambiguous calls")
    plt.tight_layout(); pp = os.path.join(OUT, "ps93_gui_selftest.png"); plt.savefig(pp, dpi=100)
    print(f"[selftest] wrote {pp} — plumbing OK; run without --selftest for the interactive GUI")

def gui():
    import matplotlib; matplotlib.use("TkAgg"); import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    refs = ref_frames()
    state = {"pos": 0}
    fig = plt.figure(figsize=(15, 8.5))
    try: fig.canvas.manager.set_window_title("PS93 8/5 spout-position review")
    except Exception:
        try: fig.canvas.set_window_title("PS93 8/5 spout-position review")
        except Exception: pass
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 3, 0.5], hspace=0.25, wspace=0.15)
    ref_ax = [fig.add_subplot(gs[0, j]) for j in range(6)]
    main_ax = fig.add_subplot(gs[1, :4]); info_ax = fig.add_subplot(gs[1, 4:]); info_ax.axis("off")
    for j, k in enumerate(range(6)):
        ref_ax[j].axis("off")
        if k in refs: ref_ax[j].imshow(refs[k][0], cmap="gray"); ref_ax[j].axvline(refs[k][1], c="lime", lw=1)
        ref_ax[j].set_title(NM[k], fontsize=9)

    def draw():
        i = AMB[state["pos"]]; code = int(DC[i]); a, b = CAND[code]
        main_ax.clear(); main_ax.imshow(grab(CF[i]), cmap="gray")
        main_ax.axvline(XS[i], c="orange", lw=1.5, label=f"detected x={XS[i]:.0f}")
        main_ax.axvline(THR[code], c="cyan", ls=":", lw=1.2, label=f"threshold={THR[code]:.0f}")
        main_ax.legend(loc="upper right", fontsize=8); main_ax.set_xticks([]); main_ax.set_yticks([])
        main_ax.set_title(f"ambiguous {state['pos']+1}/{len(AMB)}  |  trial {i}  deg{code}  "
                          f"margin={CONF[i]:.0f}px", fontsize=11)
        cur = int(REVIEWED[i]); auto = int(TRUE[i])
        info_ax.clear(); info_ax.axis("off")
        lines = [f"DAQ code deg{code}  ->  candidates:", "",
                 f"  [1] {NM[a]}" + ("   <-- current" if cur == a else ""),
                 f"  [2] {NM[b]}" + ("   <-- current" if cur == b else ""), "",
                 f"auto-label: {NM[auto]}",
                 f"current:    {NM[cur]}" + ("  (CHANGED)" if cur != auto else ""), "",
                 "keys: 1/2 assign  |  ->/Enter accept+next",
                 "      <- back  |  s save  |  q save+quit",
                 "", f"changes so far: {int((REVIEWED != TRUE).sum())}"]
        info_ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace")
        for j, k in enumerate(range(6)):
            ref_ax[j].patch.set_edgecolor("red" if k in (a, b) else "none")
            ref_ax[j].patch.set_linewidth(3 if k in (a, b) else 0)
        fig.canvas.draw_idle()

    def on_key(ev):
        i = AMB[state["pos"]]; code = int(DC[i]); a, b = CAND[code]
        if ev.key in ("1",): REVIEWED[i] = a; draw()
        elif ev.key in ("2",): REVIEWED[i] = b; draw()
        elif ev.key in ("right", "enter", " "):
            state["pos"] = min(state["pos"]+1, len(AMB)-1); draw()
        elif ev.key == "left": state["pos"] = max(state["pos"]-1, 0); draw()
        elif ev.key == "s": save_review()
        elif ev.key == "q": save_review(); plt.close(fig)
    fig.canvas.mpl_connect("key_press_event", on_key)
    # on-screen buttons mirror the keys
    def mk(rect, label, cb):
        bax = fig.add_axes(rect); btn = Button(bax, label); btn.on_clicked(cb); return btn
    btns = [mk([0.10, 0.04, 0.12, 0.05], "1: cand A", lambda e: (REVIEWED.__setitem__(AMB[state["pos"]], CAND[int(DC[AMB[state["pos"]]])][0]), draw())),
            mk([0.23, 0.04, 0.12, 0.05], "2: cand B", lambda e: (REVIEWED.__setitem__(AMB[state["pos"]], CAND[int(DC[AMB[state["pos"]]])][1]), draw())),
            mk([0.40, 0.04, 0.10, 0.05], "< back", lambda e: (state.__setitem__("pos", max(state["pos"]-1, 0)), draw())),
            mk([0.51, 0.04, 0.14, 0.05], "accept + next >", lambda e: (state.__setitem__("pos", min(state["pos"]+1, len(AMB)-1)), draw())),
            mk([0.70, 0.04, 0.10, 0.05], "save", lambda e: save_review()),
            mk([0.81, 0.04, 0.10, 0.05], "save + quit", lambda e: (save_review(), plt.close(fig)))]
    fig._btns = btns  # keep refs alive
    draw(); plt.show()

if __name__ == "__main__":
    if "--selftest" in sys.argv: selftest()
    else: gui()
