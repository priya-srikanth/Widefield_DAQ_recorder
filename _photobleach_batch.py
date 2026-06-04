"""Photobleaching across several sessions: per-session figures + one summary.

For each (label, DAT, DAQ): label every camera frame 415/470 from the DAQ LED
TTLs (DAQ = ground truth, so channel identity is correct regardless of parity),
restrict to a brain ROI, and track the time-binned MEDIAN intensity per channel.
Saves a per-session PNG and finally a combined summary (normalized trends +
%-drift bars). Pairs whose DAQ doesn't match the DAT (frame-count mismatch or
missing LED channels) are skipped with a printed warning.

Run with an env that has h5py+scipy+matplotlib but WITHOUT importing wfield
(avoids the wfield/h5py DLL clash):
    C:\\ProgramData\\anaconda3\\envs\\wfield\\python.exe _photobleach_batch.py
"""
import os, re, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion

OUT = r"C:\Github\Widefield_DAQ_recorder\_photobleach_out"
os.makedirs(OUT, exist_ok=True)
NSAMP, NB = 3000, 40
MIN_DUR_MIN = 15.0  # skip short baseline blips (unreliable trend)

SESSIONS = [
    ("PS92_0603", r"E:\labcams_data\20260603\PS92\PS92_20260603_104008\raw_widefield_data\pco_edge_run000_00000000_2_477_464_uint16.dat",
                  r"E:\DAQ_recorder_output\20250603\PS92_20260603_104607.h5"),
    ("PS94_0603", r"E:\labcams_data\20260603\PS94\raw_widefield_data\pco_edge_run000_00000000_2_462_464_uint16.dat",
                  r"E:\DAQ_recorder_output\20250603\PS94_20260603_175946.h5"),
    ("PS95_0603", r"E:\labcams_data\20260603\PS95\PS95\PS95_20260603_194442\raw_widefield_data\pco_edge_run000_00000000_2_462_464_uint16.dat",
                  r"E:\DAQ_recorder_output\20250603\PS95_20260603_194902.h5"),
    ("PS92_0602", r"E:\labcams_data\20260602\PS92\PS92_20260602_151820\raw_widefield_data\pco_edge_run001_00000000_2_487_480_uint16.dat",
                  r"E:\DAQ_recorder_output\20250602\PS92_20260602_152607.h5"),
    ("PS94_0601", r"E:\labcams_data\20260601\PS94_20260601_141614\pco_edge_run004_00000000_2_540_640_uint16.dat",
                  r"E:\DAQ_recorder_output\PS94_baseline_20260601_141642.h5"),
    ("PS95_0601", r"E:\labcams_data\20260601\PS95_20260601_153653\pco_edge_run000_00000000_2_540_640_uint16.dat",
                  r"E:\DAQ_recorder_output\PS95_baseline_20260601_153627.h5"),
    ("PS92_0601", r"E:\labcams_data\20260601\PS92_20260601\pco_edge_run000_00000000_2_540_640_uint16.dat",
                  r"E:\DAQ_recorder_output\PS92_baseline_20260601_122510.h5"),
]

COL = {"415": "violet", "470": "royalblue"}


def analyze(label, dat, daq):
    if not (os.path.exists(dat) and os.path.exists(daq)):
        print(f"[{label}] SKIP missing file"); return None
    m = re.search(r"_(\d+)_(\d+)_(\d+)_uint16", os.path.basename(dat))
    H, W = int(m.group(2)), int(m.group(3))
    import h5py
    with h5py.File(daq, "r") as f:
        fs = float(f.attrs["sample_rate_hz"])
        an = [s.decode() for s in f["analog/channel_names"][:]]
        di = [s.decode() for s in f["digital/channel_names"][:]]
        if not {"led415_ttl", "led470_ttl"} <= set(an) or "pco_exposure" not in di:
            print(f"[{label}] SKIP DAQ missing LED/pco channels"); return None
        packed = f["digital/packed_samples"][:, 0]
        sc = f["analog/int16_scale_volts_per_count"][:]; of = f["analog/int16_offset_volts"][:]
        def ac(n): i = an.index(n); return f["analog/samples_int16"][:, i].astype(np.float32) * sc[i] + of[i]
        led415, led470 = ac("led415_ttl"), ac("led470_ttl")

    def db(b): return (packed >> b) & 1
    def rises(x, t=0.5): b = (np.asarray(x) > t).astype(np.int8); return np.flatnonzero(np.diff(b) == 1) + 1
    pco = rises(db(di.index("pco_exposure")))
    i2 = np.clip(pco + int(0.002 * fs), 0, len(packed) - 1)
    code = np.where((led415[i2] > 1.5) & ~(led470[i2] > 1.5), 415,
            np.where((led470[i2] > 1.5) & ~(led415[i2] > 1.5), 470, 0))

    nphys = os.path.getsize(dat) // (H * W * 2)
    if len(pco) == 0 or abs(len(pco) - nphys) / max(nphys, 1) > 0.2:
        print(f"[{label}] SKIP frame/pco mismatch (dat={nphys} pco={len(pco)})"); return None
    mm = np.memmap(dat, mode="r", dtype=np.uint16, shape=(nphys, H, W))
    n = min(nphys, len(pco))
    if (((pco / fs)[min(n - 1, len(pco) - 1)] - (pco / fs)[0]) / 60.0) < MIN_DUR_MIN:
        print(f"[{label}] SKIP too short (<{MIN_DUR_MIN:.0f} min)"); return None
    samp = np.linspace(0, n - 1, min(NSAMP, n)).astype(int)
    avg = np.zeros((H, W), np.float64)
    for k in samp[::8]: avg += mm[k]
    avg /= len(samp[::8])
    P = binary_erosion(avg > (0.45 * avg.max()), iterations=6).ravel()
    if P.sum() < 200:
        print(f"[{label}] SKIP tiny ROI ({int(P.sum())} px)"); return None

    roi = np.array([mm[k].reshape(-1)[P].mean() for k in samp], np.float64)
    lab = code[np.clip(samp, 0, len(code) - 1)]
    t = (pco / fs)[np.clip(samp, 0, len(pco) - 1)]
    dur_min = (t.max() - t.min()) / 60.0

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    edges = np.linspace(t.min(), t.max(), NB + 1); ctr = 0.5 * (edges[:-1] + edges[1:])
    res = {"label": label, "dat": dat, "daq": daq, "n_frames": int(nphys),
           "dur_min": float(dur_min), "roi_px": int(P.sum()), "channels": {}}
    norm = {}
    for c, name in [(415, "415"), (470, "470")]:
        msk = lab == c
        if msk.sum() < 30: continue
        tt, vv = t[msk], roi[msk]
        bmed = np.array([np.median(vv[(tt >= edges[i]) & (tt < edges[i + 1])])
                         if np.any((tt >= edges[i]) & (tt < edges[i + 1])) else np.nan for i in range(NB)])
        good = np.isfinite(bmed)
        p = np.polyfit(ctr[good], bmed[good], 1)
        start, end = np.polyval(p, ctr[good][0]), np.polyval(p, ctr[good][-1])
        pct = 100 * (end - start) / start
        res["channels"][name] = dict(median=float(np.median(vv)), pct=float(pct),
                                      per_min=float(p[0] * 60), start=float(start), end=float(end))
        ax[0].plot(tt, vv, ".", ms=2, color=COL[name], alpha=0.15)
        ax[0].plot(ctr[good], bmed[good], "-o", ms=4, lw=2, color=COL[name],
                   label=f"{name}  ({pct:+.1f}%)")
        nrm = bmed[good] / bmed[good][0]
        norm[name] = (ctr[good] - ctr[good][0], nrm)
        ax[1].plot(ctr[good] - ctr[good][0], nrm, "-o", ms=4, lw=2, color=COL[name], label=name)
        print(f"[{label}] {name}: median={np.median(vv):.0f} drift={pct:+.1f}% ({p[0]*60:+.1f}/min)")
    ax[0].set_xlabel("session time (s)"); ax[0].set_ylabel("brain-ROI mean"); ax[0].legend()
    ax[0].set_title(f"{label}  ROI intensity (binned median + fit)")
    ax[1].axhline(1.0, color="k", lw=0.6)
    allnrm = np.concatenate([y for _, y in norm.values()]) if norm else np.array([1.0])
    pad = max(0.01, 0.05 * (allnrm.max() - allnrm.min()))
    ax[1].set_ylim(allnrm.min() - pad, max(allnrm.max() + pad, 1.0 + pad))
    ax[1].set_xlabel("time since start (s)"); ax[1].set_ylabel("normalized (median / first bin)")
    ax[1].legend(); ax[1].set_title(f"{label}  normalized trend  ({dur_min:.1f} min)")
    plt.tight_layout()
    fp = os.path.join(OUT, f"photobleach_{label}.png"); plt.savefig(fp, dpi=120); plt.close(fig)
    print(f"[{label}] saved {fp}")
    res["_norm"] = {k: (x.tolist(), y.tolist()) for k, (x, y) in norm.items()}
    return res


def summary(results):
    results = [r for r in results if r]
    if not results:
        print("no sessions analyzed"); return
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
    cmap = plt.cm.tab10(np.linspace(0, 1, len(results)))
    # shared y-range across both channel panels so they stay comparable AND fit data
    ally = [v for r in results for nm in ("470", "415")
            for v in r.get("_norm", {}).get(nm, ([], []))[1]]
    ally = np.array(ally) if ally else np.array([1.0])
    pad = max(0.01, 0.05 * (ally.max() - ally.min()))
    ylo, yhi = ally.min() - pad, max(ally.max() + pad, 1.0 + pad)
    for ci, name in enumerate(["470", "415"]):
        for r, col in zip(results, cmap):
            if name in r.get("_norm", {}):
                x, y = r["_norm"][name]
                ax[ci].plot(np.array(x) / 60.0, y, "-", lw=1.8, color=col, label=r["label"])
        ax[ci].axhline(1.0, color="k", lw=0.6); ax[ci].set_ylim(ylo, yhi)
        ax[ci].set_xlabel("time since start (min)"); ax[ci].set_ylabel("normalized intensity")
        ax[ci].set_title(f"{name} nm  normalized trend (all sessions)"); ax[ci].legend(fontsize=8)
    labels = [r["label"] for r in results]
    x = np.arange(len(labels)); w = 0.38
    p415 = [r["channels"].get("415", {}).get("pct", np.nan) for r in results]
    p470 = [r["channels"].get("470", {}).get("pct", np.nan) for r in results]
    ax[2].bar(x - w / 2, p415, w, color="violet", label="415")
    ax[2].bar(x + w / 2, p470, w, color="royalblue", label="470")
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set_xticks(x); ax[2].set_xticklabels(labels, rotation=45, ha="right")
    ax[2].set_ylabel("% change over session (linear fit)")
    ax[2].set_title("Per-channel drift by session"); ax[2].legend()
    plt.suptitle("Photobleaching summary across sessions  (415 isosbestic vs 470 functional)", fontsize=13)
    plt.tight_layout()
    fp = os.path.join(OUT, "photobleach_SUMMARY.png"); plt.savefig(fp, dpi=130); plt.close(fig)
    print("saved", fp)
    print("\n=== drift table (% over session) ===")
    print(f"{'session':12s} {'min':>5s} {'415%':>8s} {'470%':>8s}")
    for r in results:
        c = r["channels"]
        print(f"{r['label']:12s} {r['dur_min']:5.1f} "
              f"{c.get('415',{}).get('pct',float('nan')):8.1f} {c.get('470',{}).get('pct',float('nan')):8.1f}")


if __name__ == "__main__":
    out = [analyze(*s) for s in SESSIONS]
    summary(out)
    with open(os.path.join(OUT, "photobleach_results.json"), "w") as fh:
        json.dump([{k: v for k, v in r.items() if k != "_norm"} for r in out if r], fh, indent=2)
    print("done ->", OUT)
