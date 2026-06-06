"""Photobleaching for 2026-06-05: continuous (PS93/94/95) vs trial-triggered (PS92).

Reuses analyze() from _photobleach_batch (per-session 415/470 ROI-median trend from
the raw .dat + DAQ LED TTLs) and adds a recording-type-grouped summary to test
whether CONTINUOUS recording bleaches worse than TRIAL-TRIGGERED. Compares both
%-decline over the session and %/min rate (the fairer metric: continuous
illuminates 100% of wall-clock time, trial-triggered only during trials).

Outputs to _photobleach_out_0605\ : per-session PNGs + photobleach_0605_continuous_vs_trial.png
Run in the wfield env (h5py+scipy+matplotlib, no wfield import).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _photobleach_batch as pb

OUT = r"C:\Github\Widefield_DAQ_recorder\_photobleach_out_0605"
os.makedirs(OUT, exist_ok=True)
pb.OUT = OUT  # redirect analyze()'s per-session PNGs here

D = r"E:\labcams_data\20260605"
Q = r"E:\DAQ_recorder_output\20260605"
DAT = r"raw_widefield_data\pco_edge_run000_00000000_2_460_480_uint16.dat"

# (label, recording_type, dat, daq)
SESSIONS = [
    ("PS92_0605", "trial-triggered", fr"{D}\PS92_20260605_125023\{DAT}", fr"{Q}\PS92_20260605_125301.h5"),
    ("PS93_0605", "continuous",      fr"{D}\PS93_20260605_174659\{DAT}", fr"{Q}\PS93_20260605_175452.h5"),
    ("PS94_0605", "continuous",      fr"{D}\PS94_20260605_142009\{DAT}", fr"{Q}\PS94_20260605_142249.h5"),
    ("PS95_0605", "continuous",      fr"{D}\PS95_20260605_163102\{DAT}", fr"{Q}\PS95_20260605_163405.h5"),
]
TYPE_COL = {"trial-triggered": "#d1701a", "continuous": "#1f6fd1"}


def main():
    results = []
    for label, rtype, dat, daq in SESSIONS:
        r = pb.analyze(label, dat, daq)
        if r:
            r["rtype"] = rtype
            results.append(r)
    if not results:
        print("no sessions analyzed"); return 1

    fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
    # panels 1-2: normalized 470 / 415 trends, colored by recording type
    for ci, name in enumerate(["470", "415"]):
        seen = set()
        for r in results:
            if name not in r.get("_norm", {}):
                continue
            x, y = r["_norm"][name]
            rt = r["rtype"]
            ax[ci].plot(np.array(x) / 60.0, y, "-", lw=1.8, color=TYPE_COL[rt],
                        alpha=0.9, label=rt if rt not in seen else None)
            seen.add(rt)
            ax[ci].annotate(r["label"].split("_")[0], (np.array(x)[-1] / 60.0, y[-1]),
                            fontsize=7, color=TYPE_COL[rt])
        ax[ci].axhline(1.0, color="k", lw=0.6)
        ax[ci].set_xlabel("time since start (min)"); ax[ci].set_ylabel("normalized intensity")
        ax[ci].set_title(f"{name} nm normalized trend  (by recording type)"); ax[ci].legend(fontsize=9)

    # panel 3: 470 %/min = (%-over-session / duration) -- duration-normalized, fair
    # across the very different session lengths (PS93 ran ~2x longer).
    def rate_pm(r):
        c = r["channels"].get("470", {})
        return c["pct"] / r["dur_min"] if ("pct" in c and r["dur_min"]) else np.nan
    labels = [r["label"].split("_")[0] for r in results]
    rates470 = [rate_pm(r) for r in results]
    cols = [TYPE_COL[r["rtype"]] for r in results]
    x = np.arange(len(results))
    ax[2].bar(x, rates470, 0.6, color=cols)
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set_xticks(x); ax[2].set_xticklabels(labels)
    ax[2].set_ylabel("470 nm drift (%/min = %over-session / duration)")
    ax[2].set_title("Functional-channel (470) bleaching rate")
    for r, rt470 in zip(results, rates470):
        ax[2].annotate(f"{rt470:+.3f}", (x[results.index(r)], rt470), ha="center",
                       va="bottom" if rt470 >= 0 else "top", fontsize=7)
    # per-type mean lines
    for rt, col in TYPE_COL.items():
        vals = [rate_pm(r) for r in results if r["rtype"] == rt]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            ax[2].axhline(np.mean(vals), color=col, ls="--", lw=1.2,
                          label=f"{rt} mean {np.mean(vals):+.3f}%/min")
    ax[2].legend(fontsize=8)
    plt.suptitle("2026-06-05 photobleaching: continuous (PS93/94/95) vs trial-triggered (PS92)", fontsize=13)
    plt.tight_layout()
    fp = os.path.join(OUT, "photobleach_0605_continuous_vs_trial.png")
    plt.savefig(fp, dpi=130); plt.close(fig)
    print("saved", fp)

    print("\n=== 6/5 drift: %over session and duration-normalized %/min, by type ===")
    print(f"{'session':10s} {'type':16s} {'min':>5s} {'415%':>7s} {'470%':>7s} {'470%/min':>9s}")
    for r in results:
        c = r["channels"]
        print(f"{r['label']:10s} {r['rtype']:16s} {r['dur_min']:5.1f} "
              f"{c.get('415',{}).get('pct',float('nan')):7.1f} "
              f"{c.get('470',{}).get('pct',float('nan')):7.1f} "
              f"{rate_pm(r):9.3f}")
    for rt in TYPE_COL:
        rr = [rate_pm(r) for r in results if r["rtype"] == rt and "470" in r["channels"]]
        rr = [v for v in rr if np.isfinite(v)]
        if rr:
            print(f"  {rt}: mean 470 rate {np.mean(rr):+.3f}%/min (n={len(rr)})")
    import json
    with open(os.path.join(OUT, "photobleach_0605_results.json"), "w") as fh:
        json.dump([{k: v for k, v in r.items() if k != "_norm"} for r in results], fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
