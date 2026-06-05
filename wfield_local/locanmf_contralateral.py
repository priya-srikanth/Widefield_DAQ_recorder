"""Contralateral somatosensory tuning of LocaNMF SSp components (validates vs SVD maps).

The pixel/SVD post-lick maps show more somatosensory activation for CONTRALATERAL-spout
licks. This checks the same in LocaNMF component space: for each SSp component, the
post-lick 150 ms response by spout position, split by hemisphere, and a contralateral index
(contra-minus-ipsi). Because it is a WITHIN-component contrast across positions, it is
immune to the per-component normalization (the quiet-SD problem that biased cross-animal
magnitudes does not affect a within-component position comparison).

Reads the lick-aligned npz (per-position 150 ms-post means). Writes a bar figure
(SSp area x hemisphere, positions; contralateral spouts highlighted) + prints the index.

    python -m wfield_local.locanmf_contralateral --root "M:/.../labcams" --output "<dir>"
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

POS = ["close_L", "close_center", "close_R", "far_L", "far_center", "far_R"]
SSP = {5: "SSp-n", 6: "SSp-m"}  # +label = left hemi, -label = right hemi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="M:/MICROSCOPE/Priya/Widefield/labcams")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--post-ms", type=float, default=150.0)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    npzs = sorted(glob.glob(f"{args.root}/**/locanmf_lick_aligned_affine8v1/*_locanmf_lick_aligned.npz",
                            recursive=True))
    rows = []  # (area, hemi, animal, per-position 150ms response dict)
    for f in npzs:
        z = np.load(f); t = z["time"]; reg = z["regions"]
        cnt = json.loads(Path(f.replace(".npz", "_summary.json")).read_text())["counts_by_position"]
        pm = (t >= 0) & (t <= args.post_ms / 1000.0)
        animal = Path(f).name.split("_")[0]
        for k, nm in SSP.items():
            for lab, hemi in [(k, "L"), (-k, "R")]:
                for i in np.where(reg == lab)[0]:
                    rows.append((nm, hemi, animal,
                                 {p: (z[f"{p}_mean"][i][pm].mean() if cnt.get(p, 0) > 0 else np.nan) for p in POS}))

    def contra_index(hemi, v):
        lsp = np.nanmean([v["close_L"], v["far_L"]]); rsp = np.nanmean([v["close_R"], v["far_R"]])
        return (rsp - lsp) if hemi == "L" else (lsp - rsp)

    print("contralateral index (contra-ipsi, post-lick 150ms):")
    for nm in ["SSp-n", "SSp-m"]:
        for hemi in ["L", "R"]:
            ci = np.array([contra_index(hemi, v) for (a, h, an, v) in rows if a == nm and h == hemi])
            ci = ci[np.isfinite(ci)]
            print(f"  {nm}_{hemi}: mean={ci.mean():+.3f}  n={ci.size}  frac>0={100*np.mean(ci>0):.0f}%")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for r, nm in enumerate(["SSp-n", "SSp-m"]):
        for c, hemi in enumerate(["L", "R"]):
            ax = axes[r][c]
            m, e = [], []
            for p in POS:
                vv = [v[p] for (a, h, an, v) in rows if a == nm and h == hemi and np.isfinite(v[p])]
                m.append(np.mean(vv) if vv else 0); e.append(np.std(vv) / np.sqrt(len(vv)) if vv else 0)
            colors = ["tab:red" if (("R" in p and hemi == "L") or ("L" in p and hemi == "R")) else "tab:grey"
                      for p in POS]
            ax.bar(range(6), m, yerr=e, color=colors, capsize=2)
            ax.set_xticks(range(6)); ax.set_xticklabels(POS, rotation=45, ha="right", fontsize=7)
            ax.set_title(f"{nm}_{hemi}  (red = contralateral spout)", fontsize=10)
            ax.set_ylabel(f"{args.post_ms:.0f}ms post-lick (quiet z)", fontsize=8); ax.axhline(0, color="k", lw=0.5)
    fig.suptitle("LocaNMF SSp components: post-lick by spout position, by hemisphere (contralateral tuning)",
                 fontsize=13)
    fig.tight_layout()
    png = args.output / "locanmf_SSp_contralateral.png"
    fig.savefig(png, dpi=130); plt.close(fig)
    print("wrote", png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
