"""Cross-session generalization of the position DECODER and ENCODER, on a common Allen-ROI basis.

Tests the FROZEN pre-stroke-model premise directly: train a model on session i, apply it to session j
(within animal), for all baseline-day pairs. The off-diagonal (cross-session) vs diagonal (within-session,
held-out) accuracy gap = the cost of freezing one model across days -- i.e. how much day-to-day drift
would degrade a frozen pre->post decoder/encoder. Uses Allen-ROI features (atlas-anchored, identical
dimensions across sessions, matched by region label) because LocaNMF components are not shared across days.

  decoder  : multinomial logistic regression (position classifier). transfer = train-i -> classify-j.
  encoder  : ridge position->ROI-activity; templates B (6 x nROI) = expected pattern per position.
             transfer = classify j's trials by NEAREST encoder-i template (a generative/prototype readout,
             directly comparable to the decoder accuracy).
  diagonal : held-out within-session block-CV for both (fair reference, not in-sample).

    python -m wfield_local.locanmf_xsession_generalization --output "<dir>" [--align lick|cue]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score

from wfield_local.locanmf_cue_lick_analysis import SESSIONS
from wfield_local.locanmf_position_decoder import _trial_features
from wfield_local.plot_lick_aligned_averages import POSITION_NAMES, DISPLAY_ORDER

FS = 31.23
ANIMALS = ["PS92", "PS93", "PS94", "PS95"]
DATES = ["0605", "0606", "0607", "0608"]
DLAB = {"0605": "6/5", "0606": "6/6", "0607": "6/7", "0608": "6/8"}
POSN = [POSITION_NAMES[c] for c in DISPLAY_ORDER]


def _roi(label, align):
    s = next(x for x in SESSIONS if x["label"] == label)
    a = SimpleNamespace(source="roi", align=align, baseline="none", cv="block",
                        fs=FS, pre_s=1.0, post_s=2.0, max_rt=2.0)
    X, y, g, _, _, reg = _trial_features(s, a)
    return X, y, np.asarray(g), np.asarray(reg)


def _onehot(y):
    return np.stack([(y == p).astype(float) for p in DISPLAY_ORDER], 1)


def _templates(Xs, y):                                   # (6, nfeat) expected pattern per position
    return Ridge(alpha=1.0).fit(_onehot(y), Xs).coef_.T


def _match(Xs, B):                                       # nearest-template (prototype) classification
    d = ((Xs[:, None, :] - B[None, :, :]) ** 2).sum(2)
    return np.array(DISPLAY_ORDER)[d.argmin(1)]


def _diag_cv(X, y, g, kind):
    """Held-out within-session accuracy (block-CV) for decoder or encoder-template readout."""
    ub = np.unique(g); k = min(5, len(ub))
    if k < 2:
        return np.nan
    yhat = np.empty_like(y)
    for f in range(k):
        te = np.isin(g, ub[f::k]); tr = ~te
        if tr.sum() < 6 or te.sum() < 1 or len(np.unique(y[tr])) < 2:
            yhat[te] = -1; continue
        sc = StandardScaler().fit(X[tr])
        if kind == "decoder":
            clf = LogisticRegression(C=0.5, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
            yhat[te] = clf.predict(sc.transform(X[te]))
        else:
            B = _templates(sc.transform(X[tr]), y[tr])
            yhat[te] = _match(sc.transform(X[te]), B)
    ok = yhat != -1
    return accuracy_score(y[ok], yhat[ok]) if ok.any() else np.nan


def _matrices(animal, align):
    """4x4 (train x test) accuracy matrices for decoder and encoder, on common ROIs across the animal's days."""
    data = {}
    for d in DATES:
        try:
            X, y, g, reg = _roi(f"{animal}_{d}", align)
            data[d] = (X, y, g, reg)
        except Exception as ex:
            print(f"  {animal}_{d} skip: {type(ex).__name__}: {str(ex)[:60]}", flush=True)
    days = [d for d in DATES if d in data]
    if len(days) < 2:
        return None
    common = set.intersection(*[set(data[d][3].tolist()) for d in days])
    common = sorted(common)
    idx = {d: np.array([list(data[d][3]).index(r) for r in common]) for d in days}
    n = len(days); Dm = np.full((n, n), np.nan); Em = np.full((n, n), np.nan)
    for i, di in enumerate(days):
        Xi, yi, gi, _ = data[di]; Xi = Xi[:, idx[di]]
        sc = StandardScaler().fit(Xi)
        clf = LogisticRegression(C=0.5, max_iter=2000).fit(sc.transform(Xi), yi)
        B = _templates(sc.transform(Xi), yi)
        for j, dj in enumerate(days):
            Xj, yj, gj, _ = data[dj]; Xj = Xj[:, idx[dj]]
            if i == j:
                Dm[i, j] = _diag_cv(Xi, yi, gi, "decoder")
                Em[i, j] = _diag_cv(Xi, yi, gi, "encoder")
            else:
                Xt = sc.transform(Xj)
                Dm[i, j] = accuracy_score(yj, clf.predict(Xt))
                Em[i, j] = accuracy_score(yj, _match(Xt, B))
    return days, common, Dm, Em


def _offdiag(M):
    n = M.shape[0]; off = [M[i, j] for i in range(n) for j in range(n) if i != j]
    return np.nanmean(off)


def fig_generalization(out, align="lick"):
    out = Path(out)
    res = {}
    for a in ANIMALS:
        r = _matrices(a, align)
        if r is not None:
            res[a] = r
            days, _, Dm, Em = r
            print(f"{a}: decoder diag={np.nanmean(np.diag(Dm)):.2f} off={_offdiag(Dm):.2f} | "
                  f"encoder diag={np.nanmean(np.diag(Em)):.2f} off={_offdiag(Em):.2f}", flush=True)
    # per-animal 2-heatmap figure
    for a, (days, common, Dm, Em) in res.items():
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))
        for ax, M, ttl in [(axes[0], Dm, "DECODER (logistic)"), (axes[1], Em, "ENCODER (template-match)")]:
            im = ax.imshow(M, vmin=1 / 6, vmax=1, cmap="viridis")
            ax.set_xticks(range(len(days))); ax.set_xticklabels([DLAB[d] for d in days])
            ax.set_yticks(range(len(days))); ax.set_yticklabels([DLAB[d] for d in days])
            ax.set_xlabel("TEST session"); ax.set_ylabel("TRAIN session")
            for i in range(len(days)):
                for j in range(len(days)):
                    v = M[i, j]
                    ax.text(j, i, f"{v:.2f}" if np.isfinite(v) else "", ha="center", va="center",
                            fontsize=9, color="white" if (not np.isfinite(v) or v < 0.62) else "black")
            ax.set_title(f"{ttl}\ndiag(within)={np.nanmean(np.diag(M)):.2f}  off(cross)={_offdiag(M):.2f}", fontsize=10)
            fig.colorbar(im, ax=ax, shrink=0.8)
        fig.suptitle(f"{a} — cross-session generalization on common Allen-ROI basis ({align}-aligned 2 s, "
                     f"n_ROI={len(common)}); chance=0.17. off≈diag => a frozen model transfers across days", fontsize=11)
        fig.tight_layout()
        p = out / f"locanmf_xsession_generalization_{a}.png"; fig.savefig(p, dpi=130); plt.close(fig)
        print("wrote", p.name, flush=True)
    # cross-animal summary: within (diag) vs cross (off-diag), decoder & encoder
    if res:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6)); x = np.arange(len(res)); w = 0.38
        an = list(res)
        for ax, mi, ttl in [(axes[0], 2, "DECODER"), (axes[1], 3, "ENCODER")]:
            diag = [np.nanmean(np.diag(res[a][mi])) for a in an]
            off = [_offdiag(res[a][mi]) for a in an]
            ax.bar(x - w / 2, diag, w, label="within-session (block-CV)", color="#2980b9")
            ax.bar(x + w / 2, off, w, label="cross-session (frozen transfer)", color="#c0392b")
            ax.axhline(1 / 6, color="k", ls="--", lw=0.8, label="chance")
            ax.set_xticks(x); ax.set_xticklabels(an); ax.set_ylim(0, 1); ax.set_ylabel("accuracy")
            ax.set_title(ttl); ax.legend(fontsize=8)
        fig.suptitle(f"Frozen-model feasibility: within- vs cross-session accuracy (Allen-ROI, {align} 2 s). "
                     f"Small gap => one pre-stroke model can be applied across sessions.", fontsize=11)
        fig.tight_layout()
        p = out / "locanmf_xsession_generalization_summary.png"; fig.savefig(p, dpi=130); plt.close(fig)
        print("wrote", p.name, flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--align", default="lick", choices=("lick", "cue"))
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    fig_generalization(args.output, args.align)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
