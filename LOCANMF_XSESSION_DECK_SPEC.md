# Cross-session deck (6/5 onward) — spec

Plan-of-record for the NEW PowerPoint deck Priya requested on 2026-06-10, built on the **freshly
re-registered** LocaNMF outputs (each session re-aligned to that animal's 6/6 image; 12-session
re-run, of which 10 actually re-LocaNMF'd — see below). Source-of-truth analysis docs unchanged:
`LOCANMF_LICK_CUE_ANALYSIS.md` (F1–F17), `LOCANMF_NIGHTLY_PIPELINE.md`. This file = how the deck is
organized and which figure each slide pulls.

## Data scope
- Dates: **6/5, 6/6, 6/7, 6/8 only** (no 6/1–6/4).
- Animals: PS92, PS93, PS94, PS95 (PS93 new from 6/5; RIGHT orofacial deficit).
- Re-registration re-run (2026-06-10): re-LocaNMF'd 10 sessions whose aligned inputs were newer than
  output — all 6/5 (×4) + PS92/PS93 6/6,6/7,6/8 (×6). **PS94/PS95 6/6 left as-is** (already current,
  re-running would desync from their un-rerun 6/7–6/8). PS94/PS95 6/7,6/8 unchanged (v1).
- **ENL (lick-free pre-cue interval):** PS94/PS95 2–3 s from 6/5; PS92 2–3 s from 6/6 (PS92 6/5 was
  1.5–2.5 s). ⇒ a **2 s pre-cue window is clean everywhere except possibly PS92 6/5** (flag that slide).

## Deck grouping (Priya's spec, verbatim intent)
Grouped **by animal first, date second** for the per-animal sections (PS92 6/5, PS92 6/6, … then
PS93 6/5 …). Interleaved with two per-date (all-animals) sections and a cross-mouse closing section.
Order:

### Section A — PER ANIMAL → date: per-session decode (cue-anchored)
For each animal, for each of its dates, the decoding matrices + recall. >1 date per slide if it fits.
Three window placements:
- **post-cue 2 s** — confusion matrix + per-position recall (engaged; + no-lick generalization).
- **rolling temporal bins** — per-session cue-aligned **sliding-window** accuracy curve spanning the
  full 2–3 s ENL → post-cue. This doubles as the **rolling pre-cue decoder** (Priya 6/10): strictly
  more informative than a single pre-cue mean because the maintained code RAMPS through the ENL — it
  separates a flat "maintained throughout ENL" code (the motor-independent readout) from a near-cue
  "anticipatory ramp" (possible movement prep), and exposes the earliest lick-free decodable bin.
  Optional later upgrade: cross-temporal generalization (train bin i → test bin j) = is it a stable
  code vs a dynamic sequence.
- **pre-cue 2 s** — confusion matrix + per-position recall (engaged + no-lick = maintained code). Kept
  alongside the rolling curve: the curve shows WHEN, the confusion matrix shows WHICH positions confuse
  + the no-lick generalization.
> NEW figure code: the decoder currently renders ONE figure per date with all animals as columns.
> Need a **per-(animal,date)** confusion+recall render and a **per-session** rolling-accuracy render.
> OPEN FORK: "post-cue 2 s" alignment — see Decisions below.

### Section B — PER DATE → all animals: cue-aligned rolling dynamics
For each date (6/5–6/8), one figure with all animals on that date:
- **cue-aligned rolling temporal dynamics** (pre-cue ENL → post-cue), one line per animal.
  → existing `locanmf_decoder_rolling_cue_{date}.png` (fig_rolling_cue).
- **accuracy: 2 s window vs rolling** comparison.
  → existing window-sweep / `fig_temporal_dynamics_{date}` material; may need a per-date 2s-vs-rolling panel.

### Section C — PER ANIMAL across sessions: weights + ablation
For each animal, pooled/overlaid across its sessions:
- **per-position decoder weights** across sessions. → NEW per-animal-across-sessions weights fig
  (existing `fig_weights_by_region` is per-date-all-animals).
- **region importance by ablation** — **leave-one-out and leave-two-out** — across sessions.
  → existing ablation is per-(animal,date) (`fig_region_ablation_{label}`) + one grouped all-animals
  (`fig_ablation_grouped_unilateral`, which already does leave-1 vs leave-2). NEW: per-animal
  across-sessions summary (mean ± spread of region-importance over that animal's sessions).

### Section D — PER DATE → all animals: encoder quiet baseline
For each date, encoder baseline with **time-local quiet (rest) reference** across the session, all
animals on that date. → existing `locanmf_encoder_quiet_drift_{date}.png` (fig_quiet_drift).

### Section E — PER ANIMAL: encoder expectations
For each animal:
- **expected MO and SS pooled activity by position** (encoder predicted time-course / per-position).
  → existing per-session `fig_temporal_encoder_{label}` (SSp/MO pooled), grouped by animal; consider a
  per-animal pooled-across-sessions version.
- **encoder explained variance per region — absolute AND normalized-to-1.0 (FEVE).**
  → existing per-session `fig_encoding_r2_by_region_{label}` already has both panels; group by animal.

### Section F — closing, cross-mouse / cross-session (all sessions)
- **cross-mouse cortical representation of spout position** → `locanmf_cross_mouse_comparison.png`.
- **within-animal consistency** of per-position decode/encode → `locanmf_within_animal_consistency*.png`
  (all-sessions + 6/5–6/8 matched-engagement subset).
- **RSA / RDM** → `locanmf_rsa_sessions.png`, `locanmf_rsa_rdms.png`, hemisphere-resolved
  (`locanmf_rsa_hemisphere_summary.png`, `_rdms.png`).

## Figure inventory: existing vs NEW
EXISTING (regenerate on fresh LocaNMF, re-use as-is): rolling_cue, temporal_dynamics, window_sweep,
quiet_drift, encoder temporal/r2_by_region/feve, cross_mouse, within_animal_consistency, rsa* .
NEW figure code required:
- **N1** per-(animal,date) decode confusion+recall (post-cue 2 s, pre-cue 2 s).
- **N2** per-(animal,date) rolling-accuracy curve (single session).
- **N3** per-animal across-sessions decoder weights.
- **N4** per-animal across-sessions ablation (leave-1 + leave-2) summary.
- **DECK** new builder module `locanmf_xsession_deck.py` (or a `build_xsession_ppt` in the existing
  ppt module) with animal-first ordering and the A–F sections; output e.g.
  `spout_position_decoder_xsession_6on.pptx`.

## Decisions / open forks
1. **Section A "post-cue 2 s" alignment** — DECIDED (Priya 6/10): **BOTH** cue-aligned (0–2 s after
   cue, symmetric with pre-cue, no-lick generalizable) AND first-lick-aligned 2 s (headline decoder).
   So each (animal,date) shows cue-2s + first-lick-2s confusion+recall, the rolling curve, and the
   pre-cue-2s matrix.
2. **Pre-cue window = 2 s** (was 1 s in pipeline) — justified by 2–3 s ENL on these sessions; flag PS92
   6/5 (ENL down to 1.5 s). CONFIRMED by Priya's request.
3. Per-animal sections: include all 4 of that animal's dates (PS94/PS95 have 6/5–6/8; PS92/PS93 too).

## How to rebuild (reproducible)
LocaNMF re-run (10 sessions: all 6/5 + PS92/PS93 6/6,6/7,6/8) and the figure build use two local drivers
in `~/source/` (the established home for run drivers, alongside `run_lick_batch.py`):
1. `python ~/source/rerun_locanmf_xreg.py` — re-LocaNMF the re-registered sessions (overwrite in place,
   r2=0.95 loc=80 maxrank=20). Resumable; skips sessions whose summary is newer than inputs.
2. `python ~/source/build_xsession_figs.py` — regenerate ALL deck figures into the OUT dir (sections A–F).
3. `python -c "from pathlib import Path; from wfield_local.locanmf_xsession_deck import build_xsession_ppt; build_xsession_ppt(Path('C:/Users/sabatini/source/cue_lick'))"`
   → `spout_position_decoder_xsession_6on.pptx` (59 slides).
The committed pieces (this spec + the figure functions N1–N4 + date filters + `locanmf_xsession_deck.py`)
fully define the deck; the two `~/source/` drivers are thin orchestration over them.

## Build result (2026-06-10)
Deck built: **59 slides, 112 embedded figures**, 0 empty content slides. Decoding sane on the fresh
re-registration (first-lick 2 s: 6/5 0.66–0.83, 6/8 0.81–0.89; SSp ≫ chance ⇒ regime B confirmed).
**PS93 SSp-left ≪ SSp-right lateralization (F15/F15a) SURVIVES re-registration**: SSp-L/R = PS92 .50/.50,
**PS93 .48/.60 (L−R −0.12)**, PS94 .50/.53, PS95 .66/.59 — same direction/magnitude as before, n=4 (6/5–6/8).
