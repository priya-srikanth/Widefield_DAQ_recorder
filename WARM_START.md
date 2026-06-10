# WARM START — Widefield LocaNMF spout-position decoder/encoder study

Paste-able context for a fresh Claude Code session continuing this project (Priya, Sabatini lab).
Read the two companion docs FIRST — they are the source of truth and more detailed than this file:
- [`LOCANMF_NIGHTLY_PIPELINE.md`](LOCANMF_NIGHTLY_PIPELINE.md) — the nightly runbook
- [`LOCANMF_LICK_CUE_ANALYSIS.md`](LOCANMF_LICK_CUE_ANALYSIS.md) — decisions + findings F1–F17

Also skim `DECISIONS.md` (server layout, event→frame regimes A/B).

## What the study is
Pre/post-stroke widefield calcium imaging in mice doing a 6-position spout-licking task. Goal: decode
INTENDED spout position from cortex (LocaNMF components) and build the reverse encoder, to later detect
how a ventrolateral-striatal stroke changes cortical movement / motor-planning representations. Mice:
PS92, PS93, PS94, PS95. **PS93 has a RIGHT orofacial deficit** (tongue deviates right, minimal right
whisking) — the lesion-relevant animal and focus of the cross-mouse / lateralization work.
State: pre-stroke baseline week **6/1–6/8 complete** (25 sessions; 6/2–6/8 all regime B). Repo tagged
`locanmf-stroke-decoder-v1.4`; latest deck = 85 slides.

## Environment / key paths
- Repo: `C:/Users/sabatini/GitHub/Widefield_DAQ_recorder` (work on `main`)
- Python: `C:/Users/sabatini/.conda/envs/locanmf/python.exe`, `PYTHONPATH=<repo>`
- Working figure dir (OUT): `C:/Users/sabatini/source/cue_lick`
- MICROSCOPE = `M:` = `\\research.files.med.harvard.edu\Neurobio`; bash mount `/m`
- Widefield sessions: `/m/MICROSCOPE/Priya/Widefield/labcams/<YYYYMMDD>/PS9*_<YYYYMMDD>*/motion_corrected/`
- LocaNMF inputs: `wfield_local_results/allen_aligned_affine8v1/U_atlas.npy` + `../../SVTcorr.npy`
- DAQ h5: `/m/MICROSCOPE/Priya/Widefield/DAQ_recorder_output/<YYYYMMDD>/`
- Deck/figures dest: `/m/MICROSCOPE/Priya/Widefield/labcams/locanmf_lick_pooled/cue_analysis/`

## Rules (non-negotiable)
- **Safety:** never delete anything on MICROSCOPE/N:; only ever WRITE inside `MICROSCOPE/Priya/`; never
  touch another person's folder. Don't delete on D: until Priya confirms copies, and only after
  per-folder verification (recursive file count + bytes match), deleting exact matches only.
- **Rig coordination** (two machines push here): before any push `git fetch origin && git rebase
  origin/main`; push = `git add -A && commit` → fetch → rebase → push. **NEVER force-push.** The imaging
  machine may push small commits — always re-fetch/rebase if rejected.
- **Lane:** do NOT edit `archive_day.py`, `framemap_event_maps.py`,
  `plot_spout_trial_averages_shared_scale.py`, `qc_motion_correction.py`, or top-level `_*.py` drivers.
- Pre-push hook needs `export CONDA_PREFIX=C:/Users/sabatini/.conda/envs/locanmf`.
- On `DECISIONS.md` / `LOCANMF_LICK_CUE_ANALYSIS.md` / `LOCANMF_NIGHTLY_PIPELINE.md` conflict, keep BOTH sides.

## Nightly tasks
Each evening Priya records 4 mice; the imaging computer motion-corrects → SVD → Allen-aligns → uploads to
MICROSCOPE. Upload is often **STAGED**: `wfield_local_results/` lands first, then `*cleanpairs_frame_map.npz`
+ the rest of `motion_corrected/`; mice arrive at different times; folders may be renamed on upload.

**(A) Evening, while waiting** — today's camera at `D:\camera`, behavior at `D:\behavior_logs`:
- Dropped-frame QC: `python C:/Users/sabatini/source/dropframe_check_all.py "D:\camera"` → writes
  `dropped_frames_summary_<date>.{csv,txt}` into the date folder (frame-id gap = dropped frame; a TRUE
  drop has timestamp gap ≈ missing×4 ms). Report counts.
- `robocopy D:\camera M:\MICROSCOPE\Priya\Behavior_Cameras /E /COPY:DAT /R:2 /W:5 /MT:16` (copy-only, never `/MIR`)
- `robocopy D:\behavior_logs M:\MICROSCOPE\Priya\Behavior_logs /E /COPY:DAT /R:2 /W:5 /MT:16`
- Do NOT delete D: until Priya confirms.

**(B) ~1 AM** — poll every 30 min for the day's widefield inputs, then run the full pipeline (per
`LOCANMF_NIGHTLY_PIPELINE.md`) for whichever mice are ready. LocaNMF needs only `wfield_local_results`;
**DECODING needs the frame_map** — hold decode until it's present, and do NOT guess regime A (that caused a
chance-decoding bug on 6/5). Steps: LocaNMF (`batch_locanmf`, r2 0.95 / loc 80 / maxrank 20) → add to
`SESSIONS` with regime (cleanpairs present → B/`fmdir=None`, else A) → VALIDATE regime by sensible decoding
(SSp ≫ chance), not RT → rolling cue/first-lick/laterality + pre-cue + single-window decoding → encoder
(raw + normalized-to-1.0 FEVE) → cross-mouse + within-animal consistency (all + the 6/5→ matched-engagement
window) + RSA (incl hemisphere-resolved) → bump deck refs in `locanmf_decoder_ppt.py` (add day to `DAYS`;
newest-day rolling/encoder refs; the consistency-subset slide ref **and** title — note dash-separated dates
like `_0605-06xx` are NOT caught by a `"_06xx"` search-replace, fix by hand) → `build_ppt` → copy pptx+PNGs
to `cue_analysis/` → commit via rig procedure → tag if substantial → `CronDelete` when all of the day's mice
are done.

## Canonical decoder/encoder choices
Individual LocaNMF components (not region-pooled), NO per-trial baseline, block-aware CV (GroupKFold by
~6-trial position blocks), first-lick-aligned 2 s window. Chance = 0.167 (6-way). Regime B alignment uses
the cleanpairs frame_map + `chosen_exposure_offset`.

## The scientifically interesting analyses (current findings)
1. **Decoding** works well every baseline day (first-lick up to ~0.91). Contralateral SSp orofacial
   subfields dominate; MOp/MOs secondary. Position info is highly REDUNDANT across cortex (small
   leave-region-out drops) — a movement-confound signature (the animal licks toward the spout; this is
   movement-linked, not a pure planning code). Post-cue no-lick decodes at chance (control); **PRE-CUE
   no-lick decodes ABOVE chance** = a motor-independent maintained position code (the key post-stroke readout).
2. **Encoder** (position→activity, ridge): FEVE = captured/explainable per region, normalized to 1.0. Low
   absolute R² is mostly a low noise ceiling (trial noise), not a bad model — most regions capture
   ~0.88–0.96 of EXPLAINABLE variance. "Explainable" = between-position SS / single-trial SS.
3. **PS93 lateralization (headline):** PS93 is the ONLY mouse with a large SSp-LEFT ≪ SSp-RIGHT decoding
   asymmetry (L−R ≈ −0.12, stable across 6/5–6/8). Left hemisphere = contralateral to the right orofacial
   deficit. Behavioral R-spout recall is intact → the signature is CORTICAL-HEMISPHERE, not spout-side.
   The most robust PS93 result (F15/F15a).
4. **Within-animal consistency:** per-position decode/encode is only moderately reproducible across ALL
   baseline days, but restricting to consecutive matched-engagement days (6/5–6/8) tightens the noise floor
   2–4× (decode per-position SD ~0.04–0.08). The representation is stable when engagement is matched; the
   early-day variability was the low-engagement days (6/1, 6/4). Implication: a post-stroke per-position
   change of ~0.10–0.15 is detectable → use multiple engagement-matched baselines and aggregate/pattern
   endpoints over single-session per-position values.
5. **RSA of representational geometry** (6×6 position RDM, basis-free 2nd-order RSA): each animal has a
   stable INDIVIDUAL geometry (within-animal RDM similarity > across-animal for all 4); calibrated by a
   split-half noise ceiling, PS92/PS94 are essentially AT ceiling (geometry as stable across days as within
   a session), PS95 shows genuine cross-day drift. DISSOCIATION: **PS93 is NOT the geometric outlier** (RDM
   most like PS92's; PS94 most distinct) — so PS93's deficit is lateralized, not a global geometry change.
6. **Hemisphere-resolved RDM** (the PS93 probe; F17): position RDM built separately from left- vs
   right-hemisphere components. At n=3 (6/5–6/7) PS93's disattenuated L-vs-R agreement was 0.44 (clear
   outlier); at n=4 (+6/8) it softened to 0.63 → session-sensitive, treat as suggestive (the SSp decode
   asymmetry #3 is the robust one). **Next ideas (F17):** per-cell `RDM_L − RDM_R` to localize WHICH
   position contrasts diverge in PS93's left hemisphere; cross-validated (crossnobis) RDM.

## Modules (all `wfield_local/`)
`locanmf_cue_lick_analysis.py` (SESSIONS, regimes) · `locanmf_position_decoder.py` (--align lick/cue/precue)
· `locanmf_position_encoder.py` (FEVE, heatmaps) · `locanmf_decoder_weights.py` (rolling/temporal figs,
adaptive y-axis) · `locanmf_decoder_ablation.py` · `locanmf_cross_mouse.py` (cross-mouse + within-animal
consistency, dates/tag subset) · `locanmf_rsa.py` (RSA + noise ceiling + hemisphere-resolved) ·
`locanmf_decoder_ppt.py` (`build_ppt`).

## Start by
Read the two docs above; `git -C <repo> fetch && git log --oneline -8`; check `labcams/<today>` for new
inputs. If it's not a nightly run, ask Priya what to focus on. (Crons are session-only and die with the
session — if a nightly check should run tonight, set one up.)
