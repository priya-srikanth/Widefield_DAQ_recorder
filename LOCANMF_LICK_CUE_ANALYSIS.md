# LocaNMF lick/cue analysis & position-decoding — decisions, findings, plan

Working notes for the widefield LocaNMF behavioral analysis (lick/cue-evoked activity, spout
position, and the per-position decoder for the planned stroke study). Written 2026-06-04 so a
future session can resume without re-deriving. Companion to `GPU_LOCANMF_RUNLOG.md` (the run/
env/params) and `DECISIONS.md`.

## 0. The endpoint this is building toward
Compare **cortex-wide activity when the animal tries to lick to each spout position, pre vs
post stroke**. Stroke = **ventrolateral striatum (subcortical)** → the imaged cortex map is
structurally intact, so this is a *functional* reorganization question, not lesioned-cortex.

## 1. Data / scope
- 6 Allen-aligned sessions: PS92/PS94/PS95 × {6/1 full-FOV regime A, 6/2 & 6/3 cleanpairs
  regime B}. **6/1 and 6/2 are noisy — analysis is built on the 6/3 sessions; 6/4 pending.**
- LocaNMF components (final params `r2=0.95, loc_thresh=80, maxrank=20`) in each session's
  `motion_corrected/locanmf_affine8v1_final/` (`*_locanmf_{A,C,regions}.npy`).
- DAQ h5 (lick_analog, cue TTL, spout strobe/bits, pco) in
  `MICROSCOPE/Priya/Widefield/DAQ_recorder_output/` (6/1 at root; 6/2,6/3 in `<YYYYMMDD>`
  subfolders with a **2025 year-typo**: `20250602`, `20250603`).
- Behavior video on `MICROSCOPE/Priya/Behavior_Cameras/` (DLC + facerhythm planned; not yet
  analyzed). **The cue is a 1 kHz, 75 ms tone.**

## 2. Decisions (with rationale)
- **Lick events split by the 6 spout positions** (close/far × L/center/R) via `_classify_events`
  / `_classify_cues` (spout strobe + bits). Cue = the *intended* position.
- **Components kept INDIVIDUAL, labeled by Allen region — NOT pooled.** Pooling components
  within a region ≈ an Allen-ROI average (see Finding F5), which throws away LocaNMF's only
  value-add. Region label is for cross-animal *identity*, not for averaging.
- **Normalization (this was a journey — see Findings F3/F4):**
  - quiet-period z-score was the first choice but is **unusable for cross-animal magnitude**
    (PS92's quiet mask is contaminated → deflated z).
  - a data-driven "quietest-frames SD" **over-corrects** (selection-biased SD → z inflated
    ~10–45×, non-uniformly).
  - **pre-cue-1 s** z (subtract pre-cue mean, ÷ pooled pre-cue SD) is the comparable baseline.
  - **physical ΔF/F** via the scale-invariant footprint weight `s_i = Σ Aᵢ²/Σ Aᵢ`
    (`dff_i = s_i·C_i`) is the Churchland-convention unit and the best for cross-animal — it
    removes LocaNMF's scale ambiguity and needs no SD. **Use ΔF/F for cross-animal magnitude.**
  - **Within-component contrasts (e.g. contralateral index) are normalization-robust** (the
    per-component scale cancels), so those are trustworthy regardless.
- **Stats: animal is the unit of replication** (mean ± SEM across mice, n = mice), not across
  components/trials.
- **Engagement (critical for the stroke comparison):** exclude *disengaged blocks* (extended
  no-lick **and** no-movement), but define engagement by **movement/arousal, NOT by per-trial
  lick success** — because post-stroke a no-lick trial is a *failed attempt* (the deficit you
  want to keep). Lick-density gating is a first pass; upgrade to movement-gated once DLC is up.
  Baseline disengaged time here ≈ 0–13 % (PS95 6/3 highest).

## 3. Findings
- **F1. Orofacial responses are predominantly lick(motor)-driven**, not cue(sensory): on
  cue-no-lick trials the orofacial cortex is ~flat while cue+lick is large.
- **F2. Cue→first-lick RT ≈ 0.16 s** (very fast) → cue and lick are tightly collinear; hard to
  separate sensory from motor.
- **F3. The "PS95 > PS94 > PS92" cross-animal amplitude order was mostly a normalization
  artifact.** PS92's quiet mask has quiet-SD/total-SD ≈ 0.97 (quiet ≈ whole-session variance)
  so its quiet-z denominator is inflated (z deflated ~45× vs ~10× for others). In **ΔF/F +
  animal-level stats the three mice are tightly consistent.**
- **F4. NOT a 470/415 frame-parity artifact** (checked: lag-1 autocorr +0.99, ~0 % Nyquist
  power; a parity artifact would be ≈ −1 / ~80 %). Frames are indexed correctly.
- **F5. Region-pooled LocaNMF ≡ Allen-ROI ΔF/F** (`mean_pixels(U_region)·SVTcorr`): trace
  correlation median **r ≈ 0.99** (SS) / 0.985 (MO). So **for region-level work use ROIs**;
  reserve LocaNMF for per-component analyses.
- **F6. Contralateral somatosensory tuning is real and reproduced by LocaNMF.** Within-
  component post-lick-150 ms position×hemisphere contrast: contralateral-spout licks larger in
  **100 % of SSp components**. Motor contralateral *modulation* (scale-free index) is
  **comparable to SSp**, just more variable — the earlier "motor less lateralized" claim was an
  **absolute-amplitude artifact and is retracted**.
- **F7. far_center outlier = PS95-specific** (both its sessions ~1.8–2.2× others), not sampling.
- **F8. Cue+lick FIR encoding model**: R² ~3–6 % only, cue/lick collinear → the sensory/motor
  *split is unstable*. Consistent with Musall (cortex movement-dominated); a real separation
  needs **video-derived movement regressors**.
- **F9. Auditory positive-control inconclusive due to coverage.** Cue is auditory (tone), but
  AUD ROI signal is **~3–4× weaker than SSp** every session (auditory cortex under-sampled at
  the lateral edge of the dorsal window). Not evidence against an auditory response.
- **F10. Spout position decodes strongly from cortex.** 6-way logistic-regression, cue-aligned,
  engaged trials, 5-fold: **0.59 / 0.69 / 0.79** (chance 0.17) for PS92/PS94/PS95 6/3. **SSp
  carries it (0.51–0.63) >> MO (0.30–0.47).** Confusions are between adjacent spouts. LocaNMF
  features modestly beat SVD-ROI (esp. MO); cue-aligned ≈ lick-aligned.
- **F11. No-lick trials decode at ≈ chance** (train on engaged, apply to no-lick: 0.16/0.23/0.26
  LocaNMF, 0.13–0.21 ROI; chance 0.17). (a) A clean **negative control** — the decoder isn't
  exploiting a confound (else no-lick would still decode). (b) Confirms the position code is
  **lick/engagement-driven** (F1). (c) IMPORTANT: baseline no-lick = **disengagement** (chose
  not to engage → no attempt → no code); post-stroke no-lick = **failed attempts** (tried,
  motor failed) — those are the real test (if they decode *above* this chance baseline, intent
  is preserved in cortex despite motor failure). **Must separate failed-attempt from disengaged
  via movement/video.** `locanmf_position_decoder.py` now emits engaged + no-lick recall as a
  standard output.

## 4. When is LocaNMF actually helpful (the synthesis)
- **Not** for region-level evoked responses / ROI summaries → that equals an Allen-ROI average
  (F5) with extra scale-ambiguity headaches; use ROIs/pixel maps there.
- **Yes** for: (a) demixing overlapping/contaminated sources, (b) sub-region structure
  (multiple components/region), and especially (c) a **compact, denoised, interpretable,
  atlas-anchored basis for models** — decoding/encoding/connectivity/single-trial — where it
  beats SVD (interpretability) and ROIs (single-trial SNR, multi-area joint model). The
  position decoder (F10) is the canonical good use.

## 5. Stroke-study analysis plan (decided)
- **Model = the per-position decoder** (multinomial **logistic regression**, L2, standardized,
  5-fold). Linear + interpretable (weights = which areas code each position), and correct for
  the n/p regime (~300–450 trials, 66–152 features) — complex models would overfit.
- **Anchor on the cue** (intended position) so it applies to post-stroke **no-lick failed
  attempts**; train on baseline **engaged (cue+lick)** trials.
- **Two complementary comparisons:**
  1. **Per-session decoders (primary)** — train per session, compare **per-position recall +
     confusion + SSp/MO breakdown** pre vs post. *No common feature space needed* (each decoder
     in its own components) → sidesteps cross-day component correspondence. Also apply each
     session's decoder to its *own* no-lick trials ("is intended position still readable when
     the lick fails?").
  2. **Frozen pre-stroke decoder (confirmatory)** — needs a common basis: either **fixed pre-
     stroke `A`, refit `C`** (`C_new = pinv(A_ref)·U_new·SVTcorr_new`; valid because the stroke
     is subcortical) or **Allen-ROI features**. Tests whether the pre-stroke readout transfers.
- **Prerequisites:** cross-day vasculature registration (`cross_day_align.py`); DLC/facerhythm
  movement regressors **time-synced** to widefield+DAQ (to separate "cortex codes position
  differently" from "the movement just changed").

## 6. Module inventory (all on `main`)
`wfield_local/`:
- `locanmf_lick_aligned.py` — lick/cue-triggered component traces by spout position, quiet-z;
  `--event lick|cue`. Emits per-session npz + orofacial quick-look.
- `locanmf_cards.py` … `locanmf_lick_cards.py` — per-component spatial map + lick traces.
- `locanmf_lick_pool.py` — cross-session per-component pooled npz + per-animal overlay + area×
  session responsiveness heatmap.
- `locanmf_event_master.py` — one combined lick+cue per-animal figure.
- `locanmf_cue_lick_analysis.py` — position-specific cue + cue-with/without-lick (holds the
  shared `SESSIONS` config used by later modules).
- `locanmf_cue_auc.py` — cue-evoked AUC (0–2 s) by position per animal (pre-cue z).
- `locanmf_firstlick_aligned.py` — pre-cue-normalized, aligned to first lick after cue.
- `locanmf_crossanimal_dff.py` — cross-animal ΔF/F (footprint scale) + animal-level stats;
  exports `_footprint_scale`, `_frames`.
- `locanmf_dff_by_position.py` — ΔF/F by position (cross-mouse + per-animal).
- `locanmf_contralateral.py` — per-component contralateral modulation index (scale-free) +
  position profiles; `--event lick|cue`.
- `locanmf_encoding_model.py` — cue+lick FIR ridge encoding model (cue vs lick kernels, R²).
- `locanmf_position_decoder.py` — **the decoder**; `--source locanmf|roi`, `--align cue|lick`,
  per-area accuracy + per-position recall + confusion (the pre/post-diffable outputs).
- (local helpers in `~/source/`: `run_lick_batch.py`, `norm_compare.py`, `auditory_cue.py`.)

## 7. Output locations
Figures + tables on MICROSCOPE: `labcams/locanmf_lick_pooled/` (overlays, master, cards/) and
`labcams/locanmf_lick_pooled/cue_analysis/` (cue, normalization, AUC, first-lick, contralateral,
encoding kernels, auditory, **decoder + recall** figures). Final per-session LocaNMF outputs in
each session's `motion_corrected/locanmf_affine8v1_final/`.
