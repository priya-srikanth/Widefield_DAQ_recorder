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
- **F10. Spout position decodes strongly from cortex.** 6-way logistic-regression on individual
  LocaNMF components, engaged trials. **Canonical method (see F12): no per-trial baseline,
  block-aware CV, first-lick-aligned, 2 s window** → **0.67 / 0.83 / 0.85** (PS92/PS94/PS95 6/3;
  chance 0.17), 0.56 / 0.29 / 0.49 on 6/4 (PS94 6/4 is a low-engagement outlier day — its
  between-position component signal is ~4× weaker than 6/3, not a decoder bug). **SSp carries it
  (SSp-only 0.62–0.77) >> MO (MO-only 0.33–0.52, still above chance).** Top features are
  **orofacial SSp subfields (SSp-m mouth, SSp-n nose, SSp-un, SSp-bfd barrel), CONTRALATERAL to
  the spout** (left spouts→right-hemi SSp, right→left); MOp/MOs contribute ~9–16 % of weight,
  most for *far* positions. Confusions are between adjacent spouts. First-lick > cue alignment
  (6/3 0.67–0.85 vs 0.64–0.79); cue alignment kept only for the no-lick test (F11).
  **Window: ~2 s is the optimum** (integrates the lick bout; 1 s under-reads — PS94 0.76→0.83 —
  and >~2.5 s dilutes the transient; ITI ~7 s so not cross-trial bleed). *(The earlier 0.59/0.69/0.79 used random-CV + pre-cue baseline and
  was both deflated by over-subtraction and distorted by block leakage — superseded by F12.)*
- **F11. No-lick trials decode at ≈ chance** (train on engaged, apply to no-lick, cue-aligned:
  6/3 0.29/0.11/0.20, 6/4 0.21/0.14/0.12; chance 0.17). (a) A clean **negative control** — the
  decoder isn't exploiting a confound (else no-lick would still decode). (b) Confirms the position
  code is **lick/engagement-driven** (F1). (c) IMPORTANT: baseline no-lick = **disengagement**
  (chose not to engage → no attempt → no code); post-stroke no-lick = **failed attempts** (tried,
  motor failed) — those are the real test (if they decode *above* this chance baseline, intent
  is preserved in cortex despite motor failure). **Must separate failed-attempt from disengaged
  via movement/video.** No-lick is intrinsically **cue-aligned** (no lick to align to).
- **F12. Decoder methodology (load-bearing for the stroke pre/post).** (i) **Positions are
  presented in ~6-trial BLOCKS** (P(stay)≈0.84). With random k-fold, same-block trials land in
  train *and* test, so the decoder reads each block's **slow-drift fingerprint** rather than
  evoked coding → inflation. Symptom: the *pre-cue* window "decoded" 0.47–0.65 under random-CV.
  **Use block-aware CV (leave-whole-blocks-out).** (ii) **No per-trial baseline.** A
  *session-constant* baseline (quiet-period) is removed by feature standardization — *identical*
  decoding to none (proven: 0.692=0.692). A *per-trial pre-cue* baseline **over-subtracts** real
  anticipatory position signal (block-CV no-base 0.71–0.82 vs pre-cue-sub 0.61–0.79 on 6/3).
  (iii) The pre-cue window decodes position **above chance even under block-CV** (6/3 0.40–0.56)
  = genuine **anticipatory/preparatory** coding (blocked design); so no-baseline measures *total*
  position info at lick time, pre-cue-sub isolates the *evoked* change. `locanmf_position_decoder.py`
  defaults: `--baseline none --cv block`; toggles `--baseline precue`, `--cv random` for contrasts.
- **F13. Pre-cue (anticipatory) decoding — the motor-independent post-stroke readout.** Train the
  decoder on engaged (cue+lick) trials' **pre-cue 1 s window**, block-CV. Engaged: 6/3 0.40/0.55/0.56.
  Crucially, applied to **no-lick** trials it decodes **above chance** (6/3 0.26/0.34/0.22) —
  unlike the *post*-cue no-lick decode (F11, ≈chance). So the **maintained position code is readable
  without a lick**, the ideal readout for post-stroke failed attempts (train on baseline engaged →
  apply to post-stroke no-lick; intact ⇒ preparation survived, collapsed ⇒ representation degraded).
  **Validated by a time-only control**: position is *not* decodable from cue-time alone (0.02–0.11),
  so pre-cue signal is genuine cross-block coding, not slow drift. *Caveat:* pre-cue may also carry
  ongoing inter-trial movement to the current spout → a post-stroke drop could be loss-of-movement,
  not loss-of-representation; needs video.
- **F14. Baseline (pre-stroke) variability is LARGE — 3 days/animal.** Post-lick 2 s decoding across
  baseline days: PS92 (6/2,6/3,6/4) 0.46/0.67/0.56 (range 0.21); PS94 (6/1,6/3,6/4) 0.73/0.83/**0.29**
  (range **0.55**); PS95 (6/1,6/3,6/4) **0.91**/0.85/0.49 (range 0.41). **6/4 is the low day for
  PS94 & PS95** and tracks engagement (no-lick counts: PS94 6/1 n=5 → 6/4 n=145). 6/1 itself is *not*
  too noisy to decode (PS95 6/1 = 0.91, the best). **Design implication:** a single pre-vs-post
  contrast is uninterpretable; need multiple baseline days to set each animal's baseline distribution,
  **engagement/movement matching**, and likely per-position/per-region contrasts (more stable than
  overall accuracy) + the F13 pre-cue readout. Figures: `locanmf_decoder_baseline_variability.png`.
- **F15. Cross-mouse cortical-representation comparison — PS93's hemisphere asymmetry (all sessions
  pooled per mouse; `locanmf_cross_mouse.py`).** Motivated by **PS93's RIGHT orofacial deficit**
  (tongue deviates right, minimal right whisking). Per-mouse SSp-hemisphere-only decoding (first-lick
  2 s, block-CV): **PS93 SSp-LEFT 0.40 << SSp-RIGHT 0.52 (L−R = −0.12)** vs near-symmetric in the
  others (PS92 +0.02, PS94 +0.05, PS95 +0.01) — **PS93 is the only mouse with a large SSp left-vs-
  right asymmetry**, and its LEFT hemisphere (contralateral to the right-side deficit) is the weakest
  for position decoding while its right is the strongest of any mouse's. Behavioral R-spout recall is
  intact (0.79), so the signature is **cortical-hemisphere, not spout-side** — consistent with a
  left-hemisphere substrate for the right orofacial deficit. Encoding EV/position reinforces it:
  PS93 close_R is **negative** (−0.06, poorly encoded) while close_L is well-encoded (0.15). **n=2 for
  PS93 (6/5, 6/6); n=2–6 for the others.** Figure: `locanmf_cross_mouse_comparison.png` (6-panel:
  overall/per-position decoding, L-vs-R spout decodability, SSp-left-vs-right hemisphere, encoding EV,
  L/R asymmetry indices). 6/6 added all four mice as regime B (cleanpairs frame_map present); decoding
  sensible (first-lick all 0.73/0.73/0.78/0.87) confirming the frame mapping.

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
