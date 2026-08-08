# LocaNMF nightly pipeline (spout-position decoder/encoder study)

Durable record of the overnight pipeline run for each new widefield session day. The live executor is a
**session-only cron** (created via CronCreate each evening, auto-expires in 7 days) — this file is the
canonical, version-controlled description so the cron prompt can be rebuilt from it. Companion docs:
`LOCANMF_LICK_CUE_ANALYSIS.md` (decisions/findings F1–F15a), `DECISIONS.md` (server layout, regimes).

## Cadence
Priya kicks the imaging computer (motion-correct → SVD → Allen-align → upload to MICROSCOPE) in the
evening. A recurring 30-min cron starts ~1 AM and polls for that day's inputs, then runs everything for
whichever mice are ready (later fires pick up stragglers), and **deletes itself** once all of that day's
sessions are processed. Mice: PS92/PS93/PS94/PS95. PS93 has a RIGHT orofacial deficit (tongue deviates
right, minimal right whisking) — the cross-mouse / lateralization angle.

## Paths & environment
- Widefield: `/m/MICROSCOPE/Priya/Widefield/labcams/<YYYYMMDD>/PS9*_<YYYYMMDD>*/motion_corrected/`
- LocaNMF inputs: `wfield_local_results/allen_aligned_affine8v1/U_atlas.npy` + sibling `../../SVTcorr.npy`
  (NB inputs sometimes appear under a temp name then get renamed; if the session folder exists but
  `allen_aligned_affine8v1/` is missing, also `ls wfield_local_results/` for any `allen_aligned_*`. Use
  `allen_aligned_affine8v1` for consistency with all prior sessions — there is also a cross-day
  `allen_aligned_xday6` variant, currently unused.)
- DAQ h5: `/m/MICROSCOPE/Priya/Widefield/DAQ_recorder_output/<YYYYMMDD>/`
- Python: `C:/Users/sabatini/.conda/envs/locanmf/python.exe`, `PYTHONPATH=C:/Users/sabatini/GitHub/Widefield_DAQ_recorder`
- Working figure dir (OUT): `C:/Users/sabatini/source/cue_lick`
- Deck + figures destination: `/m/MICROSCOPE/Priya/Widefield/labcams/locanmf_lick_pooled/cue_analysis/`
- Pre-push hook needs `export CONDA_PREFIX=C:/Users/sabatini/.conda/envs/locanmf`

## Steps

**1. Detect inputs.** For each mouse check `allen_aligned_affine8v1/U_atlas.npy` + `SVTcorr.npy`. If none
ready, print `still waiting <HH:MM>` and stop (re-fires in 30 min).

**2a. LocaNMF.** Write `~/source/locanmf_batch_<MMDD>.json` (one `{label,allen_dir,output}` per ready
session; `output=.../motion_corrected/locanmf_affine8v1_final`). Run:
`python -u -m wfield_local.batch_locanmf --manifest <m> --r2 0.95 --loc 80 --maxrank 20`. Confirm
component counts (typically ~90–180/session).

**2b. Register sessions.** Append to `SESSIONS` in `wfield_local/locanmf_cue_lick_analysis.py`. **Regime:**
`*cleanpairs_frame_map.npz` present in `motion_corrected/` → regime `"B"` (`fmdir=None`); absent → `"A"`.
(6/2–8/7 have all been B.) **Frame-mapping is validated by SENSIBLE DECODING, not by RT** (RT is in DAQ
samples). If decoding collapses / SSp → chance, the regime is wrong → try the other regime. (6/5 bug:
regime A gave chance, B fixed it.)

**2b-bis. Spout-position source + behavior-log BACKUP.** Positions come from the DAQ spout-strobe bits
(`_classify_cues`). The **Aug-2026 sessions had DAQ `spout_bit1` (line5) dead** → dropped close_R +
far_center, mis-coded onto other positions. `wfield_local/behavior_position.classify_cues_with_backup`
(wired into `_trial_features`, so decoder/encoder/cross-mouse/RSA all inherit it) auto-repairs this: if the
DAQ shows <6 positions AND the task controller's `trials.csv` (`pos_idx`) aligns to the DAQ's good
positions at ≥0.9 by an integer trial-offset, it substitutes the behavior-log positions; otherwise it keeps
DAQ untouched (so good sessions are never altered). Behavior log:
`MICROSCOPE/Priya/Behavior_logs/Widefield/<mouse>_<YYYYMMDD>_*/trials.csv`. **8/5–8/6 needed the backup on
all mice; 8/7 the DAQ delivered all 6 positions directly (bit1 fixed) — no backup fired.** PS93 8/5 had a
dead bit1 AND an empty behavior log, so it was recovered from the cam1 video (human-verified, 0 corrections;
see STROBE_BIT1_RECOVERY.md): its SESSIONS entry carries `behavior_trials=<...spout_position_recovery_cam1/
ps93_reviewed_trials.csv>`, which classify_cues_with_backup honors via framemap_event_maps._behavior_cue_codes
(order+bitmask, >=98%) so decoder/encoder/cross-session agree with the maps. NOTE: the DAQ cue/strobe stream tracks the
**rewarded-trial subset** (reward held after ~6 misses in a row), which conveniently doubles as an
engagement filter dropping the disengaged session tail — keep this scoping; do NOT fold in unrewarded
trials (that is the separate future post-stroke "failed-attempt" analysis, which is movement-gated).

**ORCHESTRATOR: steps 2c–2e are run by one in-repo module** — `python -m wfield_local.nightly_figs
<MMDD>` (repo root derived from `__file__`; `--output` defaults to `~/source/cue_lick`; `--from` overrides
the cross-session span). It runs the three decode alignments, the rolling/top-component figs, the encoder, and the
cross-mouse/within-animal/RSA. Per-day figures use `<MMDD>`; the **cross-session comparisons (cross-mouse,
within-animal, RSA, pooled encoder FEVE) span ALL registered sessions** (dates computed from `SESSIONS`;
tag `0601-<MMDD>`). Mouse panels are ordered **PS92, PS93, PS94, PS95** everywhere (per-day decode sorts
`sess` by mouse; cross-mouse/RSA already sort). The individual-module commands below are what it calls.

**2c. Decoding** (individual LocaNMF components, no-baseline, block-CV, first-lick 2 s):
- rolling cue-aligned: `locanmf_decoder_weights.fig_rolling_cue(_avail("<MMDD>"), OUT, "<MMDD>")`
- rolling first-lick: `fig_temporal_dynamics(_avail("<MMDD>"), OUT)` and `fig_rolling_laterality(..., "<MMDD>")`
- pre-cue: `python -m wfield_local.locanmf_position_decoder --date <MMDD> --align precue --post-s 1.0 --output OUT`
- single-window: `--align lick` and `--align cue` (write the per-day decoder+recall figs the deck reads).

**2d. Encoding** (`python -m wfield_local.locanmf_position_encoder --date <MMDD> --output OUT`): per-session
`r2_by_region` two-panel (absolute explainable-vs-captured **and** FEVE normalized-to-1.0); FEVE-by-region
heatmaps (pooled-per-animal + per-session, all 64 atlas regions); predicted maps; EV-by-position; ceiling;
temporal; encoder-vs-SVD validation.

**2e. Cross-mouse / cross-session (ALL registered sessions, tag `0601-<MMDD>`).**
- `python -m wfield_local.locanmf_cross_mouse --output OUT --dates <all> --tag 0601-<MMDD>` → cross-mouse
  6-panel (bars = mean ± SEM with session points) **and** within-animal per-position consistency across all
  sessions. Early June has partial mouse coverage (6/1 = PS94/PS95, 6/2 = PS92 only); the per-mouse
  aggregation handles that. A supplementary matched-engagement `_0605-0608` within-animal slide (PS92 6/5
  trial-triggered but comparable) is kept from an earlier run.
- RSA: `python -m wfield_local.locanmf_rsa --output OUT --dates <all> --tag 0601-<MMDD>` → session×session
  2nd-order RSA (Spearman of 6×6 position RDMs; sessions animal-blocked then date-ordered) + within/
  across-animal stability vs split-half noise ceiling + animal×animal RDM similarity, **and**
  hemisphere-resolved RDMs (left-hem vs right-hem position geometry, disattenuated L-vs-R agreement; the
  PS93 lateralization probe). Crossnobis (noise-unbiased) variant available via `fig_rsa_crossnobis`.
  On the RDM vs RSM question: for the correlation metric they are the same information (RDM = 1 − RSM);
  the RDM framing only earns its keep with the **crossnobis** distance, which is noise-unbiased and on a
  ratio scale (0 = identical), unlike the positively-biased 1−corr.

**2f. Deck + commit.** In `wfield_local/locanmf_decoder_ppt.py`: add `("<MMDD>","M/D")` to `DAYS`; bump the
newest-day rolling/encoder (`rc`/`lr`/`encs`/`vs`/`evp`/`evc`) refs and their title text to `<MMDD>`; set
the cross-mouse/within-animal/RSA (`cm`/`wac`/`rsa`/`rsr`/`hsum`/`hrdm`) refs to `_0601-<MMDD>`.
Rebuild via `locanmf_decoder_ppt.build_ppt(Path(OUT))`. Copy `.pptx` + new PNGs (robocopy `/MAXAGE:<today>`)
to the cue_analysis dir.
Commit to `main` via the **rig procedure**: `git add -A && commit` → `git fetch origin` → `git rebase
origin/main` → `git push` (NEVER force-push; if rejected, re-fetch/rebase/push). Stay in the `locanmf_*`
lane — do NOT edit rig-owned files (`archive_day.py`, `framemap_event_maps.py`,
`plot_spout_trial_averages_shared_scale.py`, `qc_motion_correction.py`, or top-level `_*.py` drivers). On
`DECISIONS.md` / `LOCANMF_LICK_CUE_ANALYSIS.md` conflict, keep BOTH sides.

**3. Cleanup.** When all of the day's sessions are done, `CronDelete` the job.

## Camera + behavior transfers (done in the evening, before the cron)
- Dropped-frame QC: `python C:/Users/sabatini/source/dropframe_check_all.py "D:\camera"` → writes
  `dropped_frames_summary_<date>.{csv,txt}` into each date folder (one row per cam recording; gap in the
  frame-id sequence = dropped frame; col1 timestamp is ns, nominal Δt = 4 ms / ~250 fps; verify the
  timestamp gap scales as missing×4 ms = a true drop). D:\camera may contain an empty literal
  `YYYYMMDD` template folder — skip it.
- **Destinations are the `Widefield\` SUBDIR** (as of 2026-08-06; NOT the top level):
  `robocopy D:\camera\<YYYYMMDD> \\research.files.med.harvard.edu\Neurobio\MICROSCOPE\Priya\Behavior_Cameras\Widefield\<YYYYMMDD> /E /MT:16 /R:2 /W:5`
  and `robocopy D:\behavior_logs\<session> ...\Priya\Behavior_logs\Widefield\<session> /E /R:2 /W:5`
  (camera = date-folder layout; behavior = per-session folders). Robocopy exit code 1 = "files copied OK".
- **Never `/MIR`. Do NOT delete anything on D: until Priya confirms** — then re-verify per-folder
  (recursive file count + total bytes, src==dst) immediately before deleting, and delete only exact matches.

## Standing safety rules
Never delete anything on MICROSCOPE/N:. Only ever write inside `MICROSCOPE/Priya/`. Never another
person's folder. Don't delete on D: without explicit confirmation + per-folder verification.
