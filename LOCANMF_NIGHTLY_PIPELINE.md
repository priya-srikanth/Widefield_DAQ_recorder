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
(6/2–6/7 were all B.) **Frame-mapping is validated by SENSIBLE DECODING, not by RT** (RT is in DAQ
samples). If decoding collapses / SSp → chance, the regime is wrong → try the other regime. (6/5 bug:
regime A gave chance, B fixed it.)

**2c. Decoding** (individual LocaNMF components, no-baseline, block-CV, first-lick 2 s):
- rolling cue-aligned: `locanmf_decoder_weights.fig_rolling_cue(_avail("<MMDD>"), OUT, "<MMDD>")`
- rolling first-lick: `fig_temporal_dynamics(_avail("<MMDD>"), OUT)` and `fig_rolling_laterality(..., "<MMDD>")`
- pre-cue: `python -m wfield_local.locanmf_position_decoder --date <MMDD> --align precue --post-s 1.0 --output OUT`
- single-window: `--align lick` and `--align cue` (write the per-day decoder+recall figs the deck reads).

**2d. Encoding** (`python -m wfield_local.locanmf_position_encoder --date <MMDD> --output OUT`): per-session
`r2_by_region` two-panel (absolute explainable-vs-captured **and** FEVE normalized-to-1.0); FEVE-by-region
heatmaps (pooled-per-animal + per-session, all 64 atlas regions); predicted maps; EV-by-position; ceiling;
temporal; encoder-vs-SVD validation.

**2e. Cross-mouse / cross-session.**
- `python -m wfield_local.locanmf_cross_mouse --output OUT` → cross-mouse 6-panel (bars = mean ± SEM with
  session points) **and** within-animal per-position consistency (all sessions).
- Matched-engagement window: `locanmf_cross_mouse.fig_within_animal_consistency(OUT, dates={"0605".."<MMDD>"}, tag="0605-<MMDD>")`.
  Window starts **6/5 for all four animals** — PS92's 6/5 was trial-triggered (others continuous) but was
  assessed comparable (6/5 first-lick 0.77 between 6/6 0.73 and 6/7 0.84; 6/5–6/7 SD 0.05); re-verify each
  run and exclude PS92 6/5 only if it becomes an outlier.
- RSA: `python -m wfield_local.locanmf_rsa --output OUT` → session×session 2nd-order RSA (Spearman of 6×6
  position RDMs) + within/across-animal stability vs split-half noise ceiling + animal×animal RDM
  similarity, **and** hemisphere-resolved RDMs (left-hem vs right-hem position geometry, disattenuated
  L-vs-R agreement; the PS93 lateralization probe).

**2f. Deck + commit.** In `wfield_local/locanmf_decoder_ppt.py`: add `("<MMDD>","M/D")` to `DAYS`; bump the
newest-day rolling/encoder refs to `<MMDD>`; bump the consistency-subset slide ref to `_0605-<MMDD>`.
Rebuild via `locanmf_decoder_ppt.build_ppt(OUT)`. Copy `.pptx` + all new PNGs to the cue_analysis dir.
Commit to `main` via the **rig procedure**: `git add -A && commit` → `git fetch origin` → `git rebase
origin/main` → `git push` (NEVER force-push; if rejected, re-fetch/rebase/push). Stay in the `locanmf_*`
lane — do NOT edit rig-owned files (`archive_day.py`, `framemap_event_maps.py`,
`plot_spout_trial_averages_shared_scale.py`, `qc_motion_correction.py`, or top-level `_*.py` drivers). On
`DECISIONS.md` / `LOCANMF_LICK_CUE_ANALYSIS.md` conflict, keep BOTH sides.

**3. Cleanup.** When all of the day's sessions are done, `CronDelete` the job.

## Camera + behavior transfers (done in the evening, before the cron)
- Dropped-frame QC: `python C:/Users/sabatini/source/dropframe_check_all.py "D:\camera"` → writes
  `dropped_frames_summary_<date>.{csv,txt}` into each date folder (one row per cam recording; gap in the
  frame-id sequence = dropped frame; verify the timestamp gap scales as missing×4 ms = a true drop).
- `robocopy D:\camera M:\MICROSCOPE\Priya\Behavior_Cameras /E /COPY:DAT /R:2 /W:5 /MT:16` (copy-only).
- `robocopy D:\behavior_logs M:\MICROSCOPE\Priya\Behavior_logs /E /COPY:DAT /R:2 /W:5 /MT:16`.
- **Never `/MIR`. Do NOT delete anything on D: until Priya confirms** — then verify per-folder
  (recursive count + bytes) and delete only exact matches.

## Standing safety rules
Never delete anything on MICROSCOPE/N:. Only ever write inside `MICROSCOPE/Priya/`. Never another
person's folder. Don't delete on D: without explicit confirmation + per-folder verification.
