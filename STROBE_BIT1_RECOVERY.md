# Strobe bit1 dead (2026-08-05/06) + behavior-log position recovery

## What happened
On **2026-08-05 and 2026-08-06**, the widefield DAQ recorded only spout strobe codes
**0,1,4,5** — strobe **bit1 (value 2) never toggled** (a dead/miswired strobe line). Spout
position is encoded `code = bit0 + 2*bit1 + 4*bit2`, so the two positions whose code needs
bit1 were lost:
- **close_R (code 2 = 010)** -> read as 0 -> merged into **close_center**
- **far_center (code 3 = 011)** -> read as 1 -> merged into **close_L**

Result: cue/lick maps showed only 4 of 6 positions, and close_center/close_L were
contaminated. **Not recoverable from the DAQ alone.** Confirmed: 6/8 (and earlier) recorded
all of 0-5; the fault started 8/5. Functional data (motion/SVD/allen/LocaNMF/photobleach/QC)
is UNAFFECTED — only spout-position labeling.

Hardware fix: repair the strobe bit1 line to the widefield DAQ (owner fixing for 2026-08-07+).

## Recovery from behavior logs
Behavior logs (per-session dirs) live at:
`N:\MICROSCOPE\Priya\Behavior_logs\Widefield\<PSxx>_<YYYYMMDD>_<hhmmss>\trials.csv`
`trials.csv` has the TRUE per-trial `pos_idx`/`pos_name` (all 6 positions) in trial order,
matching the DAQ cue count.

`framemap_event_maps.py` now accepts **`--behavior-trials <trials.csv>`**:
- aligns the behavior `pos_idx` sequence to the DAQ cues by order and verifies
  `DAQ_code == (true_code & ~dead_bit)` (>=98% match; auto-detects the dead bit),
- uses the true positions for cue maps, and assigns each lick to its most-recent cue's true
  position for lick maps.

Batch re-run: **`_maps_behavior_run.py <date>`** re-generates the full cue/lick/quiet map
suite for 8/5 + 8/6 with recovered positions (session -> DAQ + behavior-log mapping inside),
reading/writing N:. Verified: PS92 8/5 aligned offset=0, dead-bit=2, 100% match; all 6
positions restored (close_R=58, far_center=63).

Going forward: once bit1 is fixed the DAQ strobe suffices; `--behavior-trials` is only needed
for 8/5-8/6 (or any future session with a dead strobe bit).

## Fallback when the behavior log is ALSO missing (PS93 2026-08-05) — cam1 video recovery
PS93 8/5 had the dead strobe bit1 **and** an empty behavior log (trials.csv + events.csv both
0 rows), so neither the DAQ nor the log carried the two lost positions. Recovered instead from
the **head-on behavior camera (cam1)** by reading the spout's x-position per trial:
1. **Sync-align DAQ<->cam1.** DAQ digital `sync` line (line0, 5000 Hz) rising edges and cam1
   CSV col3 LSB rising edges are the SAME pulse train (13714 edges each, exact). Map DAQ time
   -> cam frame via `np.interp(t, sync_daq_times, sync_cam_frames)`. (Behavior-log sync events
   also match: 13715.) This is the same alignment that makes DAQ+camera+log agree exactly.
2. **Detect spout x.** For each DAQ cue, grab the cam1 frame at cue+0.6 s (ffmpeg `-ss` seek;
   cv2 lacks the AVI demuxer) and take `spout_x = argmin` of the darkest column over the bright
   head band (`y[140:270], x[170:440]`). The mechanical spout has fixed detents, so spout_x is
   effectively **quantized** into 5 discrete, well-gapped clusters (far_L~425, close_L~374,
   center~327, close_R~251, far_R~202; L=high x, R=low x — head-on view is mirrored).
3. **Classify.** deg4=far_L, deg5=far_R are unambiguous. The two contaminated codes split by a
   2-cluster threshold: deg0 -> close_center (x>=289) vs close_R (x<289); deg1 -> close_L
   (x>=351) vs far_center (x<351). Min per-trial margin from threshold = 10 px (no borderline
   trials). Counts (cc73/cL71/cR67/fc68/fL64/fR68, n=411) match the balanced-cycle marginals.
4. **Apply.** Write a synthetic `trials.csv` from the recovered codes and feed it through the
   existing `--behavior-trials` path (the recovered codes satisfy `DAQ_code==(true & ~bit1)` by
   construction, so the order+bitmask aligner accepts them at 100%, offset=0, dead-bit=2).

Scripts (repo): `_ps93_autodetect.py` (sync map + detect + classify + validation scatter),
`_ps93_verify.py` (reference/closest-call montage), `_ps93_apply.py` (synthetic trials.csv +
regen maps). Validation artifacts archived to N: at
`...\20260805\PS93_20260805_201110\motion_corrected\spout_position_recovery_cam1\`
(`ps93_autolabels.png/.npz`, `ps93_verify_montage.png`, `ps93_recovered_trials.csv`).
Reference position frames (calibration) from PS92 8/5: `_ps93_gui_refs.py`.

**Human-verified.** All 279 ambiguous trials were reviewed trial-by-trial in `_ps93_gui.py`
(closest-to-threshold first) against per-position reference frames: **0 corrections** — the
auto-recovery was confirmed exactly. Final confirmed labels: `ps93_reviewed_trials.csv`
(identical to the auto version). Both are on N: under the session's `spout_position_recovery_cam1\`
(with a `README.txt`), i.e. on MICROSCOPE where the GPU reads (`M:` on the GPU box).

**GPU / LocaNMF.** Position-dependent analysis MUST use the recovered CSV, not the raw DAQ
strobe (bit1 dead -> merged codes). `wfield_local.locanmf_cue_lick_analysis` now honors a
per-session `behavior_trials` key: set it to `...\spout_position_recovery_cam1\ps93_reviewed_trials.csv`
for PS93 8/5 and it overrides the DAQ codes via `_behavior_cue_codes` (order + bitmask, >=98%).
The cue/lick MAP outputs already on N: were generated with these positions.
