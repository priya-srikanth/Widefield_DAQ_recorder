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
