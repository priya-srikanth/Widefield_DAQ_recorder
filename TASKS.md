# Ongoing tasks & decisions (widefield pipeline)

## Standing daily workflow (per recording day)
1. DAQ recorder `.h5` -> `N:\MICROSCOPE\Priya\Widefield\DAQ_recorder_output\<date>\`.
2. **Fixed** motion correction (`run_wfield_motion` -> `wfield_local/motion_correct_fixed.py`)
   + SVD/hemo (`run_wfield_local`, k=100, fc=1, hp0.1/lp14) per session. relabel rescue.
3. **Alignment policy (NEW, 2026-06-07):** do **cross-session registration to that animal's
   2026-06-06 session** (reference-native, FFT phase-correlation) and use the **6/6
   session's Allen CCF alignment for every session from here on out** — no new per-session
   landmark placement. The 6/6 landmark fit is the single CCF bridge per animal.
4. Copy to MICROSCOPE (N:): non-bin outputs (SVD, Allen/CCF-aligned U+atlas+mask, motion
   summary/shifts, cleanpairs frame_map/summary), camlogs, any landmark JSONs.
   **Prioritize LocaNMF inputs** (SVTcorr + the CCF-aligned U_atlas/atlas/mask) so the GPU
   can start per session as they land.
5. Photobleaching (per-session 415/470 trend) + motion-correction QC -> deck.
   Also track **cross-day raw fluorescence intensity** (caveat: LED power is manually
   titrated day to day, so a trend may reflect LED setting, not bleaching).
6. Cue (2 s post / 2 s pre delta) + lick (150 ms, +/- quiet-period) maps -> deck.
7. Raw `.dat` + corrected `.bin` -> M: standby (`M:\Widefield\labcams\<date>\<session>\`).
8. **Do NOT delete anything on E: until checked in** that copies are verified.

Deck: `N:\MICROSCOPE\Priya\Widefield\labcams\PS92_94_95_affine8v1.pptx`.

## Motion-correction sign-bug remediation (in progress)
wfield 0.4.2 doubled drift (sign error); fixed in `wfield_local/motion_correct_fixed.py`
(see `MOTION_CORRECTION_SIGN_BUG.md`). Re-processing ALL prior sessions with the fix,
newest -> oldest (`_redo_motion_all.py`, resumable; status in `MOTION_REDO_STATUS.md`).
- DONE: PS93 2026-06-06.
- PENDING: PS94 2026-06-05 (`_redo_ps94_0605.py`); then the bulk batch for the rest.
**After all sessions are re-corrected**, run cross-session registration of every session
to its 2026-06-06 reference and Allen-align using the 6/6 reference (per the policy above),
then refresh the deck + cross-day QC.

## Servers
- M: standby = raw `.dat` + corrected `.bin` (huge files), `M:\Widefield\labcams\<date>\<session>\`.
- N: MICROSCOPE = analyzed (SVD, CCF-aligned, maps, QC, DAQ, deck); NOT the `.bin`.

## Allen-dir naming (GPU/LocaNMF)
Cross-session-to-6/6 emits the CCF allen dir as **`allen_aligned_affine8v1`** (the
standard name the GPU/LocaNMF, maps, and deck all expect) -- it CONTAINS the 6/6-CCF
alignment. Do not use a custom name (e.g. xday6) on N: or the GPU won't find it.
