# Ongoing tasks & decisions (widefield pipeline)

## Standing nightly pipeline
The full per-night runbook (steps, scripts, params, conventions) lives in
**`NIGHTLY_PIPELINE.md`** -- that is the source of truth. Summary: DAQ->N; fixed motion +
SVD; cross-register each session to that animal's 2026-06-06 session and apply 6/6's Allen
CCF (emit `allen_aligned_affine8v1`, no new landmarks); LocaNMF inputs->N first; cue/lick
maps; photobleaching + cross-day intensity; deck; raw+bin->M. Don't delete E: until checked in.

Deck: `N:\MICROSCOPE\Priya\Widefield\labcams\PS92_94_95_affine8v1.pptx`.

## 6/6 reference landmarks for cross-register-all (use v2 where present)
When doing the final cross-register-all-to-6/6, the per-animal 6/6 reference Allen CCF uses
these landmark JSONs (in each 6/6 session's `raw_widefield_data\`, on N:):
- **PS92 -> `dorsal_cortex_landmarks_v2.json`** (re-done 2026-06-09)
- **PS93 -> `dorsal_cortex_landmarks_v2.json`** (re-done 2026-06-09)
- PS94 -> `dorsal_cortex_landmarks_v1.json`
- PS95 -> `dorsal_cortex_landmarks_v1.json`
Also recompute PS92 & PS93's 6/6 reference `allen_aligned_affine8v1` with their v2 landmarks
so the reference session itself is consistent with what the other days are aligned to.

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

## Re-run LocaNMF after a motion redo (GPU lane)
Any session whose motion correction is re-done with the sign-fixed code gets a NEW
`SVTcorr.npy` + `allen_aligned_affine8v1/` (its SVD and CCF-aligned U change). **Any
LocaNMF result computed on the pre-fix inputs is stale and must be re-run** against the
corrected inputs now on N:. This applies to every session in the bulk redo batch
(`_redo_motion_all.py`) once it lands, plus PS93 6/6 (already corrected) and PS94 6/5.
GPU: re-run LocaNMF for a session after its `wfield_local_results` mtime updates on N:.

## Allen-dir naming (GPU/LocaNMF)
Cross-session-to-6/6 emits the CCF allen dir as **`allen_aligned_affine8v1`** (the
standard name the GPU/LocaNMF, maps, and deck all expect) -- it CONTAINS the 6/6-CCF
alignment. Do not use a custom name (e.g. xday6) on N: or the GPU won't find it.
