# Nightly widefield pipeline (runbook)

Run after each recording day, once imaging is done and the rig is free. `<DATE>` =
labcams folder name, `YYYYMMDD`. Sessions are `PSxx_<DATE>_<hhmmss>`. Default dims
`2_460_480` (2 channels: ch0 = 415 isosbestic, ch1 = 470 functional).

**Paths**
- Raw + DAQ on E: `E:\labcams_data\<DATE>\<session>\raw_widefield_data\...`,
  `E:\DAQ_recorder_output\PSxx_<DATE>_*.h5` (DAQ files sit loose, not per-session).
- MICROSCOPE (analysis): `N:\MICROSCOPE\Priya\Widefield\labcams\<DATE>\<session>\...`
- Standby (huge files): `M:\Widefield\labcams\<DATE>\<session>\...`
- Deck: `N:\MICROSCOPE\Priya\Widefield\labcams\PS92_94_95_affine8v1.pptx`
- wfield python: `C:\ProgramData\anaconda3\envs\wfield\python.exe`; run with
  `PYTHONPATH=C:\Github\Widefield_DAQ_recorder`.

**Hard rules**
- Use the **sign-fixed** motion correction (`run_wfield_motion` -> `motion_correct_fixed.py`).
- Alignment: **cross-register each session to that animal's 2026-06-06 session** and apply
  6/6's Allen CCF -- the emitted dir MUST be named `allen_aligned_affine8v1` (GPU/LocaNMF
  expects that name). No new per-session landmark placement.
- **Never delete from E:** until checked in that copies are byte-verified. Never delete from
  N: (MICROSCOPE) or any non-Priya folder without explicit per-time permission.
- After any motion redo, the GPU must **re-run LocaNMF** on the corrected inputs.

## Steps (each night)
0. **Discover** sessions/DAQ/dims; confirm no per-session landmark JSONs (expected).
1. **DAQ -> N:** copy `E:\DAQ_recorder_output\*<DATE>*.h5` ->
   `N:\...\DAQ_recorder_output\<DATE>\` (size-verified).
2. **Motion (fixed) + SVD** -- `_mc_svd_<DATE>_run.py` (template: `_mc_svd_0608_run.py`).
   relabel `rescue`; `--mode 2d`. SVD: `run_wfield_local` k=100, `--functional-channel 1`,
   `--fs 31.23 --freq-highpass 0.1 --freq-lowpass 14.0`. (Background; long pole.)
3. **Cross-register to 6/6 + emit allen dir** -- per animal, `wfield_local.cross_day_align`
   with a config: `mode reference-native`, `reference PSxx_0606`, `warp_u true`, sessions =
   {6/6 (results+landmarks on N), <DATE> (results on E)}. Emits
   `wfield_local_results\allen_aligned_affine8v1\` (warped U + both-channel mean to 6/6 CCF +
   6/6 atlas/mask). Target NCC ~0.99. (See the `_xday_PSxx_<DATE>.json` configs.)
4. **LocaNMF inputs -> N: FIRST** (prioritized). The GPU needs ALL of:
   - `wfield_local_results\SVTcorr.npy`
   - `wfield_local_results\allen_aligned_affine8v1\` (U_atlas/atlas/mask)
   - the cleanpairs **frame_map** + summary, which live in the session's `motion_corrected\`
     (NOT in `wfield_local_results\`): `*_cleanpairs_frame_map.npz` + `*_cleanpairs_summary.json`
     -- maps SVT frames <-> physical/DAQ frames so LocaNMF traces align to behavior.
   - DAQ `.h5` (already copied in step 1).
   GPU runs with `--allen-dir ...\allen_aligned_affine8v1`, SVTcorr in the parent dir.
   NOTE: pushing only `wfield_local_results\` MISSES the frame_map -- push the
   `motion_corrected\*cleanpairs_frame_map.npz`+summary too (the step-8 archive copies them
   later, but the GPU needs them with the prioritized push, before archive).
5. **Maps + QC** -- `_maps_<DATE>_run.py` (template: `_maps_0608_run.py`), allen dir =
   `allen_aligned_affine8v1`: cue (`--pre-s 2.0 --post-s 2.0`, shared-scale + contrasts),
   lick (`--post-s 0.15`, contrasts, cue-vs-lick), quiet + quiet-normalized lick, motion QC.
6. **Photobleaching + cross-day intensity** (run while step 2 is going):
   - `_photobleach_<DATE>_run.py` (per-session 415/470 trend + %drift).
   - `_crossday_intensity.py` (brain-ROI median raw counts per animal across days from each
     session's `frames_average`; CAVEAT on the figure: LED is manually titrated day-to-day,
     so a trend may reflect LED, not bleaching).
7. **Deck** -- `_update_ppt_affine8v1.py`: add `<DATE>` sessions to SESSIONS, add a
   photobleach `<DATE>` section, refresh cross-day intensity; it swaps map pictures + refreshes
   motion-QC images in place. Idempotent; writes a `.bak`.
8. **Archive** -- `wfield_local.archive_day archive --date <DATE>`: raw + corrected `.bin`
   -> M:, all other outputs -> N: (LocaNMF inputs first). Leaves reproducible cleanpairs on E:.
9. **Then** return to the prior-session motion redo (`_redo_motion_all.py`), and -- once all
   prior sessions are re-corrected -- cross-register everything to 6/6 + Allen via 6/6.

## E: cleanup (only after explicit check-in)
`wfield_local.archive_day clean --date <DATE>` (dry-run; add `--execute`). It re-verifies the
M:/N: copy (byte size) before deleting and only removes confirmed-copied files + reproducible
intermediates. For a single re-corrected session, delete session-scoped after byte-verifying
its bin on M: + outputs on N:. Keep any new correction not yet transferred to standby.

See also: `TASKS.md` (decisions), `MOTION_CORRECTION_SIGN_BUG.md`, `MOTION_REDO_STATUS.md`.
