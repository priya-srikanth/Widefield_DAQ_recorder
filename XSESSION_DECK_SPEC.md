# cross-session_aligned deck spec

New deck: `N:\MICROSCOPE\Priya\Widefield\labcams\cross-session_aligned.pptx`.
Built AFTER the bulk motion redo + cross-register-all-to-6/6 (so maps/alignment are in the
6/6 CCF frame). Generator: `_build_xsession_deck.py` (to write).

## Scope
- Dates: **6/5, 6/6, 6/7, 6/8** (per animal, where available).
- Animals: PS92, PS93, PS94, PS95.
- Alignment = each session cross-registered to that animal's 6/6, in 6/6 CCF
  (`allen_aligned_affine8v1`; PS92/PS93 6/6 reference uses v2 landmarks).

## Layout: per-animal sections
Title slide, then the single **cross-date photobleaching** figure (overview), then for each
animal a section with these families in order, each family showing all dates sequentially
(6/5 -> 6/6 -> 6/7 -> 6/8):
1. Mean 415/470 + Allen overlay (from `allen_aligned_affine8v1` / frames_average_atlas).
2. Cue maps (shared-scale delta).
3. Lick maps (150 ms by spout).
4. Quiet-normalized lick maps.
5. Per-session photobleach QC (`_photobleach_out_<date>\photobleach_PSxx_<date>.png`).
6. Alignment image with reference/Allen overlay (cross-day overlay + Allen boundaries).
7. Motion QC (`motion_qc\PSxx_<date>_affine8v1_motion_qc.png`).

## Prereqs to generate before building
- Cross-register-all-to-6/6 done for 6/5-6/8 (allen overlays in 6/6 frame).
- **Regenerate 6/5 maps at the 2 s/2 s cue window** (orig 6/5 used 2 s/1 s) + lick +
  quietnorm lick + motion QC for all four 6/5 sessions, so 6/5 is equivalent to 6/6-6/8.
- Motion QC regenerated for any redone session lacking a fresh `_motion_qc.png`.

## Notes
- Keep the existing `PS92_94_95_affine8v1.pptx` deck as-is; this is a separate, results-
  focused, animal-grouped view.
