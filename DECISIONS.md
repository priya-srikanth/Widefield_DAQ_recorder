# Analysis decisions: global vs session-specific

This records the choices behind the widefield analysis pipeline so future runs are
reproducible and the per-session quirks are explicit. "Global" = applies to every
session; "Session-specific" = decided per recording because the data differ.

Last updated: 2026-06-03.

## Global decisions

- **Dual-wavelength**: 470 nm = functional (GCaMP), 415 nm = isosbestic reference.
  Hemodynamic correction = 470 − β·415, fit in SVD space by `wfield`
  (`hemodynamic_correction`), which highpass-filters both channels at 0.1 Hz
  (this already removes slow LED drift — see photobleaching note below).
- **Channel identity comes from the DAQ LED TTLs**, not frame parity. The
  relabel step (`trim_illuminated_labcams`) assigns 415/470 from
  `led415_ttl`/`led470_ttl`, so channel identity is correct regardless of the
  per-session parity ambiguity.
- **Allen alignment grid**: all spatial maps are warped to the **540×640 Allen
  atlas grid** (`apply_allen_transform --dims 540 640`), not the native ROI size,
  with the atlas built in **reference space** (`do_transform=False`). This keeps
  ROI-cropped recordings aligned with the atlas.
- **Alignment transform (this batch)**: **8-point affine** with the lateral
  anchors (OB_center/L/R, RSP_base, MOp_L/R, SS_L/R), from the hand-placed
  `dorsal_cortex_landmarks_v1.json` per session. The lateral MOp/SS points break
  the medial collinearity so an affine (independent AP/ML scale + shear) is
  well-constrained, vs the earlier 4-point similarity used in the original deck.
  Output dirs/labels use the tag **`affine8v1`** so nothing prior is overwritten.
- **Cue-aligned maps**: per spout position, mean over 1 s pre-cue and 1 s
  post-cue, plus the post−pre **delta**. Spout position from
  `spout_strobe` + `spout_bit0/1/2` (code = bit0 + 2·bit1 + 4·bit2 at the most
  recent strobe before the cue).
- **Lick-aligned maps**: per spout position, mean over 150 ms post-lick. Licks
  from `lick_analog` (upper/lower thresholds 2.5/1.0 V, 1–20 ms lockout, 100 ms
  refractory). **Delta-position lick maps** = pairwise position contrasts
  (position A post-lick − position B post-lick).
- **Mean image + Allen overlay**: 415/470 mean motion-corrected frames warped to
  the atlas grid, with Allen region outlines (from `frames_average_atlas.npy`).
- **Frame rate**: 31.23 Hz per channel.
- **Versioning of figure dirs/labels** has historically tracked **code
  iteration** (e.g. `_v2`, `_v6`), NOT the landmark-JSON version. This batch uses
  `affine8v1` to denote the 8-pt-affine landmarks-v1 alignment.

## Event→frame mapping: two regimes (session-specific)

The corrected movie (SVTcorr) is indexed by paired 415/470 timepoints. Mapping a
DAQ event (cue/lick) to a corrected-frame index depends on whether the movie was
relabeled:

- **Regime A — raw recording (no relabel)**: corrected frame = (nearest
  `pco_exposure` pulse to the event) // 2. Used when the movie is the full,
  contiguous recording. (`frame_align=pco` in the stock plotters.)
- **Regime B — relabeled "cleanpairs" movie (`--relabel-mode rescue`)**: the
  movie is a non-contiguous subset of kept 415/470 pairs, so raw//2 is wrong.
  Each corrected frame `t` maps to DAQ sample
  `pco_samples[frame_map["original_frame_index_ch0"][t] + chosen_exposure_offset]`;
  events map to the nearest such sample. `chosen_exposure_offset` is read from the
  per-session `*_cleanpairs_summary.json` (**differs per session**). A contiguity
  guard rejects windows that cross a trial/kept-frame boundary.

## Per-session table

| Session | FOV (native) | Relabel | Mapping | DAQ h5 | Notes |
|---|---|---|---|---|---|
| 6/1 PS94 (`20260601/PS94_20260601_141614`) | 540×640 full | no | A (raw//2) | `PS94_baseline_20260601_141642.h5` | "baseline"-named file but contains task cue/strobe/lick |
| 6/1 PS95 (`20260601/PS95_20260601_153653`) | 540×640 full | no | A (raw//2) | `PS95_baseline_20260601_153627.h5` | same |
| 6/2 PS92 (`20260602/PS92/PS92_20260602_151820/illuminated_rescue`) | 487×480 ROI | yes (offset 1) | B (frame_map) | `PS92_20260602_152607.h5` | **functional-channel swap fixed**: use `SVTcorr.npy` (the `*_functional1_WRONG.npy` are bad) |
| 6/3 PS92 (`20260603/PS92/PS92_20260603_104008`) | 477×464 ROI | yes (offset 0) | B (frame_map) | `PS92_20260603_104607.h5` | functional channel assumed correct via DAQ relabel (verify) |
| 6/3 PS94 (`20260603/PS94`) | 462×464 ROI | yes | B (frame_map) | `PS94_20260603_175946.h5` | SVD pending at time of writing |
| 6/3 PS95 (`20260603/PS95/.../PS95_20260603_194442`) | 462×464 ROI | yes | B (frame_map) | `PS95_20260603_194902.h5` | motion+SVD pending at time of writing |

## Photobleaching / LED drift (context)

Across sessions the **isosbestic 415 declines ~9–16%** over a session while the
**functional 470 is stable (±2–3%, two longest sessions −6 to −7%)**. Because true
GCaMP photobleaching would hit 470 hardest, the 415-specific decline is attributed
to **violet-LED drift**, not fluorophore bleaching. The hemo-correction's 0.1 Hz
highpass already removes this slow drift, so it does not contaminate ΔF/F.
`run_wfield_local` also exposes `--detrend-order` + `--freq-highpass` for cases
where a gentler highpass is wanted (keep slow signal, still remove LED drift).

## Things still to verify

- 6/3 PS92 / PS95 functional-channel identity (PS94 6/3 was verified correct).
- 6/3 PS94 & PS95 SVD completion before their maps can be made.
