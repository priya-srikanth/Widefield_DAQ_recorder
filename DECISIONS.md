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

## Decomposition: SVD + atlas now; PMD/LocaNMF not yet

Our local pipeline (`run_wfield_local`) does **SVD** (wfield `approximate_svd`,
mean-centered ΔF/F, `divide_by_average=True`, k≈100) → **hemodynamic correction in
SVD space** → **Allen landmark alignment**. Activity maps are `U @ SVTcorr` averaged
over event windows. We do **NOT** currently run **PMD** (penalized matrix
decomposition denoising) or **LocaNMF** (localized semi-NMF), which the wfield /
NeuroCAAS protocol (Couto et al., *Nat Protoc* — PMC8788140) recommends as the next
steps after SVD: PMD denoises/compresses, then LocaNMF re-factorizes the low-rank
data into **non-negative, anatomically-localized components anchored to Allen
regions** (Saxena et al., *PLoS Comput Biol* 2020, pcbi.1007791). Versus raw SVD
components (delocalized, not reproducible across sessions), LocaNMF components are
interpretable and **reproducible across sessions and animals** — the natural unit
for cross-animal and functional-subnetwork analyses (e.g. Nat Neurosci
s41593-022-01245-9).

Decision: stay on SVD + atlas for evoked maps and within-animal work (adequate).
Add LocaNMF when we move to cross-animal / subnetwork analysis. Constraint: this
machine has **no CUDA GPU and no torch/locanmf installed**; LocaNMF is GPU-oriented,
so the practical paths are (a) **NeuroCAAS cloud** (the intended wfield route; we
have `wfield_local/wfield_ncaas_fixed.py`), or (b) a GPU box with `locanmf`+`torch`.
LocaNMF consumes exactly what we already produce (low-rank `U`/`SVTcorr` + the
`allen_area_atlas_native_grid` atlas), so it is a clean bolt-on.

## Cross-day and cross-animal alignment policy

- **Within animal, across days**: register the motion-corrected **mean 470 nm
  vasculature** to a single chosen reference session (`cross_day_align.py`):
  landmark-init → intensity-based ECC affine refine (SIFT+RANSAC fallback),
  composed into the reference/CCF frame. Vasculature is a far denser, more
  repeatable fiducial than the ~8 landmarks, whose independent per-session errors
  otherwise compound day-to-day. Keep the **same ROI/zoom** across days (full-FOV↔ROI
  pairs register poorly). QC = masked NCC + red/green vessel overlay.
- **Across animals**: vasculature is not shared, so the **only** common frame is the
  Allen atlas (landmarks). Compare at the **ROI / Allen-area level** (or via LocaNMF
  components), not pixelwise; group pixel maps are for visualization only.

## Outline rendering fix (atlas overlay)

Region outlines are now drawn by the shared `wfield_local/atlas_overlay.region_edges`
(used by all plot modules). The earlier per-module version marked only the upper/left
pixel of each label transition then masked to labeled pixels, dropping the brain's
**left and anterior (top) outer borders** (the open left-anterior / olfactory-bulb
edge). The shared version marks both pixels of each transition before masking, so the
outline closes all around (verified left border 96/524 → 524/524).

## Server layout, archiving & safety

Two institutional file servers (copy-only; **never delete from a server without
explicit per-action permission**, and **only ever write inside the `Priya\` folder**):

- **M: (standby)** = `\\standby.files.med.harvard.edu\hms\neurobio\sabatini\collaborations\Priya`.
  **Raw, un-motion-corrected** `.dat` (raw_widefield_data + the 6/1 full-FOV session
  files) → `M:\Widefield\labcams_raw_data\<date>\<animal>\`. Only the raw camera
  `.dat`; not camlogs/cleanpairs/motioncorrect/analysis. Copy → verify sizes → confirm
  → then delete E: originals.
- **N: (MICROSCOPE)** = `\\research.files.med.harvard.edu\Neurobio`, folder
  `N:\MICROSCOPE\Priya\`. **Analyzed data** (motion-corrected `.bin` videos, SVD,
  Allen alignment, maps/QC, PPTs) → `N:\MICROSCOPE\Priya\Widefield\labcams\<rel path>\`.
  Copy excludes the regenerable raw + cleanpairs `*_uint16.dat`. This is also where the
  GPU machine reads inputs for LocaNMF.

See the [[microscope-server-safety]] memory for the hard rules.

## LocaNMF (run on the GPU machine)

`wfield_local/run_locanmf.py` runs `wfield.local_nmf.compute_locaNMF` on an
`allen_aligned_*` folder (`U_atlas` + `allen_area_atlas_native_grid` +
`allen_brain_mask_native_grid` + `SVTcorr`) → localized components `A`/`C`/`regions` +
montage. Needs PyTorch (the `torch` package) + the `locanmf` package + a CUDA GPU; this
rig PC has none, so it runs on the NVIDIA box. `GPU_LOCANMF_KICKOFF.md` is the paste-ready
kickoff (clone repo → set up torch+locanmf env matching the GPU's CUDA → read data from
`N:\MICROSCOPE\Priya\...` → run). There is no maintained newer-Python *prebuilt* locanmf;
newer Python compiles the extension from source (see the script header).

## Significant local-analysis modules (added during this work)

- `wfield_local/atlas_overlay.py` — shared region-outline helper (the fix above).
- `wfield_local/framemap_event_maps.py` — cue/lick maps for **relabeled cleanpairs**
  movies (regime B), generalizing the one-off `_ps92_spout/_ps92_lick`; emits the
  same filenames as the stock plotters so downstream contrast/mean/cue-vs-lick steps
  are reused. `chosen_exposure_offset` is read per session.
- `wfield_local/qc_motion_correction.py` — per-session motion-correction QC (shift
  traces + magnitude histogram, raw-vs-corrected sharpness, residual-motion std,
  pass/warn verdict).
- `wfield_local/cross_day_align.py` — within-animal cross-day vasculature registration
  (above).
- `wfield_local/run_locanmf.py` + `GPU_LOCANMF_KICKOFF.md` — LocaNMF (and sNMF via
  `--mode`) on the GPU box.
- `wfield_local/roi_activity.py` — CPU Allen-area ROI traces (region-averaged ΔF/F)
  + optional cue/lick per-region responses; lightweight baseline alongside LocaNMF.
- `wfield_local/quiet_periods.py` — quiet-period (not running/licking/peri-reward)
  per-frame mask for behavior-controlled baseline F0; ported from the
  stroke_orofacial_pipeline `find_quiet_bouts`, adapted for one spout.

## Quiet-period baseline (and params to tune later)

Trial-triggered acquisition records no true inter-trial rest, so for a behavior-
controlled baseline we detect "quiet" frames within the recording (not running, not
near a lick, not peri-reward) and intersect with the pre-cue ENL window (or pool as
F0). Logic is ported from the stroke pipeline (`find_quiet_bouts`). Two rig-specific
decisions: (1) **grooming OFF by default** — the stroke detector needs two spouts
(bilateral conjunction); single-spout long-touch is unreliable because a true long
lick at our close spouts also looks long. (2) **thresholds are provisional** —
running/quiet speed, min durations, and lick/reward/treadmill buffers are stroke
defaults (the 8 s reward buffer is generous for short ENL); **tune per rig/task**,
ideally validated against DLC/FaceRhythm movement (the future movement regressor) —
not yet available. Done on the `quiet-period-baseline` branch to avoid colliding with
the GPU machine's LocaNMF work on `main`.
- `run_wfield_local` — added `--detrend-order` and exposed `--freq-highpass` /
  `--freq-lowpass` (the default 0.1 Hz highpass already removes the slow 415 LED
  drift; detrend is for when a gentler highpass is wanted).

## Data lifecycle, archival & deletions (2026-06-04)

Storage now has three tiers; **new analysis outputs go to N: (`...\Priya\...`)** going forward:
- **Raw** `.dat` -> **M:** standby, verified, then deleted from E: (~648 GB freed).
- **Analyzed** (motion-corrected `.bin`, SVD/alignment, maps, QC, decks) -> **N:**
  `MICROSCOPE\Priya\Widefield\labcams`, verified (0 missing).
- **DAQ** `.h5` -> **N:** `MICROSCOPE\Priya\Widefield\DAQ_recorder_output`, verified,
  then deleted from E: (4.5 GB).
- Deleted from E: after verification: the motion-corrected `.bin` (~621 GB, on N:) and
  the **cleanpairs movies** `*_cleanpairs_*_uint16.dat` (~340 GB, regenerable from the
  M: raw via relabel; intentionally NOT archived).
- **Kept on E:** SVD/alignment/maps/QC outputs (for fast local re-analysis) and the
  small `*_cleanpairs_frame_map.npz/.csv` + `*_cleanpairs_summary.json` (needed for
  regime-B event alignment; these ARE also on N:). All server ops are copy-only and
  only inside `Priya\` (see [[microscope-server-safety]]).

## Relabel step for future recordings (latest firmware)

The relabel/cleanpairs step is still recommended even with the current trial-gated
acquire-enable firmware: the 6/3 acquire-enable recordings still had ~100-180 stray
illuminated/dark frames that relabel dropped, and relabel guarantees deterministic
415/470 pairing + the `frame_map` that regime-B event alignment needs. Use
`--relabel-mode acquire-enable` for trial-gated recordings (the lighter mode);
`rescue` is for the older continuously-saved sessions. Note: the cleanpairs **movie**
is a deletable, regenerable intermediate, but the relabel **step** and its small
`frame_map` stay in the pipeline. (If a future session's saved `.dat` is provably
clean — DAQ pco count == saved frame count, consistent parity, no stray frames — the
standard raw//2 regime-A mapping like the 6/1 sessions can be used instead.)

## Quiet-normalized lick activity (workflow)

`quiet_periods.py` -> `*_quiet_frame.npy`; pass it as `--quiet-frame` to the lick
plotter (`plot_lick_aligned_averages` regime A / `framemap_event_maps --what lick`
regime B) to emit both the raw post-lick map and a `*_quietnorm*` map (post-lick minus
the mean quiet-period baseline = lick-evoked relative to the not-running/not-licking
state). Quiet-period thresholds are provisional (see the quiet-period section).

## Things still to verify

- 6/3 PS92 / PS95 functional-channel identity (PS94 6/3 was verified correct).
  (6/3 PS94 & PS95 SVD + maps + QC are now complete and in the deck.)
