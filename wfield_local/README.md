# Local wfield Processing Helpers

This folder contains local processing scripts used with `jcouto/wfield` for the widefield behavior rig. They are designed to run in the `wfield` conda environment and to keep large imaging outputs outside git.

Typical environment:

```powershell
conda activate wfield
cd "C:\Github\Widefield_DAQ_recorder"
```

## Processing Overview

1. Motion-correct the labcams `.dat` file (optionally QC it, §13).
2. Run local SVD and dual-color hemodynamic correction.
3. Apply Allen/wfield landmark alignment.
4. Generate cue-aligned spout-position maps (relabeled sessions: §12).
5. Generate optional alignment diagnostics and comparison PowerPoints.
6. Generate optional lick-aligned post-event maps.
7. Optionally align the same animal across days on the mean 470 nm vasculature (§14).

> Decomposition stops at SVD + hemo + atlas here; PMD/LocaNMF are not run locally
> (see the "Decomposition note" at the bottom and `DECISIONS.md`).

Large outputs should stay in the recording folder, usually under:

```text
E:\labcams_data\YYYYMMDD\SESSION\motion_corrected
```

## Notebooks

- `notebooks/widefield_local_analysis_walkthrough.ipynb` - annotated end-to-end workflow for generating motion/SVD outputs, Allen alignment, cue maps, lick maps, cue-vs-lick maps, alignment diagnostics, ROI label maps, and comparison PowerPoints.
- `notebooks/local_wfield_processing_template.ipynb` - shorter command template for starting a new session.

## 1. Motion Correction

Example:

```powershell
python .\wfield_local\run_wfield_motion.py `
  "E:\labcams_data\20260601\PS95_20260601_153653\raw_widefield_data\pco_edge_run000_00000000_2_540_640_uint16.dat" `
  --output-dir "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected" `
  --mode 2d
```

Important adjustable parameters:

- `--mode 2d`: lean rigid XY correction. This is the default practical choice.
- `--mode ecc`: slower, less lean option that can estimate a richer transform. Try on a subset first.
- `--output-dir`: where corrected `.bin` and shift summaries are written.

The raw `.dat` file is not overwritten. (The script's flag is `--output`.)

### TTL-based LED relabeling (trial-gated recordings)

For trial-gated recordings, run motion correction with `--daq-h5` so the DAT is
relabeled to DAQ-confirmed 415/470 pairs BEFORE motion correction and SVD. The
saved `.dat` channel split is only a frame-parity interpretation, which can drift
from the true LED across trials; relabeling from the DAQ `pco_exposure` /
`led415_ttl` / `led470_ttl` channels makes channel 0 = 415, channel 1 = 470
deterministically (and drops any dark inter-trial frames).

```powershell
python .\wfield_local\run_wfield_motion.py `
  "E:\labcams_data\...\raw_widefield_data\pco_edge_run000_00000000_2_H_W_uint16.dat" `
  --daq-h5 "E:\DAQ_recorder_output\<session>.h5" `
  --relabel-mode acquire-enable `
  --output "E:\labcams_data\...\motion_corrected"
```

This writes a `*_daq_led_cleanpairs_*.dat` (plus a frame map) into the output
folder and motion-corrects that. Use `--relabel-mode rescue` for older
continuously-saved (LED-gated) sessions that contain dark inter-trial frames.
The relabel can also be run standalone via
`python -m wfield_local.trim_illuminated_labcams <dat> <daq.h5> --output-dir <dir> --mode acquire-enable`.

## 2. SVD And Hemodynamic Correction

Example:

```powershell
python .\wfield_local\run_wfield_local.py `
  "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\motioncorrect_2_540_640_uint16.bin" `
  --output-dir "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\wfield_local_results" `
  --functional-channel 1 `
  --n-components 100
```

Important adjustable parameters:

- `--functional-channel 1`: for this rig, channel 0 is 415 nm and channel 1 is 470 nm, so 470 is the functional channel. (PS92 6/2 rescue was recorrected with `--functional-channel 0` after a swap.)
- `--n-components`: SVD component count. `100` has been used for PS94/PS95.
- Chunk/memory parameters in the script can be adjusted if a computer runs out of RAM.
- `--freq-highpass` / `--freq-lowpass`: hemo-correction filter cutoffs (defaults 0.1 / 14 Hz). The default 0.1 Hz highpass already removes the slow 415 nm LED drift, so it does not leak into ΔF/F.
- `--detrend-order N`: optional per-component polynomial detrend (per channel) applied before the β regression. Use with a lowered `--freq-highpass` to strip slow LED drift while keeping slow neural signal. Default off; the running pipeline behavior is unchanged.

> The activity maps are `U @ SVTcorr` averaged over an event window, in **fractional ΔF/F** relative to the **session-mean image** (`divide_by_average=True`), hemo-corrected and high-pass filtered. The cue figure's `post-pre` delta is the baselined evoked map; the lick maps are post-only (vs session mean), so treat them as activity relative to session mean unless a pre-lick baseline is added.

Outputs include:

- `U.npy`
- `SVT.npy`
- `SVTcorr.npy`
- `frames_average.npy`
- `rcoeffs.npy`
- `T.npy`

## 3. Allen Alignment

After making or revising `dorsal_cortex_landmarks.json` in the wfield/NeuroCAAS GUI, apply it locally:

```powershell
python .\wfield_local\apply_allen_transform.py `
  "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\wfield_local_results" `
  --landmarks "E:\labcams_data\20260601\PS95_20260601_153653\raw_widefield_data\dorsal_cortex_landmarks.json" `
  --output "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\wfield_local_results\allen_aligned_v6"
```

Use a versioned output folder, such as `allen_aligned_v6`, whenever you are comparing landmark attempts.

Important note:

- `wfield` stores a transform that maps reference/atlas landmark coordinates to clicked image coordinates.
- Image warping uses that transform through `skimage.warp`, which treats the transform as an output-to-input inverse map.
- The helper diagnostic scripts account for this when plotting where clicked points land after warping.

## 4. Cue-Aligned Spout-Position Averages

By default, DAQ events are aligned to imaging frames using the DAQ-recorded
`pco_exposure` rising edges. This is more robust than wall-clock timestamps:
the cue sample is mapped to the nearest PCO exposure pulse index, then divided
by two because each 415/470 raw frame pair becomes one hemodynamic-corrected
timepoint in `SVTcorr.npy`. Passing `--camlog` is still useful because the
summary JSON records frame-count QC against the labcams camlog.

Example:

```powershell
python .\wfield_local\plot_spout_trial_averages.py `
  --label PS95_v6 `
  --daq-h5 "E:\DAQ_recorder_output\PS95_baseline_20260601_153627.h5" `
  --wfield-results "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\wfield_local_results" `
  --allen-dir "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\wfield_local_results\allen_aligned_v6" `
  --camlog "E:\labcams_data\20260601\PS95_20260601_153653\pco_edge_run000_00000000.camlog" `
  --output "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\spout_trial_averages_allen_v6" `
  --frame-align pco `
  --pre-s 1.0 `
  --post-s 1.0 `
  --fs 31.23
```

Adjustable parameters:

- `--pre-s`: seconds before cue.
- `--post-s`: seconds after cue.
- `--fs`: hemodynamic-corrected paired-frame sampling rate. For PS94/PS95 this was `31.23`.
- `--frame-align pco`: use DAQ `pco_exposure` pulse order. Use `camlog` only for legacy wall-clock reproduction.
- `--activity-percentile`: display scaling percentile for pre/post panels.

Spout position is assigned using the most recent `spout_strobe` before each cue:

```text
code = spout_bit0 + 2*spout_bit1 + 4*spout_bit2
```

## 5. Shared-Scale Spout Figures

The original cue-aligned plot uses one scale for pre/post and a separate scale for post-minus-pre. To make pre, post, and delta visually comparable, regenerate a shared-scale figure:

```powershell
python .\wfield_local\plot_spout_trial_averages_shared_scale.py `
  --label PS95_v6 `
  --trial-maps "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\spout_trial_averages_allen_v6\PS95_v6_spout_positions_1s_pre_post_delta_maps.npz" `
  --allen-dir "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\wfield_local_results\allen_aligned_v6" `
  --summary "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\spout_trial_averages_allen_v6\PS95_v6_spout_positions_1s_pre_post_delta_summary.json" `
  --output "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\spout_trial_averages_allen_v6"
```

## 6. Alignment Diagnostics

Plot clicked landmark points before and after the transform:

```powershell
python .\wfield_local\plot_alignment_before_after.py `
  --label PS95 `
  --json-dir "E:\labcams_data\20260601\PS95_20260601_153653\raw_widefield_data" `
  --results "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\wfield_local_results" `
  --output "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\alignment_before_after" `
  --current-version v6
```

Plot mean 470 nm with Allen outlines and landmark points:

```powershell
python .\wfield_local\plot_alignment_landmark_overlays.py `
  --label PS95 `
  --json-dir "E:\labcams_data\20260601\PS95_20260601_153653\raw_widefield_data" `
  --results "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\wfield_local_results" `
  --output "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\alignment_landmark_overlays" `
  --current-version v6
```

Plot the fixed Allen/wfield reference landmarks:

```powershell
python .\wfield_local\plot_allen_reference_landmarks.py `
  --landmarks "E:\labcams_data\20260601\PS95_20260601_153653\raw_widefield_data\dorsal_cortex_landmarks.json" `
  --output "E:\labcams_data\20260601\allen_wfield_reference_landmark_targets.png"
```

Plot a color-coded Allen ROI label map:

```powershell
python .\wfield_local\plot_allen_roi_labels.py `
  --allen-dir "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\wfield_local_results\allen_aligned_v6" `
  --output "E:\labcams_data\20260601\allen_wfield_roi_labels_ps95_v6.png" `
  --title "Allen/wfield ROI labels from PS95 v6 atlas"
```

## 7. Alignment Comparison PowerPoint

Build a comparison deck from the generated PNGs:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  "C:\Github\Widefield_DAQ_recorder\wfield_local\build_alignment_comparison_ppt.ps1" `
  -OutputPath "E:\labcams_data\20260601\alignment_comparison_PS94_PS95_shared_scale.pptx"
```

This uses local PowerPoint COM automation. PowerPoint must be installed on the machine.

## 8. Post-Lick Averages

Post-lick averages use analog `lick_analog` falling threshold crossings. No pre-lick baseline is used by default, because lick bouts can make pre-lick windows hard to interpret.

Like cue-aligned maps, lick events default to `--frame-align pco`, so lick
sample indices are mapped through DAQ-recorded PCO exposure pulses rather than
labcams wall-clock timestamps. The optional `--camlog` path adds frame-count QC
to the output summary.

Example:

```powershell
python .\wfield_local\plot_lick_aligned_averages.py `
  --label PS95_v6 `
  --daq-h5 "E:\DAQ_recorder_output\PS95_baseline_20260601_153627.h5" `
  --wfield-results "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\wfield_local_results" `
  --allen-dir "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\wfield_local_results\allen_aligned_v6" `
  --camlog "E:\labcams_data\20260601\PS95_20260601_153653\pco_edge_run000_00000000.camlog" `
  --output "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\lick_aligned_v6" `
  --frame-align pco `
  --lick-thresh-upper-v 2.5 `
  --lick-thresh-lower-v 1.0 `
  --refractory-s 0.10 `
  --post-s 0.150 `
  --fs 31.23
```

Adjustable parameters:

- `--lick-thresh-upper-v`: lick onset threshold. For PS95, licks dropped from ~5.5 V to ~0 V, so `2.5` V is a reasonable default.
- `--lick-thresh-lower-v`: lick offset threshold for hysteresis. `1.0` V has been used for PS94/PS95.
- `--refractory-s`: minimum separation between lick events. `0.10` s avoids counting one lick as many threshold crossings.
- `--post-s`: post-lick window duration. `0.150` s was requested for PS95.
- `--fs`: paired-frame hemodynamic-corrected sampling rate.

Outputs:

- `*_lick_aligned_150ms_post_by_spout.png`
- `*_lick_aligned_150ms_post_by_spout_maps.npz`
- `*_lick_aligned_150ms_post_by_spout_summary.json`

The lick detector is in `lick_detection.py`. It ports the double-threshold
hysteresis + lockout logic from the stroke/orofacial pipeline:

- onset: voltage crosses below `thresh_upper`
- offset: voltage crosses above `thresh_lower`
- cleanup: drop onset events inside a post-offset lockout window
- optional refractory: collapse dense lick bouts for imaging-triggered averages

## 9. Cue vs Lick Spout-Position Comparisons

Compare cue-aligned and lick-aligned maps for the same aligned session:

```powershell
python .\wfield_local\plot_lick_vs_cue_spout_maps.py `
  --label PS95_v6 `
  --cue-maps "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\spout_trial_averages_allen_v6\PS95_v6_spout_positions_1s_pre_post_delta_maps.npz" `
  --lick-maps "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\lick_aligned_v6\PS95_v6_lick_aligned_150ms_post_by_spout_maps.npz" `
  --allen-dir "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\wfield_local_results\allen_aligned_v6" `
  --cue-summary "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\spout_trial_averages_allen_v6\PS95_v6_spout_positions_1s_pre_post_delta_summary.json" `
  --lick-summary "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\lick_aligned_v6\PS95_v6_lick_aligned_150ms_post_by_spout_summary.json" `
  --output "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\lick_aligned_v6"
```

The figure columns are:

- cue-aligned post map
- lick-aligned post map
- `lick - cue`

By default the labels are `1 s post-cue` and `150 ms post-lick`; these are not identical windows, so interpret the third column as a descriptive contrast rather than a pure event-type subtraction.

## 10. Sync-Pulse Timebase Alignment

`frame_sync.py` ports the sync-pulse alignment algorithm from the
stroke/orofacial pipeline. Use it when you need to align DAQ/WaveSurfer-like
sync channels to camera timestamp CSVs without relying only on wall-clock
timestamps.

Main entry point:

```python
from wfield_local.frame_sync import make_alignment_template

template = make_alignment_template(
    ws_signals={
        "Sync_signal": sync_trace,
        "sample_rate_input": 5000.0,
    },
    cam_csv=cam_timestamp_dataframe,
    params={
        "csv_idx_sync": 0,
        "csv_idx_time": 1,
        "key_sync": "Sync_signal",
        "fps_cam": 62.46,
        "window": 20,
        "p": 0.1,
        "min_matched_edges": 5,
        "edge_threshold": 0.5,
    },
)
```

Core outputs:

- `sig_camIdx__idx_ws`: for each DAQ/WaveSurfer sample, estimated camera frame index
- `sig_wsIdx__idx_cam`: for each camera frame, estimated DAQ/WaveSurfer sample index
- affine fit parameters and matched edge diagnostics

Save templates with:

```python
import numpy as np
np.savez_compressed("alignment_template.npz", **template)
```

## 11. Treadmill Velocity And Running Bout QC

`treadmill.py` ports the treadmill pieces from the stroke/orofacial pipeline:

```text
raw voltage -> calibrated speed in mm/s -> Gaussian smoothing -> running bout mask
```

Calibration uses:

```text
speed_mm_s = (voltage - offset_v) * (1 / volt_sec_per_rot) * mm_per_rot
```

Run a QC plot from a DAQ recorder file:

```powershell
python .\wfield_local\plot_treadmill_running_qc.py `
  --label PS95 `
  --daq-h5 "E:\DAQ_recorder_output\PS95_baseline_20260601_153627.h5" `
  --output "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\treadmill_qc" `
  --channel treadmill `
  --smoothing-sigma-s 0.15 `
  --thresh-speed 5.0 `
  --max-gap-duration 0.3 `
  --min-duration 2.0
```

Important parameters:

- `--offset-v`: voltage zero offset. Default is the legacy cohort value `1.2587643276652853`.
- `--volt-sec-per-rot`: encoder calibration constant. Default is the legacy cohort value `0.382`.
- `--mm-per-rot`: wheel circumference. Default is the legacy cohort value `29.25`.
- `--smoothing-sigma-s`: Gaussian smoothing sigma in seconds. Default is `0.15`.
- `--thresh-speed`: speed threshold in mm/s for running. Default is `5.0`.
- `--max-gap-duration`: fill below-threshold gaps shorter than this duration. Default is `0.3`.
- `--min-duration`: discard bouts shorter than this duration. Default is `2.0`.

Default legacy cohort constants:

```text
OFFSET_IN_VOLTS = 1.2587643276652853
VOLT_SEC_PER_ROT = 0.382
MM_PER_ROT = 29.25
SMOOTHING_SIGMA_SEC = 0.15
THRESH_SPEED_MM_PER_S = 5.0
MAX_GAP_DURATION_SEC = 0.3
MIN_DURATION_SEC = 2.0
```

Outputs:

- `*_treadmill_running_bout_overview.png`
- `*_treadmill_running_not_running_examples.png`
- `*_treadmill_running_bouts.npz`
- `*_treadmill_running_bout_summary.json`

After QC looks sensible, generate running-vs-not-running widefield maps:

```powershell
python .\wfield_local\plot_running_activity_maps.py `
  --label PS95_v6 `
  --daq-h5 "E:\DAQ_recorder_output\PS95_baseline_20260601_153627.h5" `
  --wfield-results "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\wfield_local_results" `
  --allen-dir "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\wfield_local_results\allen_aligned_v6" `
  --output "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\treadmill_qc" `
  --channel treadmill `
  --smoothing-sigma-s 0.15 `
  --thresh-speed 5.0 `
  --max-gap-duration 0.3 `
  --min-duration 2.0
```

This maps corrected imaging frames to DAQ samples through the DAQ-recorded
`pco_exposure` pulse train, then averages `SVTcorr` frames classified as
running versus not running. Treat `--offset-v auto` as a QC convenience; for
final analysis, prefer the legacy default or a known rig/session override.

## 12. Relabeled (cleanpairs) cue/lick maps

For trial-gated recordings relabeled with `--relabel-mode rescue`, the corrected
movie is a non-contiguous subset of kept 415/470 pairs, so the stock plotters'
`raw//2` event→frame mapping is wrong. `framemap_event_maps.py` maps each event to
the nearest kept corrected frame via the cleanpairs frame map + DAQ `pco_exposure`
pulses (`chosen_exposure_offset` read from the `*_cleanpairs_summary.json`), and
writes the **same filenames/npz keys** as the stock plotters, so the downstream
`plot_spout_position_contrasts`, `plot_lick_position_contrasts`, and
`plot_lick_vs_cue_spout_maps` steps run unchanged.

```powershell
python -m wfield_local.framemap_event_maps --what cue `
  --daq-h5 <session.h5> --wfield-results <...\wfield_local_results> `
  --allen-dir <...\allen_aligned_*> --frame-map <...\*_cleanpairs_frame_map.npz> `
  --cleanpairs-summary <...\*_cleanpairs_summary.json> --output <...\spout_trial_averages_*> --label <LABEL>
# --what lick  (add --post-s 0.15) for the post-lick maps
```

Full-FOV (non-relabeled) sessions still use the stock `plot_spout_trial_averages` /
`plot_lick_aligned_averages` with `--frame-align pco`.

## 13. Motion-Correction QC

`qc_motion_correction.py` reads the saved per-frame shifts and the pre/post movies
and emits one QC figure (shift traces + magnitude histogram, mean-image sharpness
raw vs corrected, corrected temporal-std residual-motion image) plus a pass/warn
JSON. Verdict comes from the shift distribution; the sharpness ratio only downgrades
when there was real motion to remove (sub-pixel sessions are not penalized for warp
interpolation softening).

```powershell
python -m wfield_local.qc_motion_correction `
  --motion-dir "E:\labcams_data\...\motion_corrected" --label <LABEL> `
  --output "E:\labcams_data\...\motion_corrected\motion_qc"
```

## 14. Cross-Day Alignment (within animal)

Register each day's motion-corrected **mean 470 nm vasculature** to one reference
session so all days share the reference/CCF frame (`cross_day_align.py`). The
reference is CCF-aligned by its landmarks; other days are landmark-initialized then
refined by intensity-based ECC affine (SIFT+RANSAC fallback), with a greedy
keep-best on masked NCC so refinement never worsens the init. Outputs a red/green
vessel-overlay QC + NCC table + per-session transforms; `"warp_u": true` also warps
each day's `U` into the common frame.

```powershell
python -m wfield_local.cross_day_align config.json
```

Config (one per animal): `{"animal","func_channel","reference","output","warp_u",
"sessions":{"<id>":{"results":"...wfield_local_results","landmarks":"...v1.json",
"func_channel":<optional override>}, ...}}`. Keep the same ROI/zoom across days
(full-FOV↔ROI pairs register poorly). Across **animals**, vasculature is not shared —
use CCF/landmarks + Allen-area (ROI) or LocaNMF-component comparison instead.

## 15. ROI-based (Allen-area) activity extraction

`roi_activity.py` is a lightweight CPU baseline alongside LocaNMF: it averages the
Allen-aligned `U` over each atlas region to get a region x time trace
(`U_bar @ SVTcorr`, no pixel reconstruction), and optionally aligns to cue/lick
events by spout position. Runs in the wfield CPU env (numpy + h5py; no torch/GPU).
The wfield atlas is already lateralized (`MOp_left` / `MOp_right`), so you get one
trace per area per hemisphere — useful for stroke laterality.

```powershell
# region traces only
python -m wfield_local.roi_activity --allen-dir <...\allen_aligned_affine8v1> `
  --label PS94_0603 --output <...\roi_activity_affine8v1>
# + cue/lick per-region responses by spout position (regime B: pass --frame-map + --cleanpairs-summary)
python -m wfield_local.roi_activity --allen-dir <...> --label PS94_0603 --output <...> `
  --daq-h5 <session.h5> --what both `
  --frame-map <...\*_cleanpairs_frame_map.npz> --cleanpairs-summary <...\*_cleanpairs_summary.json>
```

Outputs: `*_roi_traces.npy` (R x T) + meta, `*_{cue,lick}_roi_by_position.npz`
(regions x positions: post + delta) + named-region heatmaps (cue and lick use the
same region names/order), `*_allen_reference_labeled.png` (all regions colored +
labeled at their centroids — the key for reading the heatmaps), and a
`*_roi_overview.png` that shows the locations + traces of `--overview-regions`
(default `MOp_left,MOp_right,MOs_left,MOs_right`). This is
the simple "one signal per area" baseline; `run_locanmf.py` gives the denoised,
region-anchored, multi-component version. Use ROI traces as a fast cross-check and
for quick per-area trial stats; use LocaNMF components for cross-animal claims.

## 16. Quiet-period detection (baseline selection)

`quiet_periods.py` builds a per-sample and per-corrected-frame "quiet" mask (animal
not running and not licking, not in a peri-reward window) for behavior-controlled
baseline (F0) selection — useful because trial-triggered acquisition records no true
inter-trial rest. Ported from the stroke pipeline's `find_quiet_bouts`
(`quiet = slow-treadmill AND not-near-lick AND not-near-reward`, buffered) and adapted
for ONE spout. Reuses the ported `treadmill` + `lick_detection` helpers; CPU env.

```powershell
python -m wfield_local.quiet_periods --daq-h5 <session.h5> --label PS94_0603 `
  --output <...\quiet_affine8v1> `
  --frame-map <...\*_cleanpairs_frame_map.npz> --cleanpairs-summary <...\*_cleanpairs_summary.json>
```
Outputs `*_quiet_sample.npy` (DAQ-rate bool), `*_quiet_frame.npy` (per corrected
frame), a summary, and a QC plot (speed + lick + quiet shading). Intersect
`quiet_frame` with the pre-cue ENL window, or pool it, to define F0.

- **Grooming is OFF by default.** The stroke pipeline detects grooming via *bilateral*
  two-spout conjunction, which doesn't apply here. Single-spout "long-touch" is the
  only proxy, but a TRUE long lick at close spouts also looks long, so it is
  unreliable — enable only experimentally with `--grooming`.
- **TUNE LATER:** running/quiet speed, min durations, and lick/reward/treadmill time
  buffers are stroke-pipeline starting points (e.g. the 8 s reward buffer is generous
  for short ENL windows). Revisit per rig/task, ideally validated against
  DLC/FaceRhythm movement once available (the future movement regressor).

## Decomposition note: SVD vs LocaNMF

This local pipeline stops at **SVD + hemodynamic correction + Allen alignment**. It
does **not** run **PMD** denoising or **LocaNMF** (localized semi-NMF), which the
wfield/NeuroCAAS protocol adds after SVD to produce anatomically-localized,
cross-session/animal-reproducible components. For evoked maps and within-animal work,
SVD + atlas is adequate; for cross-animal / functional-subnetwork analysis, add
LocaNMF (runs on the existing low-rank `U`/`SVTcorr` + the `allen_area_atlas`). It is
GPU-oriented and not installed here — use NeuroCAAS (`wfield_ncaas_fixed.py`) or a GPU
machine. See `DECISIONS.md`.

## NeuroCAAS Compatibility Launcher

If `wfield ncaas` crashes during upload with `QProgressBar.setValue` receiving a `numpy.float64`, launch through:

```powershell
python .\wfield_local\wfield_ncaas_fixed.py "E:\labcams_data\20260601\PS94_20260601_141614"
```

This launcher also supports AWS session tokens in the local credentials file.
