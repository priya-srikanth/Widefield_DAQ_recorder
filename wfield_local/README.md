# Local wfield Processing Helpers

This folder contains local processing scripts used with `jcouto/wfield` for the widefield behavior rig. They are designed to run in the `wfield` conda environment and to keep large imaging outputs outside git.

Typical environment:

```powershell
conda activate wfield
cd "C:\Github\Widefield_DAQ_recorder"
```

## Processing Overview

1. Motion-correct the labcams `.dat` file.
2. Run local SVD and dual-color hemodynamic correction.
3. Apply Allen/wfield landmark alignment.
4. Generate cue-aligned spout-position maps.
5. Generate optional alignment diagnostics and comparison PowerPoints.
6. Generate optional lick-aligned post-event maps.

Large outputs should stay in the recording folder, usually under:

```text
E:\labcams_data\YYYYMMDD\SESSION\motion_corrected
```

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

The raw `.dat` file is not overwritten.

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

- `--functional-channel 1`: for this rig, channel 0 is 415 nm and channel 1 is 470 nm, so 470 is the functional channel.
- `--n-components`: SVD component count. `100` has been used for PS94/PS95.
- Chunk/memory parameters in the script can be adjusted if a computer runs out of RAM.

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

Example:

```powershell
python .\wfield_local\plot_spout_trial_averages.py `
  --label PS95_v6 `
  --daq-h5 "E:\DAQ_recorder_output\PS95_baseline_20260601_153627.h5" `
  --wfield-results "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\wfield_local_results" `
  --allen-dir "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\wfield_local_results\allen_aligned_v6" `
  --camlog "E:\labcams_data\20260601\PS95_20260601_153653\pco_edge_run000_00000000.camlog" `
  --output "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\spout_trial_averages_allen_v6" `
  --pre-s 1.0 `
  --post-s 1.0 `
  --fs 31.23
```

Adjustable parameters:

- `--pre-s`: seconds before cue.
- `--post-s`: seconds after cue.
- `--fs`: hemodynamic-corrected paired-frame sampling rate. For PS94/PS95 this was `31.23`.
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

Example:

```powershell
python .\wfield_local\plot_lick_aligned_averages.py `
  --label PS95_v6 `
  --daq-h5 "E:\DAQ_recorder_output\PS95_baseline_20260601_153627.h5" `
  --wfield-results "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\wfield_local_results" `
  --allen-dir "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\wfield_local_results\allen_aligned_v6" `
  --camlog "E:\labcams_data\20260601\PS95_20260601_153653\pco_edge_run000_00000000.camlog" `
  --output "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\lick_aligned_v6" `
  --lick-threshold-v 2.5 `
  --refractory-s 0.10 `
  --post-s 0.150 `
  --fs 31.23
```

Adjustable parameters:

- `--lick-threshold-v`: falling threshold for lick detection. For PS95, licks dropped from ~5.5 V to ~0 V, so `2.5` V is a reasonable default.
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

## NeuroCAAS Compatibility Launcher

If `wfield ncaas` crashes during upload with `QProgressBar.setValue` receiving a `numpy.float64`, launch through:

```powershell
python .\wfield_local\wfield_ncaas_fixed.py "E:\labcams_data\20260601\PS94_20260601_141614"
```

This launcher also supports AWS session tokens in the local credentials file.
