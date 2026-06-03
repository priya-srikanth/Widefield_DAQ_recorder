# Tools

Small analysis/QC scripts for widefield + DAQ recordings. Run them in an
environment with `h5py` + `numpy` (the `wfield` conda env on the rig):

```powershell
$py = "C:\ProgramData\anaconda3\envs\wfield\python.exe"
```

## check_trial_gating.py

Verify that a session was trial-gated (camera frames only during behavior
trials) and that the 415/470 LEDs alternated cleanly. The robust signal is the
`pco_exposure` train: with working acquire-enable gating it breaks into one
burst per trial separated by multi-second pauses; broken gating is one
continuous train.

```powershell
& $py tools\check_trial_gating.py --daq E:\DAQ_recorder_output\<session>.h5 `
    --camlog C:\path\to\<run>.camlog
```

PASS looks like: `gating: GATED - N exposure bursts match M trial_starts` and
`LED alternation: clean`. `NOT GATED - exposure train is continuous` means the
camera free-ran (e.g. acquire mode never set to external).

## estimate_storage.py

List large `.dat` files under a labcams root and estimate ROI + trial-gating
storage savings versus full-FOV continuous acquisition. Pass a DAQ behavior
session to measure the real trial duty cycle.

```powershell
& $py tools\estimate_storage.py --labcams-dir E:\labcams_data --min-gb 50 `
    --full 540x640 --roi 487x480 --daq E:\DAQ_recorder_output\<behavior>.h5
```

Model: `saved_size = baseline * (roi_pixels / full_pixels) * (in-trial_time / total_time)`.

## inspect_recording.py

Dump a DAQ recorder HDF5 structure and/or check a labcams `.camlog` against its
`.dat` (frame count, inter-frame gaps, size consistency).

```powershell
& $py tools\inspect_recording.py --daq E:\DAQ_recorder_output\<session>.h5
& $py tools\inspect_recording.py --camlog <run>.camlog --dat <run>_..._2_540_640_uint16.dat
```
