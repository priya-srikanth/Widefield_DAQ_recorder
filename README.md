# Widefield DAQ Recorder

Python NI-DAQ recorder for the widefield imaging + behavior rig. It replaces the subset of WaveSurfer used here: continuous synchronized DAQ recording, live strip charts, simple controls, saved config, and HDF5 output.

The current working hardware target is an **NI USB-6366 (BNC), configured as `Dev2`** on the DAQ computer.

## Current Design Goals

- Record from one NI multifunction DAQ, currently the USB-6366 on `Dev2`.
- Use the AI sample clock as the timing master.
- Record synchronized analog and digital channels into one self-contained `.h5` file.
- Provide a lightweight Tk GUI with `Play`, `Record`, `Stop`, config save/load, and live per-channel visualization.
- Keep the app focused on recording only; cameras, LEDs, and behavior control are handled by other systems.

## Current Rig Config

Default USB-6366 config: `usb6366_config.json`

Analog channels:

- `ai0` = `lick_analog`
- `ai1` = `led405_ttl`
- `ai2` = `led470_ttl`
- `ai3` = `treadmill`
- `ai4` = `reward_ttl`

Digital channels:

- `port0/line0` = `sync`
- `port0/line1` = `cue`
- `port0/line2` = `trial_start`
- `port0/line3` = `spout_strobe`
- `port0/line4` = `spout_bit0`
- `port0/line5` = `spout_bit1`
- `port0/line6` = `spout_bit2`
- `port0/line7` = `pco_exposure`

Typical settings:

- sample rate: `5000 Hz`
- block size: `1000`
- display window: `60 s`
- default output folder: `E:/DAQ_recorder_output`
- analog storage: `int16_scaled`

## Repository Files

Core app:

- `run_daq_recorder.py` - main GUI launcher. Use this for normal Play/Record/Stop operation.
- `daq_recorder/config.py` - dataclasses and JSON load/save helpers for recorder configuration.
- `daq_recorder/acquisition.py` - NI-DAQmx acquisition workers plus simulated acquisition mode.
- `daq_recorder/writer.py` - HDF5 writer, compact analog/digital storage, metadata, and periodic flushes.
- `daq_recorder/gui.py` - Tk GUI, live strip charts, controls, config editing, and display ordering.

Configs:

- `usb6366_config.json` - current working USB-6366 rig config for `Dev2`.
- `20260529_config.json` - saved working config snapshot from the initial USB-6366 testing day.
- `default_config.json` - general default config; currently also aligned with the USB-6366 setup.

Convenience launchers:

- `launch_usb6366_hardware.bat` - launches the GUI with `usb6366_config.json` in hardware mode.
- `launch_hardware.bat` - generic hardware launcher using the default config.
- `launch_simulate.bat` - launches simulated acquisition for GUI testing without NI hardware.

Diagnostics and utilities:

- `arduino/treadmill_rh/treadmill_rh.ino` - Teensy treadmill encoder firmware that outputs speed on DAC/A14 for DAQ recording.
- `arduino/stim_camera_trigger_dual_wavelength/stim_camera_trigger_dual_wavelength.ino` - Teensy camera/dual-wavelength trigger firmware used by labcams excitation triggering.
- `arduino/constant_camera_dual_wavelength/constant_camera_dual_wavelength.ino` - labcams-compatible imaging Teensy firmware for PCO exposure-gated dual-wavelength LED triggering with this rig's pin map.
- `arduino/trial_gated_acquire_enable/trial_gated_acquire_enable.ino` - current trial-gated firmware. Drives PCO Acquire Enable as a level signal on pin 18 (HIGH from behavior `trial_start` on pin 20 to `trial_stop` on pin 19); the free-running camera produces frames only while Acquire Enable is HIGH. LEDs are gated closed-loop from PCO Status Expos on pin 3 (same as `constant_camera_dual_wavelength`), so the camera and LEDs cannot desynchronize. `pulse_count` resets on `trial_start` so each trial starts on 415 (isosbestic), and the stop is deferred until the current 415/470 pair completes (ends on 470), so every trial has an even frame count and a consistent channel order — keeping the live preview and saved stream aligned trial to trial.
- `diagnose_hardware.py` - short hardware acquisition diagnostic for checking whether NI-DAQmx can acquire from the configured device.
- `scan_ai.py` - helper for scanning analog input behavior while troubleshooting wiring/ranges.
- `tools/` - analysis/QC scripts (run in the `wfield` env): `check_trial_gating.py` (verify the camera was gated to trials and LEDs alternated), `estimate_storage.py` (ROI + trial-gating storage savings), `inspect_recording.py` (HDF5/camlog/dat structure). See `tools/README.md`.
- `wfield_local/trim_illuminated_labcams.py` - rescue a continuously-saved session by trimming the labcams DAT to DAQ-confirmed illuminated 415/470 frame pairs (for recordings made before camera trial-gating worked).
- `labcams/labcams_widefield_pco_only.json` - PCO-only labcams config for this rig. It includes `allow_missing_camera` for offline GUI testing through `labcams_ps`.
- `labcams/labcams_widefield_pco_trial_gated_acquire_enable.json` - current trial-gated config. Sets `trigger_mode: "auto sequence"` and `acquire_mode: "external"` so the camera free-runs and is gated by the Teensy Acquire Enable line.
- `labcams_ps/` - small repo-owned wrapper around upstream `labcams` that adds opt-in offline PCO GUI launch behavior without modifying the installed upstream package.
- `wfield_local/` - local widefield processing helpers for motion correction, SVD/hemodynamic correction, Allen alignment, cue/lick-aligned plots, alignment diagnostics, and NeuroCAAS compatibility launch.
- `requirements.txt` - Python package dependencies.
- `DECISIONS.md` - analysis decisions log (global vs session-specific): transforms, event→frame mapping regimes, functional-channel notes, photobleaching/LED drift, cross-day/cross-animal alignment, SVD-vs-LocaNMF, raw-data archive location.
- `.gitignore` - excludes HDF5 recordings, Python caches, logs, and local environment folders.

## Install Dependencies

In the Python environment used on the DAQ machine:

```powershell
pip install -r C:\Github\Widefield_DAQ_recorder\requirements.txt
```

Core packages:

- `nidaqmx`
- `h5py`
- `numpy`

The GUI uses standard-library `tkinter`.

## Launch

Recommended USB-6366 hardware launch:

```powershell
cd "C:\Github\Widefield_DAQ_recorder"
& "C:\Users\Optiplex 7090 Tower\AppData\Local\Programs\Python\Python313\python.exe" ".\run_daq_recorder.py" --config ".\usb6366_config.json" --hardware
```

Other useful commands:

```powershell
python .\run_daq_recorder.py --simulate
python .\run_daq_recorder.py --config .\usb6366_config.json --hardware
python .\diagnose_hardware.py --seconds 10
```

## labcams PS Wrapper

Use the dedicated conda environment for camera acquisition. The repo includes a small `labcams_ps` launcher that imports upstream `labcams` and adds one rig-specific behavior: if a PCO camera entry has `"allow_missing_camera": true`, the GUI can open with a clear warning when the camera is powered off or disconnected. Recording is disabled for that placeholder camera. Remove or set that flag to `false` for real acquisition if you want missing-camera errors to stop the launch.

PCO-only labcams launch. Use this wrapper, not the upstream `labcams.exe`, when you want the optional offline-PCO placeholder:

```powershell
cd "C:\Github\Widefield_DAQ_recorder"
& "C:\ProgramData\anaconda3\envs\labcams\python.exe" -m labcams_ps.gui ".\labcams\labcams_widefield_pco_only.json" -w
```

For trial-gated PCO acquisition (camera frames only during behavior trials), see [Trial-gated acquisition via PCO Acquire Enable (current method)](#trial-gated-acquisition-via-pco-acquire-enable-current-method) below for the launch command, wiring, and verification.

The `LED Control` dock has a `Trial-triggered` checkbox. Leave it unchecked for alignment/preview: the Teensy holds Acquire Enable high so the camera free-runs and the PCO Status Expos line gates the selected LED. Check it immediately before the behavior task: the Teensy then raises Acquire Enable on behavior `trial_start` (pin 20) and lowers it on `trial_stop` (pin 19).

The `Max trial` value is a safety timeout for trial-triggered mode. The default is 5 s; set it to 0 to disable. If `trial_stop` is missed, the Teensy lowers Acquire Enable when this timeout expires.

If the labcams console shows `[CamStimTrigger] Unknown message: ...` after launch, the Teensy is running firmware that does not understand the trial-trigger command; flash the current `trial_gated_acquire_enable.ino`. If preview free-runs the camera but LEDs stay off, check that PCO SMA output #4 is physically routed to Teensy pin 3 and that camera/Teensy grounds are shared. If trial starts are detected but trial stops are not, check the behavior `trial_stop` -> Teensy pin 19 wiring and shared ground.

The upstream `labcams` package remains installed in the conda `labcams` environment; this repository does not rename or vendor the upstream package. For convenience, `launch_labcams_ps.bat` runs the same wrapper command.

The `labcams_ps` GUI also adds a `Session Save` dock. Choose an output folder, enter a prefix such as `PS94_pre_stroke`, and click `Apply Save Name` before recording. The wrapper sets the labcams session name to `prefix_YYYYMMDD_HHMMSS` and updates camera writers to use the selected folder. The labcams configs default `recorder_path` to `E:\labcams_data`; the Session Save dock overrides it per session.

Per-animal configs: the Session Save dock has `Save Config As...` and `Load Config...`. Set an animal's ROI (Camera Crop/ROI dock) and folder, then `Save Config As...` writes the current config to a new JSON (default `labcams\animals\<name>.json`) and makes it active, so the animal's ROI persists there day to day. `Load Config...` picks a config and relaunches labcams with it (a relaunch is required for camera settings like ROI to take effect). For first launch you can also use `launch_labcams_animal.ps1 <ANIMAL>`, which clones the template to `labcams\animals\<ANIMAL>.json` (full-frame) and launches it:

```powershell
cd "C:\Github\Widefield_DAQ_recorder"
.\launch_labcams_animal.ps1 PS92
```

The `Alignment Preview` dock lets you choose a prior alignment snapshot; it displays the saved reference in red over the live preview in green so the animal/window can be physically aligned before recording. `Load Reference` enables the overlay and `Clear` reliably removes it (both force a redraw, so the overlay also turns off correctly via the right-click `alignment reference` action). This mode changes display only and does not alter recorded frames.

The `Camera Crop / ROI` dock lets you set a PCO ROI in **absolute full-FOV coordinates** (`x0,y0,x1,y1`, 1-based binned-sensor pixels), so the same animal gets the exact same ROI day to day without redrawing a box. The dock shows the full binned FOV (read from the camera). Type the bounds and `Accept ROI` to write them into the active labcams JSON config (a drawn box is optional and is converted to absolute coordinates using the current crop offset). `Clear ROI` removes the crop. Restart labcams before recording after changing ROI so the PCO camera initializes with the requested hardware crop.

## Trial-gated acquisition via PCO Acquire Enable (current method)

This is the recommended way to record widefield only during behavior trials. It replaces an earlier open-loop frame-trigger method, which dimmed the live image at random times: the Teensy drove PCO frame-start triggers at a period equal to the exposure time, so whenever sensor readout had not finished the PCO dropped that trigger; the surviving exposures then all landed on the same (isosbestic, dim) LED for the rest of the trial.

In the acquire-enable method the camera free-runs internally (`trigger_mode: "auto sequence"`) and the Teensy gates acquisition with the PCO Acquire Enable input (`acquire_mode: "external"`). The Teensy holds Acquire Enable HIGH for the duration of each trial, so the camera produces zero frames between trials and a clean continuous-looking stream during each trial. LEDs are gated closed-loop from PCO Status Expos, exactly as in continuous acquisition, so the camera and LEDs cannot desynchronize. Per-channel rate is the full 31.25 Hz (62.5 fps total).

Launch:

```powershell
cd "C:\Github\Widefield_DAQ_recorder"
& "C:\ProgramData\anaconda3\envs\labcams\python.exe" -m labcams_ps.gui ".\labcams\labcams_widefield_pco_trial_gated_acquire_enable.json" -w
```

Flash `arduino/trial_gated_acquire_enable/trial_gated_acquire_enable.ino` to the labcams Teensy and wire:

- behavior Arduino `trial_start` TTL -> Teensy pin 20
- behavior Arduino `trial_stop` TTL -> Teensy pin 19
- Teensy pin 18 -> **PCO SMA input #2, Acquire Enable** (level signal, HIGH during trial). Leave SMA input #1 unconnected. This is the one wiring change from the open-loop method, which used SMA input #1.
- PCO SMA output #4, Status Expos -> Teensy pin 3
- Teensy pin 5 -> 415 nm/violet LED TTL input
- Teensy pin 6 -> 470 nm/blue LED TTL input
- optional Teensy pin 7 -> DAQ for a direct copy of Status Expos (`pco_exposure`)

CamWare notes: `pco.Camera()` calls `reset_settings_to_default()` on every connect, so trigger/acquire mode set in CamWare do not survive a labcams launch -- only the config + wrapper set them at runtime. What still matters in CamWare is the hardware I/O: the Acquire Enable input enabled with `High` polarity, and SMA output #4 as `Status Exposure`, `Show common time of 'All lines'`, `On`, `High`.

### Why the wrapper applies patches at import time

labcams forces multiprocessing start method `spawn` (see `labcams/cams.py`), so the PCO camera runs in a spawned child interpreter that re-imports `labcams.pco` fresh and never calls `main()`/`apply_patches()`. Monkey-patches applied only in `main()` are therefore absent in that child, which then runs the unpatched upstream `PCOCam._cam_init` -- whose only acquire-mode line is the upstream typo `self.cam.set_acquire_mode = self.acquire_mode` (an attribute assignment, not an SDK call). Combined with the reset-to-default on connect, the camera silently stayed in `acquire_mode="auto"` and free-ran continuously, ignoring Acquire Enable.

The fix (`labcams_ps/gui.py`) moves the camera-process patches into `apply_camera_process_patches()` and calls it at module import time, not just inside `main()`. The spawn child re-imports this module to unpickle the camera Process, so the call runs there too and the child's `_cam_init` actually issues `set_acquire_mode("external")` / `set_trigger_mode("auto sequence")`. The patches are idempotent, so re-application from `main()` in the parent is a no-op.

### Verifying gating worked

After a short test recording, run `tools/check_trial_gating.py` (or compare the DAQ `pco_exposure` digital line against `trial_start`/`trial_end` and the labcams `.camlog` by hand):

```powershell
& "C:\ProgramData\anaconda3\envs\wfield\python.exe" tools\check_trial_gating.py --daq E:\DAQ_recorder_output\<session>.h5 --camlog C:\path\to\<run>.camlog
```

The checks:

- the camlog inter-frame intervals should show multi-second gaps at trial boundaries (not a single uninterrupted 16 ms train), and
- essentially no `pco_exposure` pulses should fall outside trial windows.

On a successful 7-trial test (2026-06-02), the camlog showed 6 inter-trial pauses of ~2.4-2.6 s, in-trial rate was 62.5 fps, and LED alternation was clean (0 frames with both LEDs on). Before the fix the same setup produced a gapless exposure train with 47% of frames falling between trials.

## wfield NeuroCAAS Launcher

If `wfield ncaas` crashes during upload with `QProgressBar.setValue` receiving a `numpy.float64`, launch NeuroCAAS through the repo compatibility wrapper:

```powershell
conda activate wfield
cd "C:\Github\Widefield_DAQ_recorder"
python .\wfield_local\wfield_ncaas_fixed.py
```

To start in a specific recording folder:

```powershell
python .\wfield_local\wfield_ncaas_fixed.py "E:\labcams_data\20260601\PS94_20260601_141614"
```

The wrapper does not modify the installed `wfield` package. It only coerces PyQt progress-bar values to plain Python integers in that launched process.

## Local wfield Processing

See [wfield_local/README.md](wfield_local/README.md) for local processing instructions covering:

- motion correction (+ optional motion-correction QC, `qc_motion_correction.py`)
- SVD and dual-color hemodynamic correction (with optional `--detrend-order` / exposed `--freq-highpass`/`--freq-lowpass`)
- Allen landmark alignment (shared outline helper `atlas_overlay.py`)
- cue-aligned spout-position maps + post-pre delta; shared-scale figure regeneration
- post-lick 150 ms maps and delta-position contrasts by spout position
- cue/lick maps for relabeled "cleanpairs" movies (`framemap_event_maps.py`)
- within-animal cross-day alignment on the mean 470 nm vasculature (`cross_day_align.py`)
- alignment diagnostics and comparison PowerPoints
- imported sync-pulse and hysteresis lick-detection helpers from the stroke/orofacial workflow

Analysis design choices (global vs session-specific, event→frame mapping regimes,
functional-channel notes, photobleaching/LED drift, cross-day/cross-animal alignment
policy, and the SVD-vs-LocaNMF decomposition decision) are recorded in
[DECISIONS.md](DECISIONS.md).

## Timing Notes

The app uses analog input acquisition as the master timing source. Digital input is hardware-timed from the device AI sample clock so analog and digital samples share one sample timeline.

On devices that support it, DI start can be aligned to the AI start trigger. On devices that reject a DI start trigger, the app starts DI before AI; samples remain aligned because DI is still clocked by the AI sample clock.

The current validated path is the NI USB-6366 on `Dev2`. Earlier PCIe-6259/BNC-2110 work remains useful context, but is not the default target for this repo anymore.

## Display Order

Checked channels in the Analog/Digital channel tables define what is acquired and recorded. The Display Order box defines which enabled channels are visible in the live GUI and in what order. Removing a channel from Display Order hides it from the live display but does not stop it from being recorded, as long as it remains checked in the channel table. Use `Use Current Channels` to repopulate the display list with all enabled channels.

The top status bar shows sample count and elapsed recording/play time in minutes.
## Finite Session Duration

The `Max duration (s)` field controls optional finite sessions. Leave it blank or set it to `0` for continuous acquisition. Set it to a positive number to auto-stop once that many seconds have been acquired. This works in both Play and Record modes.

## HDF5 Layout

Each recording session writes one self-contained `.h5` file. No JSON sidecar is required; the active config is stored in the root `config_json` attribute.

Current compact layout:

- `/analog/samples_int16` for `int16_scaled` analog storage
- `/analog/int16_scale_volts_per_count`
- `/analog/int16_offset_volts`
- `/analog/channel_names`
- `/analog/physical_channels`
- `/analog/input_range`
- `/digital/packed_samples` with digital line bits packed into `uint8`
- `/digital/channel_names`
- `/digital/physical_channels`
- root attrs:
  - `device`
  - `sample_rate_hz`
  - `sample_count`
  - `sample_index_start`
  - `sample_index_is_contiguous`
  - `created_at`
  - `file_prefix`
  - `config_json`

Analog reconstruction formula:

```text
volts = raw_int16 * int16_scale_volts_per_count + int16_offset_volts
```

Digital unpacking:

```python
digital = np.unpackbits(packed_samples, axis=1, count=n_digital_channels, bitorder="little")
```

## Loading Recordings

Use `daq_recorder.io.load_recording()` to load either old or new recorder HDF5 files without manually handling storage details:

```python
from daq_recorder.io import load_recording

rec = load_recording(r"E:\DAQ_recorder_output\example.h5")
analog_volts = rec["analog"]
digital = rec["digital"]
time_s = rec["time_s"]
analog_names = rec["analog_channel_names"]
digital_names = rec["digital_channel_names"]
```

The loader reconstructs `int16_scaled` analog data back into volts and unpacks packed digital bits into one 0/1 column per digital channel.

## Storage And Compression

The GUI/config supports two analog storage modes:

- `int16_scaled` - default. Analog voltages are quantized into signed 16-bit integers using each channel's configured min/max range. The HDF5 stores the scale and offset needed to reconstruct volts. This is much smaller and was tested on representative PS92 data with sub-0.1 mV reconstruction error and unchanged lick/reward threshold event counts.
- `float32` - stores analog samples directly as 32-bit volts. This is larger, but useful as a conservative fallback while debugging.

Both analog modes use HDF5 `lzf` compression with shuffle enabled. `lzf` is lossless and fast, so it should reduce file size without adding much CPU load. Digital data is always packed into bits and stored with `lzf` compression.

The old writer format stored unpacked digital columns and a full per-sample `sample_index` dataset. New files omit that redundant index and instead store `sample_rate_hz`, `sample_count`, and `sample_index_start` metadata.

## Crash Safety

During recording, the writer updates `sample_count` and flushes the HDF5 file about every 2 seconds, plus once on clean close. This improves recoverability if the GUI or Python process crashes. A clean `Stop` is still preferred whenever possible.

## Current Limitations

- Focused on continuous acquisition and simple session recording.
- No WaveSurfer-style stimulus generation or trigger protocol system.
- No camera, LED, or behavior control.
- HDF5 files created by older versions of this app may use `/analog/samples`, `/digital/samples`, and `/sample_index` instead of the compact layout above.
