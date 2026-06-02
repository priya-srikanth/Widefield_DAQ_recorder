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
- `arduino/trial_gated_camera_dual_wavelength/trial_gated_camera_dual_wavelength.ino` - labcams-compatible Teensy firmware that waits for behavior `trial_start` on pin 20, emits PCO frame-start trigger pulses on pin 18 during the trial, gates 415/470 LEDs from PCO Status Expos on pin 3, and stops on behavior `trial_stop` on pin 22.
- `diagnose_hardware.py` - short hardware acquisition diagnostic for checking whether NI-DAQmx can acquire from the configured device.
- `scan_ai.py` - helper for scanning analog input behavior while troubleshooting wiring/ranges.
- `labcams/labcams_widefield_pco_only.json` - PCO-only labcams config for this rig. It includes `allow_missing_camera` for offline GUI testing through `labcams_ps`.
- `labcams/labcams_widefield_pco_trial_gated.json` - PCO labcams config for trial-gated acquisition using the Teensy as the external PCO frame trigger source.
- `labcams_ps/` - small repo-owned wrapper around upstream `labcams` that adds opt-in offline PCO GUI launch behavior without modifying the installed upstream package.
- `wfield_local/` - local widefield processing helpers for motion correction, SVD/hemodynamic correction, Allen alignment, cue/lick-aligned plots, alignment diagnostics, and NeuroCAAS compatibility launch.
- `requirements.txt` - Python package dependencies.
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

Trial-gated PCO launch. Use this when the behavior rig sends trial-start/trial-stop TTLs to the labcams Teensy and the Teensy triggers the PCO:

```powershell
cd "C:\Github\Widefield_DAQ_recorder"
& "C:\ProgramData\anaconda3\envs\labcams\python.exe" -m labcams_ps.gui ".\labcams\labcams_widefield_pco_trial_gated.json" -w
```

For trial-gated imaging, flash `arduino/trial_gated_camera_dual_wavelength/trial_gated_camera_dual_wavelength.ino` to the labcams Teensy and wire:

- behavior Arduino pin 6 `trial_start` TTL -> Teensy pin 20
- behavior Arduino pin 9 `trial_stop` TTL -> Teensy pin 22
- Teensy pin 18 -> PCO SMA input #1, Exposure Trigger
- PCO SMA output #4, Status Expos -> Teensy pin 3
- Teensy pin 5 -> 415 nm/violet LED TTL input
- Teensy pin 6 -> 470 nm/blue LED TTL input
- optional Teensy pin 7 -> DAQ if you want a direct copy of the generated frame trigger

The trial-gated config asks the wrapper to set the PCO trigger mode to external exposure start. Pin 18 sends short frame-start pulses; it should not be held high continuously. PCO SMA output #4 should be configured in Camware as `Status Exposure`, `Show common time of 'All lines'`, `On`, `High`.

The `LED Control` dock has a `Trial-triggered` checkbox. Leave it unchecked for alignment/preview: when armed, the Teensy free-runs camera trigger pulses and the PCO Status Expos line gates the selected LED. Check it immediately before the behavior task: the Teensy then waits for behavior `trial_start` on pin 20 and stops frame triggers/LED gates on behavior `trial_stop` on pin 22.

The upstream `labcams` package remains installed in the conda `labcams` environment; this repository does not rename or vendor the upstream package. For convenience, `launch_labcams_ps.bat` runs the same wrapper command.

The `labcams_ps` GUI also adds a `Session Save` dock. Choose an output folder, enter a prefix such as `PS94_pre_stroke`, and click `Apply Save Name` before recording. The wrapper sets the labcams session name to `prefix_YYYYMMDD_HHMMSS` and updates camera writers to use the selected folder.

The `Alignment Preview` dock lets you choose a prior alignment snapshot; it displays the saved reference in red over the live preview in green so the animal/window can be physically aligned before recording. `Clear` removes the overlay. This mode changes display only and does not alter recorded frames.

The `Camera Crop / ROI` dock lets you draw or enter a PCO ROI rectangle. `Accept ROI` writes that ROI into the active labcams JSON config, and `Clear ROI` removes it. Restart labcams before recording after changing ROI so the PCO camera initializes with the requested hardware crop.

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

- motion correction
- SVD and dual-color hemodynamic correction
- Allen landmark alignment
- cue-aligned spout-position maps
- shared-scale figure regeneration
- alignment diagnostics and comparison PowerPoints
- post-lick 150 ms maps by spout position
- imported sync-pulse and hysteresis lick-detection helpers from the stroke/orofacial workflow

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
