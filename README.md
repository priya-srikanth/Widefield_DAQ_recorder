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
- `diagnose_hardware.py` - short hardware acquisition diagnostic for checking whether NI-DAQmx can acquire from the configured device.
- `scan_ai.py` - helper for scanning analog input behavior while troubleshooting wiring/ranges.
- `labcams/labcams_widefield_pco_only.json` - latest labcams config found for the PCO-only widefield camera setup.
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
