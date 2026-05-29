# Widefield DAQ Recorder

Python NI-DAQ recorder intended to cover the subset of WaveSurfer functionality needed for the widefield + behavior rig:

- continuous AI + DI recording from one NI device
- synchronized analog and digital acquisition
- live per-channel visualization
- `Play`, `Record`, and `Stop`
- JSON config save/load
- HDF5 output

## Current design goals

- NI multifunction DAQ with one device, usually `Dev1`
- analog channels for lick, LEDs, treadmill, and optional reward
- digital channels for behavior TTLs and `PCO` exposure
- simple, inspectable file format
- lightweight Tk GUI with no heavy GUI dependencies

## Files

- `default_config.json`
- `requirements.txt`
- `run_daq_recorder.py`
- `diagnose_hardware.py`
- `launch_hardware.bat`
- `launch_simulate.bat`

## Install dependencies

In the Python environment you plan to use on the DAQ machine:

```powershell
pip install -r C:\Github\Widefield_DAQ_recorder\requirements.txt
```

Core packages:

- `nidaqmx`
- `h5py`
- `numpy`

The GUI uses standard-library `tkinter`.

## Launch

```powershell
cd C:\Github\Widefield_DAQ_recorder
python .\run_daq_recorder.py --hardware
```

Optional:

```powershell
python .\run_daq_recorder.py --simulate
python .\run_daq_recorder.py --config C:\path\to\config.json
python .\diagnose_hardware.py --seconds 10
```

## Important DAQ note for M-series

This app assumes an NI M-series style synchronization approach in which:

- analog input uses the onboard AI sample clock
- digital input is hardware-timed from `/<device>/ai/SampleClock`
- when supported, digital input start is aligned to `/<device>/ai/StartTrigger`
- on devices that reject DI start triggers, DI starts before AI and remains aligned because the DI task is clocked by the AI sample clock

For PCIe-6259/M-series style boards, digital lines on one port are read as one port-wide hardware-timed channel and unpacked into individual 0/1 traces in software.

## HDF5 layout

The recorder writes:

- `/analog/samples` shape `(n_samples, n_ai)`
- `/digital/samples` shape `(n_samples, n_di)`
- `/analog/channel_names`
- `/digital/channel_names`
- `/analog/physical_channels`
- `/digital/physical_channels`
- `/analog/scale`
- `/analog/input_range`
- root metadata attrs including:
  - device
  - sample rate
  - start time
  - config JSON

## Current limitations

- Hardware smoke-tested on `Dev1` with 5 AI and 8 DI channels at 5000 Hz.
- It does not yet provide WaveSurfer-style trigger protocols or stimulus generation.
- It is aimed at continuous acquisition and simple session recording.
- Digital traces are rendered as 0/1 line plots, not staircase plots.
