# 2pRAM DAQ Recorder Profile

This folder contains the machine profile for the 2pRAM behavior rig. It uses the repository's shared DAQ recorder GUI with the installed NI PCIe-6353 named `PCIe-6353`.

The channel names and terminal assignments were transcribed from the rig's read-only Janelia WaveSurfer profile `20260120_WS_SI_Teensy-LLLR.wsp`. Sampling, buffering, display, and HDF5 storage settings retain the recorder defaults used by the widefield profile.

## Hardware and timing

- NI device: `PCIe-6353`
- sample rate: `5000 Hz` per configured channel
- block size: `1000` samples
- display window: `60 s`
- analog storage: `int16_scaled`
- recording folder: `E:\2pRAM_DAQ_recorder_output\data`
- analog timing master: AI sample clock
- digital inputs: hardware-timed from the AI sample clock on Port 0

The PCIe-6353 is a multiplexed 16-bit device rated for 1.25 MS/s aggregate analog input. This profile requests 10 channels at 5 kHz, or 50 kS/s aggregate, which is well within that rate. Unlike the USB-6366 widefield target, its analog channels are not sampled simultaneously.

## Channels

| Analog input | Name |
| --- | --- |
| `ai17` | `Acc_(x)` |
| `ai18` | `Acc_(y)` |
| `ai19` | `Acc_(z)` |
| `ai7` | `Y_galvo` |
| `ai0` | `Trial_start` |
| `ai1` | `Tone` |
| `ai3` | `Left_lick` |
| `ai4` | `Right_lick` |
| `ai6` | `Treadmill` |
| `ai2` | `ENL_duration` |

| Digital input | Name |
| --- | --- |
| `port0/line0` | `Frame_clock` |
| `port0/line1` | `Confirm_acq_trigger` |
| `port0/line3` | `Left_reward` |
| `port0/line4` | `Right_reward` |
| `port0/line2` | `Sync_pulse` |

## Environment setup

Create a dedicated environment once from Anaconda PowerShell:

```powershell
conda create --name widefield-daq python=3.12 pip tk -y
conda activate widefield-daq
python -m pip install -r "C:\Github\2pRAM_DAQ_recorder\requirements.txt"
```

This environment is intentionally separate from `base` and unrelated analysis/camera environments.

## Launch

From Anaconda PowerShell:

```powershell
conda activate widefield-daq
cd "C:\Github\2pRAM_DAQ_recorder"
python .\run_daq_recorder.py --config .\2pRAM\pcie6353_config.json --hardware
```

Or use the profile launcher after activating the environment:

```powershell
cd "C:\Github\2pRAM_DAQ_recorder"
.\2pRAM\launch_hardware.ps1
```

Offline GUI test:

```powershell
.\2pRAM\launch_simulate.ps1
```

Hardware diagnostic:

```powershell
python .\diagnose_hardware.py --config .\2pRAM\pcie6353_config.json --seconds 10
```

## Validation record

On 2026-08-09 this profile was verified on the installed PCIe-6353 under NI-DAQmx 24.6 with Python 3.12.13. A three-second end-to-end acquisition recorded 15,000 samples on all 10 analog and 5 digital channels, wrote the HDF5 file, reopened it successfully, and confirmed complete recording metadata.

## References

- [NI PCIe-6353 product specifications](https://www.ni.com/en/shop/hardware/voltage/model-pcie-6353)
- [Widefield USB-6366 profile](../README.md#recorder-profiles)
