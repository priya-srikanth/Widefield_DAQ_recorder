from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover - optional runtime dependency
    h5py = None


def _decode_strings(values: np.ndarray) -> list[str]:
    return [item.decode() if isinstance(item, bytes) else str(item) for item in values]


def _read_analog(group: Any) -> np.ndarray:
    if "samples_int16" in group:
        raw = group["samples_int16"][()]
        scale = group["int16_scale_volts_per_count"][()]
        offset = group["int16_offset_volts"][()]
        return raw.astype(np.float32) * scale + offset
    if "samples" in group:
        return group["samples"][()].astype(np.float32, copy=False)
    return np.zeros((0, 0), dtype=np.float32)


def _read_digital(group: Any) -> np.ndarray:
    channel_count = len(group["channel_names"]) if "channel_names" in group else 0
    if "packed_samples" in group:
        packed = group["packed_samples"][()]
        return np.unpackbits(packed, axis=1, count=channel_count, bitorder="little").astype(np.uint8, copy=False)
    if "samples" in group:
        return group["samples"][()].astype(np.uint8, copy=False)
    return np.zeros((0, 0), dtype=np.uint8)


def load_recording(path: str | Path, include_time: bool = True) -> dict[str, Any]:
    """Load a DAQ recorder HDF5 file into analysis-friendly arrays.

    Returns analog samples reconstructed in volts and digital samples unpacked
    as one 0/1 column per configured digital channel. Supports both the compact
    current HDF5 layout and the older draft layout.
    """
    if h5py is None:
        raise RuntimeError("h5py is required to load recorder HDF5 files.")

    with h5py.File(path, "r") as h5:
        sample_rate_hz = float(h5.attrs["sample_rate_hz"])
        analog_group = h5.get("analog")
        digital_group = h5.get("digital")

        analog = _read_analog(analog_group) if analog_group is not None else np.zeros((0, 0), dtype=np.float32)
        digital = _read_digital(digital_group) if digital_group is not None else np.zeros((0, 0), dtype=np.uint8)

        analog_names = _decode_strings(analog_group["channel_names"][()]) if analog_group is not None and "channel_names" in analog_group else []
        digital_names = _decode_strings(digital_group["channel_names"][()]) if digital_group is not None and "channel_names" in digital_group else []
        analog_physical = _decode_strings(analog_group["physical_channels"][()]) if analog_group is not None and "physical_channels" in analog_group else []
        digital_physical = _decode_strings(digital_group["physical_channels"][()]) if digital_group is not None and "physical_channels" in digital_group else []

        sample_count = int(h5.attrs.get("sample_count", max(analog.shape[0], digital.shape[0])))
        sample_index_start = int(h5.attrs.get("sample_index_start", 0))
        attrs = {key: h5.attrs[key] for key in h5.attrs.keys()}

    result: dict[str, Any] = {
        "path": str(path),
        "sample_rate_hz": sample_rate_hz,
        "sample_count": sample_count,
        "sample_index_start": sample_index_start,
        "analog": analog,
        "digital": digital,
        "analog_channel_names": analog_names,
        "digital_channel_names": digital_names,
        "analog_physical_channels": analog_physical,
        "digital_physical_channels": digital_physical,
        "attrs": attrs,
    }
    if include_time:
        result["time_s"] = (np.arange(sample_count, dtype=np.float64) + sample_index_start) / sample_rate_hz
    return result