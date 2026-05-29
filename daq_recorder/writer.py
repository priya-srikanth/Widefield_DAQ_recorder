from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover - optional runtime dependency
    h5py = None

if TYPE_CHECKING:
    from .config import RecorderConfig


@dataclass
class RecordingPaths:
    file_path: Path


class HDF5Recorder:
    def __init__(self, config: "RecorderConfig", file_path: str | Path):
        if h5py is None:
            raise RuntimeError("h5py is required for HDF5 recording.")
        self.config = config
        self.file_path = Path(file_path)
        self._file = None
        self._analog = None
        self._digital = None
        self._sample_index = 0
        self._digital_bytes_per_sample = 0
        self._analog_storage = getattr(config, "analog_storage", "int16_scaled")
        self._analog_int16_scale: np.ndarray | None = None
        self._analog_int16_offset: np.ndarray | None = None
        self._analog_input_range: np.ndarray | None = None
        self._last_flush_time = time.monotonic()
        self._flush_interval_s = 2.0

    @staticmethod
    def _safe_file_prefix(prefix: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", prefix.strip())
        cleaned = cleaned.strip("._-")
        return cleaned or "daq"

    @staticmethod
    def make_default_path(config: "RecorderConfig") -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = HDF5Recorder._safe_file_prefix(getattr(config, "file_prefix", "daq"))
        return config.output_path / f"{prefix}_{stamp}.h5"

    def open(self) -> RecordingPaths:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.file_path, "w")
        analog_count = len(self.config.enabled_analog)
        digital_count = len(self.config.enabled_digital)
        block = max(1, int(self.config.block_size))

        self._file.attrs["created_at"] = datetime.now().isoformat()
        self._file.attrs["device"] = self.config.device
        self._file.attrs["sample_rate_hz"] = self.config.sample_rate_hz
        self._file.attrs["file_prefix"] = getattr(self.config, "file_prefix", "daq")
        self._file.attrs["config_json"] = self.config.to_json()
        self._file.attrs["sample_index_start"] = 0
        self._file.attrs["sample_count"] = 0
        self._file.attrs["sample_index_is_contiguous"] = True
        self._file.attrs["recording_complete"] = False

        analog_group = self._file.create_group("analog")
        digital_group = self._file.create_group("digital")

        analog_group.create_dataset(
            "channel_names",
            data=np.array([ch.name for ch in self.config.enabled_analog], dtype="S"),
        )
        analog_group.create_dataset(
            "physical_channels",
            data=np.array([f"{self.config.device}/{ch.physical_channel}" for ch in self.config.enabled_analog], dtype="S"),
        )
        analog_group.create_dataset(
            "scale",
            data=np.array([ch.scale for ch in self.config.enabled_analog], dtype=np.float32),
        )
        analog_group.create_dataset(
            "terminal_config",
            data=np.array([ch.terminal_config for ch in self.config.enabled_analog], dtype="S"),
        )
        self._analog_input_range = np.array(
            [[ch.min_val * ch.scale, ch.max_val * ch.scale] for ch in self.config.enabled_analog],
            dtype=np.float32,
        )
        analog_group.create_dataset("input_range", data=self._analog_input_range)
        analog_group.attrs["units"] = "V_after_scale"

        if self._analog_storage == "float32":
            analog_group.attrs["storage"] = "float32_lzf_shuffle"
            self._analog = analog_group.create_dataset(
                "samples",
                shape=(0, analog_count),
                maxshape=(None, analog_count),
                dtype=np.float32,
                chunks=(block, max(analog_count, 1)),
                compression="lzf",
                shuffle=True,
            )
        else:
            self._analog_storage = "int16_scaled"
            span = self._analog_input_range[:, 1] - self._analog_input_range[:, 0]
            if np.any(span <= 0):
                raise ValueError("Analog channel max must be greater than min for int16_scaled storage.")
            self._analog_int16_scale = (span / 65535.0).astype(np.float32)
            self._analog_int16_offset = (
                self._analog_input_range[:, 0] + 32768.0 * self._analog_int16_scale
            ).astype(np.float32)
            analog_group.attrs["storage"] = "int16_scaled_volts_lzf_shuffle"
            analog_group.attrs["int16_formula"] = "volts = raw_int16 * scale + offset"
            analog_group.create_dataset("int16_scale_volts_per_count", data=self._analog_int16_scale)
            analog_group.create_dataset("int16_offset_volts", data=self._analog_int16_offset)
            self._analog = analog_group.create_dataset(
                "samples_int16",
                shape=(0, analog_count),
                maxshape=(None, analog_count),
                dtype=np.int16,
                chunks=(block, max(analog_count, 1)),
                compression="lzf",
                shuffle=True,
            )

        digital_group.create_dataset(
            "channel_names",
            data=np.array([ch.name for ch in self.config.enabled_digital], dtype="S"),
        )
        digital_group.create_dataset(
            "physical_channels",
            data=np.array([f"{self.config.device}/{ch.physical_channel}" for ch in self.config.enabled_digital], dtype="S"),
        )
        digital_group.attrs["line_grouping"] = "CHAN_PER_LINE"
        digital_group.attrs["values"] = "0_or_1"
        digital_group.attrs["storage"] = "packed_uint8_lzf_shuffle"
        digital_group.attrs["bit_order"] = "little"
        digital_group.attrs["description"] = "digital packed bit n corresponds to channel_names[n]"
        self._digital_bytes_per_sample = max(1, (digital_count + 7) // 8)
        self._digital = digital_group.create_dataset(
            "packed_samples",
            shape=(0, self._digital_bytes_per_sample),
            maxshape=(None, self._digital_bytes_per_sample),
            dtype=np.uint8,
            chunks=(block, self._digital_bytes_per_sample),
            compression="lzf",
            shuffle=True,
        )
        self._file.flush()
        self._last_flush_time = time.monotonic()
        return RecordingPaths(file_path=self.file_path)

    def append(self, analog: np.ndarray, digital: np.ndarray) -> None:
        if self._file is None or self._analog is None or self._digital is None:
            raise RuntimeError("Recorder must be opened before append.")
        n = 0
        if analog.size:
            n = analog.shape[0]
        elif digital.size:
            n = digital.shape[0]
        if n == 0:
            return

        if analog.size == 0:
            analog = np.zeros((n, self._analog.shape[1]), dtype=np.float32)
        if digital.size == 0:
            digital = np.zeros((n, len(self.config.enabled_digital)), dtype=np.uint8)

        start = self._analog.shape[0]
        stop = start + n
        self._analog.resize((stop, self._analog.shape[1]))
        self._digital.resize((stop, self._digital.shape[1]))

        analog_f32 = analog.astype(np.float32, copy=False)
        if self._analog_storage == "float32":
            self._analog[start:stop, :] = analog_f32
        else:
            if self._analog_input_range is None:
                raise RuntimeError("Missing int16 analog scaling metadata.")
            lo = self._analog_input_range[:, 0]
            hi = self._analog_input_range[:, 1]
            span = hi - lo
            raw = np.rint((analog_f32 - lo) / span * 65535.0 - 32768.0)
            self._analog[start:stop, :] = np.clip(raw, -32768, 32767).astype(np.int16)

        digital_u8 = digital.astype(np.uint8, copy=False)
        digital_u8 = np.where(digital_u8 > 0, 1, 0).astype(np.uint8, copy=False)
        packed = np.packbits(digital_u8, axis=1, bitorder="little")
        if packed.shape[1] < self._digital_bytes_per_sample:
            pad = np.zeros((n, self._digital_bytes_per_sample - packed.shape[1]), dtype=np.uint8)
            packed = np.concatenate((packed, pad), axis=1)
        self._digital[start:stop, :] = packed
        self._sample_index += n
        self._file.attrs["sample_count"] = self._sample_index

        now = time.monotonic()
        if now - self._last_flush_time >= self._flush_interval_s:
            self._file.flush()
            self._last_flush_time = now

    def close(self) -> None:
        if self._file is not None:
            self._file.attrs["sample_count"] = self._sample_index
            self._file.attrs["recording_complete"] = True
            self._file.attrs["closed_at"] = datetime.now().isoformat()
            self._file.flush()
            self._file.close()
            self._file = None
            self._analog = None
            self._digital = None
