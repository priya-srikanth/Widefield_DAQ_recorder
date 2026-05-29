from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AnalogChannelConfig:
    name: str
    physical_channel: str
    enabled: bool = True
    min_val: float = -10.0
    max_val: float = 10.0
    scale: float = 1.0
    terminal_config: str = "DIFF"


@dataclass
class DigitalChannelConfig:
    name: str
    physical_channel: str
    enabled: bool = True


@dataclass
class RecorderConfig:
    app_title: str = "Widefield DAQ Recorder"
    device: str = "Dev1"
    sample_rate_hz: float = 5000.0
    block_size: int = 250
    display_seconds: float = 10.0
    max_duration_s: float = 0.0
    output_directory: str = r"C:\Data\daq_recordings"
    file_prefix: str = "daq"
    analog_storage: str = "int16_scaled"
    simulate: bool = False
    analog_channels: list[AnalogChannelConfig] = field(default_factory=list)
    digital_channels: list[DigitalChannelConfig] = field(default_factory=list)
    display_order: list[str] = field(default_factory=list)

    @property
    def enabled_analog(self) -> list[AnalogChannelConfig]:
        return [ch for ch in self.analog_channels if ch.enabled]

    @property
    def enabled_digital(self) -> list[DigitalChannelConfig]:
        return [ch for ch in self.digital_channels if ch.enabled]

    @property
    def output_path(self) -> Path:
        return Path(self.output_directory)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecorderConfig":
        return cls(
            app_title=data.get("app_title", "Widefield DAQ Recorder"),
            device=data.get("device", "Dev1"),
            sample_rate_hz=float(data.get("sample_rate_hz", 5000.0)),
            block_size=int(data.get("block_size", 250)),
            display_seconds=float(data.get("display_seconds", 10.0)),
            max_duration_s=float(data.get("max_duration_s", data.get("duration_s", 0.0)) or 0.0),
            output_directory=data.get("output_directory", r"C:\Data\daq_recordings"),
            file_prefix=data.get("file_prefix", data.get("filename_prefix", "daq")),
            analog_storage=data.get("analog_storage", "int16_scaled"),
            simulate=bool(data.get("simulate", False)),
            analog_channels=[
                AnalogChannelConfig(**item) for item in data.get("analog_channels", [])
            ],
            digital_channels=[
                DigitalChannelConfig(**item) for item in data.get("digital_channels", [])
            ],
            display_order=list(data.get("display_order", [])),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RecorderConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8-sig")))
