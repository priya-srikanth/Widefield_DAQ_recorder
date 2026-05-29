from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import RecorderConfig

try:
    import nidaqmx
    from nidaqmx.constants import AcquisitionType, Edge, LineGrouping, TerminalConfiguration
    from nidaqmx.errors import DaqError
except ImportError:  # pragma: no cover - optional runtime dependency
    nidaqmx = None
    AcquisitionType = Edge = LineGrouping = TerminalConfiguration = None
    DaqError = None


_DIGITAL_LINE_RE = re.compile(r"^(?P<port>port\d+)/line(?P<line>\d+)$", re.IGNORECASE)


@dataclass
class DataChunk:
    analog: np.ndarray
    digital: np.ndarray
    sample_index: int
    timestamp: float


class AcquisitionBackend:
    def start(self, callback: Callable[[DataChunk | Exception], None]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def stop(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class MockBackend(AcquisitionBackend):
    def __init__(self, config: RecorderConfig):
        self.config = config
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._sample_index = 0

    def start(self, callback: Callable[[DataChunk | Exception], None]) -> None:
        self._stop.clear()

        def run() -> None:
            dt = 1.0 / self.config.sample_rate_hz
            block = self.config.block_size
            t0 = time.perf_counter()
            while not self._stop.is_set():
                t = (np.arange(block) + self._sample_index) * dt
                analog = []
                for idx, _ch in enumerate(self.config.enabled_analog):
                    freq = 0.2 + 0.15 * idx
                    wave = np.sin(2 * math.pi * freq * t) + 0.05 * np.random.randn(block)
                    if idx > 0:
                        wave = (wave > 0).astype(np.float32) * 5.0
                    analog.append(wave.astype(np.float32))
                digital = []
                for idx, _ch in enumerate(self.config.enabled_digital):
                    period = max(20, 70 - idx * 5)
                    digital.append((((np.arange(block) + self._sample_index) // period) % 2).astype(np.uint8))
                analog_arr = np.stack(analog, axis=1) if analog else np.zeros((block, 0), dtype=np.float32)
                digital_arr = np.stack(digital, axis=1) if digital else np.zeros((block, 0), dtype=np.uint8)
                callback(
                    DataChunk(
                        analog=analog_arr,
                        digital=digital_arr,
                        sample_index=self._sample_index,
                        timestamp=time.time(),
                    )
                )
                self._sample_index += block
                elapsed = time.perf_counter() - t0
                target = self._sample_index * dt
                sleep_time = max(0.0, target - elapsed)
                time.sleep(min(sleep_time, 0.05))

        self._thread = threading.Thread(target=run, name="MockDAQ", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None


class NIDAQmxBackend(AcquisitionBackend):
    def __init__(self, config: RecorderConfig):
        if nidaqmx is None:
            raise RuntimeError("nidaqmx is required for hardware acquisition.")
        self.config = config
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ai_task = None
        self._di_task = None
        self._sample_index = 0

    def _make_port_digital_channel(self) -> tuple[str, list[int]] | None:
        port: str | None = None
        line_numbers: list[int] = []
        for ch in self.config.enabled_digital:
            match = _DIGITAL_LINE_RE.match(ch.physical_channel.strip())
            if match is None:
                return None
            if port is None:
                port = match.group("port")
            elif port.lower() != match.group("port").lower():
                return None
            line_numbers.append(int(match.group("line")))

        if port is None or not line_numbers:
            return None
        first = min(line_numbers)
        last = max(line_numbers)
        physical = f"{self.config.device}/{port}/line{first}:{last}"
        bit_offsets = [line - first for line in line_numbers]
        return physical, bit_offsets

    @staticmethod
    def _terminal_config(name: str):
        normalized = name.strip().upper()
        try:
            return getattr(TerminalConfiguration, normalized)
        except AttributeError as exc:
            valid = ", ".join(item.name for item in TerminalConfiguration)
            raise ValueError(f"Unknown analog terminal_config {name!r}. Use one of: {valid}") from exc

    def start(self, callback: Callable[[DataChunk | Exception], None]) -> None:
        self._stop.clear()
        self._sample_index = 0

        try:
            ai_lines = self.config.enabled_analog
            di_lines = self.config.enabled_digital
            device = self.config.device
            rate = self.config.sample_rate_hz
            block = self.config.block_size
            di_bit_offsets: list[int] | None = None

            self._ai_task = nidaqmx.Task(new_task_name="daq_recorder_ai")
            for ch in ai_lines:
                self._ai_task.ai_channels.add_ai_voltage_chan(
                    f"{device}/{ch.physical_channel}",
                    min_val=ch.min_val,
                    max_val=ch.max_val,
                    terminal_config=self._terminal_config(ch.terminal_config),
                )
            if ai_lines:
                self._ai_task.timing.cfg_samp_clk_timing(
                    rate=rate,
                    sample_mode=AcquisitionType.CONTINUOUS,
                    samps_per_chan=block * 10,
                )

            self._di_task = nidaqmx.Task(new_task_name="daq_recorder_di")
            if di_lines:
                if not ai_lines:
                    raise RuntimeError(
                        "Hardware-timed DI requires at least one enabled AI channel "
                        "to provide the sample clock."
                    )
                port_channel = self._make_port_digital_channel()
                if port_channel is None:
                    di_physical = ",".join(f"{device}/{ch.physical_channel}" for ch in di_lines)
                    self._di_task.di_channels.add_di_chan(
                        di_physical,
                        line_grouping=LineGrouping.CHAN_PER_LINE,
                    )
                else:
                    di_physical, di_bit_offsets = port_channel
                    self._di_task.di_channels.add_di_chan(
                        di_physical,
                        line_grouping=LineGrouping.CHAN_FOR_ALL_LINES,
                    )
                self._di_task.timing.cfg_samp_clk_timing(
                    rate=rate,
                    source=f"/{device}/ai/SampleClock",
                    active_edge=Edge.RISING,
                    sample_mode=AcquisitionType.CONTINUOUS,
                    samps_per_chan=block * 10,
                )
                try:
                    self._di_task.triggers.start_trigger.cfg_dig_edge_start_trig(
                        f"/{device}/ai/StartTrigger"
                    )
                except Exception as exc:
                    status = getattr(exc, "error_code", None)
                    if status != -200452:
                        raise
                    # Some M-series devices do not support DI start triggers.
                    # Starting DI before AI still keeps samples aligned because
                    # the external AI sample clock does not tick until AI starts.
                    pass
        except Exception:
            self._cleanup_tasks()
            raise

        def run() -> None:
            try:
                if di_lines:
                    self._di_task.start()
                if ai_lines:
                    self._ai_task.start()
                while not self._stop.is_set():
                    analog_arr = np.zeros((block, 0), dtype=np.float32)
                    digital_arr = np.zeros((block, 0), dtype=np.uint8)

                    if ai_lines:
                        analog = self._ai_task.read(
                            number_of_samples_per_channel=block, timeout=5.0
                        )
                        analog_arr = np.asarray(analog, dtype=np.float32)
                        if analog_arr.ndim == 1:
                            analog_arr = analog_arr[np.newaxis, :]
                        analog_arr = analog_arr.T
                        for idx, ch in enumerate(ai_lines):
                            analog_arr[:, idx] *= ch.scale

                    if di_lines:
                        digital = self._di_task.read(
                            number_of_samples_per_channel=block, timeout=5.0
                        )
                        if di_bit_offsets is None:
                            digital_arr = np.asarray(digital, dtype=np.uint8)
                            if digital_arr.ndim == 1:
                                digital_arr = digital_arr[np.newaxis, :]
                            digital_arr = digital_arr.T
                        else:
                            words = np.asarray(digital, dtype=np.uint32).reshape(-1)
                            digital_arr = np.zeros((words.size, len(di_bit_offsets)), dtype=np.uint8)
                            for col, bit in enumerate(di_bit_offsets):
                                digital_arr[:, col] = ((words >> bit) & 1).astype(np.uint8)

                    callback(
                        DataChunk(
                            analog=analog_arr,
                            digital=digital_arr,
                            sample_index=self._sample_index,
                            timestamp=time.time(),
                        )
                    )
                    self._sample_index += block
            except Exception as exc:
                callback(exc)
            finally:
                self._cleanup_tasks()

        self._thread = threading.Thread(target=run, name="NIDAQmxDAQ", daemon=True)
        self._thread.start()

    def _cleanup_tasks(self) -> None:
        for task in (self._di_task, self._ai_task):
            if task is not None:
                try:
                    task.stop()
                except Exception:
                    pass
                try:
                    task.close()
                except Exception:
                    pass
        self._di_task = None
        self._ai_task = None

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._cleanup_tasks()


def make_backend(config: RecorderConfig) -> AcquisitionBackend:
    if config.simulate:
        return MockBackend(config)
    return NIDAQmxBackend(config)
