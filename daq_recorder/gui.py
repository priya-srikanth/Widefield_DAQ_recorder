from __future__ import annotations

import argparse
import json
import queue
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np

from .acquisition import DataChunk, make_backend
from .config import AnalogChannelConfig, DigitalChannelConfig, RecorderConfig
from .writer import HDF5Recorder


PACKAGE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PACKAGE_DIR / "default_config.json"


@dataclass
class ChannelRow:
    enabled_var: tk.BooleanVar
    name_var: tk.StringVar
    physical_var: tk.StringVar
    min_var: tk.StringVar | None = None
    max_var: tk.StringVar | None = None
    scale_var: tk.StringVar | None = None
    terminal_var: tk.StringVar | None = None


class ScrollableFrame(ttk.Frame):
    def __init__(self, master: tk.Misc):
        super().__init__(master)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.vscroll = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self._is_scrollbar_visible = True
        self.canvas.configure(yscrollcommand=self.vscroll.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vscroll.grid(row=0, column=1, sticky="ns")

        self.content = ttk.Frame(self.canvas)
        self._window_id = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.content.bind("<Configure>", self._update_scroll_region)
        self.canvas.bind("<Configure>", self._resize_content)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel, add="+")

    def _update_scroll_region(self, _event: tk.Event) -> None:
        self._update_scroll_region_and_bar()

    def _resize_content(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self._window_id, width=event.width)
        self._update_scroll_region_and_bar()

    def _update_scroll_region_and_bar(self) -> None:
        bbox = self.canvas.bbox("all")
        self.canvas.configure(scrollregion=bbox)
        if bbox is None:
            return
        content_height = max(0, bbox[3] - bbox[1])
        viewport_height = max(1, self.canvas.winfo_height())
        needs_scrollbar = content_height > viewport_height + 2
        if needs_scrollbar and not self._is_scrollbar_visible:
            self.vscroll.grid(row=0, column=1, sticky="ns")
            self._is_scrollbar_visible = True
        elif not needs_scrollbar and self._is_scrollbar_visible:
            self.vscroll.grid_remove()
            self._is_scrollbar_visible = False
            self.canvas.yview_moveto(0)
    def _on_mousewheel(self, event: tk.Event) -> None:
        if self.winfo_containing(event.x_root, event.y_root) is not None:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


class StripChart(ttk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        title: str,
        color: str,
        y_min: float,
        y_max: float,
        max_samples: int,
        is_digital: bool = False,
        display_decimation: int = 1,
    ):
        super().__init__(master)
        self.current_height = 80
        self.title = ttk.Label(self, text=title, width=18)
        self.title.pack(side=tk.LEFT, padx=(0, 4))
        self.columnconfigure(1, weight=1)
        self.canvas = tk.Canvas(self, width=700, height=self.current_height, bg="white", highlightthickness=1, highlightbackground="#cccccc")
        self.canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.value_label = ttk.Label(self, text="--", width=12)
        self.value_label.pack(side=tk.LEFT, padx=(4, 0))
        self.color = color
        self.y_min = y_min
        self.y_max = y_max
        self.is_digital = is_digital
        self.display_decimation = max(1, int(display_decimation))
        self.suspend_redraw = False
        self.max_display_samples = max(2, int(max_samples) * self.display_decimation)
        self.points: deque[tuple[float, float]] = deque()
        self._next_sample = 0.0
        self._baseline_id: int | None = None
        self._trace_id: int | None = None

    def set_range(self, y_min: float, y_max: float) -> None:
        self.y_min = y_min
        self.y_max = y_max

    def _digital_segment_summary(self, segment: np.ndarray) -> list[tuple[float, float]]:
        states = (segment >= 0.5).astype(np.uint8)
        first = float(states[0])
        last = float(states[-1])
        if states.min() == states.max():
            return [(0.5, first)]
        edge_indices = np.flatnonzero(np.diff(states.astype(np.int16)) != 0) + 1
        result = [(0.0, first)]
        for edge_index in edge_indices:
            result.append((float(edge_index) / max(1, states.size - 1), float(states[edge_index])))
        if result[-1][0] < 1.0:
            result.append((1.0, last))
        return result

    def _append_point(self, sample_index: float, value: float) -> None:
        self.points.append((float(sample_index), float(value)))
        cutoff = self._next_sample - self.max_display_samples
        while self.points and self.points[0][0] < cutoff:
            self.points.popleft()

    def push(self, values: np.ndarray) -> None:
        if not values.size:
            return
        values = np.asarray(values)
        if self.display_decimation > 1:
            trim = (values.size // self.display_decimation) * self.display_decimation
            if trim:
                windows = values[:trim].reshape(-1, self.display_decimation)
                for window in windows:
                    sample_start = self._next_sample
                    if self.is_digital:
                        for frac, val in self._digital_segment_summary(window):
                            self._append_point(sample_start + frac * self.display_decimation, val)
                    else:
                        self._append_point(sample_start, float(window.min()))
                        self._append_point(sample_start + 0.5 * self.display_decimation, float(window.max()))
                    self._next_sample += float(self.display_decimation)
                remainder = values[trim:]
                if remainder.size:
                    self._append_raw_values(remainder)
            else:
                self._append_raw_values(values)
        else:
            self._append_raw_values(values)

        if self.points:
            if self.is_digital:
                self.value_label.configure(text=str(int(round(self.points[-1][1]))))
            else:
                self.value_label.configure(text=f"{self.points[-1][1]:.3f}")
        if not self.suspend_redraw:
            self.redraw()

    def _append_raw_values(self, values: np.ndarray) -> None:
        for val in values.tolist():
            self._append_point(self._next_sample, float(val))
            self._next_sample += 1.0

    def set_height(self, height: int) -> None:
        height = int(max(28, min(90, height)))
        if height == self.current_height:
            return
        self.current_height = height
        self.canvas.configure(height=height)
        if not self.suspend_redraw:
            self.redraw()

    def _thin_for_width(self, x: np.ndarray, data: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
        if data.size <= width:
            return x, data
        edges = np.linspace(0, data.size, width + 1, dtype=np.int64)
        x_out: list[float] = []
        y_out: list[float] = []
        for start, stop in zip(edges[:-1], edges[1:]):
            if stop <= start:
                continue
            x_segment = x[start:stop]
            y_segment = data[start:stop]
            if self.is_digital:
                for frac, value in self._digital_segment_summary(y_segment):
                    x_out.append(float(x_segment[0] + frac * (x_segment[-1] - x_segment[0])))
                    y_out.append(value)
            else:
                min_index = int(np.argmin(y_segment))
                max_index = int(np.argmax(y_segment))
                ordered_indices = sorted((min_index, max_index))
                for idx in ordered_indices:
                    x_out.append(float(x_segment[idx]))
                    y_out.append(float(y_segment[idx]))
        return np.asarray(x_out, dtype=np.float32), np.asarray(y_out, dtype=np.float32)

    def redraw(self) -> None:
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        if not self.is_digital:
            if self._baseline_id is None:
                self._baseline_id = self.canvas.create_line(0, height / 2, width, height / 2, fill="#e6e6e6", dash=(2, 4))
            else:
                self.canvas.coords(self._baseline_id, 0, height / 2, width, height / 2)
        if len(self.points) < 2:
            return

        point_array = np.asarray(self.points, dtype=np.float32)
        x_bins = point_array[:, 0]
        data = point_array[:, 1]
        x_bins, data = self._thin_for_width(x_bins, data, width)
        if data.size < 2:
            return
        x_min = self._next_sample - self.max_display_samples
        x_span = max(1.0, self.max_display_samples)
        x = ((x_bins - x_min) / x_span) * width

        if self.is_digital:
            high_y = height * 0.22
            low_y = height * 0.78
            states = data >= 0.5
            coords = [float(x[0]), high_y if states[0] else low_y]
            for idx in range(len(data) - 1):
                x_next = float(x[idx + 1])
                y_now = high_y if states[idx] else low_y
                y_next = high_y if states[idx + 1] else low_y
                coords.extend((x_next, y_now))
                if y_next != y_now:
                    coords.extend((x_next, y_next))
            coords.extend((float(width), high_y if states[-1] else low_y))
            line_width = 1
        else:
            if self.y_max == self.y_min:
                y = np.zeros_like(data) + height / 2
            else:
                clipped = np.clip(data, self.y_min, self.y_max)
                y = height - ((clipped - self.y_min) / (self.y_max - self.y_min)) * height
            coords = []
            for xv, yv in zip(x, y):
                if -2 <= xv <= width + 2:
                    coords.extend((float(xv), float(yv)))
            line_width = 1.5

        if len(coords) < 4:
            return
        if self._trace_id is None:
            self._trace_id = self.canvas.create_line(*coords, fill=self.color, width=line_width)
        else:
            self.canvas.coords(self._trace_id, *coords)
            self.canvas.itemconfigure(self._trace_id, fill=self.color, width=line_width)
        if self._baseline_id is not None:
            self.canvas.tag_lower(self._baseline_id, self._trace_id)

class RecorderApp(tk.Tk):
    def __init__(self, config_path: Path):
        super().__init__()
        self.title("Widefield DAQ Recorder")
        self.geometry("1260x900")
        self.minsize(900, 600)

        self.config_path = config_path
        self.config_obj = RecorderConfig.load(config_path)
        self.title(self.config_obj.app_title)
        self.backend = None
        self.writer: HDF5Recorder | None = None
        self.recording = False
        self.running = False
        self.data_queue: queue.Queue[DataChunk | Exception] = queue.Queue()
        self._pending_plot_chunks: list[DataChunk] = []
        self.channel_rows_ai: list[ChannelRow] = []
        self.channel_rows_di: list[ChannelRow] = []
        self.plots: list[StripChart] = []
        self.plot_routes: list[tuple[str, int, StripChart]] = []
        self.display_order_text: tk.Text | None = None
        self.plot_scroll: ScrollableFrame | None = None
        self.status_var = tk.StringVar(value="Ready")
        self.file_var = tk.StringVar(value="")
        self.sample_count_var = tk.StringVar(value="0")
        self.analog_stats_var = tk.StringVar(value="")
        self.digital_stats_var = tk.StringVar(value="")
        self._last_plot_update = 0.0
        self._plot_interval_s = 0.20
        self._analog_edges: np.ndarray | None = None
        self._analog_high_samples: np.ndarray | None = None
        self._analog_last: np.ndarray | None = None
        self._digital_edges: np.ndarray | None = None
        self._digital_high_samples: np.ndarray | None = None
        self._digital_last: np.ndarray | None = None
        self._resize_after_id: str | None = None
        self._max_samples: int | None = None
        self._auto_stopping = False

        self._build_ui()
        self._load_config_into_ui(self.config_obj)
        self.after(50, self._poll_queue)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)

        toolbar = ttk.Frame(root)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(toolbar, text="Play", command=self.play).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Record", command=self.record).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(toolbar, text="Stop", command=self.stop).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(toolbar, text="Load Config", command=self.load_config_dialog).pack(side=tk.LEFT, padx=(18, 0))
        ttk.Button(toolbar, text="Save Config", command=self.save_config).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(toolbar, text="Save Config As", command=self.save_config_as).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(toolbar, text="Choose Output Folder", command=self.choose_output_directory).pack(side=tk.LEFT, padx=(18, 0))

        status = ttk.Frame(root)
        status.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(status, text="Status:").pack(side=tk.LEFT)
        ttk.Label(status, textvariable=self.status_var).pack(side=tk.LEFT, padx=(4, 18))
        ttk.Label(status, text="Samples:").pack(side=tk.LEFT)
        ttk.Label(status, textvariable=self.sample_count_var).pack(side=tk.LEFT, padx=(4, 18))
        ttk.Label(status, text="Recording file:").pack(side=tk.LEFT)
        ttk.Label(status, textvariable=self.file_var).pack(side=tk.LEFT, padx=(4, 0))

        paned = tk.PanedWindow(root, orient=tk.HORIZONTAL, sashwidth=8, sashrelief=tk.RAISED, bd=0)
        paned.grid(row=2, column=0, sticky="nsew")

        config_frame = ttk.Frame(paned, padding=(0, 0, 8, 0))
        plot_frame = ttk.Frame(paned)
        paned.add(config_frame, minsize=280, stretch="never")
        paned.add(plot_frame, minsize=420, stretch="always")

        config_scroll = ScrollableFrame(config_frame)
        config_scroll.pack(fill=tk.BOTH, expand=True)
        self.plot_scroll = ScrollableFrame(plot_frame)
        self.plot_scroll.pack(fill=tk.BOTH, expand=True)
        self.plot_scroll.canvas.bind("<Configure>", self._schedule_plot_resize, add="+")

        self._build_config_panel(config_scroll.content)
        self._build_plot_panel(self.plot_scroll.content)

    def _build_config_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        general = ttk.LabelFrame(parent, text="General", padding=8)
        general.grid(row=0, column=0, sticky="ew")
        general.columnconfigure(1, weight=1)

        self.device_var = tk.StringVar()
        self.rate_var = tk.StringVar()
        self.block_var = tk.StringVar()
        self.display_seconds_var = tk.StringVar()
        self.max_duration_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.file_prefix_var = tk.StringVar()
        self.analog_storage_var = tk.StringVar(value="int16_scaled")
        self.simulate_var = tk.BooleanVar()

        general_fields = [
            ("Device", self.device_var),
            ("Sample rate (Hz)", self.rate_var),
            ("Block size", self.block_var),
            ("Display seconds", self.display_seconds_var),
            ("Max duration (s)", self.max_duration_var),
            ("Output directory", self.output_dir_var),
            ("File prefix", self.file_prefix_var),
        ]
        for row, (label, var) in enumerate(general_fields):
            ttk.Label(general, text=label).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Entry(general, textvariable=var, width=28).grid(row=row, column=1, sticky="ew", pady=2)
        storage_row = len(general_fields)
        ttk.Label(general, text="Analog storage").grid(row=storage_row, column=0, sticky="w", pady=2)
        ttk.Combobox(
            general,
            textvariable=self.analog_storage_var,
            values=("int16_scaled", "float32"),
            state="readonly",
            width=25,
        ).grid(row=storage_row, column=1, sticky="ew", pady=2)
        ttk.Checkbutton(general, text="Simulate instead of using NI hardware", variable=self.simulate_var).grid(
            row=storage_row + 1, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )

        ai_group = ttk.LabelFrame(parent, text="Analog Channels", padding=8)
        ai_group.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self._build_ai_editor(ai_group)

        di_group = ttk.LabelFrame(parent, text="Digital Channels", padding=8)
        di_group.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        self._build_di_editor(di_group)

        order_group = ttk.LabelFrame(parent, text="Display Order", padding=8)
        order_group.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        order_group.columnconfigure(0, weight=1)
        self.display_order_text = tk.Text(order_group, height=8, width=34, wrap="none")
        self.display_order_text.grid(row=0, column=0, sticky="ew")
        buttons = ttk.Frame(order_group)
        buttons.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(buttons, text="Use Current Channels", command=self.fill_display_order_from_current_ui).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Apply Display Order", command=self.apply_display_order).pack(side=tk.LEFT, padx=(6, 0))

    def _build_ai_editor(self, parent: ttk.LabelFrame) -> None:
        headers = ["On", "Name", "Physical", "Min", "Max", "Scale", "Mode"]
        for col, text in enumerate(headers):
            ttk.Label(parent, text=text).grid(row=0, column=col, sticky="w", padx=2)
        for idx in range(8):
            row = idx + 1
            enabled = tk.BooleanVar(value=False)
            name = tk.StringVar()
            physical = tk.StringVar()
            min_var = tk.StringVar(value="-5")
            max_var = tk.StringVar(value="5")
            scale = tk.StringVar(value="1.0")
            terminal = tk.StringVar(value="DIFF")
            ttk.Checkbutton(parent, variable=enabled).grid(row=row, column=0, sticky="w")
            ttk.Entry(parent, textvariable=name, width=14).grid(row=row, column=1, padx=2, pady=1)
            ttk.Entry(parent, textvariable=physical, width=12).grid(row=row, column=2, padx=2, pady=1)
            ttk.Entry(parent, textvariable=min_var, width=8).grid(row=row, column=3, padx=2, pady=1)
            ttk.Entry(parent, textvariable=max_var, width=8).grid(row=row, column=4, padx=2, pady=1)
            ttk.Entry(parent, textvariable=scale, width=8).grid(row=row, column=5, padx=2, pady=1)
            ttk.Combobox(parent, textvariable=terminal, values=("RSE", "NRSE", "DIFF", "PSEUDODIFFERENTIAL"), width=8).grid(row=row, column=6, padx=2, pady=1)
            self.channel_rows_ai.append(ChannelRow(enabled, name, physical, min_var, max_var, scale, terminal))

    def _build_di_editor(self, parent: ttk.LabelFrame) -> None:
        headers = ["On", "Name", "Physical"]
        for col, text in enumerate(headers):
            ttk.Label(parent, text=text).grid(row=0, column=col, sticky="w", padx=2)
        for idx in range(12):
            row = idx + 1
            enabled = tk.BooleanVar(value=False)
            name = tk.StringVar()
            physical = tk.StringVar()
            ttk.Checkbutton(parent, variable=enabled).grid(row=row, column=0, sticky="w")
            ttk.Entry(parent, textvariable=name, width=18).grid(row=row, column=1, padx=2, pady=1)
            ttk.Entry(parent, textvariable=physical, width=16).grid(row=row, column=2, padx=2, pady=1)
            self.channel_rows_di.append(ChannelRow(enabled, name, physical))

    def _build_plot_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        ttk.Label(parent, text="Live Channel Views", font=("", 11, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(parent, textvariable=self.analog_stats_var, justify=tk.LEFT).grid(row=1, column=0, sticky="ew", pady=(4, 0))
        ttk.Label(parent, textvariable=self.digital_stats_var, justify=tk.LEFT).grid(row=2, column=0, sticky="ew", pady=(4, 0))
        self.plot_container = ttk.Frame(parent)
        self.plot_container.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        parent.rowconfigure(3, weight=1)

    def _display_key_for_channel(self, kind: str, channel: object) -> str:
        prefix = "ai" if kind == "analog" else "di"
        return f"{prefix}:{getattr(channel, 'physical_channel', '')}"

    def _default_display_order(self, cfg: RecorderConfig) -> list[str]:
        analog = [self._display_key_for_channel("analog", ch) for ch in cfg.enabled_analog]
        digital = [self._display_key_for_channel("digital", ch) for ch in cfg.enabled_digital]
        return analog + digital

    def _set_display_order_text(self, order: list[str]) -> None:
        if self.display_order_text is None:
            return
        self.display_order_text.delete("1.0", tk.END)
        self.display_order_text.insert("1.0", "\n".join(order))

    def _display_order_from_text(self) -> list[str]:
        if self.display_order_text is None:
            return []
        text = self.display_order_text.get("1.0", tk.END)
        order: list[str] = []
        for raw_line in text.replace(",", "\n").splitlines():
            item = raw_line.strip()
            if item:
                order.append(item)
        return order

    def fill_display_order_from_current_ui(self) -> None:
        cfg = self._build_config_from_ui()
        self._set_display_order_text(self._default_display_order(cfg))

    def apply_display_order(self) -> None:
        cfg = self._build_config_from_ui()
        self.config_obj = cfg
        self._rebuild_plots(cfg)
        self.status_var.set("Applied display order")

    def _canonical_display_key(self, item: str, candidates: dict[str, tuple[str, int, object]]) -> str | None:
        item = item.strip()
        if not item:
            return None
        lowered = item.lower()
        candidate_keys = {key.lower(): key for key in candidates}
        if lowered in candidate_keys:
            return candidate_keys[lowered]

        if ":" in lowered:
            prefix, wanted = lowered.split(":", 1)
            kind_filter = "analog" if prefix == "ai" else "digital" if prefix == "di" else ""
        else:
            wanted = lowered
            kind_filter = ""

        matches = []
        for key, (kind, _, channel) in candidates.items():
            if kind_filter and kind != kind_filter:
                continue
            name = getattr(channel, "name", "").lower()
            physical = getattr(channel, "physical_channel", "").lower()
            if wanted == name or wanted == physical:
                matches.append(key)
        return matches[0] if len(matches) == 1 else None

    def _ordered_display_channels(self, cfg: RecorderConfig) -> list[tuple[str, int, object]]:
        candidates: dict[str, tuple[str, int, object]] = {}
        for idx, ch in enumerate(cfg.enabled_analog):
            candidates[self._display_key_for_channel("analog", ch)] = ("analog", idx, ch)
        for idx, ch in enumerate(cfg.enabled_digital):
            candidates[self._display_key_for_channel("digital", ch)] = ("digital", idx, ch)

        selected: list[tuple[str, int, object]] = []
        used: set[str] = set()
        for item in cfg.display_order:
            key = self._canonical_display_key(item, candidates)
            if key is not None and key not in used:
                selected.append(candidates[key])
                used.add(key)

        for key in self._default_display_order(cfg):
            if key in candidates and key not in used:
                selected.append(candidates[key])
                used.add(key)
        return selected
    def _load_config_into_ui(self, cfg: RecorderConfig) -> None:
        self.device_var.set(cfg.device)
        self.rate_var.set(str(cfg.sample_rate_hz))
        self.block_var.set(str(cfg.block_size))
        self.display_seconds_var.set(str(cfg.display_seconds))
        max_duration = getattr(cfg, "max_duration_s", 0.0)
        self.max_duration_var.set("" if max_duration <= 0 else str(max_duration))
        self.output_dir_var.set(cfg.output_directory)
        self.file_prefix_var.set(getattr(cfg, "file_prefix", "daq"))
        self.analog_storage_var.set(getattr(cfg, "analog_storage", "int16_scaled"))
        self.simulate_var.set(cfg.simulate)

        for row in self.channel_rows_ai:
            row.enabled_var.set(False)
            row.name_var.set("")
            row.physical_var.set("")
            if row.min_var:
                row.min_var.set("-5")
            if row.max_var:
                row.max_var.set("5")
            if row.scale_var:
                row.scale_var.set("1.0")
            if row.terminal_var:
                row.terminal_var.set("DIFF")
        for idx, ch in enumerate(cfg.analog_channels[: len(self.channel_rows_ai)]):
            row = self.channel_rows_ai[idx]
            row.enabled_var.set(ch.enabled)
            row.name_var.set(ch.name)
            row.physical_var.set(ch.physical_channel)
            row.min_var.set(str(ch.min_val))
            row.max_var.set(str(ch.max_val))
            row.scale_var.set(str(ch.scale))
            row.terminal_var.set(ch.terminal_config)

        for row in self.channel_rows_di:
            row.enabled_var.set(False)
            row.name_var.set("")
            row.physical_var.set("")
        for idx, ch in enumerate(cfg.digital_channels[: len(self.channel_rows_di)]):
            row = self.channel_rows_di[idx]
            row.enabled_var.set(ch.enabled)
            row.name_var.set(ch.name)
            row.physical_var.set(ch.physical_channel)

        self._set_display_order_text(cfg.display_order or self._default_display_order(cfg))

        self._rebuild_plots(cfg)

    def _build_config_from_ui(self) -> RecorderConfig:
        analog_channels = []
        for row in self.channel_rows_ai:
            if row.name_var.get().strip() or row.physical_var.get().strip():
                analog_channels.append(
                    AnalogChannelConfig(
                        name=row.name_var.get().strip(),
                        physical_channel=row.physical_var.get().strip(),
                        enabled=bool(row.enabled_var.get()),
                        min_val=float(row.min_var.get()),
                        max_val=float(row.max_var.get()),
                        scale=float(row.scale_var.get()),
                        terminal_config=row.terminal_var.get().strip() if row.terminal_var else "DIFF",
                    )
                )

        digital_channels = []
        for row in self.channel_rows_di:
            if row.name_var.get().strip() or row.physical_var.get().strip():
                digital_channels.append(
                    DigitalChannelConfig(
                        name=row.name_var.get().strip(),
                        physical_channel=row.physical_var.get().strip(),
                        enabled=bool(row.enabled_var.get()),
                    )
                )

        return RecorderConfig(
            app_title=self.config_obj.app_title,
            device=self.device_var.get().strip(),
            sample_rate_hz=float(self.rate_var.get()),
            block_size=int(self.block_var.get()),
            display_seconds=float(self.display_seconds_var.get()),
            max_duration_s=float(self.max_duration_var.get().strip() or 0.0),
            output_directory=self.output_dir_var.get().strip(),
            file_prefix=self.file_prefix_var.get().strip(),
            analog_storage=self.analog_storage_var.get().strip() or "int16_scaled",
            simulate=bool(self.simulate_var.get()),
            analog_channels=analog_channels,
            digital_channels=digital_channels,
            display_order=self._display_order_from_text(),
        )

    def _rebuild_plots(self, cfg: RecorderConfig) -> None:
        for child in self.plot_container.winfo_children():
            child.destroy()
        self.plots.clear()
        self.plot_routes.clear()
        colors = ["#1f77b4", "#2ca02c", "#9467bd", "#d62728", "#8c564b", "#17becf", "#7f7f7f", "#bcbd22"]
        display_points_per_channel = 5000
        raw_display_samples = int(max(2, round(cfg.display_seconds * cfg.sample_rate_hz)))
        display_decimation = max(1, int(np.ceil(raw_display_samples / display_points_per_channel)))
        max_samples = max(2, int(np.ceil(raw_display_samples / display_decimation)))
        for row, (kind, source_index, ch) in enumerate(self._ordered_display_channels(cfg)):
            if kind == "analog":
                plot = StripChart(
                    self.plot_container,
                    ch.name,
                    colors[row % len(colors)],
                    ch.min_val * ch.scale,
                    ch.max_val * ch.scale,
                    max_samples,
                    is_digital=False,
                    display_decimation=display_decimation,
                )
            else:
                plot = StripChart(
                    self.plot_container,
                    ch.name,
                    colors[row % len(colors)],
                    -0.1,
                    1.1,
                    max_samples,
                    is_digital=True,
                    display_decimation=display_decimation,
                )
            plot.grid(row=row, column=0, sticky="ew", pady=2)
            self.plot_container.columnconfigure(0, weight=1)
            self.plots.append(plot)
            self.plot_routes.append((kind, source_index, plot))
        self.after_idle(self._apply_plot_resize)
    def _schedule_plot_resize(self, _event: tk.Event | None = None) -> None:
        for plot in self.plots:
            plot.suspend_redraw = True
        if self._resize_after_id is not None:
            self.after_cancel(self._resize_after_id)
        self._resize_after_id = self.after(160, self._apply_plot_resize)

    def _apply_plot_resize(self) -> None:
        self._resize_after_id = None
        if not self.plots or self.plot_scroll is None:
            return
        self.plot_scroll.content.update_idletasks()
        viewport_height = max(1, self.plot_scroll.canvas.winfo_height())
        top = self.plot_container.winfo_y() if self.plot_container.winfo_exists() else 80
        row_gap = 4
        border_slop = 6
        available = max(30, viewport_height - top - border_slop)
        per_plot = int((available - row_gap * max(0, len(self.plots) - 1)) / len(self.plots))
        height = max(28, min(80, per_plot))
        for plot in self.plots:
            plot.suspend_redraw = False
            plot.set_height(height)
            plot.redraw()
        self.plot_scroll.content.update_idletasks()
        self.plot_scroll._update_scroll_region_and_bar()

    def choose_output_directory(self) -> None:
        path = filedialog.askdirectory(initialdir=self.output_dir_var.get() or str(Path.cwd()))
        if path:
            self.output_dir_var.set(path)

    def load_config_dialog(self) -> None:
        path = filedialog.askopenfilename(
            title="Load recorder config",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
            initialdir=str(self.config_path.parent),
        )
        if not path:
            return
        cfg = RecorderConfig.load(path)
        self.config_path = Path(path)
        self.config_obj = cfg
        self._load_config_into_ui(cfg)
        self.status_var.set(f"Loaded config from {path}")

    def save_config(self) -> None:
        cfg = self._build_config_from_ui()
        cfg.save(self.config_path)
        self.config_obj = cfg
        self._rebuild_plots(cfg)
        self.status_var.set(f"Saved config to {self.config_path}")

    def save_config_as(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save recorder config",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
            initialdir=str(self.config_path.parent),
            initialfile=self.config_path.name,
        )
        if not path:
            return
        self.config_path = Path(path)
        self.save_config()

    def play(self) -> None:
        confirmed = messagebox.askokcancel(
            "You are not recording",
            "You are not recording. Data will be displayed but will not be saved.\n\nClick OK to continue in Play mode, or Cancel to go back.",
            icon=messagebox.WARNING,
            default=messagebox.CANCEL,
        )
        if confirmed:
            self._start(record=False)

    def record(self) -> None:
        self._start(record=True)

    def _start(self, record: bool) -> None:
        if self.running:
            messagebox.showinfo("Already running", "Stop the current acquisition before starting a new one.")
            return
        cfg = self._build_config_from_ui()
        if not cfg.enabled_analog and not cfg.enabled_digital:
            messagebox.showerror("No channels", "Enable at least one analog or digital channel.")
            return
        self.config_obj = cfg
        self._rebuild_plots(cfg)
        self._reset_digital_stats(cfg)

        try:
            self.backend = make_backend(cfg)
        except Exception as exc:
            messagebox.showerror("Backend error", str(exc))
            return

        self.writer = None
        if record:
            try:
                default_path = HDF5Recorder.make_default_path(cfg)
                self.writer = HDF5Recorder(cfg, default_path)
                info = self.writer.open()
                self.file_var.set(str(info.file_path))
            except Exception as exc:
                self.writer = None
                messagebox.showerror("Recording error", str(exc))
                return
        else:
            self.file_var.set("")

        self._pending_plot_chunks.clear()
        self._last_plot_update = 0.0
        self._max_samples = int(round(cfg.max_duration_s * cfg.sample_rate_hz)) if cfg.max_duration_s > 0 else None
        self._auto_stopping = False
        self.running = True
        self.recording = record
        self.sample_count_var.set("0")
        self._reset_analog_stats(cfg)
        self._reset_digital_stats(cfg)
        if self._max_samples is None:
            self.status_var.set("Running (recording)" if record else "Running (play)")
        else:
            duration_text = f" for {cfg.max_duration_s:g} s"
            self.status_var.set(("Running (recording)" if record else "Running (play)") + duration_text)

        def callback(chunk: DataChunk | Exception) -> None:
            self.data_queue.put(chunk)

        try:
            self.backend.start(callback)
        except Exception as exc:
            self.running = False
            self.recording = False
            if self.writer:
                self.writer.close()
                self.writer = None
            messagebox.showerror("Start error", str(exc))

    def stop(self) -> None:
        if self.backend is not None:
            try:
                self.backend.stop()
            except Exception as exc:
                messagebox.showwarning("Stop warning", str(exc))
        self.backend = None
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        self.running = False
        self.recording = False
        self._max_samples = None
        self._auto_stopping = False
        self.status_var.set("Stopped")

    def _poll_queue(self) -> None:
        chunks: list[DataChunk] = []
        try:
            for _ in range(100):
                chunk = self.data_queue.get_nowait()
                if isinstance(chunk, Exception):
                    self._handle_backend_error(chunk)
                    continue
                chunks.append(chunk)
        except queue.Empty:
            pass

        if chunks:
            self._handle_chunks(chunks)
        self.after(50, self._poll_queue)

    def _handle_chunk(self, chunk: DataChunk) -> None:
        if isinstance(chunk, Exception):
            self._handle_backend_error(chunk)
            return
        self._handle_chunks([chunk])

    def _handle_chunks(self, chunks: list[DataChunk]) -> None:
        if not chunks:
            return
        for chunk in chunks:
            if self.writer is not None:
                self.writer.append(chunk.analog, chunk.digital)

        chunk = chunks[-1]
        total = chunk.sample_index + (chunk.analog.shape[0] if chunk.analog.size else chunk.digital.shape[0])
        self.sample_count_var.set(str(total))
        if self._max_samples is not None and total >= self._max_samples and not self._auto_stopping:
            self._auto_stopping = True
            self.after_idle(self._finish_finite_session)

        self._pending_plot_chunks.extend(chunks)
        now = time.perf_counter()
        if now - self._last_plot_update < self._plot_interval_s:
            return
        self._last_plot_update = now

        plot_chunks = self._pending_plot_chunks
        self._pending_plot_chunks = []
        analog = np.vstack([item.analog for item in plot_chunks if item.analog.size]) if any(item.analog.size for item in plot_chunks) else np.zeros((0, 0), dtype=np.float32)
        digital = np.vstack([item.digital for item in plot_chunks if item.digital.size]) if any(item.digital.size for item in plot_chunks) else np.zeros((0, 0), dtype=np.uint8)
        self._update_analog_stats(analog)
        self._update_digital_stats(digital)

        for kind, source_index, plot in self.plot_routes:
            if kind == "analog" and source_index < analog.shape[1]:
                plot.push(analog[:, source_index])
            elif kind == "digital" and source_index < digital.shape[1]:
                plot.push(digital[:, source_index])


    def _finish_finite_session(self) -> None:
        if not self.running:
            return
        self.stop()
        self.status_var.set("Stopped after finite session duration")
    def _handle_backend_error(self, exc: Exception) -> None:
        if not self.running:
            return
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        self.backend = None
        self.running = False
        self.recording = False
        self.status_var.set("Error")
        messagebox.showerror("Acquisition error", str(exc))

    def _reset_analog_stats(self, cfg: RecorderConfig) -> None:
        n = len(cfg.enabled_analog)
        self._analog_edges = np.zeros(n, dtype=np.int64)
        self._analog_high_samples = np.zeros(n, dtype=np.int64)
        self._analog_last = None
        self.analog_stats_var.set("Analog TTL counters: waiting for samples" if n else "")

    def _update_analog_stats(self, analog: np.ndarray) -> None:
        if analog.size == 0 or self._analog_edges is None or self._analog_high_samples is None:
            return
        if analog.shape[1] != self._analog_edges.shape[0]:
            self._reset_analog_stats(self.config_obj)
            return

        ttl = (analog > 2.5).astype(np.uint8)
        self._analog_high_samples += ttl.sum(axis=0).astype(np.int64)
        if self._analog_last is not None:
            transitions = np.vstack([self._analog_last, ttl])
        else:
            transitions = ttl
        self._analog_edges += np.count_nonzero(np.diff(transitions.astype(np.int16), axis=0), axis=0)
        self._analog_last = ttl[-1, :].copy()

        names = [ch.name for ch in self.config_obj.enabled_analog]
        lines = []
        for idx, name in enumerate(names):
            last_v = float(analog[-1, idx])
            edges = int(self._analog_edges[idx])
            lines.append(f"{name}={last_v:.2f}V e={edges}")
        self.analog_stats_var.set("Analog: " + " | ".join(lines))

    def _reset_digital_stats(self, cfg: RecorderConfig) -> None:
        n = len(cfg.enabled_digital)
        self._digital_edges = np.zeros(n, dtype=np.int64)
        self._digital_high_samples = np.zeros(n, dtype=np.int64)
        self._digital_last = None
        self.digital_stats_var.set("Digital: waiting for samples" if n else "")

    def _update_digital_stats(self, digital: np.ndarray) -> None:
        if digital.size == 0 or self._digital_edges is None or self._digital_high_samples is None:
            return
        if digital.shape[1] != self._digital_edges.shape[0]:
            self._reset_digital_stats(self.config_obj)
            return

        self._digital_high_samples += digital.sum(axis=0).astype(np.int64)
        if self._digital_last is not None:
            transitions = np.vstack([self._digital_last, digital])
        else:
            transitions = digital
        self._digital_edges += np.count_nonzero(np.diff(transitions.astype(np.int16), axis=0), axis=0)
        self._digital_last = digital[-1, :].copy()

        names = [ch.name for ch in self.config_obj.enabled_digital]
        lines = []
        for idx, name in enumerate(names):
            last = int(self._digital_last[idx])
            edges = int(self._digital_edges[idx])
            lines.append(f"{name}={last} e={edges}")
        self.digital_stats_var.set("Digital: " + " | ".join(lines))

    def on_close(self) -> None:
        self.stop()
        self.destroy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WaveSurfer-like NI DAQ recorder draft.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to recorder JSON config.")
    parser.add_argument("--simulate", action="store_true", help="Override the config and use simulated data.")
    parser.add_argument("--hardware", action="store_true", help="Override the config and use NI hardware.")
    args = parser.parse_args(argv)

    app = RecorderApp(args.config)
    if args.simulate or args.hardware:
        cfg = app._build_config_from_ui()
        cfg.simulate = bool(args.simulate and not args.hardware)
        app.config_obj = cfg
        app._load_config_into_ui(cfg)
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
    return 0




