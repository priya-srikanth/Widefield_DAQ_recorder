from __future__ import annotations

import argparse
import time

import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration


def terminal_config(name: str):
    return getattr(TerminalConfiguration, name.upper())


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan NI analog inputs for visible signal changes.")
    parser.add_argument("--device", default="Dev1")
    parser.add_argument("--channels", default="ai0:15", help="Example: ai0:15 or ai0,ai4,ai8")
    parser.add_argument("--mode", default="RSE", choices=["RSE", "NRSE", "DIFF"])
    parser.add_argument("--rate", type=float, default=5000.0)
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()

    if ":" in args.channels:
        prefix, last = args.channels.split(":", 1)
        first = int(prefix.lower().replace("ai", ""))
        last_i = int(last)
        chans = [f"ai{i}" for i in range(first, last_i + 1)]
    else:
        chans = [item.strip() for item in args.channels.split(",") if item.strip()]

    n_samples = max(1, int(round(args.rate * args.seconds)))
    with nidaqmx.Task(new_task_name="scan_ai") as task:
        for ch in chans:
            task.ai_channels.add_ai_voltage_chan(
                f"{args.device}/{ch}",
                min_val=-10.0,
                max_val=10.0,
                terminal_config=terminal_config(args.mode),
            )
        task.timing.cfg_samp_clk_timing(
            rate=args.rate,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=n_samples,
        )
        print(f"Reading {len(chans)} channels for {args.seconds:g}s in {args.mode} mode...")
        data = task.read(number_of_samples_per_channel=n_samples, timeout=args.seconds + 5.0)

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]

    print("Channel summary")
    for idx, ch in enumerate(chans):
        vals = arr[idx]
        ttl = (vals > 2.5).astype(np.uint8)
        edges = int(np.count_nonzero(np.diff(ttl.astype(np.int16))))
        print(
            f"{ch:6s} min={float(vals.min()):8.4f} max={float(vals.max()):8.4f} "
            f"mean={float(vals.mean()):8.4f} ptp={float(np.ptp(vals)):8.4f} "
            f">2.5V={int(ttl.sum()):7d} edges={edges:5d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
