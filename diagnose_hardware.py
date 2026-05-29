from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from daq_recorder.acquisition import NIDAQmxBackend
from daq_recorder.config import RecorderConfig


def count_edges(values: np.ndarray) -> int:
    if values.size < 2:
        return 0
    return int(np.count_nonzero(np.diff(values.astype(np.int16))))


def main() -> int:
    parser = argparse.ArgumentParser(description="Short hardware-only DAQ input diagnostic.")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "default_config.json")
    parser.add_argument("--seconds", type=float, default=5.0)
    args = parser.parse_args()

    cfg = RecorderConfig.load(args.config)
    cfg.simulate = False

    chunks = []
    errors = []
    backend = NIDAQmxBackend(cfg)
    backend.start(lambda item: errors.append(item) if isinstance(item, Exception) else chunks.append(item))
    time.sleep(args.seconds)
    backend.stop()

    if errors:
        for exc in errors:
            print(f"ERROR: {exc}")
        return 1
    if not chunks:
        print("No samples acquired.")
        return 1

    analog = np.vstack([chunk.analog for chunk in chunks])
    digital = np.vstack([chunk.digital for chunk in chunks])
    print(f"Samples: {analog.shape[0] if analog.size else digital.shape[0]}")
    print(f"Sample rate: {cfg.sample_rate_hz:g} Hz")

    print("\nAnalog")
    for idx, ch in enumerate(cfg.enabled_analog):
        vals = analog[:, idx]
        ttl = (vals > 2.5).astype(np.uint8)
        print(
            f"{ch.name:16s} {ch.physical_channel:12s} "
            f"min={float(vals.min()):8.4f} max={float(vals.max()):8.4f} "
            f"mean={float(vals.mean()):8.4f} ptp={float(np.ptp(vals)):8.4f} "
            f">2.5V={int(ttl.sum()):7d} edges={count_edges(ttl):5d}"
        )

    print("\nDigital")
    for idx, ch in enumerate(cfg.enabled_digital):
        vals = digital[:, idx]
        high = int(vals.sum())
        print(
            f"{ch.name:16s} {ch.physical_channel:12s} "
            f"high_samples={high:7d} edges={count_edges(vals):5d} "
            f"unique={sorted(set(vals.tolist()))}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
