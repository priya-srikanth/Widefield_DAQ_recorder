"""Verify a widefield session was trial-gated: camera frames only during trials.

Reads the DAQ recorder HDF5 (the `pco_exposure` digital line, `trial_start`,
`trial_end`/`trial_stop`, and the LED TTL analog lines) and, optionally, the
labcams `.camlog`, then reports whether camera exposures were gated to behavior
trials and whether the 415/470 LEDs alternated cleanly.

The primary, robust signal is the structure of the `pco_exposure` train: with
working acquire-enable gating it breaks into one burst per trial separated by
multi-second pauses; with broken gating it is a single uninterrupted train.

Usage:
  python tools/check_trial_gating.py --daq path\\to\\daq.h5
  python tools/check_trial_gating.py --daq daq.h5 --camlog path\\to\\run.camlog

Run in an environment with h5py + numpy (e.g. the `wfield` conda env).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
from pathlib import Path

import h5py
import numpy as np


def _find(names, *patterns):
    """Return the index of the first name matching any (case-insensitive) regex."""
    for i, n in enumerate(names):
        for p in patterns:
            if re.search(p, n, re.IGNORECASE):
                return i
    return None


def _rises(sig, thr):
    b = (np.asarray(sig) > thr).astype(np.int8)
    return np.flatnonzero(np.diff(b) == 1) + 1


def load_daq(h5_path: Path):
    with h5py.File(h5_path, "r") as f:
        fs = float(f.attrs["sample_rate_hz"])
        an = [s.decode() for s in f["analog/channel_names"][:]]
        raw = f["analog/samples_int16"][:].astype(np.float32)
        volts = raw * f["analog/int16_scale_volts_per_count"][:] + f["analog/int16_offset_volts"][:]
        di = [s.decode() for s in f["digital/channel_names"][:]]
        bits = np.unpackbits(f["digital/packed_samples"][:, 0][:, None], axis=1, bitorder="little")
    return fs, an, volts, di, bits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--daq", type=Path, required=True, help="DAQ recorder HDF5 file")
    ap.add_argument("--camlog", type=Path, default=None, help="labcams .camlog (optional)")
    ap.add_argument("--gap-factor", type=float, default=1.8, help="IFI multiple of median that counts as an inter-trial gap")
    ap.add_argument("--ttl-thresh", type=float, default=1.5, help="analog TTL threshold (V)")
    args = ap.parse_args()

    fs, an, volts, di, bits = load_daq(args.daq)
    N = bits.shape[0]
    T = N / fs
    print(f"=== DAQ {args.daq.name} ===")
    print(f"  duration {T:.1f}s @ {fs:g} Hz, analog={an}, digital={di}")

    pco_i = _find(di, r"pco", r"expos")
    ts_i = _find(di, r"trial_?start")
    if pco_i is None:
        print("  ERROR: no pco_exposure-like digital channel found")
        return 2
    exp_r = _rises(bits[:, pco_i].astype(float), 0.5)
    exp_t = exp_r / fs
    ts_t = _rises(bits[:, ts_i].astype(float), 0.5) / fs if ts_i is not None else np.array([])

    # trial_end / trial_stop: analog first, then digital
    te_i_an = _find(an, r"trial_?end", r"trial_?stop")
    te_i_di = _find(di, r"trial_?end", r"trial_?stop")
    if te_i_an is not None:
        te_t = _rises(volts[:, te_i_an], args.ttl_thresh) / fs
    elif te_i_di is not None:
        te_t = _rises(bits[:, te_i_di].astype(float), 0.5) / fs
    else:
        te_t = np.array([])

    # --- exposure burst structure (the robust gating signal) ---
    print(f"  trials: trial_start={len(ts_t)}  trial_end/stop={len(te_t)}")
    print(f"  pco_exposure pulses: {len(exp_t)}")
    if len(exp_t) < 2:
        print("  too few exposures to assess gating")
        return 0
    ifi = np.diff(exp_t) * 1000.0
    med = float(np.median(ifi))
    gap_idx = np.flatnonzero(ifi > args.gap_factor * med)
    print(f"  inter-frame interval: median={med:.2f} ms  min={ifi.min():.2f}  max={ifi.max():.2f}")
    print(f"  inter-trial pauses (gaps > {args.gap_factor}x median): {len(gap_idx)}")
    for i in gap_idx:
        print(f"     {ifi[i]/1000:.2f} s pause at +{exp_t[i]:.2f}s")
    n_bursts = len(gap_idx) + 1
    # burst boundaries
    starts = np.r_[0, gap_idx + 1]
    stops = np.r_[gap_idx, len(exp_t) - 1]
    burst_durs = exp_t[stops] - exp_t[starts]
    print(f"  exposure bursts: {n_bursts}  mean burst {burst_durs.mean():.2f}s  "
          f"mean in-burst rate {np.mean([(b - a + 1) / max(d, 1e-9) for a, b, d in zip(starts, stops, burst_durs) if d > 0]):.1f} fps")

    # exposures outside [trial_start, trial_end] windows (secondary; pairing can be imperfect)
    if len(ts_t) and len(te_t):
        wins = []
        for s0 in ts_t:
            later = te_t[te_t > s0]
            if len(later):
                wins.append((s0, later[0]))
        inw = np.zeros(len(exp_t), bool)
        for a, b in wins:
            inw |= (exp_t >= a - 0.01) & (exp_t <= b + 0.01)
        print(f"  exposures inside trial windows={int(inw.sum())} outside={int((~inw).sum())} "
              f"(window pairing is approximate)")

    # --- LED alternation ---
    led_idx = [i for i, n in enumerate(an) if re.search(r"led", n, re.IGNORECASE)]
    if len(led_idx) >= 2:
        a_i, b_i = led_idx[0], led_idx[1]
        samp = np.clip(exp_r + int(0.002 * fs), 0, N - 1)
        sa = volts[samp, a_i] > args.ttl_thresh
        sb = volts[samp, b_i] > args.ttl_thresh
        code = sa.astype(int) + 2 * sb.astype(int)
        single = code[(code == 1) | (code == 2)]
        consec = int(np.sum(single[1:] == single[:-1])) if len(single) > 1 else 0
        print(f"  LED @ exposure: {an[a_i]}={int(sa.sum())} {an[b_i]}={int(sb.sum())} "
              f"both={int((sa & sb).sum())} neither={int((~sa & ~sb).sum())} "
              f"consec-same(single-LED)={consec}/{len(single)}")
        led_ok = int((sa & sb).sum()) == 0 and consec <= max(2, int(0.02 * len(single)))
    else:
        led_ok = None
        print(f"  LED: found {len(led_idx)} LED channels, need 2 for alternation check")

    # --- optional camlog ---
    if args.camlog and args.camlog.exists():
        fids, cts = [], []
        for line in args.camlog.read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            p = line.split(",", 1)
            try:
                fids.append(int(p[0]))
                cts.append(_dt.datetime.fromisoformat(p[1]))
            except Exception:
                pass
        if len(cts) > 1:
            cdt = np.array([(cts[i] - cts[i - 1]).total_seconds() * 1000 for i in range(1, len(cts))])
            cgaps = int(np.sum(cdt > args.gap_factor * np.median(cdt)))
            print(f"=== camlog {args.camlog.name} === frames={len(fids)} span={(cts[-1]-cts[0]).total_seconds():.1f}s gaps={cgaps}")

    # --- verdict ---
    print("=== VERDICT ===")
    if len(gap_idx) == 0:
        gating = "NOT GATED - exposure train is continuous (no inter-trial pauses)"
    elif len(ts_t) and abs(n_bursts - len(ts_t)) <= max(1, int(0.2 * len(ts_t))):
        gating = f"GATED - {n_bursts} exposure bursts match {len(ts_t)} trial_starts"
    else:
        gating = f"PARTIAL/UNCLEAR - {n_bursts} bursts vs {len(ts_t)} trial_starts; inspect gaps above"
    print(f"  gating: {gating}")
    if led_ok is not None:
        print(f"  LED alternation: {'clean' if led_ok else 'CHECK - both-on or excessive consecutive-same'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
