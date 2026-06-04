"""Quiet-period detection for baseline (F0) selection.

Builds a per-sample (and per-corrected-frame) "quiet" mask: times when the animal
is NOT running and NOT licking (and not in a peri-reward window). These frames are a
behavior-controlled baseline for event-triggered ΔF/F — e.g. intersect with the
pre-cue ENL window, or pool as an F0, instead of relying on a true inter-trial quiet
period (which trial-triggered acquisition doesn't record).

Ported from the stroke_orofacial_pipeline (`spout_behavior/bouts.py::find_quiet_bouts`):
    quiet = slow-treadmill  AND  not-near-lick  AND  not-near-reward  [AND not grooming]
with each exclusion widened by a time buffer, then short quiet runs dropped.

Adapted for THIS rig:
- ONE spout, so the stroke pipeline's grooming detector (bilateral two-spout
  conjunction) does not apply. Grooming here would rely only on single-spout
  "long-touch" contact, but with close spouts a TRUE long lick can produce a long
  deflection -> long-touch is an unreliable grooming proxy. So grooming is OFF by
  default; enable with --grooming only as an experiment (see caveat below).
- The quiet exclusions use running (treadmill) + licking (lick_analog) + reward.

!! TUNE LATER: the defaults below (running/quiet speed, min durations, time buffers,
lick thresholds) are starting points carried over from the stroke pipeline. Revisit
"running" speed/duration and the lick/reward/treadmill buffers for this rig and task
once we have ground-truth (e.g. DLC/FaceRhythm movement) to validate against. The
stroke `min_quiet_duration` was 10 s (for long rest bouts); here it defaults small so
the mask is usable as a per-frame baseline cleanliness flag within short ENL windows.

Vendored boolean helpers (idx2bool / widen_bool_sparse / set_short_bool_to_low) are
MIT-licensed ports from bnpm (© 2021 RichieHakim) via the stroke pipeline.

Runs in the wfield CPU env (numpy + scipy + h5py; no torch/GPU).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wfield_local.treadmill import calibrate_treadmill, smooth_treadmill
from wfield_local.lick_detection import detect_licks


# --- vendored boolean helpers (MIT, bnpm © 2021 RichieHakim; via stroke pipeline) ---
def idx2bool(idx: np.ndarray, length: int) -> np.ndarray:
    out = np.zeros(length, dtype=bool)
    idx = np.asarray(idx, dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < length)]
    out[idx] = True
    return out


def widen_bool_sparse(b: np.ndarray, n_before: int, n_after: int) -> np.ndarray:
    b = np.asarray(b).astype(bool); n = b.shape[0]
    out = b.copy()
    for i in np.flatnonzero(b):
        out[max(0, i - n_before):min(n, i + n_after + 1)] = True
    return out


def set_short_bool_to_low(b: np.ndarray, n: int) -> np.ndarray:
    """Zero True runs shorter than n samples (runs of length == n are kept)."""
    b = np.asarray(b).astype(bool).copy()
    if b.size == 0:
        return b
    pad = np.concatenate(([False], b, [False])).astype(np.int8)
    d = np.diff(pad)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)  # exclusive
    for s, e in zip(starts, ends):
        if (e - s) < n:
            b[s:e] = False
    return b


def _rising(sig: np.ndarray, thr: float = 0.5) -> np.ndarray:
    bb = (np.asarray(sig) > thr).astype(np.int8)
    return np.flatnonzero(np.diff(bb) == 1) + 1


def _runs_at_least(contact: np.ndarray, n: int) -> np.ndarray:
    """Keep only True runs of length >= n (the long-touch violations)."""
    return set_short_bool_to_low(contact, n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--daq-h5", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--label", required=True)
    # per-corrected-frame mapping (regime B = cleanpairs frame-map; else regime A raw//2)
    ap.add_argument("--frame-map", type=Path, default=None)
    ap.add_argument("--cleanpairs-summary", type=Path, default=None)
    ap.add_argument("--offset", type=int, default=None)
    # treadmill calibration + running (TUNE LATER)
    ap.add_argument("--tread-channel", default="treadmill")
    ap.add_argument("--offset-v", type=float, default=1.2587643276652853)
    ap.add_argument("--volt-sec-per-rot", type=float, default=0.382)
    ap.add_argument("--mm-per-rot", type=float, default=29.25)
    ap.add_argument("--smoothing-sigma-s", type=float, default=0.15)
    ap.add_argument("--quiet-speed", type=float, default=1.0, help="mm/s; below = 'slow' (TUNE)")
    ap.add_argument("--treadmill-buffer", type=float, nargs=2, default=(3.0, 3.0))
    # licking
    ap.add_argument("--lick-channel", default="lick_analog")
    ap.add_argument("--lick-thresh-upper-v", type=float, default=2.5)
    ap.add_argument("--lick-thresh-lower-v", type=float, default=1.0)
    ap.add_argument("--lockout-s", type=float, nargs=2, default=(0.001, 0.020))
    ap.add_argument("--refractory-s", type=float, default=0.10)
    ap.add_argument("--lick-buffer", type=float, nargs=2, default=(1.0, 3.0))
    # reward
    ap.add_argument("--reward-channel", default="reward_ttl")
    ap.add_argument("--reward-thresh-v", type=float, default=2.5)
    ap.add_argument("--reward-buffer", type=float, nargs=2, default=(0.1, 8.0))
    # grooming (OFF by default; unreliable with one close spout)
    ap.add_argument("--grooming", action="store_true",
                    help="EXPERIMENTAL: exclude single-spout long-touch as grooming. "
                         "Caveat: a true long lick at close spouts also looks long.")
    ap.add_argument("--groom-contact-thresh-v", type=float, default=2.5)
    ap.add_argument("--groom-max-long-s", type=float, default=0.4)
    ap.add_argument("--groom-buffer", type=float, nargs=2, default=(6.0, 6.0))
    # output mask shaping
    ap.add_argument("--min-quiet-s", type=float, default=0.5,
                    help="drop quiet runs shorter than this (stroke rest-bout default was 10 s)")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.daq_h5, "r") as f:
        fs = float(f.attrs["sample_rate_hz"])
        an = [s.decode() for s in f["analog/channel_names"][:]]
        di = [s.decode() for s in f["digital/channel_names"][:]]
        sc = f["analog/int16_scale_volts_per_count"][:]; of = f["analog/int16_offset_volts"][:]

        def ac(name):
            i = an.index(name)
            return f["analog/samples_int16"][:, i].astype(np.float32) * sc[i] + of[i]

        tread_v = ac(args.tread_channel)
        lick_v = ac(args.lick_channel)
        reward_v = ac(args.reward_channel)
        packed = f["digital/packed_samples"][:, 0]
    n = len(packed)
    pco = _rising((packed >> di.index("pco_exposure")) & 1)

    def wid(b, buf):
        return widen_bool_sparse(b, int(buf[0] * fs), int(buf[1] * fs))

    # running / slow treadmill
    speed = smooth_treadmill(calibrate_treadmill(tread_v, args.offset_v, args.volt_sec_per_rot, args.mm_per_rot),
                             fs, args.smoothing_sigma_s)
    slow = speed < args.quiet_speed

    # licks (onsets) + reward
    lick = detect_licks(lick_v, fs, args.lick_thresh_upper_v, args.lick_thresh_lower_v,
                        tuple(args.lockout_s), args.refractory_s)
    lick_onsets = np.asarray(lick["lick_onsets"], dtype=np.int64)
    lick_bool = idx2bool(lick_onsets, n)
    reward_bool = idx2bool(_rising(reward_v, args.reward_thresh_v), n)

    quiet = (~wid(~slow, args.treadmill_buffer)
             & ~wid(lick_bool, args.lick_buffer)
             & ~wid(reward_bool, args.reward_buffer))

    groom_bool = np.zeros(n, dtype=bool)
    if args.grooming:
        contact = lick_v < args.groom_contact_thresh_v
        groom_bool = _runs_at_least(contact, int(args.groom_max_long_s * fs))
        quiet = quiet & ~wid(groom_bool, args.groom_buffer)

    quiet = set_short_bool_to_low(quiet, int(args.min_quiet_s * fs))

    # ---- map to corrected frames ----
    if args.frame_map is not None:
        from wfield_local.framemap_event_maps import _offset_from_summary
        offset = args.offset if args.offset is not None else _offset_from_summary(args.cleanpairs_summary)
        fm = np.load(args.frame_map)
        frame_samples = pco[np.clip(fm["original_frame_index_ch0"] + offset, 0, len(pco) - 1)]
        regime = "B(frame-map)"
    else:
        npairs = len(pco) // 2
        frame_samples = pco[np.arange(npairs) * 2]
        regime = "A(raw//2)"
    quiet_frame = quiet[np.clip(frame_samples, 0, n - 1)]

    np.save(args.output / f"{args.label}_quiet_sample.npy", quiet)
    np.save(args.output / f"{args.label}_quiet_frame.npy", quiet_frame)
    summary = {
        "label": args.label, "daq_h5": str(args.daq_h5), "fs": fs, "n_samples": int(n),
        "regime": regime, "n_frames": int(quiet_frame.size),
        "frac_quiet_sample": float(quiet.mean()), "frac_quiet_frame": float(quiet_frame.mean()),
        "n_licks": int(lick_onsets.size), "grooming_used": bool(args.grooming),
        "params": {k: getattr(args, k) for k in (
            "quiet_speed", "min_quiet_s", "treadmill_buffer", "lick_buffer", "reward_buffer",
            "smoothing_sigma_s", "lick_thresh_upper_v", "lick_thresh_lower_v", "refractory_s",
            "reward_thresh_v", "groom_max_long_s", "groom_buffer")},
        "tune_later": "running/quiet speed, durations, and lick/reward/treadmill buffers are "
                      "starting points (stroke-pipeline defaults); revisit for this rig/task.",
        "grooming_caveat": "single-spout long-touch is an unreliable grooming proxy (a true long "
                           "lick at close spouts also looks long); OFF by default.",
    }
    (args.output / f"{args.label}_quiet_periods_summary.json").write_text(json.dumps(summary, indent=2, default=list))

    # ---- QC: a 120 s window of speed/lick/reward with quiet shaded ----
    t = np.arange(n) / fs
    w = slice(0, min(n, int(120 * fs)))
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t[w], speed[w], lw=0.6, color="0.3", label="speed (mm/s)")
    ax.axhline(args.quiet_speed, color="green", lw=0.6, ls="--")
    for i in lick_onsets[(lick_onsets >= w.start) & (lick_onsets < (w.stop or n))]:
        ax.axvline(t[i], color="orange", lw=0.3, alpha=0.5)
    qw = quiet[w]
    ax.fill_between(t[w], 0, 1, where=qw, transform=ax.get_xaxis_transform(),
                    color="cyan", alpha=0.25, label="quiet")
    ax.set_xlabel("s"); ax.set_ylabel("mm/s"); ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"{args.label} quiet periods (first 120 s) | quiet={quiet.mean()*100:.1f}% samples, "
                 f"{quiet_frame.mean()*100:.1f}% frames | grooming={'on' if args.grooming else 'off'}")
    fig.tight_layout(); fig.savefig(args.output / f"{args.label}_quiet_periods.png", dpi=130); plt.close(fig)

    print(json.dumps(summary, indent=2, default=list), flush=True)
    print(f"[{args.label}] quiet: {quiet.mean()*100:.1f}% samples, {quiet_frame.mean()*100:.1f}% frames ({regime})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
