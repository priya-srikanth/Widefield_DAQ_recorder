"""Date-parametrized nightly analysis-figure generation for the spout-position summary deck.

  python -m wfield_local.nightly_figs <MMDD> [--from <comma MMDD>] [--output <dir>]
    e.g. python -m wfield_local.nightly_figs 0807
         python -m wfield_local.nightly_figs 0807 --from 0605,0606,0607,0608

Generates (into OUT): per-day decode (lick/cue 2 s, pre-cue 1 s), rolling cue + first-lick temporal
dynamics + laterality, top components, encoder (+ FEVE raw & normalized), and the cross-mouse /
within-animal consistency / RSA (incl. crossnobis) comparison. The per-day figures use <MMDD>; the
cross-SESSION comparisons span ALL registered sessions by default (dates read from SESSIONS, MMDD
zero-padded so string-sort == chronological), tag `<first>-<MMDD>`. Override the span with --from.

Canonical orchestrator for the nightly pipeline (see LOCANMF_NIGHTLY_PIPELINE.md). Runs the individual
locanmf_* modules as subprocesses so each gets a clean process; the repo root is derived from this file's
location, so it is portable across checkouts.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from wfield_local.locanmf_cue_lick_analysis import SESSIONS

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
DEFAULT_OUT = "C:/Users/sabatini/source/cue_lick"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def cli(*a):
    log("CLI " + " ".join(a))
    if subprocess.call([PY, "-u", "-m", *a], cwd=str(REPO)):
        log("  !! nonzero exit: " + a[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("date", nargs="?", default="0807", help="per-day figure date, MMDD")
    ap.add_argument("--from", dest="from_dates", default=None,
                    help="comma MMDD list for the cross-session comparisons (default: ALL registered sessions)")
    ap.add_argument("--output", default=DEFAULT_OUT, help="figure output dir (also where the deck is built)")
    args = ap.parse_args()

    date = args.date
    out = args.output
    all_dates = ",".join(sorted({s["label"][-4:] for s in SESSIONS}))
    from_dates = args.from_dates or all_dates
    tag = f"{from_dates.split(',')[0]}-{from_dates.split(',')[-1]}"

    log(f"date={date} cross-session dates={from_dates} tag={tag} out={out}")
    for al, ps in (("lick", "2.0"), ("cue", "2.0"), ("precue", "1.0")):
        cli("wfield_local.locanmf_position_decoder", "--date", date, "--align", al, "--post-s", ps, "--output", out)

    from wfield_local.locanmf_decoder_weights import (_avail, fig_rolling_cue, fig_temporal_dynamics,
                                                      fig_rolling_laterality, fig_top_components)
    labs = _avail(date); log(f"{date} sessions: {labs}")
    for desc, fn in (("rolling_cue", lambda: fig_rolling_cue(labs, Path(out), date)),
                     ("temporal_dynamics", lambda: fig_temporal_dynamics(labs, Path(out))),
                     ("rolling_laterality", lambda: fig_rolling_laterality(labs, Path(out), date))):
        try:
            log("wrote " + fn().name)
        except Exception as ex:
            log(f"  !! {desc}: {type(ex).__name__} {str(ex)[:70]}")
    for lab in labs:
        try:
            log("wrote " + fig_top_components(lab, Path(out)).name)
        except Exception as ex:
            log(f"  !! top_components {lab}: {str(ex)[:60]}")

    cli("wfield_local.locanmf_position_encoder", "--date", date, "--pool-dates", from_dates, "--output", out)
    cli("wfield_local.locanmf_cross_mouse", "--output", out, "--dates", from_dates, "--tag", tag)
    cli("wfield_local.locanmf_rsa", "--output", out, "--dates", from_dates, "--tag", tag)
    log(f"== nightly {date} figures complete (tag {tag}) ==")


if __name__ == "__main__":
    main()
