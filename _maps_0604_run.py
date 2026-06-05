"""Driver: cue/lick/quiet-norm maps for the three 2026-06-04 sessions (regime B).

Allen transform already done (_allen_0604_run.py), so this starts from the maps:
  1. cue maps (pre/post/delta)             framemap_event_maps --what cue
  2. cue delta SHARED-SCALE + Allen overlay + pairwise   plot_spout_position_contrasts
  3. lick maps (150 ms post) by spout       framemap_event_maps --what lick
  4. delta-position lick contrasts          plot_lick_position_contrasts
  5. cue-vs-lick                            plot_lick_vs_cue_spout_maps
  6. quiet periods -> quiet_frame.npy       quiet_periods
  7. quiet-normalized lick maps             framemap_event_maps --what lick --quiet-frame

Outputs -> E:\\...\\motion_corrected\\{spout_trial_averages,lick_aligned,quiet}_affine8v1
(the deck builder reads from E; the idempotent archival pass then syncs to N).
PS92 uses the CONCATENATED DAQ. Note: the concat's 26.8 s zero-padded gap has no
corrected frames, so it cannot contribute quiet frames (per-frame baseline unaffected).
Usage:  python _maps_0604_run.py [PS92_0604 PS94_0604 PS95_0604]
"""
import os
import sys
import subprocess

PY = r"C:\ProgramData\anaconda3\envs\wfield\python.exe"
REPO = r"C:\Github\Widefield_DAQ_recorder"
D = r"E:\labcams_data\20260604"
Q = r"E:\DAQ_recorder_output\20250604"
TAG = "affine8v1"
FM = "pco_edge_run000_00000000_2_460_480_uint16_daq_led_cleanpairs_frame_map.npz"

SESSIONS = {
    "PS92_0604": dict(sess="PS92_20260604_132934", daq=fr"{Q}\PS92_20260604_concat.h5"),
    "PS94_0604": dict(sess="PS94_20260604_151516", daq=fr"{Q}\PS94_20260604_152103.h5"),
    "PS95_0604": dict(sess="PS95_20260604_165712", daq=fr"{Q}\PS95_20260604_170729.h5"),
}


def run(cmd):
    env = dict(os.environ, PYTHONPATH=REPO)
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([PY, "-m", *cmd], cwd=REPO, env=env)
    if r.returncode != 0:
        raise SystemExit(f"step failed (exit {r.returncode}): {cmd}")


def do_session(key):
    s = SESSIONS[key]
    lab = f"{s['sess'][:4]}_0604_{TAG}"          # e.g. PS92_0604_affine8v1
    mc = fr"{D}\{s['sess']}\motion_corrected"
    results = fr"{mc}\wfield_local_results"
    allen = fr"{results}\allen_aligned_{TAG}"
    cue_out = fr"{mc}\spout_trial_averages_{TAG}"
    lick_out = fr"{mc}\lick_aligned_{TAG}"
    quiet_out = fr"{mc}\quiet_{TAG}"
    fm = fr"{mc}\{FM}"
    summ = fm.replace("_frame_map.npz", "_summary.json")
    cue_npz = fr"{cue_out}\{lab}_spout_positions_1s_pre_post_delta_maps.npz"
    lick_npz = fr"{lick_out}\{lab}_lick_aligned_150ms_post_by_spout_maps.npz"
    cue_sum = fr"{cue_out}\{lab}_spout_positions_1s_pre_post_delta_summary.json"
    lick_sum = fr"{lick_out}\{lab}_lick_aligned_150ms_post_by_spout_summary.json"
    qframe = fr"{quiet_out}\{lab}_quiet_frame.npy"
    print(f"\n================ {key} ({lab}) ================", flush=True)

    # 1. cue maps
    run(["wfield_local.framemap_event_maps", "--what", "cue", "--daq-h5", s["daq"],
         "--wfield-results", results, "--allen-dir", allen, "--frame-map", fm,
         "--cleanpairs-summary", summ, "--output", cue_out, "--label", lab])
    # 2. cue delta SHARED-SCALE (cue-aligned minus pre-cue, one color scale)
    run(["wfield_local.plot_spout_trial_averages_shared_scale", "--label", lab,
         "--trial-maps", cue_npz, "--allen-dir", allen, "--output", cue_out, "--summary", cue_sum])
    # 2b. mean overlay + pairwise cue contrasts
    run(["wfield_local.plot_spout_position_contrasts", "--label", lab,
         "--trial-maps", cue_npz, "--allen-dir", allen, "--output", cue_out])
    # 3. lick maps
    run(["wfield_local.framemap_event_maps", "--what", "lick", "--daq-h5", s["daq"],
         "--wfield-results", results, "--allen-dir", allen, "--frame-map", fm,
         "--cleanpairs-summary", summ, "--output", lick_out, "--label", lab, "--post-s", "0.15"])
    # 4. delta-position lick contrasts
    run(["wfield_local.plot_lick_position_contrasts", "--label", lab,
         "--lick-maps", lick_npz, "--allen-dir", allen, "--output", lick_out])
    # 5. cue vs lick
    run(["wfield_local.plot_lick_vs_cue_spout_maps", "--label", lab, "--cue-maps", cue_npz,
         "--lick-maps", lick_npz, "--allen-dir", allen, "--output", lick_out,
         "--cue-summary", cue_sum, "--lick-summary", lick_sum])
    # 6. quiet periods
    run(["wfield_local.quiet_periods", "--daq-h5", s["daq"], "--label", lab,
         "--output", quiet_out, "--frame-map", fm, "--cleanpairs-summary", summ])
    # 7. quiet-normalized lick maps
    run(["wfield_local.framemap_event_maps", "--what", "lick", "--daq-h5", s["daq"],
         "--wfield-results", results, "--allen-dir", allen, "--frame-map", fm,
         "--cleanpairs-summary", summ, "--output", lick_out, "--label", lab,
         "--post-s", "0.15", "--quiet-frame", qframe])
    print(f"================ {key} DONE ================", flush=True)


if __name__ == "__main__":
    keys = sys.argv[1:] or list(SESSIONS)
    for k in keys:
        do_session(k)
    print("\nALL DONE:", keys, flush=True)
