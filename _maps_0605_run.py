"""Driver: cue/lick/quiet-norm maps + motion QC for the four 2026-06-05 sessions.

Cue window per request: 2 s POST-cue minus 1 s PRE-cue (prior batches used 1s/1s).
Lick: 150 ms post + quiet-period-normalized. All regime B (every session was
relabeled to cleanpairs, so frame-map event mapping is used for all four).
Outputs -> E:\\...\\motion_corrected\\{spout_trial_averages,lick_aligned,quiet,motion_qc}_affine8v1
(the deck builder reads from E; archive_day then syncs to N).
Usage:  python _maps_0605_run.py [PS92_0605 PS93_0605 PS94_0605 PS95_0605]
"""
import os
import sys
import subprocess

PY = r"C:\ProgramData\anaconda3\envs\wfield\python.exe"
REPO = r"C:\Github\Widefield_DAQ_recorder"
D = r"E:\labcams_data\20260605"
Q = r"E:\DAQ_recorder_output\20260605"
TAG = "affine8v1"
FM = "pco_edge_run000_00000000_2_460_480_uint16_daq_led_cleanpairs_frame_map.npz"
CUE_PRE, CUE_POST = "1.0", "2.0"   # 1 s pre-cue, 2 s post-cue

SESSIONS = {
    "PS92_0605": dict(sess="PS92_20260605_125023", daq=fr"{Q}\PS92_20260605_125301.h5"),
    "PS93_0605": dict(sess="PS93_20260605_174659", daq=fr"{Q}\PS93_20260605_175452.h5"),
    "PS94_0605": dict(sess="PS94_20260605_142009", daq=fr"{Q}\PS94_20260605_142249.h5"),
    "PS95_0605": dict(sess="PS95_20260605_163102", daq=fr"{Q}\PS95_20260605_163405.h5"),
}


def run(cmd):
    env = dict(os.environ, PYTHONPATH=REPO)
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([PY, "-m", *cmd], cwd=REPO, env=env)
    if r.returncode != 0:
        raise SystemExit(f"step failed (exit {r.returncode}): {cmd}")


def do_session(key):
    s = SESSIONS[key]
    lab = f"{s['sess'][:4]}_0605_{TAG}"
    mc = fr"{D}\{s['sess']}\motion_corrected"
    results = fr"{mc}\wfield_local_results"
    allen = fr"{results}\allen_aligned_{TAG}"
    cue_out = fr"{mc}\spout_trial_averages_{TAG}"
    lick_out = fr"{mc}\lick_aligned_{TAG}"
    quiet_out = fr"{mc}\quiet_{TAG}"
    qc_out = fr"{mc}\motion_qc"
    fm = fr"{mc}\{FM}"
    summ = fm.replace("_frame_map.npz", "_summary.json")
    cue_npz = fr"{cue_out}\{lab}_spout_positions_1s_pre_post_delta_maps.npz"
    lick_npz = fr"{lick_out}\{lab}_lick_aligned_150ms_post_by_spout_maps.npz"
    cue_sum = fr"{cue_out}\{lab}_spout_positions_1s_pre_post_delta_summary.json"
    lick_sum = fr"{lick_out}\{lab}_lick_aligned_150ms_post_by_spout_summary.json"
    qframe = fr"{quiet_out}\{lab}_quiet_frame.npy"
    print(f"\n================ {key} ({lab})  cue {CUE_POST}s post / {CUE_PRE}s pre ================", flush=True)

    # 1. cue maps (2 s post, 1 s pre)
    run(["wfield_local.framemap_event_maps", "--what", "cue", "--daq-h5", s["daq"],
         "--wfield-results", results, "--allen-dir", allen, "--frame-map", fm,
         "--cleanpairs-summary", summ, "--output", cue_out, "--label", lab,
         "--pre-s", CUE_PRE, "--post-s", CUE_POST])
    # 2. cue delta SHARED-SCALE (labels read window from summary)
    run(["wfield_local.plot_spout_trial_averages_shared_scale", "--label", lab,
         "--trial-maps", cue_npz, "--allen-dir", allen, "--output", cue_out, "--summary", cue_sum])
    # 2b. mean overlay + pairwise cue contrasts
    run(["wfield_local.plot_spout_position_contrasts", "--label", lab,
         "--trial-maps", cue_npz, "--allen-dir", allen, "--output", cue_out])
    # 3. lick maps (150 ms post)
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
    # 8. motion-correction QC
    run(["wfield_local.qc_motion_correction", "--motion-dir", mc, "--label", lab, "--output", qc_out])
    print(f"================ {key} DONE ================", flush=True)


if __name__ == "__main__":
    keys = sys.argv[1:] or list(SESSIONS)
    for k in keys:
        do_session(k)
    print("\nALL DONE:", keys, flush=True)
