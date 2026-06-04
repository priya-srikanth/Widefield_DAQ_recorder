"""Driver: re-align (8-pt affine, landmarks v1) + cue/lick/delta maps per session.

For each session, in separate subprocesses (so apply_allen_transform's wfield
import never shares a process with the h5py map steps):
  1. apply_allen_transform  -> allen_aligned_affine8v1/ (U_atlas, frames_average_atlas, atlas)
  2. cue maps (pre/post/delta) -> spout_trial_averages_affine8v1/
       regime A: plot_spout_trial_averages ; regime B: framemap_event_maps --what cue
  3. mean+Allen overlay + cue pairwise -> plot_spout_position_contrasts
  4. lick maps (150 ms post) -> lick_aligned_affine8v1/
       regime A: plot_lick_aligned_averages ; regime B: framemap_event_maps --what lick
  5. delta-position lick maps (pairwise) -> plot_lick_position_contrasts
  6. cue-vs-lick -> plot_lick_vs_cue_spout_maps

Nothing existing is overwritten: outputs go to *_affine8v1 dirs. Usage:
  python _affine8v1_run.py [session ...]   (default: the 4 SVD-ready sessions)
"""
import os, sys, subprocess

PY = r"C:\ProgramData\anaconda3\envs\wfield\python.exe"
REPO = r"C:\Github\Widefield_DAQ_recorder"
TAG = "affine8v1"

D = r"E:\labcams_data"
Q = r"E:\DAQ_recorder_output"

SESSIONS = {
    "PS94_0601": dict(
        animal="PS94", label="PS94_" + TAG, regime="A",
        results=fr"{D}\20260601\PS94_20260601_141614\motion_corrected\wfield_local_results",
        landmarks=fr"{D}\20260601\PS94_20260601_141614\dorsal_cortex_landmarks_v1.json",
        daq=fr"{Q}\PS94_baseline_20260601_141642.h5",
    ),
    "PS95_0601": dict(
        animal="PS95", label="PS95_" + TAG, regime="A",
        results=fr"{D}\20260601\PS95_20260601_153653\motion_corrected\wfield_local_results",
        landmarks=fr"{D}\20260601\PS95_20260601_153653\dorsal_cortex_landmarks_v1.json",
        daq=fr"{Q}\PS95_baseline_20260601_153627.h5",
    ),
    "PS92_0602": dict(
        animal="PS92", label="PS92_0602_" + TAG, regime="B",
        results=fr"{D}\20260602\PS92\PS92_20260602_151820\illuminated_rescue\motion_corrected\wfield_local_results",
        landmarks=fr"{D}\20260602\PS92\PS92_20260602_151820\raw_widefield_data\dorsal_cortex_landmarks_v1.json",
        daq=fr"{Q}\20250602\PS92_20260602_152607.h5",
        frame_map=fr"{D}\20260602\PS92\PS92_20260602_151820\illuminated_rescue\pco_edge_run001_00000000_2_487_480_uint16_daq_led_cleanpairs_frame_map.npz",
        summary=fr"{D}\20260602\PS92\PS92_20260602_151820\illuminated_rescue\pco_edge_run001_00000000_2_487_480_uint16_daq_led_cleanpairs_summary.json",
    ),
    "PS92_0603": dict(
        animal="PS92", label="PS92_0603_" + TAG, regime="B",
        results=fr"{D}\20260603\PS92\PS92_20260603_104008\motion_corrected\wfield_local_results",
        landmarks=fr"{D}\20260603\PS92\PS92_20260603_104008\raw_widefield_data\dorsal_cortex_landmarks_v1.json",
        daq=fr"{Q}\20250603\PS92_20260603_104607.h5",
        frame_map=fr"{D}\20260603\PS92\PS92_20260603_104008\motion_corrected\pco_edge_run000_00000000_2_477_464_uint16_daq_led_cleanpairs_frame_map.npz",
        summary=fr"{D}\20260603\PS92\PS92_20260603_104008\motion_corrected\pco_edge_run000_00000000_2_477_464_uint16_daq_led_cleanpairs_summary.json",
    ),
    # ---- pending SVD; filled in for the later pass ----
    "PS94_0603": dict(
        animal="PS94", label="PS94_0603_" + TAG, regime="B",
        results=fr"{D}\20260603\PS94\motion_corrected\wfield_local_results",
        landmarks=fr"{D}\20260603\PS94\raw_widefield_data\dorsal_cortex_landmarks_v1.json",
        daq=fr"{Q}\20250603\PS94_20260603_175946.h5",
        frame_map=fr"{D}\20260603\PS94\motion_corrected\pco_edge_run000_00000000_2_462_464_uint16_daq_led_cleanpairs_frame_map.npz",
        summary=fr"{D}\20260603\PS94\motion_corrected\pco_edge_run000_00000000_2_462_464_uint16_daq_led_cleanpairs_summary.json",
    ),
    "PS95_0603": dict(
        animal="PS95", label="PS95_0603_" + TAG, regime="B",
        results=fr"{D}\20260603\PS95\PS95\PS95_20260603_194442\motion_corrected\wfield_local_results",
        landmarks=fr"{D}\20260603\PS95\PS95\PS95_20260603_194442\raw_widefield_data\dorsal_cortex_landmarks_v1.json",
        daq=fr"{Q}\20250603\PS95_20260603_194902.h5",
        frame_map=fr"{D}\20260603\PS95\PS95\PS95_20260603_194442\motion_corrected\pco_edge_run000_00000000_2_462_464_uint16_daq_led_cleanpairs_frame_map.npz",
        summary=fr"{D}\20260603\PS95\PS95\PS95_20260603_194442\motion_corrected\pco_edge_run000_00000000_2_462_464_uint16_daq_led_cleanpairs_summary.json",
    ),
}

READY = ["PS94_0601", "PS95_0601", "PS92_0602", "PS92_0603"]


def run(cmd):
    env = dict(os.environ, PYTHONPATH=REPO)
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([PY, "-m", *cmd], cwd=REPO, env=env)
    if r.returncode != 0:
        raise SystemExit(f"step failed (exit {r.returncode}): {cmd}")


def do_session(key):
    s = SESSIONS[key]
    mc = os.path.dirname(s["results"])
    allen = os.path.join(s["results"], f"allen_aligned_{TAG}")
    cue_out = os.path.join(mc, f"spout_trial_averages_{TAG}")
    lick_out = os.path.join(mc, f"lick_aligned_{TAG}")
    lab = s["label"]
    cue_npz = os.path.join(cue_out, f"{lab}_spout_positions_1s_pre_post_delta_maps.npz")
    lick_npz = os.path.join(lick_out, f"{lab}_lick_aligned_150ms_post_by_spout_maps.npz")
    cue_sum = os.path.join(cue_out, f"{lab}_spout_positions_1s_pre_post_delta_summary.json")
    lick_sum = os.path.join(lick_out, f"{lab}_lick_aligned_150ms_post_by_spout_summary.json")
    print(f"\n================ {key}  ({s['animal']}, regime {s['regime']}) ================", flush=True)

    # 1. alignment
    run(["wfield_local.apply_allen_transform", s["results"], "--landmarks", s["landmarks"], "--output", allen])

    # 2. cue maps
    if s["regime"] == "A":
        run(["wfield_local.plot_spout_trial_averages", "--daq-h5", s["daq"], "--wfield-results", s["results"],
             "--allen-dir", allen, "--output", cue_out, "--label", lab, "--pre-s", "1.0", "--post-s", "1.0", "--fs", "31.23"])
    else:
        run(["wfield_local.framemap_event_maps", "--what", "cue", "--daq-h5", s["daq"], "--wfield-results", s["results"],
             "--allen-dir", allen, "--frame-map", s["frame_map"], "--cleanpairs-summary", s["summary"],
             "--output", cue_out, "--label", lab])

    # 3. mean overlay + cue pairwise
    run(["wfield_local.plot_spout_position_contrasts", "--label", lab, "--trial-maps", cue_npz,
         "--allen-dir", allen, "--output", cue_out])

    # 4. lick maps
    if s["regime"] == "A":
        run(["wfield_local.plot_lick_aligned_averages", "--daq-h5", s["daq"], "--wfield-results", s["results"],
             "--allen-dir", allen, "--output", lick_out, "--label", lab, "--post-s", "0.15", "--fs", "31.23"])
    else:
        run(["wfield_local.framemap_event_maps", "--what", "lick", "--daq-h5", s["daq"], "--wfield-results", s["results"],
             "--allen-dir", allen, "--frame-map", s["frame_map"], "--cleanpairs-summary", s["summary"],
             "--output", lick_out, "--label", lab, "--post-s", "0.15"])

    # 5. delta-position lick maps (pairwise)
    run(["wfield_local.plot_lick_position_contrasts", "--label", lab, "--lick-maps", lick_npz,
         "--allen-dir", allen, "--output", lick_out])

    # 6. cue vs lick
    run(["wfield_local.plot_lick_vs_cue_spout_maps", "--label", lab, "--cue-maps", cue_npz, "--lick-maps", lick_npz,
         "--allen-dir", allen, "--output", lick_out, "--cue-summary", cue_sum, "--lick-summary", lick_sum])

    print(f"================ {key} DONE ================", flush=True)


if __name__ == "__main__":
    keys = sys.argv[1:] or READY
    for k in keys:
        do_session(k)
    print("\nALL DONE:", keys, flush=True)
