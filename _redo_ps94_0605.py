"""Re-process PS94 2026-06-05 with the SIGN-FIXED motion correction.

E: was already cleaned for 6/5, so this rebuilds from the raw on M: standby + the
DAQ/landmarks on N: MICROSCOPE, writing fresh outputs to an E: working dir:
  raw (M:) + DAQ (N:) -> relabel rescue + fixed 2D motion -> SVD -> Allen -> maps.
Cue window matches the original 6/5 batch: 2 s post / 1 s pre. Lick 150 ms + quiet.
Outputs go to E; archive separately (non-bin -> N, bin -> M) afterwards.
"""
import os
import subprocess

PY = r"C:\ProgramData\anaconda3\envs\wfield\python.exe"
REPO = r"C:\Github\Widefield_DAQ_recorder"
SESS = "PS94_20260605_142009"
LAB = "PS94_0605_affine8v1"
TAG = "affine8v1"
RAW = r"M:\Widefield\labcams\20260605\PS94_20260605_142009\raw_widefield_data\pco_edge_run000_00000000_2_460_480_uint16.dat"
DAQ = r"N:\MICROSCOPE\Priya\Widefield\DAQ_recorder_output\20260605\PS94_20260605_142249.h5"
LM = r"N:\MICROSCOPE\Priya\Widefield\labcams\20260605\PS94_20260605_142009\raw_widefield_data\dorsal_cortex_landmarks_v1.json"
MC = r"E:\labcams_data\20260605\PS94_20260605_142009\motion_corrected"
RESULTS = MC + r"\wfield_local_results"
ALLEN = RESULTS + r"\allen_aligned_" + TAG
BIN = MC + r"\motioncorrect_2_460_480_uint16.bin"
FM = MC + r"\pco_edge_run000_00000000_2_460_480_uint16_daq_led_cleanpairs_frame_map.npz"
SUMM = FM.replace("_frame_map.npz", "_summary.json")
CUE_OUT = MC + r"\spout_trial_averages_" + TAG
LICK_OUT = MC + r"\lick_aligned_" + TAG
QUIET_OUT = MC + r"\quiet_" + TAG
QC_OUT = MC + r"\motion_qc"
CUE_NPZ = CUE_OUT + rf"\{LAB}_spout_positions_1s_pre_post_delta_maps.npz"
CUE_SUM = CUE_OUT + rf"\{LAB}_spout_positions_1s_pre_post_delta_summary.json"
LICK_NPZ = LICK_OUT + rf"\{LAB}_lick_aligned_150ms_post_by_spout_maps.npz"
LICK_SUM = LICK_OUT + rf"\{LAB}_lick_aligned_150ms_post_by_spout_summary.json"
QFRAME = QUIET_OUT + rf"\{LAB}_quiet_frame.npy"


def run(cmd):
    env = dict(os.environ, PYTHONPATH=REPO)
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([PY, "-m", *cmd], cwd=REPO, env=env)
    if r.returncode:
        raise SystemExit(f"step failed: {cmd}")


def main():
    os.makedirs(MC, exist_ok=True)
    # 1. relabel (rescue) + sign-fixed 2D motion from the M: raw
    run(["wfield_local.run_wfield_motion", RAW, "--output", MC, "--daq-h5", DAQ,
         "--relabel-mode", "rescue", "--mode", "2d"])
    # 2. SVD + hemo
    run(["wfield_local.run_wfield_local", BIN, "--output", RESULTS, "-k", "100",
         "--functional-channel", "1", "--fs", "31.23", "--freq-highpass", "0.1", "--freq-lowpass", "14.0"])
    # 3. Allen transform
    run(["wfield_local.apply_allen_transform", RESULTS, "--landmarks", LM, "--output", ALLEN])
    # 4. cue (2 s post / 1 s pre) + shared-scale + contrasts
    run(["wfield_local.framemap_event_maps", "--what", "cue", "--daq-h5", DAQ, "--wfield-results", RESULTS,
         "--allen-dir", ALLEN, "--frame-map", FM, "--cleanpairs-summary", SUMM, "--output", CUE_OUT,
         "--label", LAB, "--pre-s", "1.0", "--post-s", "2.0"])
    run(["wfield_local.plot_spout_trial_averages_shared_scale", "--label", LAB, "--trial-maps", CUE_NPZ,
         "--allen-dir", ALLEN, "--output", CUE_OUT, "--summary", CUE_SUM])
    run(["wfield_local.plot_spout_position_contrasts", "--label", LAB, "--trial-maps", CUE_NPZ,
         "--allen-dir", ALLEN, "--output", CUE_OUT])
    # 5. lick 150 ms + contrasts + cue-vs-lick
    run(["wfield_local.framemap_event_maps", "--what", "lick", "--daq-h5", DAQ, "--wfield-results", RESULTS,
         "--allen-dir", ALLEN, "--frame-map", FM, "--cleanpairs-summary", SUMM, "--output", LICK_OUT,
         "--label", LAB, "--post-s", "0.15"])
    run(["wfield_local.plot_lick_position_contrasts", "--label", LAB, "--lick-maps", LICK_NPZ,
         "--allen-dir", ALLEN, "--output", LICK_OUT])
    run(["wfield_local.plot_lick_vs_cue_spout_maps", "--label", LAB, "--cue-maps", CUE_NPZ,
         "--lick-maps", LICK_NPZ, "--allen-dir", ALLEN, "--output", LICK_OUT,
         "--cue-summary", CUE_SUM, "--lick-summary", LICK_SUM])
    # 6. quiet + quiet-normalized lick
    run(["wfield_local.quiet_periods", "--daq-h5", DAQ, "--label", LAB, "--output", QUIET_OUT,
         "--frame-map", FM, "--cleanpairs-summary", SUMM])
    run(["wfield_local.framemap_event_maps", "--what", "lick", "--daq-h5", DAQ, "--wfield-results", RESULTS,
         "--allen-dir", ALLEN, "--frame-map", FM, "--cleanpairs-summary", SUMM, "--output", LICK_OUT,
         "--label", LAB, "--post-s", "0.15", "--quiet-frame", QFRAME])
    # 7. motion QC
    run(["wfield_local.qc_motion_correction", "--motion-dir", MC, "--label", LAB, "--output", QC_OUT])
    print("\nPS94 6/5 REDO DONE", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
