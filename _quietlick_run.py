"""Run quiet-period detection + quiet-normalized lick-by-position maps for all sessions.

Reads SVD/alignment locally from E: (kept), DAQ h5 from MICROSCOPE (N:, since the DAQ
was archived there and removed from E), frame_maps locally from E: (kept). WRITES all
new outputs to MICROSCOPE (N:\\MICROSCOPE\\Priya\\...). Produces, per session:
  - quiet_affine8v1/<label>_quiet_frame.npy (+ QC)         [quiet_periods.py]
  - lick_aligned_affine8v1/<label>_lick_aligned_150ms_post_by_spout(.png/.npz)  (raw)
    and ..._quietnorm(.png/.npz)  (post-lick minus quiet baseline)
Regime A (6/1 full-FOV): raw//2 mapping. Regime B (6/2,6/3 relabeled): frame-map.
"""
import os, subprocess, sys

PY = r"C:\ProgramData\anaconda3\envs\wfield\python.exe"
REPO = r"C:\Github\Widefield_DAQ_recorder"
E = r"E:\labcams_data"
NLAB = r"N:\MICROSCOPE\Priya\Widefield\labcams"
NDAQ = r"N:\MICROSCOPE\Priya\Widefield\DAQ_recorder_output"

S = {
 "PS94_affine8v1": dict(regime="A", rel=r"20260601\PS94_20260601_141614",
     daq=fr"{NDAQ}\PS94_baseline_20260601_141642.h5"),
 "PS95_affine8v1": dict(regime="A", rel=r"20260601\PS95_20260601_153653",
     daq=fr"{NDAQ}\PS95_baseline_20260601_153627.h5"),
 "PS92_0602_affine8v1": dict(regime="B", rel=r"20260602\PS92_20260602_151820\illuminated_rescue",
     daq=fr"{NDAQ}\20250602\PS92_20260602_152607.h5",
     fm=r"pco_edge_run001_00000000_2_487_480_uint16_daq_led_cleanpairs_frame_map.npz",
     fmdir="self"),  # frame_map lives in the session/illuminated_rescue dir
 "PS92_0603_affine8v1": dict(regime="B", rel=r"20260603\PS92_20260603_104008",
     daq=fr"{NDAQ}\20250603\PS92_20260603_104607.h5",
     fm=r"pco_edge_run000_00000000_2_477_464_uint16_daq_led_cleanpairs_frame_map.npz", fmdir="mc"),
 "PS94_0603_affine8v1": dict(regime="B", rel=r"20260603\PS94_20260603",
     daq=fr"{NDAQ}\20250603\PS94_20260603_175946.h5",
     fm=r"pco_edge_run000_00000000_2_462_464_uint16_daq_led_cleanpairs_frame_map.npz", fmdir="mc"),
 "PS95_0603_affine8v1": dict(regime="B", rel=r"20260603\PS95_20260603_194442",
     daq=fr"{NDAQ}\20250603\PS95_20260603_194902.h5",
     fm=r"pco_edge_run000_00000000_2_462_464_uint16_daq_led_cleanpairs_frame_map.npz", fmdir="mc"),
}


def run(cmd):
    env = dict(os.environ, PYTHONPATH=REPO)
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([PY, "-m", *cmd], cwd=REPO, env=env)
    if r.returncode != 0:
        print(f"!! step failed (exit {r.returncode})", flush=True)
    return r.returncode


def do(label, s):
    mcE = os.path.join(E, s["rel"], "motion_corrected")
    resultsE = os.path.join(mcE, "wfield_local_results")
    allenE = os.path.join(resultsE, "allen_aligned_affine8v1")
    mcN = os.path.join(NLAB, s["rel"], "motion_corrected")
    quietN = os.path.join(mcN, "quiet_affine8v1")
    lickN = os.path.join(mcN, "lick_aligned_affine8v1")
    qframe = os.path.join(quietN, f"{label}_quiet_frame.npy")
    print(f"\n================ {label} (regime {s['regime']}) ================", flush=True)

    qcmd = ["wfield_local.quiet_periods", "--daq-h5", s["daq"], "--label", label, "--output", quietN]
    lcmd = ["wfield_local.plot_lick_aligned_averages" if s["regime"] == "A" else "wfield_local.framemap_event_maps"]
    if s["regime"] == "A":
        lcmd += ["--daq-h5", s["daq"], "--wfield-results", resultsE, "--allen-dir", allenE,
                 "--output", lickN, "--label", label, "--post-s", "0.15", "--fs", "31.23",
                 "--frame-align", "pco", "--quiet-frame", qframe]
    else:
        fmdir = mcE if s["fmdir"] == "mc" else os.path.join(E, s["rel"])
        fm = os.path.join(fmdir, s["fm"]); summ = fm.replace("_frame_map.npz", "_summary.json")
        qcmd += ["--frame-map", fm, "--cleanpairs-summary", summ]
        lcmd += ["--what", "lick", "--daq-h5", s["daq"], "--wfield-results", resultsE, "--allen-dir", allenE,
                 "--frame-map", fm, "--cleanpairs-summary", summ, "--output", lickN, "--label", label,
                 "--post-s", "0.15", "--quiet-frame", qframe]

    if run(qcmd) == 0:
        run(lcmd)


if __name__ == "__main__":
    keys = sys.argv[1:] or list(S)
    for k in keys:
        do(k, S[k])
    print("\nALL DONE:", keys, flush=True)
