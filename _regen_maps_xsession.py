"""Regenerate cue/lick/quiet maps + motion QC for sessions whose motion or alignment
changed (for the cross-session_aligned deck). All in 6/6 CCF (allen_aligned_affine8v1),
cue 2 s post / 2 s pre, lick 150 ms +/- quiet. Reads/writes N: (E: may be cleaned).
Auto-discovers DAQ + frame_map from N. Idempotent-ish (overwrites figures).

Sessions: 6/5 x4 + 6/6 x4 (motion redone) + 6/7/6/8 PS92,PS93 (v2 realigned).
6/7/6/8 PS94,PS95 unchanged -> skipped.
"""
import os, sys, glob, subprocess
PY = r"C:\ProgramData\anaconda3\envs\wfield\python.exe"; REPO = r"C:\Github\Widefield_DAQ_recorder"
NL = r"N:\MICROSCOPE\Priya\Widefield\labcams"; ND = r"N:\MICROSCOPE\Priya\Widefield\DAQ_recorder_output"
TAG = "affine8v1"; CUE_PRE = "2.0"; CUE_POST = "2.0"
SESSIONS = [
  ("20260605","PS92_20260605_125023"),("20260605","PS93_20260605_174659"),
  ("20260605","PS94_20260605_142009"),("20260605","PS95_20260605_163102"),
  ("20260606","PS92_20260606_122451"),("20260606","PS93_20260606_180117"),
  ("20260606","PS94_20260606_140854"),("20260606","PS95_20260606_160806"),
  ("20260607","PS92_20260607_121538"),("20260607","PS93_20260607_174844"),
  ("20260607","PS94_20260607_140731"),("20260607","PS95_20260607_155000"),
  ("20260608","PS92_20260608_133759"),("20260608","PS93_20260608_195203"),
  ("20260608","PS94_20260608_153651"),("20260608","PS95_20260608_180943"),
]
env = dict(os.environ, PYTHONPATH=REPO)
def run(c):
    print("  $ "+" ".join(map(str,c)), flush=True)
    if subprocess.run([PY,"-m",*c], cwd=REPO, env=env).returncode: raise SystemExit(f"fail {c}")
def do(date, sess):
    an = sess[:4]; lab = f"{an}_{date[4:]}_{TAG}"
    mc = fr"{NL}\{date}\{sess}\motion_corrected"; results = fr"{mc}\wfield_local_results"; allen = fr"{results}\{TAG}_dummy"
    allen = fr"{results}\allen_aligned_{TAG}"
    daqs = glob.glob(fr"{ND}\{date}\{an}_{date}_*.h5"); fms = glob.glob(fr"{mc}\*cleanpairs_frame_map.npz")
    if not daqs: print(f"  [{lab}] NO DAQ on N; skip"); return
    if not fms: print(f"  [{lab}] NO frame_map on N; skip"); return
    daq = daqs[0]; fm = fms[0]; summ = fm.replace("_frame_map.npz","_summary.json")
    cue = fr"{mc}\spout_trial_averages_{TAG}"; lick = fr"{mc}\lick_aligned_{TAG}"; quiet = fr"{mc}\quiet_{TAG}"; qc = fr"{mc}\motion_qc"
    cnpz = fr"{cue}\{lab}_spout_positions_1s_pre_post_delta_maps.npz"; csum = cnpz.replace("_maps.npz","_summary.json")
    lnpz = fr"{lick}\{lab}_lick_aligned_150ms_post_by_spout_maps.npz"; lsum = lnpz.replace("_maps.npz","_summary.json")
    qf = fr"{quiet}\{lab}_quiet_frame.npy"
    print(f"\n===== {lab} (cue {CUE_POST}/{CUE_PRE}) =====", flush=True)
    run(["wfield_local.framemap_event_maps","--what","cue","--daq-h5",daq,"--wfield-results",results,"--allen-dir",allen,"--frame-map",fm,"--cleanpairs-summary",summ,"--output",cue,"--label",lab,"--pre-s",CUE_PRE,"--post-s",CUE_POST])
    run(["wfield_local.plot_spout_trial_averages_shared_scale","--label",lab,"--trial-maps",cnpz,"--allen-dir",allen,"--output",cue,"--summary",csum])
    run(["wfield_local.plot_spout_position_contrasts","--label",lab,"--trial-maps",cnpz,"--allen-dir",allen,"--output",cue])
    run(["wfield_local.framemap_event_maps","--what","lick","--daq-h5",daq,"--wfield-results",results,"--allen-dir",allen,"--frame-map",fm,"--cleanpairs-summary",summ,"--output",lick,"--label",lab,"--post-s","0.15"])
    run(["wfield_local.plot_lick_position_contrasts","--label",lab,"--lick-maps",lnpz,"--allen-dir",allen,"--output",lick])
    run(["wfield_local.plot_lick_vs_cue_spout_maps","--label",lab,"--cue-maps",cnpz,"--lick-maps",lnpz,"--allen-dir",allen,"--output",lick,"--cue-summary",csum,"--lick-summary",lsum])
    run(["wfield_local.quiet_periods","--daq-h5",daq,"--label",lab,"--output",quiet,"--frame-map",fm,"--cleanpairs-summary",summ])
    run(["wfield_local.framemap_event_maps","--what","lick","--daq-h5",daq,"--wfield-results",results,"--allen-dir",allen,"--frame-map",fm,"--cleanpairs-summary",summ,"--output",lick,"--label",lab,"--post-s","0.15","--quiet-frame",qf])
    # motion QC may need the corrected bin (not on N for all); non-fatal
    if subprocess.run([PY,"-m","wfield_local.qc_motion_correction","--motion-dir",mc,"--label",lab,"--output",qc], cwd=REPO, env=env).returncode:
        print(f"  [{lab}] motion QC skipped (bin not available from N) -- handle separately", flush=True)
    print(f"===== {lab} DONE =====", flush=True)
if __name__ == "__main__":
    only = sys.argv[1:]
    for date, sess in SESSIONS:
        if only and sess not in only: continue
        do(date, sess)
    print("\nREGEN ALL DONE", flush=True)
