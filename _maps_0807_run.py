"""8/7 maps (cue 2s/2s, lick 150ms + quiet-normalized) + motion QC, using the 6/6-CCF allen
dir (allen_aligned_affine8v1). Reads/writes N: (orchestrator already pushed results+allen+
frame_map+DAQ there). Strobe bit1 fixed -> all 6 positions native (no --behavior-trials)."""
import os, sys, glob, subprocess
PY=r"C:\ProgramData\anaconda3\envs\wfield\python.exe"; REPO=r"C:\Github\Widefield_DAQ_recorder"
NL=r"N:\MICROSCOPE\Priya\Widefield\labcams\20260807"; NDQ=r"N:\MICROSCOPE\Priya\Widefield\DAQ_recorder_output\20260807"
TAG="affine8v1"; ALLEN_NAME="allen_aligned_affine8v1"; CUE_PRE,CUE_POST="2.0","2.0"
S={"PS92_0807":dict(sess="PS92_20260807_150924",daq="PS92_20260807_151146.h5"),
   "PS93_0807":dict(sess="PS93_20260807_174403",daq="PS93_20260807_174416.h5"),
   "PS94_0807":dict(sess="PS94_20260807_104125",daq="PS94_20260807_105410.h5"),
   "PS95_0807":dict(sess="PS95_20260807_124637",daq="PS95_20260807_125106.h5")}
def run(c):
    env=dict(os.environ,PYTHONPATH=REPO); print("\n$ "+" ".join(map(str,c[:6]))+" ...",flush=True)
    if subprocess.run([PY,"-m",*c],cwd=REPO,env=env).returncode: raise SystemExit(f"fail {c[1]}")
def do(k):
    s=S[k]; lab=f"{s['sess'][:4]}_0807_{TAG}"; mc=fr"{NL}\{s['sess']}\motion_corrected"
    results=fr"{mc}\wfield_local_results"; allen=fr"{results}\{ALLEN_NAME}"; daq=fr"{NDQ}\{s['daq']}"
    cue=fr"{mc}\spout_trial_averages_{TAG}"; lick=fr"{mc}\lick_aligned_{TAG}"; quiet=fr"{mc}\quiet_{TAG}"; qc=fr"{mc}\motion_qc"
    fm=glob.glob(fr"{mc}\*cleanpairs_frame_map.npz")[0]; summ=fm.replace("_frame_map.npz","_summary.json")
    cnpz=fr"{cue}\{lab}_spout_positions_1s_pre_post_delta_maps.npz"; csum=fr"{cue}\{lab}_spout_positions_1s_pre_post_delta_summary.json"
    lnpz=fr"{lick}\{lab}_lick_aligned_150ms_post_by_spout_maps.npz"; lsum=fr"{lick}\{lab}_lick_aligned_150ms_post_by_spout_summary.json"
    qf=fr"{quiet}\{lab}_quiet_frame.npy"
    print(f"\n===== {k} ({lab}) cue {CUE_POST}/{CUE_PRE} via {ALLEN_NAME} =====",flush=True)
    run(["wfield_local.framemap_event_maps","--what","cue","--daq-h5",daq,"--wfield-results",results,"--allen-dir",allen,"--frame-map",fm,"--cleanpairs-summary",summ,"--output",cue,"--label",lab,"--pre-s",CUE_PRE,"--post-s",CUE_POST])
    run(["wfield_local.plot_spout_trial_averages_shared_scale","--label",lab,"--trial-maps",cnpz,"--allen-dir",allen,"--output",cue,"--summary",csum])
    run(["wfield_local.plot_spout_position_contrasts","--label",lab,"--trial-maps",cnpz,"--allen-dir",allen,"--output",cue])
    run(["wfield_local.framemap_event_maps","--what","lick","--daq-h5",daq,"--wfield-results",results,"--allen-dir",allen,"--frame-map",fm,"--cleanpairs-summary",summ,"--output",lick,"--label",lab,"--post-s","0.15"])
    run(["wfield_local.plot_lick_position_contrasts","--label",lab,"--lick-maps",lnpz,"--allen-dir",allen,"--output",lick])
    run(["wfield_local.plot_lick_vs_cue_spout_maps","--label",lab,"--cue-maps",cnpz,"--lick-maps",lnpz,"--allen-dir",allen,"--output",lick,"--cue-summary",csum,"--lick-summary",lsum])
    run(["wfield_local.quiet_periods","--daq-h5",daq,"--label",lab,"--output",quiet,"--frame-map",fm,"--cleanpairs-summary",summ])
    run(["wfield_local.framemap_event_maps","--what","lick","--daq-h5",daq,"--wfield-results",results,"--allen-dir",allen,"--frame-map",fm,"--cleanpairs-summary",summ,"--output",lick,"--label",lab,"--post-s","0.15","--quiet-frame",qf])
    run(["wfield_local.qc_motion_correction","--motion-dir",mc,"--label",lab,"--output",qc])
    print(f"===== {k} DONE =====",flush=True)
if __name__=="__main__":
    for k in (sys.argv[1:] or list(S)): do(k)
    print("\nALL MAPS+QC DONE",flush=True)
