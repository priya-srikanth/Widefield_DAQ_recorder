"""6/7 maps (cue 2s/2s, lick 150ms+quiet) + motion QC, using the 6/6-CCF allen dir
(allen_aligned_xday6). DAQ loose in E:\DAQ_recorder_output. Outputs -> E (deck reads
from E; archive then syncs to N)."""
import os, sys, subprocess
PY=r"C:\ProgramData\anaconda3\envs\wfield\python.exe"; REPO=r"C:\Github\Widefield_DAQ_recorder"
D=r"E:\labcams_data\20260607"; Q=r"E:\DAQ_recorder_output"; TAG="affine8v1"
ALLEN_NAME="allen_aligned_xday6"
FM="pco_edge_run000_00000000_2_460_480_uint16_daq_led_cleanpairs_frame_map.npz"
CUE_PRE,CUE_POST="2.0","2.0"
S={"PS92_0607":dict(sess="PS92_20260607_121538",daq=fr"{Q}\PS92_20260607_121551.h5"),
   "PS93_0607":dict(sess="PS93_20260607_174844",daq=fr"{Q}\PS93_20260607_174854.h5"),
   "PS94_0607":dict(sess="PS94_20260607_140731",daq=fr"{Q}\PS94_20260607_140813.h5"),
   "PS95_0607":dict(sess="PS95_20260607_155000",daq=fr"{Q}\PS95_20260607_155400.h5")}
def run(c):
    env=dict(os.environ,PYTHONPATH=REPO); print("\n$ "+" ".join(map(str,c)),flush=True)
    if subprocess.run([PY,"-m",*c],cwd=REPO,env=env).returncode: raise SystemExit(f"fail {c}")
def do(k):
    s=S[k]; lab=f"{s['sess'][:4]}_0607_{TAG}"; mc=fr"{D}\{s['sess']}\motion_corrected"
    results=fr"{mc}\wfield_local_results"; allen=fr"{results}\{ALLEN_NAME}"
    cue=fr"{mc}\spout_trial_averages_{TAG}"; lick=fr"{mc}\lick_aligned_{TAG}"; quiet=fr"{mc}\quiet_{TAG}"; qc=fr"{mc}\motion_qc"
    fm=fr"{mc}\{FM}"; summ=fm.replace("_frame_map.npz","_summary.json")
    cnpz=fr"{cue}\{lab}_spout_positions_1s_pre_post_delta_maps.npz"; csum=fr"{cue}\{lab}_spout_positions_1s_pre_post_delta_summary.json"
    lnpz=fr"{lick}\{lab}_lick_aligned_150ms_post_by_spout_maps.npz"; lsum=fr"{lick}\{lab}_lick_aligned_150ms_post_by_spout_summary.json"
    qf=fr"{quiet}\{lab}_quiet_frame.npy"
    print(f"\n===== {k} ({lab}) cue {CUE_POST}/{CUE_PRE} via {ALLEN_NAME} =====",flush=True)
    run(["wfield_local.framemap_event_maps","--what","cue","--daq-h5",s["daq"],"--wfield-results",results,"--allen-dir",allen,"--frame-map",fm,"--cleanpairs-summary",summ,"--output",cue,"--label",lab,"--pre-s",CUE_PRE,"--post-s",CUE_POST])
    run(["wfield_local.plot_spout_trial_averages_shared_scale","--label",lab,"--trial-maps",cnpz,"--allen-dir",allen,"--output",cue,"--summary",csum])
    run(["wfield_local.plot_spout_position_contrasts","--label",lab,"--trial-maps",cnpz,"--allen-dir",allen,"--output",cue])
    run(["wfield_local.framemap_event_maps","--what","lick","--daq-h5",s["daq"],"--wfield-results",results,"--allen-dir",allen,"--frame-map",fm,"--cleanpairs-summary",summ,"--output",lick,"--label",lab,"--post-s","0.15"])
    run(["wfield_local.plot_lick_position_contrasts","--label",lab,"--lick-maps",lnpz,"--allen-dir",allen,"--output",lick])
    run(["wfield_local.plot_lick_vs_cue_spout_maps","--label",lab,"--cue-maps",cnpz,"--lick-maps",lnpz,"--allen-dir",allen,"--output",lick,"--cue-summary",csum,"--lick-summary",lsum])
    run(["wfield_local.quiet_periods","--daq-h5",s["daq"],"--label",lab,"--output",quiet,"--frame-map",fm,"--cleanpairs-summary",summ])
    run(["wfield_local.framemap_event_maps","--what","lick","--daq-h5",s["daq"],"--wfield-results",results,"--allen-dir",allen,"--frame-map",fm,"--cleanpairs-summary",summ,"--output",lick,"--label",lab,"--post-s","0.15","--quiet-frame",qf])
    run(["wfield_local.qc_motion_correction","--motion-dir",mc,"--label",lab,"--output",qc])
    print(f"===== {k} DONE =====",flush=True)
if __name__=="__main__":
    for k in (sys.argv[1:] or list(S)): do(k)
    print("\nALL DONE",flush=True)
