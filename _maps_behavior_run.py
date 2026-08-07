"""Re-run cue/lick/quiet maps with TRUE spout positions recovered from behavior trials.csv
(when a DAQ strobe bit is dead). Reads/writes N:. Usage: python _maps_behavior_run.py 20260805 [PS92_...]"""
import os, sys, glob, subprocess
PY=r"C:\ProgramData\anaconda3\envs\wfield\python.exe"; REPO=r"C:\Github\Widefield_DAQ_recorder"
NL=r"N:\MICROSCOPE\Priya\Widefield\labcams"; NDQ=r"N:\MICROSCOPE\Priya\Widefield\DAQ_recorder_output"
BL=r"N:\MICROSCOPE\Priya\Behavior_logs\Widefield"; TAG="affine8v1"; CP,CO="2.0","2.0"
SESS={
 "20260805":{"PS92_20260805_181150":("PS92_20260805_182111.h5","PS92_20260805_182116"),
             "PS93_20260805_201110":("PS93_20260805_202005.h5","PS93_20260805_202013"),
             "PS94_20260805_124758":("PS94_20260805_131025.h5","PS94_20260805_131030"),
             "PS95_20260805_155615":("PS95_20260805_160437.h5","PS95_20260805_160444")},
 "20260806":{"PS92_20260806_124426":("PS92_20260806_124733.h5","PS92_20260806_124753"),
             "PS93_20260806_172253":("PS93_20260806_172316.h5","PS93_20260806_172322"),
             "PS94_20260806_074320":("PS94_20260806_074558.h5","PS94_20260806_074607"),
             "PS95_20260806_105300":("PS95_20260806_105651.h5","PS95_20260806_105657")}}
def run(c):
    env=dict(os.environ,PYTHONPATH=REPO); print("$ "+" ".join(map(str,c[:6]))+" ...",flush=True)
    if subprocess.run([PY,"-m",*c],cwd=REPO,env=env).returncode: raise SystemExit(f"fail {c[1]}")
def do(date,sess):
    daqf,logdir=SESS[date][sess]; an=sess[:4]; lab=f"{an}_{date[4:]}_{TAG}"
    mc=fr"{NL}\{date}\{sess}\motion_corrected"; res=fr"{mc}\wfield_local_results"; allen=fr"{res}\allen_aligned_{TAG}"
    daq=fr"{NDQ}\{date}\{daqf}"; bt=fr"{BL}\{logdir}\trials.csv"
    fm=glob.glob(fr"{mc}\*cleanpairs_frame_map.npz")[0]; summ=fm.replace("_frame_map.npz","_summary.json")
    cue=fr"{mc}\spout_trial_averages_{TAG}"; lick=fr"{mc}\lick_aligned_{TAG}"; quiet=fr"{mc}\quiet_{TAG}"
    cnpz=fr"{cue}\{lab}_spout_positions_1s_pre_post_delta_maps.npz"; csum=cnpz.replace("_maps.npz","_summary.json")
    lnpz=fr"{lick}\{lab}_lick_aligned_150ms_post_by_spout_maps.npz"; lsum=lnpz.replace("_maps.npz","_summary.json"); qf=fr"{quiet}\{lab}_quiet_frame.npy"
    print(f"\n===== {lab} (behavior-recovered positions) =====",flush=True)
    B=["--behavior-trials",bt]
    run(["wfield_local.framemap_event_maps","--what","cue","--daq-h5",daq,"--wfield-results",res,"--allen-dir",allen,"--frame-map",fm,"--cleanpairs-summary",summ,"--output",cue,"--label",lab,"--pre-s",CP,"--post-s",CO]+B)
    run(["wfield_local.plot_spout_trial_averages_shared_scale","--label",lab,"--trial-maps",cnpz,"--allen-dir",allen,"--output",cue,"--summary",csum])
    run(["wfield_local.plot_spout_position_contrasts","--label",lab,"--trial-maps",cnpz,"--allen-dir",allen,"--output",cue])
    run(["wfield_local.framemap_event_maps","--what","lick","--daq-h5",daq,"--wfield-results",res,"--allen-dir",allen,"--frame-map",fm,"--cleanpairs-summary",summ,"--output",lick,"--label",lab,"--post-s","0.15"]+B)
    run(["wfield_local.plot_lick_position_contrasts","--label",lab,"--lick-maps",lnpz,"--allen-dir",allen,"--output",lick])
    run(["wfield_local.plot_lick_vs_cue_spout_maps","--label",lab,"--cue-maps",cnpz,"--lick-maps",lnpz,"--allen-dir",allen,"--output",lick,"--cue-summary",csum,"--lick-summary",lsum])
    run(["wfield_local.framemap_event_maps","--what","lick","--daq-h5",daq,"--wfield-results",res,"--allen-dir",allen,"--frame-map",fm,"--cleanpairs-summary",summ,"--output",lick,"--label",lab,"--post-s","0.15","--quiet-frame",qf]+B)
    print(f"===== {lab} DONE =====",flush=True)
if __name__=="__main__":
    date=sys.argv[1]; only=sys.argv[2:]; skipped=[]
    for sess in (only or list(SESS[date])):
        try: do(date,sess)
        except SystemExit as e: skipped.append((sess,str(e))); print(f"  SKIP {sess}: {e}",flush=True)
    print(f"\nBEHAVIOR MAPS DONE  (skipped: {skipped})",flush=True)
