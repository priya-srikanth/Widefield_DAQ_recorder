"""quiet_periods + quiet-norm lick (behavior-recovered) for 8/6 (the step missed on first pass)."""
import os, glob, subprocess
PY=r"C:\ProgramData\anaconda3\envs\wfield\python.exe"; REPO=r"C:\Github\Widefield_DAQ_recorder"
NL=r"N:\MICROSCOPE\Priya\Widefield\labcams"; NDQ=r"N:\MICROSCOPE\Priya\Widefield\DAQ_recorder_output"; BL=r"N:\MICROSCOPE\Priya\Behavior_logs\Widefield"; TAG="affine8v1"
S={"PS92_20260806_124426":("PS92_20260806_124733.h5","PS92_20260806_124753"),"PS93_20260806_172253":("PS93_20260806_172316.h5","PS93_20260806_172322"),
   "PS94_20260806_074320":("PS94_20260806_074558.h5","PS94_20260806_074607"),"PS95_20260806_105300":("PS95_20260806_105651.h5","PS95_20260806_105657")}
env=dict(os.environ,PYTHONPATH=REPO)
def run(c):
    if subprocess.run([PY,"-m",*c],cwd=REPO,env=env).returncode: print(f"  FAIL {c[1]}"); return False
    return True
for sess,(daqf,logd) in S.items():
    an=sess[:4]; lab=f"{an}_0806_{TAG}"; mc=fr"{NL}\20260806\{sess}\motion_corrected"; res=fr"{mc}\wfield_local_results"; allen=fr"{res}\allen_aligned_{TAG}"
    daq=fr"{NDQ}\20260806\{daqf}"; bt=fr"{BL}\{logd}\trials.csv"; fm=glob.glob(fr"{mc}\*cleanpairs_frame_map.npz")[0]; summ=fm.replace("_frame_map.npz","_summary.json")
    quiet=fr"{mc}\quiet_{TAG}"; lick=fr"{mc}\lick_aligned_{TAG}"; qf=fr"{quiet}\{lab}_quiet_frame.npy"
    print(f"== {lab} quiet+quietnorm ==",flush=True)
    run(["wfield_local.quiet_periods","--daq-h5",daq,"--label",lab,"--output",quiet,"--frame-map",fm,"--cleanpairs-summary",summ])
    run(["wfield_local.framemap_event_maps","--what","lick","--daq-h5",daq,"--wfield-results",res,"--allen-dir",allen,"--frame-map",fm,"--cleanpairs-summary",summ,"--output",lick,"--label",lab,"--post-s","0.15","--quiet-frame",qf,"--behavior-trials",bt])
print("QUIET 8/6 DONE",flush=True)
