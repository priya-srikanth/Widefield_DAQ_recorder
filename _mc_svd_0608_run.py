"""Fixed motion correction + SVD/hemo for the four 2026-06-08 sessions."""
import os, sys, subprocess
from pathlib import Path
PY=r"C:\ProgramData\anaconda3\envs\wfield\python.exe"; REPO=r"C:\Github\Widefield_DAQ_recorder"
D=r"E:\labcams_data\20260608"; Q=r"E:\DAQ_recorder_output"; DIMS="2_460_480"
SESSIONS={
 "PS92_0608":dict(sess="PS92_20260608_133759",daq=fr"{Q}\PS92_20260608_133847.h5"),
 "PS93_0608":dict(sess="PS93_20260608_195203",daq=fr"{Q}\PS93_20260608_195350.h5"),
 "PS94_0608":dict(sess="PS94_20260608_153651",daq=fr"{Q}\PS94_20260608_153702.h5"),
 "PS95_0608":dict(sess="PS95_20260608_180943",daq=fr"{Q}\PS95_20260608_180950.h5")}
def run(cmd):
    env=dict(os.environ,PYTHONPATH=REPO); print("\n$ "+" ".join(map(str,cmd)),flush=True)
    if subprocess.run([PY,"-m",*cmd],cwd=REPO,env=env).returncode: raise SystemExit(f"fail {cmd}")
def do(key):
    s=SESSIONS[key]; raw=fr"{D}\{s['sess']}\raw_widefield_data\pco_edge_run000_00000000_{DIMS}_uint16.dat"
    mc=Path(fr"{D}\{s['sess']}\motion_corrected"); binp=mc/f"motioncorrect_{DIMS}_uint16.bin"; results=mc/"wfield_local_results"
    print(f"\n================ {key} ================",flush=True)
    if binp.exists(): print("[skip] bin exists",flush=True)
    else: run(["wfield_local.run_wfield_motion",raw,"--output",str(mc),"--daq-h5",s["daq"],"--relabel-mode","rescue","--mode","2d"])
    if (results/"SVTcorr.npy").exists(): print("[skip] SVTcorr exists",flush=True)
    else: run(["wfield_local.run_wfield_local",str(binp),"--output",str(results),"-k","100","--functional-channel","1","--fs","31.23","--freq-highpass","0.1","--freq-lowpass","14.0"])
    print(f"================ {key} DONE ================",flush=True)
if __name__=="__main__":
    for k in (sys.argv[1:] or list(SESSIONS)): do(k)
    print("\nALL DONE",flush=True)
