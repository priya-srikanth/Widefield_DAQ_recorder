"""Fixed motion + SVD for 2026-08-06 (explicit raw paths; run numbers vary)."""
import os, sys, subprocess
from pathlib import Path
PY=r"C:\ProgramData\anaconda3\envs\wfield\python.exe"; REPO=r"C:\Github\Widefield_DAQ_recorder"
D=r"E:\labcams_data\20260806"; Q=r"E:\DAQ_recorder_output"; DIMS="2_460_480"
S={"PS92_0806":dict(sess="PS92_20260806_124426",raw="pco_edge_run000_00000000_2_460_480_uint16.dat",daq="PS92_20260806_124733.h5"),
   "PS93_0806":dict(sess="PS93_20260806_172253",raw="pco_edge_run000_00000000_2_460_480_uint16.dat",daq="PS93_20260806_172316.h5"),
   "PS94_0806":dict(sess="PS94_20260806_074320",raw="pco_edge_run001_00000000_2_460_480_uint16.dat",daq="PS94_20260806_074558.h5"),
   "PS95_0806":dict(sess="PS95_20260806_105300",raw="pco_edge_run001_00000000_2_460_480_uint16.dat",daq="PS95_20260806_105651.h5")}
def run(c):
    env=dict(os.environ,PYTHONPATH=REPO); print("\n$ "+" ".join(map(str,c)),flush=True)
    if subprocess.run([PY,"-m",*c],cwd=REPO,env=env).returncode: raise SystemExit(f"fail {c}")
def do(k):
    s=S[k]; raw=fr"{D}\{s['sess']}\raw_widefield_data\{s['raw']}"; mc=Path(fr"{D}\{s['sess']}\motion_corrected"); binp=mc/f"motioncorrect_{DIMS}_uint16.bin"; results=mc/"wfield_local_results"
    print(f"\n===== {k} =====",flush=True)
    if binp.exists(): print("[skip] bin",flush=True)
    else: run(["wfield_local.run_wfield_motion",raw,"--output",str(mc),"--daq-h5",fr"{Q}\{s['daq']}","--relabel-mode","rescue","--mode","2d"])
    if (results/"SVTcorr.npy").exists(): print("[skip] SVTcorr",flush=True)
    else: run(["wfield_local.run_wfield_local",str(binp),"--output",str(results),"-k","100","--functional-channel","1","--fs","31.23","--freq-highpass","0.1","--freq-lowpass","14.0"])
    print(f"===== {k} DONE =====",flush=True)
if __name__=="__main__":
    for k in (sys.argv[1:] or list(S)): do(k)
    print("\nALL DONE",flush=True)
