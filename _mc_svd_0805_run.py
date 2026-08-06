"""Fixed motion correction + SVD for the four 2026-08-05 behavior sessions (explicit
raw paths - run numbers vary). relabel rescue. DAQ loose in E:\DAQ_recorder_output."""
import os, sys, subprocess
from pathlib import Path
PY=r"C:\ProgramData\anaconda3\envs\wfield\python.exe"; REPO=r"C:\Github\Widefield_DAQ_recorder"
D=r"E:\labcams_data\20260805"; Q=r"E:\DAQ_recorder_output"; DIMS="2_460_480"
S={
 "PS92_0805":dict(sess="PS92_20260805_181150",raw="pco_edge_run001_00000000_2_460_480_uint16.dat",daq="PS92_20260805_182111.h5"),
 "PS93_0805":dict(sess="PS93_20260805_201110",raw="pco_edge_run000_00000000_2_460_480_uint16.dat",daq="PS93_20260805_202005.h5"),
 "PS94_0805":dict(sess="PS94_20260805_124758",raw="pco_edge_run002_00000000_2_460_480_uint16.dat",daq="PS94_20260805_131025.h5"),
 "PS95_0805":dict(sess="PS95_20260805_155615",raw="pco_edge_run000_00000000_2_460_480_uint16.dat",daq="PS95_20260805_160437.h5"),
}
def run(c):
    env=dict(os.environ,PYTHONPATH=REPO); print("\n$ "+" ".join(map(str,c)),flush=True)
    if subprocess.run([PY,"-m",*c],cwd=REPO,env=env).returncode: raise SystemExit(f"fail {c}")
def do(k):
    s=S[k]; raw=fr"{D}\{s['sess']}\raw_widefield_data\{s['raw']}"
    mc=Path(fr"{D}\{s['sess']}\motion_corrected"); binp=mc/f"motioncorrect_{DIMS}_uint16.bin"; results=mc/"wfield_local_results"
    print(f"\n===== {k} =====",flush=True)
    if binp.exists(): print("[skip] bin exists",flush=True)
    else: run(["wfield_local.run_wfield_motion",raw,"--output",str(mc),"--daq-h5",fr"{Q}\{s['daq']}","--relabel-mode","rescue","--mode","2d"])
    if (results/"SVTcorr.npy").exists(): print("[skip] SVTcorr exists",flush=True)
    else: run(["wfield_local.run_wfield_local",str(binp),"--output",str(results),"-k","100","--functional-channel","1","--fs","31.23","--freq-highpass","0.1","--freq-lowpass","14.0"])
    print(f"===== {k} DONE =====",flush=True)
if __name__=="__main__":
    for k in (sys.argv[1:] or list(S)): do(k)
    print("\nALL DONE",flush=True)
