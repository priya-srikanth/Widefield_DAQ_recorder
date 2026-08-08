"""Nightly per-session chain for 2026-08-07: motion(fixed) -> SVD -> cross-register to 6/6
(emit allen_aligned_affine8v1) -> push LocaNMF inputs to N: FIRST. Processes sessions one at a
time in chronological order so the GPU can start LocaNMF on each as soon as it lands.
Strobe bit1 is fixed today -> all 6 spout positions are native in the DAQ (no recovery)."""
import os, sys, glob, json, shutil, subprocess
from pathlib import Path
PY=r"C:\ProgramData\anaconda3\envs\wfield\python.exe"; REPO=r"C:\Github\Widefield_DAQ_recorder"
D=r"E:\labcams_data\20260807"; Q=r"E:\DAQ_recorder_output"; DIMS="2_460_480"
RAW=f"pco_edge_run000_00000000_{DIMS}_uint16.dat"
NL=r"N:\MICROSCOPE\Priya\Widefield\labcams\20260807"; N6=r"N:\MICROSCOPE\Priya\Widefield\labcams\20260606"
REF={"PS92":("PS92_20260606_122451","v2"),"PS93":("PS93_20260606_180117","v2"),
     "PS94":("PS94_20260606_140854","v1"),"PS95":("PS95_20260606_160806","v1")}
S={"PS94":dict(sess="PS94_20260807_104125",daq="PS94_20260807_105410.h5"),
   "PS95":dict(sess="PS95_20260807_124637",daq="PS95_20260807_125106.h5"),
   "PS92":dict(sess="PS92_20260807_150924",daq="PS92_20260807_151146.h5"),
   "PS93":dict(sess="PS93_20260807_174403",daq="PS93_20260807_174416.h5")}
def run(c):
    env=dict(os.environ,PYTHONPATH=REPO); print("\n$ "+" ".join(map(str,c)),flush=True)
    if subprocess.run([PY,"-m",*c],cwd=REPO,env=env).returncode: raise SystemExit(f"fail {c}")
def fwd(p): return str(p).replace("\\","/")
def do(an):
    s=S[an]; sess=s['sess']; mc=Path(fr"{D}\{sess}\motion_corrected"); binp=mc/f"motioncorrect_{DIMS}_uint16.bin"
    results=mc/"wfield_local_results"; allen=results/"allen_aligned_affine8v1"
    raw=fr"{D}\{sess}\raw_widefield_data\{RAW}"; daq=fr"{Q}\{s['daq']}"
    print(f"\n################ {an} {sess} ################",flush=True)
    # 1 motion (fixed, sign-corrected)
    if binp.exists(): print("[skip] bin exists",flush=True)
    else: run(["wfield_local.run_wfield_motion",raw,"--output",str(mc),"--daq-h5",daq,"--relabel-mode","rescue","--mode","2d"])
    # 2 SVD (k=100, functional ch1)
    if (results/"SVTcorr.npy").exists(): print("[skip] SVTcorr exists",flush=True)
    else: run(["wfield_local.run_wfield_local",str(binp),"--output",str(results),"-k","100","--functional-channel","1","--fs","31.23","--freq-highpass","0.1","--freq-lowpass","14.0"])
    # 3 cross-register to that animal's 6/6 + emit allen_aligned_affine8v1
    refsess,lm=REF[an]
    cfg={"animal":an,"mode":"reference-native","func_channel":1,"reference":f"{an}_0606",
         "output":fwd(fr"{NL}\xday\{an}_0807"),"warp_u":True,
         "sessions":{f"{an}_0606":{"results":fwd(fr"{N6}\{refsess}\motion_corrected\wfield_local_results"),
                                   "landmarks":fwd(fr"{N6}\{refsess}\raw_widefield_data\dorsal_cortex_landmarks_{lm}.json")},
                     f"{an}_0807":{"results":fwd(results)}}}
    p=fr"{REPO}\_xday_{an}_0807.json"; json.dump(cfg,open(p,"w"),indent=2)
    run(["wfield_local.cross_day_align",p])
    # 4 push LocaNMF inputs to N: FIRST (full results dir + frame_map + summary; NOT the .bin)
    ndst=Path(fr"{NL}\{sess}\motion_corrected"); ndst.mkdir(parents=True,exist_ok=True)
    nres=ndst/"wfield_local_results"
    if nres.exists(): shutil.rmtree(nres)
    shutil.copytree(results,nres)  # SVT/SVTcorr/U/T/rcoeffs/frames_average/summary + allen_aligned_affine8v1
    for f in glob.glob(str(mc/"*cleanpairs_frame_map.npz"))+glob.glob(str(mc/"*cleanpairs_summary.json"))+\
             glob.glob(str(mc/"motion_correction_*")):
        shutil.copy2(f,ndst/os.path.basename(f))
    print(f"[LocaNMF] {an} inputs pushed -> N: (SVTcorr + allen_aligned_affine8v1 + frame_map + summary)",flush=True)
    print(f"################ {an} DONE ################",flush=True)
if __name__=="__main__":
    for an in (sys.argv[1:] or list(S)): do(an)
    print("\nNIGHTLY 8/7 motion->SVD->xreg->push ALL DONE",flush=True)
