"""Refresh the per-animal 'xall' cross-day vasculature-alignment overlays to include EVERY
task-era session (6/5 onward), auto-discovered from N:. QC-ONLY: cross_day_align with
warp_u=False -> writes {an}_<date>_mean470_in_ref_native/ccf.npy + the multi-day overlay QC
({an}_cross_day_alignment_qc.png) + summary to labcams\\xday\\{an}_xall. Does NOT touch/re-emit
any session's allen_aligned dir (GPU inputs untouched). Reference = each animal's 6/6.

Usage: python _xall_refresh.py            (all four animals)
       python _xall_refresh.py PS94       (one animal)
"""
import os, re, sys, json, glob, subprocess
PY=r"C:\ProgramData\anaconda3\envs\wfield\python.exe"; REPO=r"C:\Github\Widefield_DAQ_recorder"
NL=r"N:\MICROSCOPE\Priya\Widefield\labcams"
LM={"PS92":"v2","PS93":"v2","PS94":"v1","PS95":"v1"}   # 6/6 reference landmark version
MIN_DATE="20260605"                                    # task era (6/5 onward); excludes 6/1-6/4 baseline
def fwd(p): return str(p).replace("\\","/")
def results_for(an,date_dir):
    for sess in sorted(glob.glob(fr"{NL}\{date_dir}\{an}_*")):
        r=fr"{sess}\motion_corrected\wfield_local_results"
        if os.path.exists(fr"{r}\frames_average.npy"): return r
    return None
def build(an):
    dates=sorted(d for d in os.listdir(NL) if re.fullmatch(r"\d{8}",d) and d>=MIN_DATE)
    sess={}
    for d in dates:
        r=results_for(an,d)
        if r: sess[f"{an}_{d[4:]}"]=r
    if f"{an}_0606" not in sess:
        print(f"[{an}] no 6/6 reference found -> skip",flush=True); return None
    ref6=sorted(glob.glob(fr"{NL}\20260606\{an}_*"))[0]
    cfg={"animal":an,"mode":"reference-native","func_channel":1,"reference":f"{an}_0606",
         "output":fwd(fr"{NL}\xday\{an}_xall"),"warp_u":False,"sessions":{}}
    for k,r in sess.items():
        cfg["sessions"][k]={"results":fwd(r)}
    cfg["sessions"][f"{an}_0606"]["landmarks"]=fwd(fr"{ref6}\raw_widefield_data\dorsal_cortex_landmarks_{LM[an]}.json")
    p=fr"{REPO}\_xall_qc_{an}.json"; json.dump(cfg,open(p,"w"),indent=2)
    print(f"[{an}] {len(sess)} days: {sorted(sess)}",flush=True); return p
def run(cfg):
    env=dict(os.environ,PYTHONPATH=REPO)
    return subprocess.run([PY,"-m","wfield_local.cross_day_align",cfg],cwd=REPO,env=env).returncode
if __name__=="__main__":
    ans=sys.argv[1:] or ["PS92","PS93","PS94","PS95"]
    for an in ans:
        p=build(an)
        if p and run(p): print(f"[{an}] cross_day_align FAILED",flush=True)
    print("\nXALL REFRESH DONE",flush=True)
