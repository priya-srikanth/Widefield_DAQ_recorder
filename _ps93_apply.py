"""Apply cam1-recovered PS93 8/5 spout positions: write synthetic trials.csv from the
auto-detected codes, copy validation artifacts to N:, then regenerate cue/lick/quiet maps
(6 positions) via the existing --behavior-trials path. Recovered codes satisfy
DAQ_code==(true & ~bit1) by construction, so the order+bitmask aligner accepts them 100%."""
import os, glob, csv, shutil, subprocess, numpy as np
PY=r"C:\ProgramData\anaconda3\envs\wfield\python.exe"; REPO=r"C:\Github\Widefield_DAQ_recorder"
NL=r"N:\MICROSCOPE\Priya\Widefield\labcams"; NDQ=r"N:\MICROSCOPE\Priya\Widefield\DAQ_recorder_output"
OUT=r"C:\Github\Widefield_DAQ_recorder\_ps93dis"; TAG="affine8v1"; CP,CO="2.0","2.0"
SESS="PS93_20260805_201110"; DAQF="PS93_20260805_202005.h5"; DATE="20260805"; lab=f"PS93_{DATE[4:]}_{TAG}"
NM={0:"close_center",1:"close_L",2:"close_R",3:"far_center",4:"far_L",5:"far_R"}
# 1. synthetic recovered trials.csv
d=np.load(os.path.join(OUT,"ps93_autolabels.npz")); true=d["true"].astype(int)
assert (true>=0).all(), "unrecovered cue present"
btcsv=os.path.join(OUT,"ps93_recovered_trials.csv")
with open(btcsv,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["pos_idx","pos_name"])
    for c in true: w.writerow([c,NM[c]])
print(f"wrote {btcsv}  ({len(true)} trials, counts={ {NM[k]:int((true==k).sum()) for k in range(6)} })")
# 2. document validation artifacts to N: (recovery folder alongside the session)
docdir=fr"{NL}\{DATE}\{SESS}\motion_corrected\spout_position_recovery_cam1"; os.makedirs(docdir,exist_ok=True)
for fn in ("ps93_autolabels.png","ps93_autolabels.npz","ps93_verify_montage.png","ps93_recovered_trials.csv"):
    shutil.copy2(os.path.join(OUT,fn),os.path.join(docdir,fn))
print(f"copied validation artifacts -> {docdir}")
# 3. regenerate maps with recovered positions
mc=fr"{NL}\{DATE}\{SESS}\motion_corrected"; res=fr"{mc}\wfield_local_results"; allen=fr"{res}\allen_aligned_{TAG}"
daq=fr"{NDQ}\{DATE}\{DAQF}"; fm=glob.glob(fr"{mc}\*cleanpairs_frame_map.npz")[0]; summ=fm.replace("_frame_map.npz","_summary.json")
cue=fr"{mc}\spout_trial_averages_{TAG}"; lick=fr"{mc}\lick_aligned_{TAG}"; quiet=fr"{mc}\quiet_{TAG}"
cnpz=fr"{cue}\{lab}_spout_positions_1s_pre_post_delta_maps.npz"; csum=cnpz.replace("_maps.npz","_summary.json")
lnpz=fr"{lick}\{lab}_lick_aligned_150ms_post_by_spout_maps.npz"; lsum=lnpz.replace("_maps.npz","_summary.json"); qf=fr"{quiet}\{lab}_quiet_frame.npy"
B=["--behavior-trials",btcsv]
def run(c):
    env=dict(os.environ,PYTHONPATH=REPO); print("$ "+" ".join(map(str,c[:5]))+" ...",flush=True)
    if subprocess.run([PY,"-m",*c],cwd=REPO,env=env).returncode: raise SystemExit(f"fail {c[1]}")
print(f"\n===== {lab} (cam1-recovered positions) =====",flush=True)
run(["wfield_local.framemap_event_maps","--what","cue","--daq-h5",daq,"--wfield-results",res,"--allen-dir",allen,"--frame-map",fm,"--cleanpairs-summary",summ,"--output",cue,"--label",lab,"--pre-s",CP,"--post-s",CO]+B)
run(["wfield_local.plot_spout_trial_averages_shared_scale","--label",lab,"--trial-maps",cnpz,"--allen-dir",allen,"--output",cue,"--summary",csum])
run(["wfield_local.plot_spout_position_contrasts","--label",lab,"--trial-maps",cnpz,"--allen-dir",allen,"--output",cue])
run(["wfield_local.framemap_event_maps","--what","lick","--daq-h5",daq,"--wfield-results",res,"--allen-dir",allen,"--frame-map",fm,"--cleanpairs-summary",summ,"--output",lick,"--label",lab,"--post-s","0.15"]+B)
run(["wfield_local.plot_lick_position_contrasts","--label",lab,"--lick-maps",lnpz,"--allen-dir",allen,"--output",lick])
run(["wfield_local.plot_lick_vs_cue_spout_maps","--label",lab,"--cue-maps",cnpz,"--lick-maps",lnpz,"--allen-dir",allen,"--output",lick,"--cue-summary",csum,"--lick-summary",lsum])
run(["wfield_local.quiet_periods","--daq-h5",daq,"--label",lab,"--output",quiet,"--frame-map",fm,"--cleanpairs-summary",summ])
run(["wfield_local.framemap_event_maps","--what","lick","--daq-h5",daq,"--wfield-results",res,"--allen-dir",allen,"--frame-map",fm,"--cleanpairs-summary",summ,"--output",lick,"--label",lab,"--post-s","0.15","--quiet-frame",qf]+B)
print(f"===== {lab} DONE =====",flush=True)
