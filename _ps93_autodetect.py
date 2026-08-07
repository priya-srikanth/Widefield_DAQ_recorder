"""Auto-recover PS93 8/5 spout positions from cam1 (dead strobe bit1). Sync-map each DAQ
cue -> cam1 frame, detect spout x (dark bar over the bright head), classify ambiguous
codes by 2-cluster split (deg0: center=close_center vs right=close_R; deg1: left=close_L
vs center=far_center). deg4=far_L, deg5=far_R unambiguous. Saves labels + validation."""
import os, csv, subprocess, numpy as np, h5py
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import imageio_ffmpeg
from wfield_local.plot_spout_trial_averages import _load_daq_events, _classify_cues
FF=imageio_ffmpeg.get_ffmpeg_exe(); OUT=r"C:\Github\Widefield_DAQ_recorder\_ps93dis"; FPS=250.0; W=H=600
DAQ=r"N:\MICROSCOPE\Priya\Widefield\DAQ_recorder_output\20260805\PS93_20260805_202005.h5"
AVI=r"N:\MICROSCOPE\Priya\Behavior_cameras\widefield\20260805\PS93\cam1_2026-08-05T20_20_03.avi"; CSV=AVI.replace(".avi",".csv")
NM={0:"close_center",1:"close_L",2:"close_R",3:"far_center",4:"far_L",5:"far_R"}
def grab(t):
    p=subprocess.run([FF,"-nostdin","-loglevel","error","-ss",f"{t:.3f}","-i",AVI,"-frames:v","1","-f","rawvideo","-pix_fmt","gray","-"],capture_output=True)
    return np.frombuffer(p.stdout[:W*H],np.uint8).reshape(H,W).astype(np.float32) if len(p.stdout)>=W*H else None
def sx(img): return 170+int(np.argmin(img[140:270,170:440].mean(0)))
with h5py.File(DAQ,"r") as f:
    names=[n.decode() for n in f["digital"]["channel_names"][:]]; packed=np.asarray(f["digital"]["packed_samples"]).ravel(); fs=f.attrs["sample_rate_hz"]
    sd=np.where(np.diff(((packed>>names.index("sync"))&1).astype(np.int8))==1)[0]/fs
    ct=np.where(np.diff(((packed>>names.index("cue"))&1).astype(np.int8))==1)[0]/fs
c3=np.array([int(r[2]) for r in csv.reader(open(CSV)) if len(r)>=3]); sc=np.where(np.diff((c3&1).astype(np.int8))==1)[0]
n=min(len(sd),len(sc)); tocam=lambda t: float(np.interp(t,sd[:n],sc[:n]))/FPS
ev=_load_daq_events(DAQ); dc=np.asarray(_classify_cues(ev["cue_samples"],ev["strobe_samples"],ev["strobe_codes"]))
N=min(len(ct),len(dc)); dc=dc[:N]; camf=np.array([tocam(ct[k]) for k in range(N)])
xs=np.full(N,-1.0)
for k in range(N):
    img=grab(camf[k]+0.6)
    if img is not None: xs[k]=sx(img)
    if k%60==0: print(f"  {k}/{N}",flush=True)
def split2(mask, hi_code, lo_code):  # higher-x cluster -> hi_code, lower-x -> lo_code
    v=xs[mask&(xs>0)]; thr=np.mean([np.median(v[v>=np.median(v)]),np.median(v[v<np.median(v)])])
    return thr
true=dc.copy().astype(int); conf=np.zeros(N)
thr0=split2(dc==0,0,2); thr1=split2(dc==1,1,3)
for k in range(N):
    if xs[k]<0: true[k]=-1; continue
    if dc[k]==0: true[k]=0 if xs[k]>=thr0 else 2; conf[k]=abs(xs[k]-thr0)
    elif dc[k]==1: true[k]=1 if xs[k]>=thr1 else 3; conf[k]=abs(xs[k]-thr1)
    elif dc[k]==4: true[k]=4
    elif dc[k]==5: true[k]=5
    else: true[k]=-1
np.savez(os.path.join(OUT,"ps93_autolabels.npz"),xs=xs,dc=dc,true=true,conf=conf,camframe=(camf*FPS).astype(int),thr0=thr0,thr1=thr1)
import collections; print("recovered:",{NM[k]:int((true==k).sum()) for k in range(6)}, "thr0=%.0f thr1=%.0f"%(thr0,thr1))
fig,ax=plt.subplots(figsize=(13,5)); C={0:"purple",1:"blue",2:"red",3:"green",4:"olive",5:"orange"}
for k in range(6):
    m=(true==k); ax.scatter(np.where(m)[0],xs[m],s=10,c=C[k],label=f"{NM[k]} n={m.sum()}")
ax.axhline(thr0,ls=":",c="gray"); ax.axhline(thr1,ls=":",c="gray")
ax.set_xlabel("trial"); ax.set_ylabel("spout x (px)"); ax.legend(fontsize=8,ncol=2); ax.set_title("PS93 8/5 auto-recovered spout positions (cam1 spout-x)")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"ps93_autolabels.png"),dpi=110); print("wrote ps93_autolabels.png",flush=True)
