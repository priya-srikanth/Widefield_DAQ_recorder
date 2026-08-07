"""Reference cam1 frames for the 6 known spout positions (PS92 8/5), via the DAQ->cam1
sync-pulse map + fast ffmpeg time-seek. Foundation for the PS93 disambiguation GUI."""
import os, csv, subprocess, numpy as np, h5py
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import imageio_ffmpeg
FF=imageio_ffmpeg.get_ffmpeg_exe()
OUT=r"C:\Github\Widefield_DAQ_recorder\_ps93dis"; os.makedirs(OUT,exist_ok=True)
NM={0:"close_center",1:"close_L",2:"close_R",3:"far_center",4:"far_L",5:"far_R"}
DAQ=r"N:\MICROSCOPE\Priya\Widefield\DAQ_recorder_output\20260805\PS92_20260805_182111.h5"
CAMAVI=r"N:\MICROSCOPE\Priya\Behavior_cameras\widefield\20260805\PS92\cam1_2026-08-05T18_21_04.avi"
CAMCSV=CAMAVI.replace(".avi",".csv"); LOG=r"N:\MICROSCOPE\Priya\Behavior_logs\Widefield\PS92_20260805_182116\trials.csv"; FPS=250.0

def daq_cam_map(daq,camcsv):
    with h5py.File(daq,"r") as f:
        names=[n.decode() for n in f["digital"]["channel_names"][:]]; packed=np.asarray(f["digital"]["packed_samples"]).ravel(); fs=f.attrs["sample_rate_hz"]
        sync=(packed>>names.index("sync"))&1; cue=(packed>>names.index("cue"))&1
        sdaq=np.where(np.diff(sync.astype(np.int8))==1)[0]/fs; cue_t=np.where(np.diff(cue.astype(np.int8))==1)[0]/fs
    c3=np.array([int(r[2]) for r in csv.reader(open(camcsv)) if len(r)>=3]); scam=np.where(np.diff((c3&1).astype(np.int8))==1)[0]
    n=min(len(sdaq),len(scam))
    return (lambda t: float(np.interp(t,sdaq[:n],scam[:n]))/FPS), cue_t  # returns cam TIME (s)

def grab(t,out):
    subprocess.run([FF,"-nostdin","-loglevel","error","-ss",f"{t:.3f}","-i",CAMAVI,"-frames:v","1","-q:v","2","-y",out],check=False)

to_camtime,cue_t=daq_cam_map(DAQ,CAMCSV)
true=[int(r["pos_idx"]) for r in csv.DictReader(open(LOG))]
fig,ax=plt.subplots(2,3,figsize=(15,10))
for i,pos in enumerate([1,0,2,4,3,5]):
    idx=[k for k in range(min(len(cue_t),len(true))) if true[k]==pos]; k=idx[len(idx)//2]
    t=to_camtime(cue_t[k])+0.6; png=os.path.join(OUT,f"ref_{pos}.png"); grab(t,png)
    a=ax.ravel()[i]
    if os.path.exists(png): a.imshow(plt.imread(png)); 
    a.set_title(f"{NM[pos]} (pos {pos})",fontsize=12); a.axis("off")
fig.suptitle("Reference cam1 (head-on) frames — 6 known spout positions (PS92 8/5, sync-aligned)",fontsize=14)
plt.tight_layout(); fp=os.path.join(OUT,"reference_positions.png"); plt.savefig(fp,dpi=100); print("wrote",fp)
