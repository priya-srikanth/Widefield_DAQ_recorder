"""PS93 8/5 verification montage: (top) one reference cam1 frame per recovered position with
its detected spout-x; (bottom) the lowest-confidence trials (nearest a threshold) for spot-check.
Reads _ps93dis/ps93_autolabels.npz. Writes ps93_verify_montage.png."""
import os, subprocess, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import imageio_ffmpeg
FF=imageio_ffmpeg.get_ffmpeg_exe(); OUT=r"C:\Github\Widefield_DAQ_recorder\_ps93dis"; FPS=250.0; W=H=600
AVI=r"N:\MICROSCOPE\Priya\Behavior_cameras\widefield\20260805\PS93\cam1_2026-08-05T20_20_03.avi"
NM={0:"close_center",1:"close_L",2:"close_R",3:"far_center",4:"far_L",5:"far_R"}
d=np.load(os.path.join(OUT,"ps93_autolabels.npz")); xs=d["xs"]; dc=d["dc"]; true=d["true"]; conf=d["conf"]; cf=d["camframe"]; thr0=float(d["thr0"]); thr1=float(d["thr1"])
def grab(fr):
    t=fr/FPS+0.6
    p=subprocess.run([FF,"-nostdin","-loglevel","error","-ss",f"{t:.3f}","-i",AVI,"-frames:v","1","-f","rawvideo","-pix_fmt","gray","-"],capture_output=True)
    return np.frombuffer(p.stdout[:W*H],np.uint8).reshape(H,W) if len(p.stdout)>=W*H else np.zeros((H,W),np.uint8)
fig,axes=plt.subplots(2,6,figsize=(19,7))
# top: highest-confidence reference per recovered position
for j,k in enumerate(range(6)):
    idx=np.where(true==k)[0]; idx=idx[np.argsort(-conf[idx])] if k in (0,1,2,3) else idx  # amb: most confident; fL/fR: any
    i=idx[len(idx)//2] if k in (4,5) else idx[0]
    ax=axes[0,j]; ax.imshow(grab(cf[i]),cmap="gray"); ax.axvline(xs[i],c="lime",lw=1.2)
    ax.set_title(f"{NM[k]}  (x={xs[i]:.0f})",fontsize=10); ax.axis("off")
# bottom: lowest-confidence AMBIGUOUS trials (nearest threshold) for spot-check
amb=np.where(np.isin(dc,[0,1]) & (true>=0))[0]; amb=amb[np.argsort(conf[amb])][:6]
for j,i in enumerate(amb):
    thr=thr0 if dc[i]==0 else thr1; ax=axes[1,j]; ax.imshow(grab(cf[i]),cmap="gray")
    ax.axvline(xs[i],c="orange",lw=1.2); ax.axhline(0,c="none")
    ax.set_title(f"trial{i} deg{dc[i]}->{NM[true[i]]}\nx={xs[i]:.0f} thr={thr:.0f}",fontsize=8,color="darkorange"); ax.axis("off")
axes[0,0].text(-0.1,0.5,"REFERENCE\n(auto-label)",transform=axes[0,0].transAxes,rotation=90,va="center",ha="right",fontsize=9,fontweight="bold")
axes[1,0].text(-0.1,0.5,"CLOSEST CALLS\n(spot-check)",transform=axes[1,0].transAxes,rotation=90,va="center",ha="right",fontsize=9,fontweight="bold")
plt.suptitle("PS93 2026-08-05 spout-position recovery — cam1 (green=reference detection, orange=nearest-threshold)",fontsize=12)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"ps93_verify_montage.png"),dpi=105); print("wrote ps93_verify_montage.png")
# report closest-call margins
print("6 closest ambiguous calls (px from threshold):", [f"trial{i}:{conf[i]:.0f}px->{NM[true[i]]}" for i in amb])
print("min margin overall:", f"{conf[amb[0]]:.0f}px")
