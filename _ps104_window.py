"""PS104 cranial-window brightness/activity test (VGLUT1-iCre x RiboL1-jGCaMP8s).
Motion-corrected + SVD already done. Crops to the window tissue (curved rectangle +
vessel), plots 415 / raw 470 / hemo-corrected 470 dF/F over time, reports raw brightness.
No DAQ/behavior. Native frame."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R=r"E:\labcams_data\20260805\PS104_20260805_214803\motion_corrected\wfield_local_results"
OUT=r"C:\Github\Widefield_DAQ_recorder\_ps104"; os.makedirs(OUT,exist_ok=True)
FS=31.23  # corrected (pair) frame rate assumed
U=np.load(os.path.join(R,"U.npy")); SVT=np.load(os.path.join(R,"SVT.npy")); SVTcorr=np.load(os.path.join(R,"SVTcorr.npy"))
fa=np.load(os.path.join(R,"frames_average.npy"))  # (2,H,W): 0=415, 1=470
svt415=SVT[:,0::2]; svt470=SVT[:,1::2]
H,W,K=U.shape
# window ROI: box over the tissue patch, exclude the saturated bone ring (very bright)
box=np.zeros((H,W),bool); box[102:226,135:295]=True
# tissue is DARKER than the bright bone ring -> keep the darker window pixels
mask=box & (fa[1] < np.percentile(fa[1][box],72))
roi_w=U[mask].mean(0)  # (K,) ROI-mean spatial weight
F0_415=float(fa[0][mask].mean()); F0_470=float(fa[1][mask].mean())
dff=lambda svt: (roi_w@svt)*100.0  # U@SVT already fractional dF/F
TRIM=int(FS)  # drop first ~1s LED-onset artifact
t=np.arange(SVTcorr.shape[1])/FS
t=t[TRIM:]; svt415=svt415[:,TRIM:]; svt470=svt470[:,TRIM:]; SVTcorr=SVTcorr[:,TRIM:]
d415=dff(svt415); d470=dff(svt470); dcorr=dff(SVTcorr)
print(f"window pixels: {int(mask.sum())}  raw brightness (counts): 415={F0_415:.0f}  470={F0_470:.0f}")
print(f"dF/F std %: 415={d415.std():.2f}  470raw={d470.std():.2f}  470corr={dcorr.std():.2f}")
fig=plt.figure(figsize=(15,6))
ax0=fig.add_subplot(1,3,1); ax0.imshow(fa[1],cmap="gray")
import numpy as _np; ov=_np.zeros((H,W,4)); ov[mask]=[1,0,0,0.30]; ax0.imshow(ov)
ax0.set_title(f"PS104 470 mean + window ROI\n({int(mask.sum())} px; 415={F0_415:.0f}, 470={F0_470:.0f} counts)"); ax0.axis("off")
ax1=fig.add_subplot(1,3,(2,3))
ax1.plot(t,d415,lw=0.6,color="violet",label=f"415 nm (isosbestic) sd={d415.std():.2f}%")
ax1.plot(t,d470,lw=0.6,color="green",label=f"470 nm raw sd={d470.std():.2f}%")
ax1.plot(t,dcorr,lw=0.8,color="black",label=f"470 nm hemo-corrected sd={dcorr.std():.2f}%")
ax1.set_xlabel("time (s)"); ax1.set_ylabel("dF/F (%)"); ax1.set_title("PS104 window-mean dF/F over time"); ax1.legend(fontsize=9)
fig.suptitle("PS104 (VGLUT1-iCre x RiboL1-jGCaMP8s) cranial-window brightness/activity test — 415 vs 470 vs corrected",fontsize=13)
plt.tight_layout(); fp=os.path.join(OUT,"PS104_window_dff.png"); plt.savefig(fp,dpi=130); plt.savefig(fp.replace(".png",".svg")); print("wrote",fp)
