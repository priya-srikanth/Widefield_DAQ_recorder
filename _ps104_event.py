"""PS104: show spatial dF/F maps for several candidate transients (corrected 470),
looking for a licking/motor bout with medial / curved-edge (M1/ALM) localization."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
R=r"E:\labcams_data\20260805\PS104_20260805_214803\motion_corrected\wfield_local_results"
OUT=r"C:\Github\Widefield_DAQ_recorder\_ps104"; FS=31.23
U=np.load(os.path.join(R,"U.npy")); SVTcorr=np.load(os.path.join(R,"SVTcorr.npy")); SVT=np.load(os.path.join(R,"SVT.npy"))
fa=np.load(os.path.join(R,"frames_average.npy")); H,W,K=U.shape
box=np.zeros((H,W),bool); box[102:226,135:295]=True
mask=box & (fa[1]<np.percentile(fa[1][box],72)); roi_w=U[mask].mean(0)
dcorr=(roi_w@SVTcorr)*100; d470=(roi_w@SVT[:,1::2])*100
sm=uniform_filter1d(dcorr,int(0.4*FS))
pk,_=find_peaks(sm, height=sm[int(2*FS):].std()*1.6, distance=int(3*FS)); pk=pk[pk/FS>25]  # skip photobleach onset
top=pk[np.argsort(sm[pk])[::-1]][:6]; top=np.sort(top)
print("mid-recording events (t s, dF/F %):",[(round(p/FS,1),round(dcorr[p],2)) for p in top])
def smap(ev):
    e0,e1=ev-int(0.3*FS),ev+int(1.0*FS); b0,b1=ev-int(2.5*FS),ev-int(1.0*FS)
    return (U@(SVTcorr[:,e0:e1].mean(1)-SVTcorr[:,b0:b1].mean(1)))*100
fig=plt.figure(figsize=(17,7))
axT=fig.add_subplot(2,1,1); tt=np.arange(len(dcorr))/FS
axT.plot(tt,dcorr,lw=0.6,color="black")
for ev in top: axT.axvline(ev/FS,color="red",lw=0.8,alpha=0.7)
axT.set_xlabel("time (s)"); axT.set_ylabel("window dF/F (%)"); axT.set_title("PS104 corrected window trace with candidate events (red)")
maps=[smap(int(ev)) for ev in top]; lim=np.nanpercentile(np.abs(np.array([m[mask] for m in maps])),98)
for i,(ev,m) in enumerate(zip(top,maps)):
    ax=fig.add_subplot(2,6,7+i); ax.imshow(fa[1],cmap="gray")
    disp=np.full((H,W),np.nan); disp[mask]=m[mask]
    ax.imshow(disp,cmap="RdBu_r",vmin=-lim,vmax=lim,alpha=0.8); ax.axis("off"); ax.set_title(f"t={ev/FS:.0f}s ({dcorr[int(ev)]:.1f}%)",fontsize=9)
fig.suptitle(f"PS104 candidate motor/licking events: corrected dF/F over ~1.3s (event mean - pre) [+/-{lim:.2f}%]. Window over M1/ALM toward midline.",fontsize=12)
plt.tight_layout(); fp=os.path.join(OUT,"PS104_candidate_events_montage.png"); plt.savefig(fp,dpi=130); plt.savefig(fp.replace(".png",".svg")); print("wrote",fp)
