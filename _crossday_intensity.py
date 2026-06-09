"""Cross-day RAW fluorescence intensity trend per animal, from each session's
frames_average.npy (raw-count motion-corrected mean) on N: -- no raw re-read.

CAVEAT printed on the figure: LED power is manually titrated day to day, so a trend
may reflect the LED setting, not photobleaching. Output -> _crossday_intensity_out\.
"""
import os, re, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
NL=r"N:\MICROSCOPE\Priya\Widefield\labcams"; OUT=r"C:\Github\Widefield_DAQ_recorder\_crossday_intensity_out"; os.makedirs(OUT,exist_ok=True)
rows=[]
for date in sorted(os.listdir(NL)):
    dp=os.path.join(NL,date)
    if not (os.path.isdir(dp) and re.fullmatch(r"\d{8}",date)): continue
    for sess in sorted(os.listdir(dp)):
        fa=os.path.join(dp,sess,"motion_corrected","wfield_local_results","frames_average.npy")
        m=re.match(r"(PS\d+)",sess)
        if not (os.path.exists(fa) and m): continue
        favg=np.load(fa)  # (2,H,W): ch0=415, ch1=470
        ref=favg[1]; brain=binary_erosion(ref>0.45*ref.max(),iterations=6)
        if brain.sum()<200: brain=ref>np.percentile(ref,40)
        med415=float(np.median(favg[0][brain])); med470=float(np.median(favg[1][brain]))
        rows.append((m.group(1),date,med415,med470))
animals=sorted({r[0] for r in rows})
alldates=sorted({r[1] for r in rows})            # chronological x-axis (fixed order)
xpos={d:i for i,d in enumerate(alldates)}
fig,ax=plt.subplots(1,2,figsize=(15,5.5))
for ci,(nm,idx) in enumerate([("415",2),("470",3)]):
    for a in animals:
        rr=sorted([r for r in rows if r[0]==a],key=lambda r:r[1])
        x=[xpos[r[1]] for r in rr]; y=[r[idx] for r in rr]   # integer positions -> true date order
        ax[ci].plot(x,y,"-o",label=a)
    ax[ci].set_xticks(range(len(alldates))); ax[ci].set_xticklabels([d[4:6]+"/"+d[6:] for d in alldates])
    ax[ci].set_title(f"{nm} nm raw brain-ROI median across days"); ax[ci].set_xlabel("date (MM/DD)"); ax[ci].set_ylabel("raw counts"); ax[ci].legend(); ax[ci].tick_params(axis="x",rotation=45)
fig.suptitle("Cross-day RAW fluorescence intensity (frames_average brain-ROI median)\n"
             "CAVEAT: LED power is manually titrated day-to-day -> a trend may reflect LED setting, not bleaching", fontsize=12)
plt.tight_layout(); fp=os.path.join(OUT,"crossday_raw_intensity.png"); plt.savefig(fp,dpi=130); plt.close(fig)
print("wrote",fp)
print(f"{'animal':6s} {'date':9s} {'415':>8s} {'470':>8s}")
for a,d,i4,i7 in rows: print(f"{a:6s} {d:9s} {i4:8.0f} {i7:8.0f}")
