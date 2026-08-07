"""Which PS94 channel is functional (470)? The functional channel shows the
cue-evoked GCaMP transient. Reconstruct a per-channel cue-triggered ROI dF/F
from raw frames (no SVD needed)."""
import h5py, numpy as np, os, re
from scipy.ndimage import binary_erosion

DAQ = r"E:\DAQ_recorder_output\20250603\PS94_20260603_175946.h5"
DAT = r"E:\labcams_data\20260603\PS94\raw_widefield_data\pco_edge_run000_00000000_2_462_464_uint16.dat"
m = re.search(r"_(\d+)_(\d+)_(\d+)_uint16", os.path.basename(DAT)); H, W = int(m.group(2)), int(m.group(3))

with h5py.File(DAQ, "r") as f:
    fs = float(f.attrs["sample_rate_hz"])
    di = [s.decode() for s in f["digital/channel_names"][:]]; an = [s.decode() for s in f["analog/channel_names"][:]]
    packed = f["digital/packed_samples"][:, 0]; sc = f["analog/int16_scale_volts_per_count"][:]; of = f["analog/int16_offset_volts"][:]
    def ac(n): i = an.index(n); return f["analog/samples_int16"][:, i].astype(np.float32)*sc[i]+of[i]
    led415 = ac("led415_ttl"); led470 = ac("led470_ttl")
def db(b): return (packed >> b) & 1
def rises(x, t=0.5): b=(np.asarray(x)>t).astype(np.int8); return np.flatnonzero(np.diff(b)==1)+1
pco = rises(db(di.index("pco_exposure"))); cue = rises(db(di.index("cue")))
i2 = np.clip(pco+int(0.002*fs),0,len(packed)-1)
code = np.where((led415[i2]>1.5)&~(led470[i2]>1.5),415,np.where((led470[i2]>1.5)&~(led415[i2]>1.5),470,0))
nphys = os.path.getsize(DAT)//(H*W*2)
mm = np.memmap(DAT, mode="r", dtype=np.uint16, shape=(nphys,H,W))
n=min(nphys,len(pco))

# ROI
avg=np.zeros((H,W));
for k in np.linspace(0,n-1,300).astype(int): avg+=mm[k]
mask=binary_erosion(avg>(0.45*avg.max()/300*300/300*1),iterations=6) if False else binary_erosion((avg/300)>(0.45*(avg/300).max()),iterations=6)
P=mask.ravel()

# cue -> frame index (1:1 frame<->pco)
cf = np.clip(np.searchsorted(pco,cue),1,n-1)
WIN=16  # frames each side (~0.5s at 31Hz)
ncue=min(150,len(cf))
sel=cf[np.linspace(0,len(cf)-1,ncue).astype(int)]
dff={415:[],470:[]}
for ci in sel:
    a,b=ci-WIN, ci+WIN
    if a<0 or b>=n: continue
    blk_idx=np.arange(a,b)
    vals=np.array([mm[k].reshape(-1)[P].mean() for k in blk_idx])
    cc=code[blk_idx]
    for c in (415,470):
        msk=cc==c
        pre=vals[msk & (blk_idx<ci)]; post=vals[msk & (blk_idx>=ci)]
        if len(pre)>=2 and len(post)>=2:
            dff[c].append((post.mean()-pre.mean())/pre.mean())
for c in (415,470):
    d=np.array(dff[c]); print(f"labeled {c}: cue-evoked dF/F mean={np.mean(d)*100:+.2f}%  (n={len(d)})  median={np.median(d)*100:+.2f}%")
print("functional (470) = channel with larger positive cue-evoked dF/F")
