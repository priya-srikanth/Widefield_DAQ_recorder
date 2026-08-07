"""Rigorous check: which DAQ-labeled channel (415/470) is the functional one?
The functional channel carries GCaMP, so it shows the larger cue- AND lick-evoked
dF/F. Frame k <-> pco pulse k by construction (DAQ pco_exposure IS the camera
exposure signal; LED TTLs are recorded simultaneously), so labeling by code[k]
is correctly aligned (offset 0)."""
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
    led415 = ac("led415_ttl"); led470 = ac("led470_ttl"); lick = ac("lick_analog")
def db(b): return (packed >> b) & 1
def rises(x, t=0.5): b=(np.asarray(x)>t).astype(np.int8); return np.flatnonzero(np.diff(b)==1)+1
pco = rises(db(di.index("pco_exposure"))); cue = rises(db(di.index("cue")))
# lick onsets (drop below 2.5V, 100ms refractory)
lb=(lick<2.5).astype(np.int8); lon=np.flatnonzero(np.diff(lb)==1)+1
keep=[lon[0]] if len(lon) else []
for s in lon[1:]:
    if (s-keep[-1])/fs>0.1: keep.append(s)
lon=np.array(keep)
i2=np.clip(pco+int(0.002*fs),0,len(packed)-1)
code=np.where((led415[i2]>1.5)&~(led470[i2]>1.5),415,np.where((led470[i2]>1.5)&~(led415[i2]>1.5),470,0))
nphys=os.path.getsize(DAT)//(H*W*2); mm=np.memmap(DAT,mode="r",dtype=np.uint16,shape=(nphys,H,W)); n=min(nphys,len(pco))
avg=np.zeros((H,W));
for k in np.linspace(0,n-1,300).astype(int): avg+=mm[k]
P=binary_erosion((avg/300)>(0.45*(avg/300).max()),iterations=6).ravel()

def evoked(events, pre_s=0.6, post_s=0.6, maxn=300):
    ev=np.clip(np.searchsorted(pco,events),1,n-1)
    sel=ev[np.linspace(0,len(ev)-1,min(maxn,len(ev))).astype(int)]
    WIN=int(round(max(pre_s,post_s)*fs/ (1000/ (1000)) ))  # placeholder
    WIN=18
    d={415:[],470:[]}
    for ci in sel:
        a,b=ci-WIN,ci+WIN
        if a<0 or b>=n: continue
        idx=np.arange(a,b); vals=np.array([mm[k].reshape(-1)[P].mean() for k in idx]); cc=code[idx]
        for c in (415,470):
            mk=cc==c; pre=vals[mk&(idx<ci)]; post=vals[mk&(idx>=ci)]
            if len(pre)>=2 and len(post)>=2: d[c].append((post.mean()-pre.mean())/pre.mean())
    return {c:np.array(v) for c,v in d.items()}

for name,evs in [("CUE",cue),("LICK",lon)]:
    d=evoked(evs)
    a=d[415]; b=d[470]
    print(f"{name}: labeled415 dF/F={np.mean(a)*100:+.2f}% (n={len(a)})  labeled470 dF/F={np.mean(b)*100:+.2f}% (n={len(b)})  -> functional={'470' if np.mean(b)>np.mean(a) else '415'}")
print("brightness (ROI): 415=%.0f 470=%.0f" % (
    np.mean([mm[k].reshape(-1)[P].mean() for k in np.where(code[:n]==415)[0][:500]]),
    np.mean([mm[k].reshape(-1)[P].mean() for k in np.where(code[:n]==470)[0][:500]])))
