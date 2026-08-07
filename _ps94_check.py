import h5py, numpy as np, os, datetime

DAQ = r"E:\DAQ_recorder_output\20250603\PS94_20260603_175946.h5"
CAMDIR = r"E:\labcams_data\20260603\PS94\raw_widefield_data"
CAMLOG = os.path.join(CAMDIR, "pco_edge_run000_00000000.camlog")

with h5py.File(DAQ, "r") as f:
    fs = float(f.attrs["sample_rate_hz"])
    an = [s.decode() for s in f["analog/channel_names"][:]]
    di = [s.decode() for s in f["digital/channel_names"][:]]
    packed = f["digital/packed_samples"][:, 0]
    scale = f["analog/int16_scale_volts_per_count"][:]
    off = f["analog/int16_offset_volts"][:]
    def acol(name):
        i = an.index(name)
        return f["analog/samples_int16"][:, i].astype(np.float32) * scale[i] + off[i]
    led415 = acol("led415_ttl"); led470 = acol("led470_ttl")
    trial_end = acol("trial_end") if "trial_end" in an else None
N = packed.shape[0]
print("dur %.1f s  fs %g  analog %s" % (N/fs, fs, an))

def dbit(b): return ((packed >> b) & 1)
def rises(x, thr=0.5):
    b = (np.asarray(x) > thr).astype(np.int8); return np.flatnonzero(np.diff(b) == 1) + 1
pco_r = rises(dbit(di.index("pco_exposure")))
ts_r = rises(dbit(di.index("trial_start")))
te_r = rises(trial_end, 1.5) if trial_end is not None else np.array([])
print("pco pulses=%d  trial_start=%d  trial_end=%d" % (len(pco_r), len(ts_r), len(te_r)))

# LED at each exposure (sample ~2ms into exposure)
idx = np.clip(pco_r + int(0.002*fs), 0, N-1)
s415 = led415[idx] > 1.5; s470 = led470[idx] > 1.5
code = np.where(s415 & ~s470, 415, np.where(s470 & ~s415, 470, np.where(s415 & s470, 3, 0)))
print("exposures: 415=%d 470=%d both=%d dark=%d" % (int((code==415).sum()),int((code==470).sum()),int((code==3).sum()),int((code==0).sum())))

# pair trial windows
st, en = pco_r*0.0, None  # placeholder
windows = []
for s in ts_r:
    later = te_r[te_r > s]
    if len(later): windows.append((s, later[0]))
print("paired trials: %d" % len(windows))

# frames per trial + parity + first/last LED
fpt = []; first_leds = []; last_leds = []; oddcount = 0
for (a, b) in windows:
    m = (pco_r >= a) & (pco_r <= b)
    n = int(m.sum()); fpt.append(n)
    if n == 0: continue
    cc = code[m]
    first_leds.append(int(cc[0])); last_leds.append(int(cc[-1]))
    if n % 2 == 1: oddcount += 1
fpt = np.array(fpt)
if len(fpt):
    print("frames/trial: min=%d max=%d mean=%.1f  ODD trials=%d/%d" % (fpt.min(),fpt.max(),fpt.mean(),oddcount,len(fpt)))
    from collections import Counter
    print("  first LED per trial:", Counter(first_leds))
    print("  last  LED per trial:", Counter(last_leds))

# GLOBAL parity consistency (what the live preview keys on): is even global
# exposure index consistently one wavelength?
illum_mask = code != 0
ill = code[illum_mask]
ev = ill[0::2]; od = ill[1::2]
def frac(a, v): return float(np.mean(a == v)) if len(a) else 0
print("GLOBAL parity over illuminated frames: even->415 %.3f even->470 %.3f | odd->415 %.3f odd->470 %.3f"
      % (frac(ev,415),frac(ev,470),frac(od,415),frac(od,470)))
# consecutive-same among illuminated (alternation breaks)
print("consecutive-same-LED among illuminated: %d / %d" % (int(np.sum(ill[1:]==ill[:-1])), len(ill)))

# camlog burst sizes (frames between gaps) parity
if os.path.exists(CAMLOG):
    ts=[];
    for line in open(CAMLOG):
        line=line.strip()
        if not line or line.startswith("#"): continue
        p=line.split(",",1)
        try: ts.append(datetime.datetime.fromisoformat(p[1]))
        except: pass
    if len(ts)>1:
        dt=np.array([(ts[i]-ts[i-1]).total_seconds()*1000 for i in range(1,len(ts))])
        med=np.median(dt); gaps=np.flatnonzero(dt>1.8*med)
        starts=np.r_[0,gaps+1]; stops=np.r_[gaps,len(ts)-1]
        bursts=stops-starts+1
        print("camlog: frames=%d bursts=%d burst sizes: min=%d max=%d ODD bursts=%d/%d"
              %(len(ts),len(bursts),bursts.min(),bursts.max(),int(np.sum(bursts%2==1)),len(bursts)))
else:
    print("camlog not found:", CAMLOG)
