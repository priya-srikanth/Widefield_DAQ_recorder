"""Byte-identical verify E videos against M: standby, then delete from E.
Streaming 16MB compare (short-circuits on mismatch). Deletes a session's raw .dat +
corrected .bin + reproducible cleanpairs .dat ONLY if raw AND bin are byte-identical on M.
Usage: python _verify_delete_videos.py <DATE>   (e.g. 20260608)
"""
import os, sys, glob
DATE=sys.argv[1]; E=fr"E:\labcams_data\{DATE}"; M=fr"M:\Widefield\labcams\{DATE}"
def byteq(a,b,chunk=1<<24):
    if os.path.getsize(a)!=os.path.getsize(b): return False
    with open(a,"rb") as fa, open(b,"rb") as fb:
        while True:
            x=fa.read(chunk); y=fb.read(chunk)
            if x!=y: return False
            if not x: return True
freed=0
for sess in sorted(os.listdir(E)):
    se=os.path.join(E,sess)
    if not os.path.isdir(se): continue
    eraw=glob.glob(os.path.join(se,"raw_widefield_data","*uint16.dat"))
    ebin=glob.glob(os.path.join(se,"motion_corrected","motioncorrect_*uint16.bin"))
    ecp =glob.glob(os.path.join(se,"motion_corrected","*cleanpairs*uint16.dat"))
    if not (eraw and ebin): print(f"{sess[:4]}: missing raw/bin on E (skip)"); continue
    mraw=os.path.join(M,sess,"raw_widefield_data",os.path.basename(eraw[0]))
    mbin=os.path.join(M,sess,"motion_corrected",os.path.basename(ebin[0]))
    if not (os.path.exists(mraw) and os.path.exists(mbin)):
        print(f"{sess[:4]}: raw/bin NOT yet on M (skip)"); continue
    print(f"{sess[:4]}: byte-verifying raw...",flush=True); rok=byteq(eraw[0],mraw)
    print(f"{sess[:4]}: byte-verifying bin...",flush=True); bok=byteq(ebin[0],mbin)
    if rok and bok:
        for p in eraw+ebin+ecp:
            g=os.path.getsize(p); os.remove(p); freed+=g
        print(f"{sess[:4]}: BYTE-IDENTICAL on M -> deleted raw+bin+cleanpairs ({(os.path.getsize(mraw)+os.path.getsize(mbin))/1e9:.0f}GB videos +cp)")
    else:
        print(f"{sess[:4]}: MISMATCH (raw_ok={rok} bin_ok={bok}) -> KEPT")
print(f"\nfreed ~{freed/1e9:.0f} GB from E")
