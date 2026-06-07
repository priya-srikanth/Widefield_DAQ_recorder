"""Finish a redone session: FORCE-copy non-bin outputs -> N and corrected .bin -> M
(force, because corrected files match the buggy ones in size), update redo status.
Usage: python _finish_redo_session.py <date> <session>
"""
import os, sys, shutil, json
E=r"E:\labcams_data"; NL=r"N:\MICROSCOPE\Priya\Widefield\labcams"; MROOT=r"M:\Widefield\labcams"
date, sess = sys.argv[1], sys.argv[2]
mc=os.path.join(E,date,sess,"motion_corrected")
n_mc=os.path.join(NL,date,sess,"motion_corrected"); m_mc=os.path.join(MROOT,date,sess,"motion_corrected")
nN=nM=0
for root,_d,files in os.walk(mc):
    rel=os.path.relpath(root,mc)
    for f in files:
        s=os.path.join(root,f)
        if f.endswith(".dat"): continue  # cleanpairs intermediate
        if f.startswith("motioncorrect_") and f.endswith(".bin"):
            d=os.path.join(m_mc,rel,f) if rel!="." else os.path.join(m_mc,f); nM+=1
        else:
            d=os.path.join(n_mc,rel,f) if rel!="." else os.path.join(n_mc,f); nN+=1
        os.makedirs(os.path.dirname(d),exist_ok=True)
        shutil.copy2(s,d)  # FORCE overwrite (corrected == buggy in size)
        if os.path.getsize(d)!=os.path.getsize(s): print("SIZE MISMATCH",d)
print(f"{sess}: FORCE-copied {nN} outputs->N, {nM} bin->M")
st_path=os.path.join(r"C:\Github\Widefield_DAQ_recorder","_redo_motion_state.json")
st=json.load(open(st_path)) if os.path.exists(st_path) else {}
st[sess]="done"; json.dump(st,open(st_path,"w"),indent=2)
print("marked done in state")
