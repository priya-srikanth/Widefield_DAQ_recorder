import os, subprocess
PY=r"C:\ProgramData\anaconda3\envs\wfield\python.exe"; REPO=r"C:\Github\Widefield_DAQ_recorder"
E=r"E:\labcams_data"; NL=r"N:\MICROSCOPE\Priya\Widefield\labcams"
S=[("20260606","PS92_20260606_122451"),("20260606","PS94_20260606_140854"),("20260606","PS95_20260606_160806"),
   ("20260605","PS94_20260605_142009"),("20260605","PS95_20260605_163102")]
env=dict(os.environ,PYTHONPATH=REPO)
for date,sess in S:
    lab=f"{sess[:4]}_{date[4:]}_affine8v1"; emc=fr"{E}\{date}\{sess}\motion_corrected"; nqc=fr"{NL}\{date}\{sess}\motion_corrected\motion_qc"
    print(f"\n=== QC {lab} (motion-dir=E, out=N) ===",flush=True)
    r=subprocess.run([PY,"-m","wfield_local.qc_motion_correction","--motion-dir",emc,"--label",lab,"--output",nqc],cwd=REPO,env=env)
    print(f"  {'OK' if r.returncode==0 else 'FAIL'}",flush=True)
print("\nQC5 DONE",flush=True)
