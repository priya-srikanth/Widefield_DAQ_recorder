import os, subprocess
PY=r"C:\ProgramData\anaconda3\envs\wfield\python.exe"; REPO=r"C:\Github\Widefield_DAQ_recorder"; NL=r"N:\MICROSCOPE\Priya\Widefield\labcams\20260806"
S={"PS92":"PS92_20260806_124426","PS93":"PS93_20260806_172253","PS94":"PS94_20260806_074320","PS95":"PS95_20260806_105300"}
env=dict(os.environ,PYTHONPATH=REPO)
for a,sess in S.items():
    mc=fr"{NL}\{sess}\motion_corrected"; lab=f"{a}_0806_affine8v1"
    r=subprocess.run([PY,"-m","wfield_local.qc_motion_correction","--motion-dir",mc,"--label",lab,"--output",fr"{mc}\motion_qc"],cwd=REPO,env=env)
    print(f"{a}: {'OK' if r.returncode==0 else 'FAIL'}",flush=True)
print("QC0806 DONE",flush=True)
