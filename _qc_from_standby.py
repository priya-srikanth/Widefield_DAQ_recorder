"""Regenerate motion-correction QC figures (with bottom example images) for dates whose movies
were cleaned from E:. Reads the corrected .bin AND raw .dat from STANDBY, shifts + output on N:.
Auto-discovers sessions. Usage: python _qc_from_standby.py 20260605 20260606 20260607 20260608"""
import os, sys, glob, subprocess
PY=r"C:\ProgramData\anaconda3\envs\wfield\python.exe"; REPO=r"C:\Github\Widefield_DAQ_recorder"
NL=r"N:\MICROSCOPE\Priya\Widefield\labcams"
STB=r"\\standby.files.med.harvard.edu\hms\neurobio\sabatini\collaborations\Priya\Widefield\labcams"
ANIMALS=["PS92","PS93","PS94","PS95"]
def run(c):
    env=dict(os.environ,PYTHONPATH=REPO); print("$ ...qc "+os.path.basename(c[c.index("--label")+1]),flush=True)
    return subprocess.run([PY,"-m",*c],cwd=REPO,env=env).returncode
for dd in sys.argv[1:]:
    for an in ANIMALS:
        for s in sorted(glob.glob(fr"{NL}\{dd}\{an}_*")):
            mc=fr"{s}\motion_corrected"
            if not os.path.exists(fr"{mc}\motion_correction_shifts.npy"): continue
            sess=os.path.basename(s); lab=f"{an}_{dd[4:]}_affine8v1"
            bins=glob.glob(fr"{STB}\{dd}\{sess}\motion_corrected\motioncorrect_*.bin")
            raws=[r for r in glob.glob(fr"{STB}\{dd}\{sess}\raw_widefield_data\*uint16.dat") if "cleanpairs" not in r]
            if not bins: print(f"  [skip] no standby bin for {sess}",flush=True); continue
            cmd=["wfield_local.qc_motion_correction","--motion-dir",mc,"--label",lab,
                 "--output",fr"{mc}\motion_qc","--cor-bin",bins[0]]
            if raws: cmd+=["--raw-dat",raws[0]]
            if run(cmd): print(f"  FAIL {lab}",flush=True)
            else: print(f"  OK {lab}",flush=True)
print("QC-FROM-STANDBY DONE",flush=True)
