"""Regenerate 8/6 + 8/7 motion-correction QC figures WITH the bottom example images
(raw mean / corrected mean / corrected temporal std). The earlier runs pointed --motion-dir
at N: (which never holds the .bin), so those panels were blank. 8/7 has both movies on E:;
8/6's .bin was cleaned from E: -> read it from standby via --cor-bin. Writes to N: motion_qc."""
import os, subprocess
PY=r"C:\ProgramData\anaconda3\envs\wfield\python.exe"; REPO=r"C:\Github\Widefield_DAQ_recorder"
NL=r"N:\MICROSCOPE\Priya\Widefield\labcams"
STB=r"\\standby.files.med.harvard.edu\hms\neurobio\sabatini\collaborations\Priya\Widefield\labcams"
BIN="motioncorrect_2_460_480_uint16.bin"
DAYS={
 "20260807":{"PS92":"PS92_20260807_150924","PS93":"PS93_20260807_174403","PS94":"PS94_20260807_104125","PS95":"PS95_20260807_124637"},
 "20260806":{"PS92":"PS92_20260806_124426","PS93":"PS93_20260806_172253","PS94":"PS94_20260806_074320","PS95":"PS95_20260806_105300"},
}
def run(c):
    env=dict(os.environ,PYTHONPATH=REPO); print("\n$ "+" ".join(map(str,c[:6]))+" ...",flush=True)
    if subprocess.run([PY,"-m",*c],cwd=REPO,env=env).returncode: raise SystemExit(f"fail {c}")
for date,sess in DAYS.items():
    for an,s in sess.items():
        lab=f"{an}_{date[4:]}_affine8v1"
        emc=fr"E:\labcams_data\{date}\{s}\motion_corrected"   # has shifts + cleanpairs .dat
        out=fr"{NL}\{date}\{s}\motion_corrected\motion_qc"
        cmd=["wfield_local.qc_motion_correction","--motion-dir",emc,"--label",lab,"--output",out]
        if date=="20260806":                                  # .bin cleaned from E: -> use standby copy
            cmd+=["--cor-bin",fr"{STB}\{date}\{s}\motion_corrected\{BIN}"]
        print(f"===== {lab} =====",flush=True); run(cmd)
print("\nQC EXAMPLE-IMAGE FIX DONE",flush=True)
