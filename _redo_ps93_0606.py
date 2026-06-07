import os, subprocess
PY=r"C:\ProgramData\anaconda3\envs\wfield\python.exe"; REPO=r"C:\Github\Widefield_DAQ_recorder"
mc=r"E:\labcams_data\20260606\PS93_20260606_180117\motion_corrected"
cp=mc+r"\pco_edge_run000_00000000_2_460_480_uint16_daq_led_cleanpairs_2_460_480_uint16.dat"
binp=mc+r"\motioncorrect_2_460_480_uint16.bin"; results=mc+r"\wfield_local_results"
env=dict(os.environ,PYTHONPATH=REPO)
def run(c):
    print("\n$"," ".join(c),flush=True); r=subprocess.run([PY,"-m",*c],cwd=REPO,env=env)
    if r.returncode: raise SystemExit(f"fail {c}")
# re-motion existing cleanpairs with the SIGN-FIXED motion correction (no relabel needed)
run(["wfield_local.run_wfield_motion", cp, "--output", mc, "--mode","2d"])
run(["wfield_local.run_wfield_local", binp, "--output", results, "-k","100","--functional-channel","1","--fs","31.23","--freq-highpass","0.1","--freq-lowpass","14.0"])
print("PS93 6/6 redo motion+SVD DONE",flush=True)
