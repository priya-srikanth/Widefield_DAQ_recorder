"""Cross-register all 6/5-6/8 sessions to each animal's 6/6 reference (final standard).

PS92/PS93: 6/6 reference uses v2 landmarks -> recompute their 6/6 allen_aligned_affine8v1
  with v2, then re-align 6/5/6/7/6/8 to the v2 6/6 CCF.
PS94/PS95: 6/6 reference uses v1 (already done for 6/7/6/8) -> only align 6/5.
Emits/overwrites allen_aligned_affine8v1 per session on N (GPU reads these). reads U/mean
from N results. cross_day_align warp_u emits the LocaNMF/maps allen dir.
"""
import os, json, subprocess
PY = r"C:\ProgramData\anaconda3\envs\wfield\python.exe"
REPO = r"C:\Github\Widefield_DAQ_recorder"
NL = r"N:\MICROSCOPE\Priya\Widefield\labcams"
env = dict(os.environ, PYTHONPATH=REPO)

DAYS = {
 "PS92": {"0605":"PS92_20260605_125023","0606":"PS92_20260606_122451","0607":"PS92_20260607_121538","0608":"PS92_20260608_133759"},
 "PS93": {"0605":"PS93_20260605_174659","0606":"PS93_20260606_180117","0607":"PS93_20260607_174844","0608":"PS93_20260608_195203"},
 "PS94": {"0605":"PS94_20260605_142009","0606":"PS94_20260606_140854","0607":"PS94_20260607_140731","0608":"PS94_20260608_153651"},
 "PS95": {"0605":"PS95_20260605_163102","0606":"PS95_20260606_160806","0607":"PS95_20260607_155000","0608":"PS95_20260608_180943"},
}
LM      = {"PS92":"v2","PS93":"v2","PS94":"v1","PS95":"v1"}
REALIGN = {"PS92":["0605","0607","0608"],"PS93":["0605","0607","0608"],"PS94":["0605"],"PS95":["0605"]}

def results(an, day): return f"{NL}/20260{day}/{DAYS[an][day]}/motion_corrected/wfield_local_results".replace("\\","/")
def landmarks(an):
    return f"{NL}/20260606/{DAYS[an]['0606']}/raw_widefield_data/dorsal_cortex_landmarks_{LM[an]}.json".replace("\\","/")

def run(cmd):
    print("  $ "+" ".join(map(str,cmd)), flush=True)
    return subprocess.run([PY,"-m",*cmd], cwd=REPO, env=env).returncode

for an in ("PS92","PS93","PS94","PS95"):
    print(f"\n================ {an} (6/6 ref, {LM[an]} landmarks) ================", flush=True)
    ref_res = results(an,"0606").replace("/","\\"); lm = landmarks(an).replace("/","\\")
    # 1) recompute the 6/6 reference's own allen dir with the chosen landmarks (v2 for PS92/93)
    if LM[an] == "v2":
        print("  recompute 6/6 reference allen (v2)", flush=True)
        run(["wfield_local.apply_allen_transform", ref_res, "--landmarks", lm,
             "--output", ref_res + r"\allen_aligned_affine8v1"])
    # 2) cross-register the realign days to the 6/6 reference
    cfg = {"animal":an,"mode":"reference-native","func_channel":1,"reference":f"{an}_0606",
           "output":f"{NL}/xday/{an}_xall".replace("\\","/"),"warp_u":True,
           "sessions":{f"{an}_0606":{"results":results(an,"0606"),"landmarks":landmarks(an)}}}
    for day in REALIGN[an]:
        cfg["sessions"][f"{an}_{day}"] = {"results":results(an,day)}
    p = f"_xall_{an}.json"; json.dump(cfg, open(p,"w"), indent=2)
    if run(["wfield_local.cross_day_align", p]):
        print(f"  {an} cross_day FAILED", flush=True)
print("\nXREGISTER ALL DONE", flush=True)
