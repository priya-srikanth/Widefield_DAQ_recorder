"""Resumable batch: re-process sessions with the SIGN-FIXED motion correction.

Newest -> oldest. Per session: fixed 2D motion -> SVD -> Allen -> copy non-bin
outputs to N: MICROSCOPE + corrected .bin to M: standby (overwriting the buggy
ones). Maps are NOT regenerated here (negligible change on these <1.6px sessions);
PS93 6/6 and PS94 6/5 are handled by their dedicated drivers (they also need
maps/deck/cross-day) and are skipped here.

Source per session (auto-detected):
  - E: cleanpairs present  -> re-motion in place (fast, local), keep E session.
  - else M: raw            -> rebuild to an E working dir (relabel rescue + motion),
                              then DELETE the transient E rebuild dir after transfer
                              (it is reproducible; the original E was already cleaned).
  - 2026-06-01: regime A (full-FOV, no relabel).

Resumable: state in _redo_motion_state.json; re-running skips sessions already done.
Pausable: TaskStop; re-launch to continue. Updates MOTION_REDO_STATUS.md.
"""
import os, re, sys, glob, json, shutil, subprocess, datetime

PY = r"C:\ProgramData\anaconda3\envs\wfield\python.exe"
REPO = r"C:\Github\Widefield_DAQ_recorder"
E = r"E:\labcams_data"
NL = r"N:\MICROSCOPE\Priya\Widefield\labcams"
ND = r"N:\MICROSCOPE\Priya\Widefield\DAQ_recorder_output"
MROOT = r"M:\Widefield\labcams"
STATE = os.path.join(REPO, "_redo_motion_state.json")
SKIP = {"PS93_20260606_180117", "PS94_20260605_142009"}  # dedicated drivers (maps/deck)
DAT_RE = re.compile(r"_(\d+)_(\d+)_(\d+)_uint16")


def _run(cmd):
    env = dict(os.environ, PYTHONPATH=REPO)
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([PY, "-m", *cmd], cwd=REPO, env=env).returncode


def _state():
    return json.load(open(STATE)) if os.path.exists(STATE) else {}


def _save_state(s):
    json.dump(s, open(STATE, "w"), indent=2)


def discover():
    out = []
    for date in sorted(os.listdir(NL), reverse=True):
        dp = os.path.join(NL, date)
        if not (os.path.isdir(dp) and re.fullmatch(r"\d{8}", date)):
            continue
        for sess in sorted(os.listdir(dp)):
            if not os.path.isdir(os.path.join(dp, sess)):
                continue
            m = re.match(r"(PS\d+)", sess)
            if m:
                out.append((date, sess, m.group(1)))
    return out


def _landmarks(date, sess):
    for c in (glob.glob(os.path.join(NL, date, sess, "raw_widefield_data", "dorsal_cortex_landmarks_v1.json")) +
              glob.glob(os.path.join(E, date, sess, "raw_widefield_data", "dorsal_cortex_landmarks_v1.json")) +
              glob.glob(os.path.join(E, date, sess, "dorsal_cortex_landmarks_v1.json"))):
        if os.path.exists(c):
            return c
    return None


def _copy_outputs(mc, date, sess):
    """non-bin motion_corrected outputs -> N (mirror); corrected .bin -> M (mirror)."""
    n_mc = os.path.join(NL, date, sess, "motion_corrected")
    m_mc = os.path.join(MROOT, date, sess, "motion_corrected")
    nbin = mbin = 0
    for root, _d, files in os.walk(mc):
        rel = os.path.relpath(root, mc)
        for f in files:
            s = os.path.join(root, f)
            if f.endswith(".dat"):
                continue  # cleanpairs intermediate: neither N nor M
            if f.startswith("motioncorrect_") and f.endswith(".bin"):
                d = os.path.join(m_mc, rel, f) if rel != "." else os.path.join(m_mc, f)
                os.makedirs(os.path.dirname(d), exist_ok=True)
                if not (os.path.exists(d) and os.path.getsize(d) == os.path.getsize(s)):
                    shutil.copy2(s, d)
                mbin += 1
            else:
                d = os.path.join(n_mc, rel, f) if rel != "." else os.path.join(n_mc, f)
                os.makedirs(os.path.dirname(d), exist_ok=True)
                if not (os.path.exists(d) and os.path.getsize(d) == os.path.getsize(s)):
                    shutil.copy2(s, d)
                nbin += 1
    return nbin, mbin


def process(date, sess, animal, state):
    mc = os.path.join(E, date, sess, "motion_corrected")
    results = os.path.join(mc, "wfield_local_results")
    allen = os.path.join(results, "allen_aligned_affine8v1")
    ecp = glob.glob(os.path.join(mc, "*cleanpairs*uint16.dat"))
    mraw = glob.glob(os.path.join(MROOT, date, sess, "raw_widefield_data", "*uint16.dat"))
    rebuild = not ecp
    if ecp:
        src_dat = ecp[0]; relabel = None
    elif mraw:
        src_dat = mraw[0]; relabel = None if date == "20260601" else "rescue"
    else:
        print(f"  NO SOURCE for {sess}; skipping"); return "no-source"
    nch, h, w = DAT_RE.search(os.path.basename(src_dat)).groups()
    binp = os.path.join(mc, f"motioncorrect_{nch}_{h}_{w}_uint16.bin")
    daq = (glob.glob(os.path.join(ND, "*", f"{animal}_{date}_*.h5")) +
           glob.glob(os.path.join(ND, f"{animal}_{date}_*.h5")))
    lm = _landmarks(date, sess)

    os.makedirs(mc, exist_ok=True)
    for stale in (results,):  # force SVD/Allen recompute from the new bin
        if os.path.isdir(stale):
            shutil.rmtree(stale)
    # 1. fixed motion
    cmd = ["wfield_local.run_wfield_motion", src_dat, "--output", mc, "--mode", "2d"]
    if relabel:
        if not daq:
            print(f"  no DAQ for {sess}; cannot relabel"); return "no-daq"
        cmd += ["--daq-h5", daq[0], "--relabel-mode", relabel]
    if _run(cmd):
        return "motion-fail"
    # 2. SVD
    if _run(["wfield_local.run_wfield_local", binp, "--output", results, "-k", "100",
             "--functional-channel", "1", "--fs", "31.23", "--freq-highpass", "0.1", "--freq-lowpass", "14.0"]):
        return "svd-fail"
    # 3. Allen
    if lm:
        if _run(["wfield_local.apply_allen_transform", results, "--landmarks", lm, "--output", allen]):
            print(f"  Allen failed for {sess}")
    else:
        print(f"  no landmarks for {sess}; skipping Allen")
    # 4. transfer
    nN, nM = _copy_outputs(mc, date, sess)
    print(f"  transferred: {nN} outputs->N, {nM} bin->M", flush=True)
    # 5. transient E rebuild dir cleanup (reproducible; original E was already cleaned)
    if rebuild:
        shutil.rmtree(os.path.join(E, date, sess), ignore_errors=True)
        print(f"  cleaned transient E rebuild dir for {sess}", flush=True)
    return "done"


def write_status_md(state):
    p = os.path.join(REPO, "MOTION_REDO_STATUS.md")
    lines = ["# Motion-correction redo status (auto-updated)\n",
             f"Last update: {datetime.datetime.now().isoformat(timespec='seconds') if False else '(stamped externally)'}\n",
             "| date | session | status |", "|---|---|---|"]
    for (date, sess, _an) in discover():
        st = "UPDATED + bin->standby" if state.get(sess) == "done" else (
            "dedicated driver (PS93 6/6 / PS94 6/5)" if sess in SKIP else "OLD (pending)")
        lines.append(f"| {date} | {sess} | {st} |")
    open(p, "w").write("\n".join(lines) + "\n")


def main():
    only = sys.argv[1:]  # optionally restrict to given session dir names
    state = _state()
    for date, sess, animal in discover():
        if sess in SKIP:
            continue
        if only and sess not in only:
            continue
        if state.get(sess) == "done":
            print(f"[skip done] {sess}", flush=True); continue
        print(f"\n================ {date} {sess} ({animal}) ================", flush=True)
        r = process(date, sess, animal, state)
        state[sess] = r
        _save_state(state); write_status_md(state)
        print(f"  -> {r}", flush=True)
    print("\nBATCH DONE:", {k: v for k, v in state.items()}, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
