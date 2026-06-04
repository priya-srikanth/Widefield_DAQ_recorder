# GPU LocaNMF kickoff (run on the NVIDIA machine)

Use this to start a Claude Code session **on the GPU computer** and run LocaNMF on
the Allen-aligned sessions. LocaNMF needs an NVIDIA GPU + PyTorch + the `locanmf`
package; this repo's `wfield_local/run_locanmf.py` does the rest.

The analyzed data is on the **MICROSCOPE** server at
`N:\MICROSCOPE\Priya\Widefield\labcams\<date>\<session>\motion_corrected\` (SVD,
alignment, maps), copied from the rig PC. The GPU machine must be able to mount that
share (the `\\research.files.med.harvard.edu\Neurobio` UNC; map it to `N:` or use the
UNC path directly).

## SAFETY (read first — no exceptions)
- **NEVER delete anything on `N:\MICROSCOPE\...`** without explicit per-action OK.
- **Only ever write inside `N:\MICROSCOPE\Priya\`** — never any other person's folder.
- LocaNMF outputs go to a new `...\motion_corrected\locanmf_affine8v1\` folder
  (copy/create only).

## 1. Bootstrap — fetch the latest repo
PowerShell:
```powershell
$repo = "C:\Github\Widefield_DAQ_recorder"
if (Test-Path $repo) { git -C $repo pull } else {
  git clone https://github.com/priya-srikanth/Widefield_DAQ_recorder.git $repo
}
nvidia-smi    # confirm GPU + driver (>418.x); note the CUDA version shown
```
bash:
```bash
repo=~/Widefield_DAQ_recorder
[ -d "$repo" ] && git -C "$repo" pull || git clone https://github.com/priya-srikanth/Widefield_DAQ_recorder.git "$repo"
nvidia-smi
```

## 2. Environment (PyTorch + locanmf)
`run_locanmf.py` imports `wfield.local_nmf.compute_locaNMF`, which needs `torch`
(PyTorch) + `locanmf` + `wfield`. PyTorch *is* the `torch` package. Two options:

- **Easiest (prebuilt, Python 3.6):**
  ```
  conda create -n locanmf python=3.6 locanmf -c ss5513 -c pytorch
  conda activate locanmf
  pip install wfield        # if it won't resolve on 3.6, see note in run_locanmf.py header
  ```
- **Newer Python (build from source against your CUDA):**
  ```
  conda create -n locanmf python=3.10 && conda activate locanmf
  python -c "import urllib"   # then install a torch build matching nvidia-smi's CUDA:
  pip install torch --index-url https://download.pytorch.org/whl/cu121   # pick cuXXX
  git clone https://github.com/ikinsella/locaNMF && pip install ./locaNMF   # compiles CUDA/C++
  pip install wfield
  ```
Verify CUDA is seen:
```
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```
(There is no maintained newer-Python *prebuilt* locanmf; the newer-Python path
compiles the extension from source.)

## 3. Run LocaNMF (start with ONE session to validate)
```powershell
$env:PYTHONPATH = "C:\Github\Widefield_DAQ_recorder"
python -m wfield_local.run_locanmf `
  --allen-dir "N:\MICROSCOPE\Priya\Widefield\labcams\20260603\PS94_20260603\motion_corrected\wfield_local_results\allen_aligned_affine8v1" `
  --label PS94_0603 `
  --output  "N:\MICROSCOPE\Priya\Widefield\labcams\20260603\PS94_20260603\motion_corrected\locanmf_affine8v1" `
  --maxrank 20 --loc-thresh 70 --r2-thresh 0.99 --device auto
```
Inputs it reads (already present in each `allen_aligned_affine8v1` + its results dir):
`U_atlas.npy`, `allen_area_atlas_native_grid.npy`, `allen_brain_mask_native_grid.npy`,
`SVTcorr.npy`. Outputs: `*_locanmf_A.npy` (H,W,ncomp spatial), `*_locanmf_C.npy`
(ncomp,T temporal), `*_locanmf_regions.npy`, a `*_locanmf_components.png` montage,
and a summary JSON. Check the montage; tune `--maxrank` / `--loc-thresh` if components
look over/under-split, then run the remaining sessions (PS92/PS94/PS95 6/1–6/3).

## 4. Paste-ready prompt for the GPU-machine Claude session
> I'm on the NVIDIA/GPU machine. Read `GPU_LOCANMF_KICKOFF.md` and
> `wfield_local/run_locanmf.py` in https://github.com/priya-srikanth/Widefield_DAQ_recorder
> (clone/pull it first). Then: (1) verify the GPU with `nvidia-smi` and set up a Python
> env with PyTorch matching this machine's CUDA, the `locanmf` package, and `wfield`,
> and confirm `torch.cuda.is_available()` is True. (2) The Allen-aligned widefield data
> is on `N:\MICROSCOPE\Priya\Widefield\labcams\<date>\<session>\motion_corrected\` —
> confirm the share is reachable. (3) Run `wfield_local.run_locanmf` on ONE session
> (PS94 6/3) first, show me the component montage, and wait for my OK before doing the
> rest. SAFETY: never delete anything on the MICROSCOPE/N: server and only ever write
> inside `N:\MICROSCOPE\Priya\` — output to a new `locanmf_affine8v1\` subfolder.

## Why LocaNMF (vs our current SVD)
Our rig pipeline stops at SVD + hemo + Allen alignment. LocaNMF re-factorizes that
low-rank data into non-negative, atlas-localized components (multiple per region) that
are reproducible across sessions/animals — the basis for cross-animal and functional-
subnetwork analysis (Saxena et al. 2020; used in PMC9991922). See `DECISIONS.md`.
