# GPU LocaNMF run log — RTX 4060 box (2026-06-04)

What actually happened executing `GPU_LOCANMF_KICKOFF.md` on the NVIDIA machine, with
the decisions and fixes needed to make `wfield_local/run_locanmf.py` run. Read this
alongside the kickoff; it records the deltas from the idealized instructions.

## Machine
- GPU: **NVIDIA GeForce RTX 4060, 8 GB** (Ada, sm_89). Driver 581.95, supports CUDA 13.0.
- A standalone CUDA 12.1 toolkit (`nvcc`) is installed; VS 2022 **Community is installed
  WITHOUT the "Desktop development with C++" workload** (no usable `cl.exe`/`vcvars64`).

## Share / drive mapping (IMPORTANT, differs from kickoff)
- On this machine the MICROSCOPE / research share (`\\research.files.med.harvard.edu\Neurobio`)
  is mounted at **`M:`**, not `N:`. So data paths are **`M:\MICROSCOPE\Priya\Widefield\...`**.
- As of this run only **`labcams\20260601`** is present on the share (sessions
  `PS94_20260601_141614`, `PS95_20260601_153653`). The 6/2 and 6/3 sessions named in the
  kickoff are **not yet copied** here. Validation was therefore run on **6/1 PS94** (full-FOV
  540×640, regime A) instead of 6/3 PS94 — per-author decision.

## Environment that worked (one env: torch+CUDA + locanmf + wfield)
Took the newer-Python "build from source" path (not the py3.6 prebuilt), because our
`wfield` exposes `compute_locaNMF` and keeps everything in one env.

```powershell
conda create -n locanmf python=3.10 -y
$py = "C:\Users\sabatini\.conda\envs\locanmf\python.exe"
& $py -m pip install torch --index-url https://download.pytorch.org/whl/cu124   # torch 2.6.0+cu124
& $py -m pip install wfield                                                     # wfield 0.6.0 (pulls numpy 2.2.6 etc.)
# locaNMF from source, PURE-PYTHON (no --with-extension -> no CUDA C++ compile):
git clone https://github.com/ikinsella/locaNMF C:\Users\sabatini\source\locaNMF_src
$env:CONDA_PREFIX = "C:\Users\sabatini\.conda\envs\locanmf"   # setup.py reads os.environ["CONDA_PREFIX"]
& $py -m pip install --no-build-isolation C:\Users\sabatini\source\locaNMF_src
```
Verified: `torch 2.6.0+cu124`, `torch.cuda.is_available()=True`, device `RTX 4060`;
`wfield.local_nmf.compute_locaNMF` importable; `locanmf.demix.use_cuhals=False`.

Two install gotchas:
- `setup.py` does `os.environ["CONDA_PREFIX"]` → **export `CONDA_PREFIX`** in the (non-activated)
  shell or metadata generation `KeyError`s.
- Use `--no-build-isolation` so the build sees the env's `torch` (setup.py imports
  `torch.utils.cpp_extension` at top level).

## locanmf needed 3 torch-compatibility patches (modern torch breaks the 2020 code)
locanmf 1.0 predates several torch API tightenings; without these it raises mid-run.
Captured in **`wfield_local/locanmf_torch_compat.patch`** (apply to a fresh
`ikinsella/locaNMF` clone, then reinstall). All three are mechanical and do NOT change the
algorithm/results:
1. `demix.py` `set_from_regions`: `masked_scatter_` requires a **bool** mask; `support` is
   `uint8` → `.bool()` the mask. (torch ≥1.2)
2. `factor.py` `permute_`: `torch.index_select(x, ..., out=x)` aliases input→output, now
   forbidden → `x.copy_(torch.index_select(x, ...))` (preserves the `.set_` storage tricks).
   (torch ≥1.5)
3. `demix.py` `TemporalLS._temporal_update`: `torch.inverse(x, out=x)` aliases input→output
   → `x.copy_(torch.inverse(x))`. (torch ≥1.5)

## cuhals (CUDA HALS kernel): NOT built — deferred
- `locanmf` runs fine without the optional `cuhals` extension: `demix.py` does
  `try: import cuhals / except ImportError: use_cuhals=False` and falls back to a native
  PyTorch HALS that **still runs on the GPU**.
- **Does cuhals change the computation? No.** `cuhals.update()` and `native_update()` take
  identical args and implement the same HALS coordinate-descent sweep; cuhals' batching only
  reorganizes the work (precompute cross-batch residual via one `addmm`, then sequential
  within-batch sweep), parallelizing over *pixels*, not component order. Same fixed point,
  same components — differences are floating-point-rounding level only. It is purely a speed
  optimization.
- **Why not built here:** compiling the CUDA C++ extension needs a host MSVC compiler
  (`cl.exe`), which is absent (no VS C++ workload). It also needs `mkl.h` (not in the env)
  and a Windows flag port (`-fopenmp` → `/openmp` in `setup.py`). Building requires installing
  the MSVC C++ Build Tools first. Revisit only if the native-PyTorch runtime is too slow.

## Performance note
Pure-PyTorch HALS on a full session is slow on this box: 6/1 PS94 (U 540×640×100,
SVTcorr 100×**90,046** frames, `--maxrank 20 --loc-thresh 70 --r2-thresh 0.99`) ran well past
an hour at a steady ~80–88 % GPU / one saturated CPU core (kernel-launch bound). Levers if
needed: build `cuhals`, lower `--maxrank` (e.g. 12–15), or subsample frames. The large frame
count (T≈90 k, ~48 min recording) is the main driver.

## Component montages: region-ordered + energy-ranked
`run_locanmf.py` writes two montages of the same components:
- `<tag>_components.png` — ordered by **Allen region** (seed label); good for anatomy.
- `<tag>_components_byenergy.png` — ordered by **descending component energy**
  `||A_i||_F * ||C_i||_2` (importance rank), strongest first. Emitted via
  `wfield_local/montage_by_energy.py`, which is pure post-processing of the saved
  `A`/`C`/`regions` (no recompute) and can backfill existing dirs:
  `python -m wfield_local.montage_by_energy <output_dir> <tag>`.
The raw component array (and `regions.npy`) stays region-grouped, ascending by seed
label; the energy view is just a re-plot, it does not reorder the saved arrays.

## Runtime / launch tip
Launch the run so stdout streams to a log (e.g. `*> run.log 2>&1`), **not** through
PowerShell `| Out-String`, which buffers all of locanmf's per-rank-iteration progress prints
until the process exits (no live progress / ETA).
