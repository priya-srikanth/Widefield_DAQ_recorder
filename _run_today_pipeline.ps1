# Sequential motion-correction + SVD for today's (2026-06-03) recordings.
# Per session: TTL relabel (separate h5py process) -> motion correction -> SVD.
$ErrorActionPreference = "Continue"
$repo = "C:\Github\Widefield_DAQ_recorder"
$py = "C:\ProgramData\anaconda3\envs\wfield\python.exe"
$env:PYTHONPATH = $repo
Set-Location $repo

$jobs = @(
  @{ name = "PS92";
     dat = "E:\labcams_data\20260603\PS92\PS92_20260603_104008\raw_widefield_data\pco_edge_run000_00000000_2_477_464_uint16.dat";
     daq = "E:\DAQ_recorder_output\20250603\PS92_20260603_104607.h5";
     mc  = "E:\labcams_data\20260603\PS92\PS92_20260603_104008\motion_corrected" },
  @{ name = "PS94";
     dat = "E:\labcams_data\20260603\PS94\raw_widefield_data\pco_edge_run000_00000000_2_462_464_uint16.dat";
     daq = "E:\DAQ_recorder_output\20250603\PS94_20260603_175946.h5";
     mc  = "E:\labcams_data\20260603\PS94\motion_corrected" },
  @{ name = "PS95";
     dat = "E:\labcams_data\20260603\PS95\PS95\PS95_20260603_194442\raw_widefield_data\pco_edge_run000_00000000_2_462_464_uint16.dat";
     daq = "E:\DAQ_recorder_output\20250603\PS95_20260603_194902.h5";
     mc  = "E:\labcams_data\20260603\PS95\PS95\PS95_20260603_194442\motion_corrected" }
)

foreach ($j in $jobs) {
  Write-Output "================ $($j.name): relabel + motion correction  [$(Get-Date)] ================"
  & $py -m wfield_local.run_wfield_motion --daq-h5 $j.daq $j.dat --output $j.mc --relabel-mode rescue
  if ($LASTEXITCODE -ne 0) { Write-Output "!!! $($j.name) motion FAILED (exit $LASTEXITCODE); skipping SVD"; continue }
  $bin = Get-ChildItem $j.mc -Filter "motioncorrect_*.bin" -ErrorAction SilentlyContinue | Select-Object -First 1
  if (-not $bin) { Write-Output "!!! $($j.name) no motioncorrect .bin found; skipping SVD"; continue }
  Write-Output "================ $($j.name): SVD + hemodynamic correction  [$(Get-Date)] ================"
  & $py -m wfield_local.run_wfield_local $bin.FullName --output "$($j.mc)\wfield_local_results"
  if ($LASTEXITCODE -ne 0) { Write-Output "!!! $($j.name) SVD FAILED (exit $LASTEXITCODE)"; continue }
  Write-Output "================ $($j.name) COMPLETE  [$(Get-Date)] ================"
}
Write-Output "ALL DONE  [$(Get-Date)]"
