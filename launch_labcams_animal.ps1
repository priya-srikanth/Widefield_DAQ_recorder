# Launch labcams (acquire-enable trial-gated) with a PER-ANIMAL config.
#
# Usage:
#   .\launch_labcams_animal.ps1 PS92
#
# First run for an animal clones the template config to labcams\animals\<ANIMAL>.json
# (with any ROI stripped, so it starts full-frame). Set that animal's ROI once via
# the Camera Crop/ROI dock; it is written back into the animal's JSON and persists,
# so the same animal gets the identical ROI day to day. Subsequent runs reuse it.

param([Parameter(Mandatory = $true)][string]$Animal)

$ErrorActionPreference = "Stop"
$repo = "C:\Github\Widefield_DAQ_recorder"
$template = Join-Path $repo "labcams\labcams_widefield_pco_trial_gated_acquire_enable.json"
$animalDir = Join-Path $repo "labcams\animals"
$cfg = Join-Path $animalDir "$Animal.json"
$python = "C:\ProgramData\anaconda3\envs\labcams\python.exe"

if (-not (Test-Path $animalDir)) { New-Item -ItemType Directory -Path $animalDir | Out-Null }

if (-not (Test-Path $cfg)) {
    # Clone the template, but drop any ROI so a new animal starts full-frame.
    $json = Get-Content $template -Raw | ConvertFrom-Json
    foreach ($c in $json.cams) { $c.PSObject.Properties.Remove('roi') }
    $json | ConvertTo-Json -Depth 20 | Set-Content $cfg -Encoding utf8
    Write-Host "Created per-animal config (full-frame): $cfg" -ForegroundColor Green
    Write-Host "Set this animal's ROI once via the Camera Crop/ROI dock; it will persist here."
}
else {
    Write-Host "Using existing per-animal config: $cfg" -ForegroundColor Cyan
}

Set-Location $repo
& $python -m labcams_ps.gui $cfg -w
