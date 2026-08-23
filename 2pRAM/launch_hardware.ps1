$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot
python ".\run_daq_recorder.py" --config ".\2pRAM\pcie6353_config.json" --hardware
