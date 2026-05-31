@echo off
cd /d "C:\Github\Widefield_DAQ_recorder"
"C:\ProgramData\anaconda3\envs\labcams\python.exe" -m labcams_ps.gui ".\labcams\labcams_widefield_pco_only.json" -w
pause