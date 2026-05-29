@echo off
set "PYTHON_EXE=C:\Users\Optiplex 7090 Tower\AppData\Local\Programs\Python\Python313\python.exe"
cd /d "%~dp0"
"%PYTHON_EXE%" "%~dp0run_daq_recorder.py" --simulate
pause
