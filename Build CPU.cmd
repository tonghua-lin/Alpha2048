@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0packaging\build_cpu.ps1"
if errorlevel 1 echo Build failed. See reports\packaging for the log.
pause
