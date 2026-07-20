@echo off
REM Launch the device monitor PowerShell script
powershell.exe -ExecutionPolicy Bypass -File "%~dp0monitor.ps1"
pause
