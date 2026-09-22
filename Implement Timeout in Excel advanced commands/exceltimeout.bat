@echo off
setlocal enabledelayedexpansion

REM Check if all 3 parameters are provided
if "%~1"=="" (
    echo Usage: KillHungExcel.bat ^<TimeoutSeconds^> ^<FileName^> ^<LoggingPath^>
    echo.
    echo Example:
    echo KillHungExcel.bat 120 "FreezeExcel_Workbook 1 - Excel" "C:\Users\AlenSunny\Downloads\excel_logging.txt"
    exit /b 1
)

if "%~2"=="" (
    echo Error: FileName parameter is required
    exit /b 1
)

REM Set parameters
set TimeoutSeconds=%~1
set FileName=%~2
set LoggingPath=%~3

REM If LoggingPath not provided, use default
if "!LoggingPath!"=="" (
    set LoggingPath=%USERPROFILE%\Downloads\ExcelKiller_Log.txt
)

echo.
echo ============================================
echo Excel Killer Script
echo ============================================
echo Timeout Seconds: !TimeoutSeconds!
echo File Name: !FileName!
echo Logging Path: !LoggingPath!
echo ============================================
echo.

REM Run PowerShell script
powershell.exe -ExecutionPolicy Bypass -File "C:\Users\AlenSunny\Downloads\KillHungExcel 1.ps1" -WaitSeconds !TimeoutSeconds! -FileName "!FileName!" -LoggingPath "!LoggingPath!"

REM Check if script executed successfully
if %ERRORLEVEL% EQU 0 (
    echo.
    echo Script completed successfully.
    exit /b 0
) else (
    echo.
    echo Script failed with error code: %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)