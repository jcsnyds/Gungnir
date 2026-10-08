@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    python setup.py %*
    exit /b %ERRORLEVEL%
)

echo Python 3 was not found on PATH.
echo Activate your existing virtual environment first, then run setup.bat again.
exit /b 1
