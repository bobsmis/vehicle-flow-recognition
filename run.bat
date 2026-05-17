@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE="

where python >nul 2>nul
if %ERRORLEVEL%==0 set "PYTHON_EXE=python"

if not defined PYTHON_EXE (
    if exist "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" (
        set "PYTHON_EXE=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    )
)

if not defined PYTHON_EXE (
    echo Python was not found. Please install Python 3.10+ and enable Add Python to PATH.
    pause
    exit /b 1
)

"%PYTHON_EXE%" -c "import cv2, numpy, PIL" >nul 2>nul
if not %ERRORLEVEL%==0 (
    echo Missing dependencies. Please run:
    echo "%PYTHON_EXE%" -m pip install -r requirements.txt
    pause
    exit /b 1
)

"%PYTHON_EXE%" main.py
endlocal
