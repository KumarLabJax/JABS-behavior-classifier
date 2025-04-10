@echo off
setlocal EnableDelayedExpansion

REM Save the current working directory
set "initialDir=%CD%"

REM Change to the script's directory
cd /d "%~dp0"

REM discontinue support for this script once we are delivering wheels for installing JABS with pip

REM Check for skip version check argument
set "SKIP_VERSION_CHECK=0"
for %%i in (%*) do (
    if "%%~i"=="--skip-version-check" (
        set "SKIP_VERSION_CHECK=1"
    )
)

if "!SKIP_VERSION_CHECK!"=="0" (
    REM Check for Python Installation
    echo Checking for python
    set "VER="
    for /f "usebackq tokens=*" %%i in (`python --version 2^>nul`) do set "VER=%%i"

    if "!VER!"=="" (
        echo Python is not installed or not in PATH.
        REM restore working directory
        cd /d "%initialDir%"
        exit /b 1
    )

    set OK=0

    REM Supported versions of Python
    if "!VER:~7,4!"=="3.10" set OK=1
    if "!VER:~7,4!"=="3.11" set OK=1
    if "!VER:~7,4!"=="3.12" set OK=1

    if "!OK!"=="1" (
        echo Found !VER!
    ) else (
        echo JABS Requires Python 3.10, 3.11, or 3.12
        REM restore working directory
        cd /d "%initialDir%"
        exit /b 1
    )
) else (
    echo Skipping Python version check
)

echo Setting up Python Virtualenv...
python -m venv jabs.venv
call jabs.venv\Scripts\activate.bat && pip install .

REM restore working directory
cd /d "%initialDir%"