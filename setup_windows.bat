@echo off
REM Save the current working directory
set "initialDir=%CD%"

REM Change to the script's directory
cd /d "%~dp0"

REM discontinue support for this script once we are delivering wheels for installing JABS with pip

REM Check for skip version check argument
set "SKIP_VERSION_CHECK=0"
for %%i in (%*) do (
    if "%%i"=="--skip-version-check" set "SKIP_VERSION_CHECK=1"
)

if %SKIP_VERSION_CHECK%==0 (
    :: Check for Python Installation
    echo Checking for python
    for /f "tokens=*" %%i in ('python --version 2^>nul') do set VER=%%i

    SET OK=0

    :: Supported versions of Python
    if "%VER:~7,3%"=="3.10" SET OK=1
    if "%VER:~7,3%"=="3.11" SET OK=1
    if "%VER:~7,3%"=="3.12" SET OK=1

    if %OK% == 1 (
        echo Found %VER%
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
jabs.venv\Scripts\activate.bat && pip install .

REM restore working directory
cd /d "%initialDir%"
