@echo off
REM Save the current working directory
set "initialDir=%CD%"

REM Change to the script's directory
cd /d "%~dp0"

REM discontinue support for this script once we are delivering wheels for installing JABS with pip

:: Check for Python Installation
echo Checking for python
for /f "tokens=*" %%i in ('python --version 2^>nul') do set VER=%%i

SET OK=0

:: Supported versions of Python
if "%VER:~7,3%"=="3.10" SET OK=1

if %OK% == 1 (
  echo Found %VER%
  echo Setting up Python Virtualenv...
  python -m venv jabs.venv
  jabs.venv\Scripts\activate.bat && pip install .
) else (
  echo JABS Requires Python 3.10
)

REM restore working directory
cd /d "%initialDir%"