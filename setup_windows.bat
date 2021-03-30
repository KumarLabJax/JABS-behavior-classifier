@echo off
:: Check for Python Installation
echo Checking for python
for /f "tokens=*" %%i in ('python --version 2^>nul') do set VER=%%i

SET OK=0

:: Supported versions of Python
if "%VER:~7,3%"=="3.7" SET OK=1
if "%VER:~7,3%"=="3.8" SET OK=1
if "%VER:~7,3%"=="3.9" SET OK=1

if %OK% == 1 (
  echo Found %VER%
  echo Setting up Python Virtualenv...
  python -m venv aba.venv
  aba.venv\Scripts\activate.bat & pip install -r requirements.txt
) else (
  echo JAX-ABA Requires Python 3.7, 3.8, or 3.9
)
