@echo off
REM Save the current working directory
set "initialDir=%CD%"

REM Change to the script's directory
cd /d "%~dp0"


echo Starting JAX Animal Behavior System App...
jabs.venv\Scripts\activate && jabs

REM restore working directory
cd /d "%initialDir%"