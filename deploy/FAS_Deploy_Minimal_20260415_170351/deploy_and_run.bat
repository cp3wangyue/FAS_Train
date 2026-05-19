@echo off
setlocal
chcp 65001 >nul

set "ROOT=%~dp0"
set "CONDA_ENV_NAME=FAS_Train"
set "ENV_CONFIG=%ROOT%env_name.txt"
set "CONDA_BAT="

if exist "%ENV_CONFIG%" (
  set /p CONDA_ENV_NAME=<"%ENV_CONFIG%"
)

echo [INFO] Checking Miniconda environment ...
echo [INFO] Target conda env: %CONDA_ENV_NAME%

if defined CONDA_EXE (
  set "CONDA_BAT=%CONDA_EXE:conda.exe=condabin\conda.bat%"
)

if not defined CONDA_BAT (
  if exist "%UserProfile%\miniconda3\condabin\conda.bat" set "CONDA_BAT=%UserProfile%\miniconda3\condabin\conda.bat"
)

if not defined CONDA_BAT (
  if exist "C:\ProgramData\miniconda3\condabin\conda.bat" set "CONDA_BAT=C:\ProgramData\miniconda3\condabin\conda.bat"
)

if not defined CONDA_BAT (
  echo [ERROR] conda.bat not found.
  echo Please make sure Miniconda is installed correctly.
  pause
  exit /b 1
)

call "%CONDA_BAT%" env list
call "%CONDA_BAT%" env list | findstr /R /C:"^[ ]*%CONDA_ENV_NAME% " /C:"^[*][ ]*%CONDA_ENV_NAME% " >nul
if errorlevel 1 (
  echo.
  echo [ERROR] Conda environment not found: %CONDA_ENV_NAME%
  echo [HINT] Please open env_name.txt and replace it with your real environment name.
  echo [HINT] Example: if your environment is named fas, then env_name.txt should contain only:
  echo         fas
  echo.
  pause
  exit /b 1
)

call "%CONDA_BAT%" activate "%CONDA_ENV_NAME%"
if errorlevel 1 (
  echo [ERROR] Failed to activate conda environment: %CONDA_ENV_NAME%
  echo [HINT] Please check whether the environment name is correct and dependencies are installed.
  pause
  exit /b 1
)

python -c "import sys; print('[INFO] Python:', sys.executable)"
if errorlevel 1 (
  echo [ERROR] Python is unavailable in current conda environment.
  pause
  exit /b 1
)

echo [INFO] Launching FAS desktop GUI ...
python "%ROOT%ui\app.py"
if errorlevel 1 (
  echo [ERROR] GUI exited with an error.
  pause
  exit /b 1
)
