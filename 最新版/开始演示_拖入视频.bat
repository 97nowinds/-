@echo off
setlocal EnableExtensions
title MiniCPM-V 4.6 Interaction Demo

set "PACKAGE_ROOT=%~dp0"
set "PYTHON_EXE=%PACKAGE_ROOT%.venv_vlm_next\Scripts\python.exe"
set "MODULE_DIR="

for /d %%D in ("%PACKAGE_ROOT%*") do (
    if exist "%%~fD\v5\minicpm_vrm_hybrid.py" set "MODULE_DIR=%%~fD"
)

if not defined MODULE_DIR goto missing_module
if not exist "%PYTHON_EXE%" goto missing_environment
if "%~1"=="" goto usage
if not exist "%~1" goto missing_video

echo.
echo Input video: %~1
echo Backend: MiniCPM-V 4.6 + constrained wrist/forearm VRM
echo Mode: camera-view-2 multi-instrument hybrid
echo Analysis is running. Keep this window open.
echo.

pushd "%MODULE_DIR%"
"%PYTHON_EXE%" -m v5.minicpm_vrm_hybrid "%~1" 2> "%PACKAGE_ROOT%launcher_error.log"
set "ANALYSIS_EXIT=%ERRORLEVEL%"
popd

if not "%ANALYSIS_EXIT%"=="0" goto analysis_error

echo.
echo Analysis completed successfully.
echo Opening the output folder now.
start "" explorer.exe "%MODULE_DIR%\outputs\minicpm_v46_vrm_hybrid"
echo Press any key to close this window.
pause >nul
exit /b 0

:usage
echo.
echo No input video was received.
echo Drag one MP4 video onto this BAT file and try again.
goto keep_open

:missing_video
echo.
echo The input video was not found:
echo %~1
goto keep_open

:missing_environment
echo.
echo The packaged Python environment was not found:
echo %PYTHON_EXE%
goto keep_open

:missing_module
echo.
echo The program folder could not be found inside this package.
goto keep_open

:analysis_error
echo.
echo Analysis failed. Error code: %ANALYSIS_EXIT%
echo Error details were saved to:
echo %PACKAGE_ROOT%launcher_error.log

:keep_open
echo.
echo Press any key to close this window.
pause >nul
exit /b 1
