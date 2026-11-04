@echo off
setlocal
cd /d "%~dp0"
title MiniCPM-V 4.6 Interaction Analysis

set "PYTHON_EXE=..\.venv_vlm_next\Scripts\python.exe"
if not exist "%PYTHON_EXE%" goto missing_environment
if "%~1"=="" goto usage
if not exist "%~1" goto missing_video

echo.
echo Input video: %~1
echo Model: MiniCPM-V 4.6
echo Target: high-speed centrifuge
echo Analysis is running. Keep this window open.
echo.

"%PYTHON_EXE%" -m v5.minicpm_video_interaction "%~1"
set "ANALYSIS_EXIT=%ERRORLEVEL%"
if not "%ANALYSIS_EXIT%"=="0" goto analysis_error

echo.
echo Analysis completed successfully.
echo Opening the output folder now.
start "" explorer.exe "%CD%\outputs\minicpm_v46_interaction"
echo Press any key to close this window.
pause >nul
exit /b 0

:usage
echo.
echo No input video was received.
echo Drag one MP4 video onto this BAT file and try again.
echo Press any key to close this window.
pause >nul
exit /b 1

:missing_video
echo.
echo The input video was not found: %~1
echo Press any key to close this window.
pause >nul
exit /b 1

:missing_environment
echo.
echo MiniCPM Python environment was not found:
echo %PYTHON_EXE%
echo Keep D:\vision-model\.venv_vlm_next in place.
echo Press any key to close this window.
pause >nul
exit /b 1

:analysis_error
echo.
echo Analysis failed. Error code: %ANALYSIS_EXIT%
echo Take a screenshot of this window before closing it.
echo Press any key to close this window.
pause >nul
exit /b %ANALYSIS_EXIT%
