@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"

rem === 改这一行，指向你自己的 ComfyUI models 目录 ===
set "MODELS=E:\Comfy-Desktop\ComfyUI-Shared\models"

set "LIST=%TEMP%\mssort_args.txt"
if exist "%LIST%" del "%LIST%"
:collect
if "%~1"=="" goto run
>>"%LIST%" echo "%~1"
shift
goto collect
:run
python "sorter.py" place --list "%LIST%" --models-root "%MODELS%"
if errorlevel 1 pause
echo.
pause
