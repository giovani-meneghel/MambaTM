@echo off
setlocal

:: Path to the Python executable
set PYTHON_EXE=C:\ComfyUI_portable\python_embeded\python.exe

:: Path to the inference script and default weights/inputs (using batch folder %~dp0)
set SCRIPT_PATH=%~dp0code\inference_mambaTM_dynamic.py
set DEFAULT_WEIGHTS=%~dp0code\model_zoo\MambaTM_dynamic.pth
set DEFAULT_INPUT=%~dp0videos\G00071.mp4
set DEFAULT_OUTPUT=%~dp0videos\G00071_restored.mp4

:: Check if arguments were passed to the batch file
if "%~1"=="" (
    echo No arguments provided. Running default restoration process on example video...
    echo Input: %DEFAULT_INPUT%
    echo Output: %DEFAULT_OUTPUT%
    "%PYTHON_EXE%" "%SCRIPT_PATH%" -f "%DEFAULT_WEIGHTS%" --input_path "%DEFAULT_INPUT%" --out_path "%DEFAULT_OUTPUT%" --num_frames 50
) else (
    echo Running restoration process with custom arguments...
    :: If custom args are passed but weight file is not specified, let's make it flexible or forward directly
    "%PYTHON_EXE%" "%SCRIPT_PATH%" -f "%DEFAULT_WEIGHTS%" %*
)

pause
