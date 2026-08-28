@echo off
setlocal
chcp 65001 >nul
set "ENV_NAME=brepprediff-infer"
set "PROMPT_MODE=0"
set "CONDA_CMD="
if not defined CONDA_CMD if defined CONDA_EXE if exist "%CONDA_EXE%" set "CONDA_CMD=%CONDA_EXE%"
if not defined CONDA_CMD for /f "delims=" %%I in ('where conda 2^>nul') do if not defined CONDA_CMD set "CONDA_CMD=%%I"
if not defined CONDA_CMD if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set "CONDA_CMD=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if not defined CONDA_CMD if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" set "CONDA_CMD=%USERPROFILE%\anaconda3\Scripts\conda.exe"
if not defined CONDA_CMD if exist "%LOCALAPPDATA%\miniconda3\Scripts\conda.exe" set "CONDA_CMD=%LOCALAPPDATA%\miniconda3\Scripts\conda.exe"
if not defined CONDA_CMD if exist "%LOCALAPPDATA%\anaconda3\Scripts\conda.exe" set "CONDA_CMD=%LOCALAPPDATA%\anaconda3\Scripts\conda.exe"
if not defined CONDA_CMD if exist "%ProgramData%\miniconda3\Scripts\conda.exe" set "CONDA_CMD=%ProgramData%\miniconda3\Scripts\conda.exe"
if not defined CONDA_CMD if exist "%ProgramData%\anaconda3\Scripts\conda.exe" set "CONDA_CMD=%ProgramData%\anaconda3\Scripts\conda.exe"
if not defined CONDA_CMD (
    echo [ERROR] 未找到 conda。请先执行 install_env.bat。
    pause
    exit /b 1
)

if "%~1"=="" (
    set "PROMPT_MODE=1"
    set /p "INPUT_PATH=请输入包含 STEP/STP 文件的目录："
) else (
    set "INPUT_PATH=%~1"
)
set "INPUT_PATH=%INPUT_PATH:"=%"

if not defined INPUT_PATH (
    echo [ERROR] 输入路径不能为空。
    if "%PROMPT_MODE%"=="1" pause
    exit /b 1
)

echo 开始推理：%INPUT_PATH%
if "%~2"=="" (
    call "%CONDA_CMD%" run --no-capture-output -n "%ENV_NAME%" python "%~dp0app\launcher.py" "%INPUT_PATH%"
) else (
    call "%CONDA_CMD%" run --no-capture-output -n "%ENV_NAME%" python "%~dp0app\launcher.py" "%INPUT_PATH%" --output-dir "%~2"
)
set "EXIT_CODE=%ERRORLEVEL%"

if "%EXIT_CODE%"=="0" (
    echo [OK] 推理完成。
) else (
    echo [ERROR] 推理未完全成功，退出码：%EXIT_CODE%。请查看输出目录中的 prediction_manifest.json。
)
if "%PROMPT_MODE%"=="1" pause
exit /b %EXIT_CODE%
