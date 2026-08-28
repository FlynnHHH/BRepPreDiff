@echo off
setlocal
chcp 65001 >nul
set "ENV_NAME=brepprediff-infer"
set "CONDA_CMD="
if defined CONDA_EXE if exist "%CONDA_EXE%" set "CONDA_CMD=%CONDA_EXE%"
if not defined CONDA_CMD for /f "delims=" %%I in ('where conda 2^>nul') do if not defined CONDA_CMD set "CONDA_CMD=%%I"
if not defined CONDA_CMD if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set "CONDA_CMD=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if not defined CONDA_CMD if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" set "CONDA_CMD=%USERPROFILE%\anaconda3\Scripts\conda.exe"
if not defined CONDA_CMD if exist "%LOCALAPPDATA%\miniconda3\Scripts\conda.exe" set "CONDA_CMD=%LOCALAPPDATA%\miniconda3\Scripts\conda.exe"
if not defined CONDA_CMD if exist "%LOCALAPPDATA%\anaconda3\Scripts\conda.exe" set "CONDA_CMD=%LOCALAPPDATA%\anaconda3\Scripts\conda.exe"
if not defined CONDA_CMD if exist "%ProgramData%\miniconda3\Scripts\conda.exe" set "CONDA_CMD=%ProgramData%\miniconda3\Scripts\conda.exe"
if not defined CONDA_CMD if exist "%ProgramData%\anaconda3\Scripts\conda.exe" set "CONDA_CMD=%ProgramData%\anaconda3\Scripts\conda.exe"
if not defined CONDA_CMD (
    echo [ERROR] 未找到 conda。请先安装 Miniconda 或 Anaconda，并从 Anaconda Prompt 运行本脚本。
    echo 下载地址：https://docs.conda.io/projects/miniconda/en/latest/
    pause
    exit /b 1
)

call "%CONDA_CMD%" env list | findstr /I /B /C:"%ENV_NAME% " >nul
if errorlevel 1 (
    echo 正在创建 %ENV_NAME% 环境，首次安装需要联网并可能耗时数分钟……
    call "%CONDA_CMD%" create --yes --name "%ENV_NAME%" --override-channels --channel pytorch --channel conda-forge python=3.10 "numpy>=1.24,<3" "pyyaml>=6" pytorch=2.5.1 cpuonly "pythonocc-core>=7.8,<8"
) else (
    echo 检测到已有 %ENV_NAME% 环境，正在校准依赖……
    call "%CONDA_CMD%" install --yes --name "%ENV_NAME%" --override-channels --channel pytorch --channel conda-forge python=3.10 "numpy>=1.24,<3" "pyyaml>=6" pytorch=2.5.1 cpuonly "pythonocc-core>=7.8,<8"
)
if errorlevel 1 (
    echo [ERROR] 环境安装失败。请检查网络连接和上方 conda 错误信息。
    pause
    exit /b 1
)

echo 正在验证推理依赖……
call "%CONDA_CMD%" run --no-capture-output -n "%ENV_NAME%" python -c "import torch, OCC; print('PyTorch', torch.__version__); print('OpenCascade', OCC.VERSION)"
if errorlevel 1 (
    echo [ERROR] 环境验证失败。
    pause
    exit /b 1
)

echo.
echo [OK] 安装完成。现在可以运行 run_inference.bat。
pause
exit /b 0
