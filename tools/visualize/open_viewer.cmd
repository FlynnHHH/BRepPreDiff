@echo off
setlocal

set "ROOT=%~dp0"
set "PORT=8061"

for /f "tokens=5" %%p in ('netstat -ano ^| findstr /R /C:":%PORT% .*LISTENING"') do (
  taskkill /F /PID %%p >nul 2>&1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%open_viewer.ps1"
if errorlevel 1 echo Failed to run open_viewer.ps1, exit code: %errorlevel%

endlocal
pause
