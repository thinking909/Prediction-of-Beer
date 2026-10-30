@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found.
    echo Please double-click install_env.bat first.
    echo.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" run_all.py %*
echo.
echo Finished. Press any key to close.
pause
