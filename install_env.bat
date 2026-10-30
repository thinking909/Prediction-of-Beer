@echo off
setlocal
cd /d "%~dp0"
echo ============================================
echo  Install offline environment for this project
echo ============================================
echo.

if not exist "wheels" (
    echo [ERROR] Missing "wheels" folder.
    pause
    exit /b 1
)

if exist ".venv\Scripts\python.exe" (
    echo [1/2] .venv already exists, skip creation.
) else (
    echo [1/2] Creating .venv ...
    py -3.11 -m venv .venv 2>nul
    if errorlevel 1 python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Could not create .venv. Install Python 3.11 64-bit first.
        pause
        exit /b 1
    )
)

echo [2/2] Installing from local wheels ...
call ".venv\Scripts\activate.bat"
python -m pip install --no-index --find-links="wheels" -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Offline install failed.
    pause
    exit /b 1
)

echo.
echo Done. Double-click run.bat to reproduce.
pause
exit /b 0
