@echo off
REM Voice Assistant Launcher - Double-click to run or run from command line

set "SCRIPT_DIR=%~dp0"
set "VENV_PYTHON=%SCRIPT_DIR%.va-env\Scripts\python.exe"
set "VENV_PYTHON_ALT=%SCRIPT_DIR%va-env\Scripts\python.exe"

REM Find virtual environment Python
if exist "%VENV_PYTHON%" (
    set "PYTHON_PATH=%VENV_PYTHON%"
) else if exist "%VENV_PYTHON_ALT%" (
    set "PYTHON_PATH=%VENV_PYTHON_ALT%"
) else (
    echo [ERROR] Virtual environment not found!
    echo Please create it first:
    echo   python -m venv .va-env
    echo   .va-env\Scripts\activate
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

echo ========================================
echo       Voice Assistant Launcher
echo ========================================
echo.
echo   1. va1.py - Basic version
echo   2. va2.py - Enhanced (recommended)
echo   3. va3.py - Model selector version
echo   4. va4.py - Legacy version
echo.
echo   q. Quit
echo.
echo ========================================

set /p choice="Select script to run: "

if "%choice%"=="1" (
    "%PYTHON_PATH%" "%SCRIPT_DIR%va1.py"
) else if "%choice%"=="2" (
    "%PYTHON_PATH%" "%SCRIPT_DIR%va2.py"
) else if "%choice%"=="3" (
    "%PYTHON_PATH%" "%SCRIPT_DIR%va3.py"
) else if "%choice%"=="4" (
    "%PYTHON_PATH%" "%SCRIPT_DIR%va4.py"
) else if /i "%choice%"=="q" (
    echo Goodbye!
    exit /b 0
) else (
    echo Invalid choice!
    pause
    exit /b 1
)
