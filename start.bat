@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo    Starting Trae Local API
echo ============================================
echo.

if not exist "node_modules\" (
    echo [WARN] node_modules not found, running npm install...
    call npm install
    if errorlevel 1 (
        echo [ERROR] npm install failed!
        pause
        exit /b 1
    )
)

echo Starting server...
echo API: http://localhost:19900
echo Press Ctrl+C to stop
echo.

call npm start
if errorlevel 1 (
    echo.
    echo [ERROR] Server exited with error.
    pause
)
