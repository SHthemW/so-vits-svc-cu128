@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul

set "SCRIPT_DIR=%~dp0"
call "%SCRIPT_DIR%scripts\sovits.bat" start webui %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo [Error] WebUI exited with code %EXIT_CODE%.
)
pause
exit /b %EXIT_CODE%
