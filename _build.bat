@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul

set "BUILD_SCRIPT=%~dp0scripts\build_release.ps1"
set "BUILD_OPTION="

if "%~1"=="" goto :run
if /I "%~1"=="--dry-run" (
    if not "%~2"=="" goto :usage
    set "BUILD_OPTION=-DryRun"
    goto :run
)
if /I "%~1"=="--build-only" (
    if not "%~2"=="" goto :usage
    set "BUILD_OPTION=-BuildOnly"
    goto :run
)

:usage
echo [Error] Unknown argument: %* 1>&2
echo Usage: %~nx0 [--dry-run^|--build-only] 1>&2
exit /b 2

:run
where.exe powershell.exe >nul 2>&1
if errorlevel 1 (
    echo [Error] PowerShell is required to build the release archive. 1>&2
    exit /b 1
)

if not exist "%BUILD_SCRIPT%" (
    echo [Error] Build helper is missing: %BUILD_SCRIPT% 1>&2
    exit /b 1
)

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%BUILD_SCRIPT%" %BUILD_OPTION%
exit /b %ERRORLEVEL%
