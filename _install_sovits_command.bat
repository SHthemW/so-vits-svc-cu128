@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul

set "ROOT_DIR=%~dp0"
set "COMMAND_DIR=%ROOT_DIR%scripts"
set "SOVITS_COMMAND_DIR=%COMMAND_DIR%"

if not exist "%COMMAND_DIR%\sovits.bat" (
    echo [Error] Cannot find command wrapper: %COMMAND_DIR%\sovits.bat 1>&2
    pause
    exit /b 1
)

where powershell.exe >nul 2>&1
if errorlevel 1 (
    echo [Error] PowerShell is required to update the user PATH. 1>&2
    pause
    exit /b 1
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$dir = [IO.Path]::GetFullPath($env:SOVITS_COMMAND_DIR).TrimEnd('\'); $path = [Environment]::GetEnvironmentVariable('Path', 'User'); $items = @($path -split ';' | ForEach-Object { $_.Trim() } | Where-Object { $_ }); if (-not ($items | Where-Object { $_.TrimEnd('\') -ieq $dir })) { [Environment]::SetEnvironmentVariable('Path', (($items + $dir) -join ';'), 'User') }"
if errorlevel 1 (
    echo [Error] Failed to update the user PATH. 1>&2
    pause
    exit /b 1
)

echo Installed the sovits command from:
echo   %COMMAND_DIR%\sovits.bat
echo.
echo Open a new terminal before running:
echo   sovits start webui
echo   sovits start cmd
echo   sovits start tensorboard
echo.
pause
exit /b 0
