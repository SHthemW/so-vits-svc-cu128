@echo off
chcp 65001 > nul

echo.
echo # So-Vits-SVC 4.1 with Cuda 12.8
echo.
echo - Intergrated by SHW / SHthemW@Github
echo - So-Vits-SVC 为开源软件, 本整合包亦完全免费, 请勿用于商用.
echo.

set "script_dir=%~dp0"
set "script_dir=%script_dir:~0,-1%"

set "python_home=%script_dir%\python_env\Python"

if not exist "%python_home%\python.exe" (
    echo [错误] 找不到捆绑的 Python 环境: %python_home%
    echo 请确保 python_env\Python\ 目录完整。
    pause
    exit /b 1
)

set "pyvenv_config=%script_dir%\python_env\pyvenv.cfg"
break > "%pyvenv_config%"
echo home = %python_home%>> "%pyvenv_config%"
echo include-system-site-packages = false>> "%pyvenv_config%"
echo version = 3.9.8>> "%pyvenv_config%"

"%script_dir%\python_env\Scripts\python.exe" webUI.py

pause
