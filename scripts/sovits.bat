@echo off
setlocal EnableExtensions DisableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "ROOT_DIR=%%~fI"

if "%~1"=="" goto usage_error
set "COMMAND=%~1"
shift

if /i "%COMMAND%"=="start" goto start_command
if /i "%COMMAND%"=="help" goto usage
if /i "%COMMAND%"=="-h" goto usage
if /i "%COMMAND%"=="--help" goto usage
goto unknown_command

:start_command
if "%~1"=="" goto missing_target
set "TARGET=%~1"
shift

if /i "%TARGET%"=="webui" goto webui_options
if /i "%TARGET%"=="cmd" goto start_cmd
if /i "%TARGET%"=="tensorboard" goto tensorboard_options
if /i "%TARGET%"=="help" goto usage
if /i "%TARGET%"=="-h" goto usage
if /i "%TARGET%"=="--help" goto usage
goto unknown_target

:webui_options
if "%~1"=="" goto start_webui
if /i "%~1"=="--local" (
    shift
    goto webui_options
)
if /i "%~1"=="-h" goto usage
if /i "%~1"=="--help" goto usage
goto unknown_webui_option

:start_webui
if not exist "%ROOT_DIR%\webUI.py" goto missing_webui
call :prepare_python_env
call :select_python
if errorlevel 1 exit /b 1
pushd "%ROOT_DIR%"
echo Starting So-VITS-SVC WebUI...
call :run_python "%ROOT_DIR%\webUI.py"
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%

:start_cmd
if not exist "%ROOT_DIR%\python_env\Scripts\activate.bat" (
    echo sovits: cannot find Python environment activation script: %ROOT_DIR%\python_env\Scripts\activate.bat 1>&2
    exit /b 1
)
call :prepare_python_env
call :select_python
if errorlevel 1 exit /b 1
pushd "%ROOT_DIR%"
call :run_python -c "from startup_banner import emit_startup_banner; emit_startup_banner('# Command Line')"
call "%ROOT_DIR%\python_env\Scripts\activate.bat"
echo So-VITS-SVC environment activated.
echo Project directory: %ROOT_DIR%
cmd /k
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%

:tensorboard_options
if "%~1"=="" goto start_tensorboard
if /i "%~1"=="--local" (
    shift
    goto tensorboard_options
)
if /i "%~1"=="--share" (
    echo sovits: --share is not supported by the Windows command wrapper. Starting locally.
    shift
    goto tensorboard_options
)
if /i "%~1"=="-h" goto usage
if /i "%~1"=="--help" goto usage
goto unknown_tensorboard_option

:start_tensorboard
if not exist "%ROOT_DIR%\logs" mkdir "%ROOT_DIR%\logs"
call :prepare_python_env
call :select_python
if errorlevel 1 exit /b 1
pushd "%ROOT_DIR%"
call :run_python -c "from startup_banner import emit_startup_banner; emit_startup_banner('# TensorBoard')"
echo Starting TensorBoard for %ROOT_DIR%\logs\44k ...
call :run_python -m tensorboard.main --logdir "%ROOT_DIR%\logs\44k" --host 127.0.0.1 --port 6006
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%

:select_python
set "PYTHON_EXE="
set "PYTHON_ARGS="
if exist "%ROOT_DIR%\python_env\Scripts\python.exe" (
    set "PYTHON_EXE=%ROOT_DIR%\python_env\Scripts\python.exe"
    exit /b 0
)
if exist "%ROOT_DIR%\python_env\Python\python.exe" (
    set "PYTHON_EXE=%ROOT_DIR%\python_env\Python\python.exe"
    exit /b 0
)
where py >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_EXE=py"
    set "PYTHON_ARGS=-3"
    exit /b 0
)
where python >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_EXE=python"
    exit /b 0
)
echo sovits: cannot find a Python interpreter. 1>&2
exit /b 1

:prepare_python_env
if not exist "%ROOT_DIR%\python_env\Python\python.exe" exit /b 0
> "%ROOT_DIR%\python_env\pyvenv.cfg" echo home = %ROOT_DIR%\python_env\Python
>> "%ROOT_DIR%\python_env\pyvenv.cfg" echo include-system-site-packages = false
>> "%ROOT_DIR%\python_env\pyvenv.cfg" echo version = 3.9.8
exit /b 0

:run_python
if defined PYTHON_ARGS (
    "%PYTHON_EXE%" %PYTHON_ARGS% %*
) else (
    "%PYTHON_EXE%" %*
)
exit /b %ERRORLEVEL%

:usage_error
echo sovits: missing command. 1>&2
goto usage

:missing_target
echo sovits: missing target after "start". 1>&2
goto usage

:unknown_command
echo sovits: unknown command "%COMMAND%". 1>&2
goto usage

:unknown_target
echo sovits: unknown start target "%TARGET%". 1>&2
goto usage

:unknown_webui_option
echo sovits: unknown webui option "%~1". 1>&2
goto usage

:unknown_tensorboard_option
echo sovits: unknown tensorboard option "%~1". 1>&2
goto usage

:missing_webui
echo sovits: cannot find WebUI entry point: %ROOT_DIR%\webUI.py 1>&2
exit /b 1

:usage
echo Usage:
echo   sovits start webui [--local]
echo   sovits start cmd
echo   sovits start tensorboard [--local^|--share]
echo.
echo Commands:
echo   webui        Start the WebUI locally.
echo   cmd          Open a shell with the bundled Python environment activated.
echo   tensorboard  Start TensorBoard for logs/44k.
exit /b 0
