chcp 65001 > nul

echo.
echo # So-Vits-SVC 4.1 with Cuda 12.8
echo.
echo - Intergrated by SHW / SHthemW@Github
echo - So-Vits-SVC 为开源软件, 本整合包亦完全免费, 请勿用于商用.
echo.

set pyvenv_config=python_env\pyvenv.cfg

break > %pyvenv_config%

echo home = %cd%\python_env\Python>> %pyvenv_config%
echo include-system-site-packages = false>> %pyvenv_config%
echo version = 3.9.8>> %pyvenv_config%

python_env\Scripts\python.exe webUI.py