import os
import sys
import subprocess

if getattr(sys, 'frozen', False):
    script_dir = os.path.dirname(os.path.abspath(sys.executable))
else:
    script_dir = os.path.dirname(os.path.abspath(__file__))

venv_dir = os.path.join(script_dir, "python_env")
pyvenv_config = os.path.join(venv_dir, "pyvenv.cfg")
python_home = os.path.join(venv_dir, "Python")
python_exe = os.path.join(python_home, "python.exe")

if not os.path.exists(python_exe):
    print(f"\033[31m[错误] 找不到捆绑的 Python 环境: {python_home}\033[0m")
    print("请确保 python_env\\Python\\ 目录完整。")
    os.system("pause")
    sys.exit(1)

with open(pyvenv_config, "w") as f:
    f.write(f"home = {python_home}\n")
    f.write("include-system-site-packages = false\n")
    f.write("version = 3.9.8\n")

print()
print("# So-Vits-SVC 4.1 with Cuda 12.8")
print()
print("- Intergrated by SHW / SHthemW@Github")
print("- So-Vits-SVC 为开源软件, 本整合包亦完全免费, 请勿用于商用.")
print()

activate_bat = os.path.join(venv_dir, "Scripts", "activate.bat")
print("虚拟环境已成功激活.")
print()
subprocess.run(["cmd", "/k", f'call "{activate_bat}"'])
