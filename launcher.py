import os
import sys
import subprocess
from launcher_update import auto_update_from_gitee

if getattr(sys, 'frozen', False):
    script_dir = os.path.dirname(os.path.abspath(sys.executable))
else:
    script_dir = os.path.dirname(os.path.abspath(__file__))

venv_dir = os.path.join(script_dir, "python_env")
auto_update_from_gitee(script_dir, venv_dir)
pyvenv_config = os.path.join(venv_dir, "pyvenv.cfg")
python_exe = os.path.join(venv_dir, "Python", "python.exe")
scripts_python = os.path.join(venv_dir, "Scripts", "python.exe")

if not os.path.exists(python_exe):
    print(f"\033[31m[错误] 找不到捆绑的 Python 环境: {os.path.join(venv_dir, 'Python')}\033[0m")
    print("请确保 python_env\\Python\\ 目录完整。")
    os.system("pause")
    sys.exit(1)

with open(pyvenv_config, "w") as f:
    f.write(f"home = {os.path.join(venv_dir, 'Python')}\n")
    f.write("include-system-site-packages = false\n")
    f.write("version = 3.9.8\n")

print()
print("# So-Vits-SVC 4.1 with Cuda 12.8")
print()
print("- Intergrated by SHW / SHthemW@Github")
print("- So-Vits-SVC 为开源软件, 本整合包亦完全免费, 请勿用于商用.")
print()

webui_path = os.path.join(script_dir, "webUI.py")
subprocess.run([scripts_python, webui_path])
os.system("pause")
