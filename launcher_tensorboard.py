import os
import sys
import subprocess
import webbrowser
import time

if getattr(sys, 'frozen', False):
    script_dir = os.path.dirname(os.path.abspath(sys.executable))
else:
    script_dir = os.path.dirname(os.path.abspath(__file__))

venv_dir = os.path.join(script_dir, "python_env")
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

logdir = os.path.join(script_dir, "logs", "44k")
print()
print("# So-Vits-SVC 4.1 - TensorBoard")
print()
print(f"Log directory: {logdir}")
print("TensorBoard 正在启动, 请稍候...")
print()

proc = subprocess.Popen([scripts_python, "-m", "tensorboard.main", "--logdir", logdir])

time.sleep(3)
print("正在打开浏览器 http://localhost:6006 ...")
webbrowser.open("http://localhost:6006")
print("按 Ctrl+C 可停止 TensorBoard")
print()

proc.wait()
os.system("pause")
