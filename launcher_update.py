import os
import shutil
import subprocess


def _run_git(git_exe, repo_dir, args, env):
    return subprocess.run(
        [git_exe, "-C", repo_dir, *args],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        text=True,
    )


def _print_output(result):
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())


def _git_environment(git_dir):
    env = os.environ.copy()
    paths = [
        os.path.join(git_dir, "cmd"),
        os.path.join(git_dir, "bin"),
        os.path.join(git_dir, "usr", "bin"),
        os.path.join(git_dir, "mingw64", "bin"),
    ]
    env["PATH"] = os.pathsep.join(paths + [env.get("PATH", "")])
    return env


def _find_git_exe(venv_dir):
    bundled_git = os.path.join(venv_dir, "Git", "cmd", "git.exe")
    if os.path.exists(bundled_git):
        return bundled_git, os.path.join(venv_dir, "Git")

    system_git = shutil.which("git")
    if system_git:
        return system_git, os.path.dirname(os.path.dirname(system_git))

    return None, None


def auto_update_from_gitee(repo_dir, venv_dir):
    print()
    print("[自动更新] 正在检查远程仓库 'gitee'...")

    if not os.path.isdir(os.path.join(repo_dir, ".git")):
        print("[自动更新] 已跳过：当前目录不是 Git 仓库。")
        return

    git_exe, git_dir = _find_git_exe(venv_dir)
    if not git_exe:
        print("[自动更新] 已跳过：未在 python_env\\Git 或 PATH 中找到 Git。")
        return

    env = _git_environment(git_dir)
    print(f"[自动更新] Git: {git_exe}")

    remote = _run_git(git_exe, repo_dir, ["remote", "get-url", "gitee"], env)
    if remote.returncode != 0:
        print("[自动更新] 已跳过：未配置名为 'gitee' 的远程仓库。")
        _print_output(remote)
        return

    print(f"[自动更新] 远程仓库: {remote.stdout.strip()}")

    branch = _run_git(git_exe, repo_dir, ["rev-parse", "--abbrev-ref", "HEAD"], env)
    if branch.returncode != 0:
        print("[自动更新] 获取当前分支失败。")
        _print_output(branch)
        print("[自动更新] 将继续启动应用。")
        return

    branch_name = branch.stdout.strip()
    pull_args = ["pull", "--ff-only", "gitee"]
    if branch_name and branch_name != "HEAD":
        pull_args.append(branch_name)
        print(f"[自动更新] 当前分支: {branch_name}")
    else:
        print("[自动更新] 当前分支: detached HEAD")

    print("[自动更新] 正在执行 git pull...")
    pull = _run_git(git_exe, repo_dir, pull_args, env)
    _print_output(pull)

    if pull.returncode == 0:
        print("[自动更新] 更新检查完成。")
    else:
        print(f"[自动更新] 更新失败，退出码: {pull.returncode}")
        print("[自动更新] 将继续启动应用。")
    print()
