from __future__ import annotations

import ctypes
import os
import stat
import sys
from pathlib import Path

from launcher_runtime import configure_console_encoding, find_project_root
from startup_banner import emit_startup_banner

WINDOWS_COMMAND_NAME = "sovits.bat"
POSIX_COMMAND_NAME = "sovits"
POSIX_INSTALL_DIR = Path.home() / ".local" / "bin"


def _normalized_windows_path(path: str) -> str:
    expanded = os.path.expandvars(path.strip().strip('"'))
    return os.path.normcase(os.path.normpath(expanded))


def _broadcast_environment_change() -> None:
    try:
        result = ctypes.c_size_t()
        ctypes.windll.user32.SendMessageTimeoutW(
            0xFFFF,
            0x001A,
            0,
            "Environment",
            0x0002,
            5000,
            ctypes.byref(result),
        )
    except (AttributeError, OSError):
        pass


def _install_windows(project_root: Path) -> None:
    import winreg

    command_dir = (project_root / "scripts").resolve()
    command_source = command_dir / WINDOWS_COMMAND_NAME
    if not command_source.is_file():
        raise RuntimeError(f"找不到命令入口: {command_source}")

    with winreg.CreateKeyEx(
        winreg.HKEY_CURRENT_USER,
        "Environment",
        0,
        winreg.KEY_QUERY_VALUE | winreg.KEY_SET_VALUE,
    ) as environment_key:
        try:
            current_path, value_type = winreg.QueryValueEx(environment_key, "Path")
        except FileNotFoundError:
            current_path, value_type = "", winreg.REG_EXPAND_SZ

        path_items = [item.strip() for item in current_path.split(";") if item.strip()]
        normalized_command_dir = _normalized_windows_path(str(command_dir))
        already_installed = any(
            _normalized_windows_path(item) == normalized_command_dir for item in path_items
        )

        if not already_installed:
            path_items.append(str(command_dir))
            if value_type not in (winreg.REG_SZ, winreg.REG_EXPAND_SZ):
                value_type = winreg.REG_EXPAND_SZ
            winreg.SetValueEx(environment_key, "Path", 0, value_type, ";".join(path_items))

    if not already_installed:
        _broadcast_environment_change()

    status = "已安装" if not already_installed else "已经安装"
    print(f"{status} sovits 命令: {command_source}")
    print("请打开新终端后运行:")
    print("  sovits start webui")
    print("  sovits start cmd")
    print("  sovits start tensorboard")


def _replace_symlink(source: Path, target: Path) -> None:
    if target.is_symlink() or target.exists():
        if target.is_dir() and not target.is_symlink():
            raise RuntimeError(f"安装目标是目录，无法覆盖: {target}")
        target.unlink()
    target.symlink_to(source)


def _shell_configuration() -> tuple[Path, str]:
    shell_name = Path(os.environ.get("SHELL", "bash")).name.lower()
    if shell_name == "zsh":
        return Path.home() / ".zshrc", 'export PATH="$HOME/.local/bin:$PATH"'
    if shell_name == "fish":
        return Path.home() / ".config" / "fish" / "config.fish", "fish_add_path $HOME/.local/bin"
    return Path.home() / ".bashrc", 'export PATH="$HOME/.local/bin:$PATH"'


def _ensure_shell_path() -> tuple[Path, str, bool]:
    config_path, path_line = _shell_configuration()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    existing_content = ""
    if config_path.is_file():
        existing_content = config_path.read_text(encoding="utf-8", errors="replace")

    if path_line in existing_content.splitlines():
        return config_path, path_line, False

    needs_newline = bool(existing_content) and not existing_content.endswith("\n")
    with config_path.open("a", encoding="utf-8", newline="\n") as config_file:
        if needs_newline:
            config_file.write("\n")
        config_file.write("\n# So-VITS-SVC command\n")
        config_file.write(f"{path_line}\n")

    return config_path, path_line, True


def _install_posix(project_root: Path) -> None:
    command_source = (project_root / "scripts" / POSIX_COMMAND_NAME).resolve()
    if not command_source.is_file():
        raise RuntimeError(f"找不到命令入口: {command_source}")

    command_source.chmod(
        command_source.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    )
    POSIX_INSTALL_DIR.mkdir(parents=True, exist_ok=True)

    targets = (POSIX_INSTALL_DIR / "sovits", POSIX_INSTALL_DIR / "Sovits")
    for target in targets:
        _replace_symlink(command_source, target)

    config_path, path_line, path_added = _ensure_shell_path()
    for target in targets:
        print(f"已安装: {target}")

    if path_added:
        print(f"已更新 shell 配置: {config_path}")
    print("请重启终端，或在当前终端运行:")
    print(f"  {path_line}")
    print("随后可以运行: sovits start webui")


def main(arguments: tuple[str, ...] | None = None) -> int:
    configure_console_encoding()
    emit_startup_banner("# Command Installer")
    try:
        arguments = tuple(arguments or ())
        project_root = find_project_root()
        if arguments == ("--self-test",):
            command_name = WINDOWS_COMMAND_NAME if os.name == "nt" else POSIX_COMMAND_NAME
            command_source = project_root / "scripts" / command_name
            if not command_source.is_file():
                raise RuntimeError(f"找不到命令入口: {command_source}")
            print(f"命令安装器自检通过，程序目录: {project_root}")
            return 0
        if arguments:
            raise RuntimeError(f"不支持的参数: {' '.join(arguments)}")

        if os.name == "nt":
            _install_windows(project_root)
        else:
            _install_posix(project_root)
        return 0
    except (OSError, RuntimeError) as error:
        print(f"安装失败: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(tuple(sys.argv[1:])))
