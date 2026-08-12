from __future__ import annotations

import ctypes
import os
import shutil
import sys
from pathlib import Path


def configure_console_encoding() -> None:
    """Use UTF-8 consistently in Windows console builds."""
    if os.name != "nt":
        return

    try:
        ctypes.windll.kernel32.SetConsoleCP(65001)
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    except (AttributeError, OSError):
        pass

    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")


def find_project_root() -> Path:
    """Locate the application root from source or a PyInstaller bundle."""
    candidates: list[Path] = []

    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent)

    candidates.extend((Path(__file__).resolve().parent, Path.cwd().resolve()))

    visited: set[Path] = set()
    for candidate in candidates:
        for directory in (candidate, *candidate.parents):
            if directory in visited:
                continue
            visited.add(directory)

            if (directory / "webUI.py").is_file() and (directory / "resource").is_dir():
                return directory

    searched = ", ".join(str(path) for path in candidates)
    raise RuntimeError(f"找不到 So-VITS-SVC 程序目录，已检查: {searched}")


def prepare_portable_python(project_root: Path) -> None:
    """Refresh the relocatable Windows environment configuration."""
    python_home = project_root / "python_env" / "Python"
    if not (python_home / "python.exe").is_file():
        return

    config_path = project_root / "python_env" / "pyvenv.cfg"
    content = (
        f"home = {python_home}\n"
        "include-system-site-packages = false\n"
        "version = 3.9.8\n"
    )
    with config_path.open("w", encoding="utf-8", newline="\n") as config_file:
        config_file.write(content)


def find_runtime_python(project_root: Path) -> list[str]:
    """Return the preferred Python command for the current platform."""
    environment_root = project_root / "python_env"
    if os.name == "nt":
        bundled_candidates = (
            environment_root / "Scripts" / "python.exe",
            environment_root / "Python" / "python.exe",
        )
    else:
        bundled_candidates = (
            environment_root / "bin" / "python",
            environment_root / "bin" / "python3",
        )

    for candidate in bundled_candidates:
        if candidate.is_file() and (os.name == "nt" or os.access(candidate, os.X_OK)):
            return [str(candidate)]

    if not getattr(sys, "frozen", False):
        return [sys.executable]

    for command_name in ("python3", "python"):
        executable = shutil.which(command_name)
        if executable:
            return [executable]

    raise RuntimeError("找不到可用的 Python 解释器，请检查 python_env 是否完整")
