from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence

from launcher_runtime import (
    configure_console_encoding,
    find_project_root,
    find_runtime_python,
    prepare_portable_python,
)
from startup_banner import emit_startup_banner


def main(arguments: Sequence[str] | None = None) -> int:
    configure_console_encoding()
    emit_startup_banner("# WebUI")
    try:
        arguments = tuple(arguments or ())
        project_root = find_project_root()
        webui_path = project_root / "webUI.py"
        if not webui_path.is_file():
            raise RuntimeError(f"找不到 WebUI 入口: {webui_path}")

        python_command = find_runtime_python(project_root)
        if arguments == ("--self-test",):
            print(f"启动器自检通过，程序目录: {project_root}")
            print(f"运行时 Python: {' '.join(python_command)}")
            return 0

        prepare_portable_python(project_root)
        command = [*python_command, str(webui_path), *arguments]

        environment = os.environ.copy()
        environment.setdefault("PYTHONUTF8", "1")

        print(f"正在启动 So-VITS-SVC WebUI: {webui_path}")
        return subprocess.call(command, cwd=project_root, env=environment)
    except (OSError, RuntimeError) as error:
        print(f"启动失败: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
