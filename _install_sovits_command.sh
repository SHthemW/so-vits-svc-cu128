#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMAND_SOURCE="$ROOT_DIR/scripts/sovits"
INSTALL_DIR="$HOME/.local/bin"
COMMAND_TARGET="$INSTALL_DIR/sovits"
COMMAND_TARGET_TITLE="$INSTALL_DIR/Sovits"

if [ ! -f "$COMMAND_SOURCE" ]; then
    echo "Cannot find command script: $COMMAND_SOURCE" >&2
    exit 1
fi

mkdir -p "$INSTALL_DIR"
chmod +x "$COMMAND_SOURCE"
ln -sfn "$COMMAND_SOURCE" "$COMMAND_TARGET"
ln -sfn "$COMMAND_SOURCE" "$COMMAND_TARGET_TITLE"

ensure_path_in_file() {
    local shell_file="$1"
    local line='export PATH="$HOME/.local/bin:$PATH"'

    touch "$shell_file"
    if ! grep -Fq "$line" "$shell_file"; then
        {
            echo
            echo "# So-VITS-SVC command"
            echo "$line"
        } >> "$shell_file"
    fi
}

case "$(basename "${SHELL:-bash}")" in
    zsh)
        ensure_path_in_file "$HOME/.zshrc"
        ;;
    *)
        ensure_path_in_file "$HOME/.bashrc"
        ;;
esac

echo "Installed: $COMMAND_TARGET"
echo "Installed: $COMMAND_TARGET_TITLE"
echo
echo "Restart your terminal, or run:"
echo '  export PATH="$HOME/.local/bin:$PATH"'
echo
echo "Available commands:"
echo "  sovits start webui"
echo "  sovits start webui --local"
echo "  sovits start cmd"
echo "  sovits start tensorboard"
