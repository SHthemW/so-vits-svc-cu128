import glob
import json
import os
import shutil
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog

import gradio as gr

ROOT = Path(__file__).parent
LOGS_DIR = ROOT / "logs" / "44k"
DIFF_DIR = LOGS_DIR / "diffusion"
TRAINED_DIR = ROOT / "trained"
CONFIG_PATH = ROOT / "configs" / "config.json"
DIFF_CONFIG_PATH = ROOT / "configs" / "diffusion.yaml"


def _fmt_size(size_bytes: int) -> str:
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.0f} KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def _fmt_time(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def _step_from_name(name: str) -> int:
    stem = Path(name).stem
    for part in stem.split("_"):
        if part.isdigit():
            return int(part)
    return -1


# ── Checkpoint scanning ─────────────────────────────────────────────────────

def scan_checkpoints() -> list[str]:
    if not LOGS_DIR.exists():
        return []
    files = sorted(LOGS_DIR.glob("G_*.pth"), key=lambda f: _step_from_name(f.name))
    choices = []
    for f in files:
        step = _step_from_name(f.name)
        size = _fmt_size(f.stat().st_size)
        mtime = _fmt_time(f.stat().st_mtime)
        d_exists = (LOGS_DIR / f"D_{step}.pth").exists()
        d_mark = "D✓" if d_exists else "D✗"
        label = f"G_{step}.pth | {size} | {mtime} | {d_mark}"
        if step == 0:
            label += " [底模]"
        choices.append(label)
    return choices


def scan_diff_checkpoints() -> list[str]:
    if not DIFF_DIR.exists():
        return []
    files = sorted(DIFF_DIR.glob("model_*.pt"), key=lambda f: _step_from_name(f.name))
    choices = []
    for f in files:
        step = _step_from_name(f.name)
        size = _fmt_size(f.stat().st_size)
        mtime = _fmt_time(f.stat().st_mtime)
        label = f"model_{step}.pt | {size} | {mtime}"
        if step == 0:
            label += " [底模]"
        choices.append(label)
    return choices


def _parse_selection(label: str) -> str:
    return label.split("|")[0].strip() if label else ""


def _get_spk_name() -> str:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        spks = list(cfg.get("spk", {}).keys())
        return spks[0] if spks else "model"
    except (OSError, json.JSONDecodeError, IndexError):
        return "model"


# ── Checkpoint info ──────────────────────────────────────────────────────────

def get_ckpt_info(selection: str) -> str:
    name = _parse_selection(selection)
    if not name:
        return ""
    step = _step_from_name(name)
    g_path = LOGS_DIR / name
    d_path = LOGS_DIR / f"D_{step}.pth"
    lines = [f"文件: {g_path}"]
    if g_path.exists():
        lines.append(f"大小: {_fmt_size(g_path.stat().st_size)}")
        lines.append(f"修改时间: {_fmt_time(g_path.stat().st_mtime)}")
        lines.append(f"训练步数: {step}")
    lines.append(f"对应判别器: {'存在 (' + _fmt_size(d_path.stat().st_size) + ')' if d_path.exists() else '不存在'}")
    if step == 0:
        lines.append("⚠ 此为训练底模，不可删除")
    return "\n".join(lines)


def get_diff_info(selection: str) -> str:
    name = _parse_selection(selection)
    if not name:
        return ""
    f = DIFF_DIR / name
    step = _step_from_name(name)
    lines = [f"文件: {f}"]
    if f.exists():
        lines.append(f"大小: {_fmt_size(f.stat().st_size)}")
        lines.append(f"修改时间: {_fmt_time(f.stat().st_mtime)}")
        lines.append(f"训练步数: {step}")
    if step == 0:
        lines.append("⚠ 此为训练底模，不可删除")
    return "\n".join(lines)


# ── Delete checkpoints ───────────────────────────────────────────────────────

def delete_checkpoint(selection: str):
    name = _parse_selection(selection)
    if not name:
        return "请先选择一个检查点", gr.Dropdown(choices=scan_checkpoints())
    step = _step_from_name(name)
    if step == 0:
        return "❌ 底模 (step 0) 不允许删除", gr.Dropdown(choices=scan_checkpoints())
    deleted = []
    g_path = LOGS_DIR / name
    d_path = LOGS_DIR / f"D_{step}.pth"
    if g_path.exists():
        os.remove(g_path)
        deleted.append(g_path.name)
    if d_path.exists():
        os.remove(d_path)
        deleted.append(d_path.name)
    if deleted:
        return f"✓ 已删除: {', '.join(deleted)}", gr.Dropdown(choices=scan_checkpoints(), value=None)
    return "文件不存在", gr.Dropdown(choices=scan_checkpoints())


def delete_diff_checkpoint(selection: str):
    name = _parse_selection(selection)
    if not name:
        return "请先选择一个检查点", gr.Dropdown(choices=scan_diff_checkpoints())
    step = _step_from_name(name)
    if step == 0:
        return "❌ 底模 (step 0) 不允许删除", gr.Dropdown(choices=scan_diff_checkpoints())
    f = DIFF_DIR / name
    if f.exists():
        os.remove(f)
        return f"✓ 已删除: {name}", gr.Dropdown(choices=scan_diff_checkpoints(), value=None)
    return "文件不存在", gr.Dropdown(choices=scan_diff_checkpoints())


# ── Export ────────────────────────────────────────────────────────────────────

def browse_export_dir():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title="选择导出目录")
    root.destroy()
    return folder if folder else ""


def export_model(ckpt_selection: str, diff_selection: str, export_dir: str):
    ckpt_name = _parse_selection(ckpt_selection)
    if not ckpt_name:
        return "请选择要导出的主模型检查点"

    export_dir = export_dir.strip()
    if not export_dir:
        return "请指定导出目录"

    g_path = LOGS_DIR / ckpt_name
    if not g_path.exists():
        return f"检查点文件不存在: {g_path}"
    if not CONFIG_PATH.exists():
        return f"配置文件不存在: {CONFIG_PATH}"

    spk = _get_spk_name()
    out_dir = Path(export_dir) / spk
    out_dir.mkdir(parents=True, exist_ok=True)

    step = _step_from_name(ckpt_name)
    out_pth = out_dir / f"{spk}_G{step}.pth"

    from compress_model import removeOptimizer
    try:
        removeOptimizer(str(CONFIG_PATH), str(g_path), False, str(out_pth))
    except Exception as e:
        return f"压缩模型失败: {e}"

    shutil.copy2(str(CONFIG_PATH), str(out_dir / "config.json"))

    result_lines = [
        f"✓ 主模型已导出到: {out_pth}",
        f"  压缩前: {_fmt_size(g_path.stat().st_size)} → 压缩后: {_fmt_size(out_pth.stat().st_size)}",
        f"  配置文件: {out_dir / 'config.json'}",
    ]

    diff_name = _parse_selection(diff_selection) if diff_selection else ""
    if diff_name:
        diff_src = DIFF_DIR / diff_name
        if diff_src.exists():
            diff_step = _step_from_name(diff_name)
            diff_out = out_dir / f"diffusion_{diff_step}.pt"
            shutil.copy2(str(diff_src), str(diff_out))
            if DIFF_CONFIG_PATH.exists():
                shutil.copy2(str(DIFF_CONFIG_PATH), str(out_dir / "diffusion.yaml"))
            result_lines.append(f"✓ 扩散模型已导出: {diff_out}")
        else:
            result_lines.append(f"⚠ 扩散模型文件不存在: {diff_src}")

    return "\n".join(result_lines)


# ── Exported models (trained/) ────────────────────────────────────────────────

def scan_exported_models() -> list[str]:
    if not TRAINED_DIR.exists():
        return []
    choices = []
    candidates = glob.glob(os.path.join(str(TRAINED_DIR), "**", "*.json"), recursive=True)
    dirs = set(os.path.dirname(c) for c in candidates)
    for d in sorted(dirs):
        jsons = glob.glob(os.path.join(d, "*.json"))
        pths = glob.glob(os.path.join(d, "*.pth"))
        if len(jsons) >= 1 and len(pths) >= 1:
            rel = os.path.relpath(d, str(TRAINED_DIR))
            total_size = sum(os.path.getsize(f) for f in pths + jsons)
            choices.append(f"{rel} | {len(pths)} pth | {_fmt_size(total_size)}")
    return choices


def get_exported_info(selection: str) -> str:
    if not selection:
        return ""
    rel_dir = selection.split("|")[0].strip()
    full_dir = TRAINED_DIR / rel_dir
    if not full_dir.exists():
        return "目录不存在"
    lines = [f"目录: {full_dir}"]
    for f in sorted(full_dir.iterdir()):
        lines.append(f"  {f.name}  ({_fmt_size(f.stat().st_size)})")
    return "\n".join(lines)


def delete_exported_model(selection: str):
    if not selection:
        return "请先选择一个模型", gr.Dropdown(choices=scan_exported_models())
    rel_dir = selection.split("|")[0].strip()
    full_dir = TRAINED_DIR / rel_dir
    if not full_dir.exists():
        return "目录不存在", gr.Dropdown(choices=scan_exported_models())
    shutil.rmtree(str(full_dir))
    return f"✓ 已删除: {full_dir}", gr.Dropdown(choices=scan_exported_models(), value=None)


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_management_tab():
    gr.Markdown("## 模型管理\n"
                "管理训练检查点和已导出的模型。")

    with gr.Accordion("训练检查点 (logs/44k/)", open=True):
        gr.Markdown("**主模型检查点**")
        with gr.Row():
            ckpt_dd = gr.Dropdown(label="选择检查点", choices=scan_checkpoints(),
                                  interactive=True, scale=3)
            ckpt_refresh = gr.Button("刷新", scale=1)
        ckpt_info = gr.Textbox(label="详情", interactive=False, lines=5)
        with gr.Row():
            ckpt_del_btn = gr.Button("删除选中检查点 (G+D)")
        ckpt_status = gr.Textbox(label="操作结果", interactive=False)

        gr.Markdown("---")
        gr.Markdown("**扩散模型检查点**")
        with gr.Row():
            diff_dd = gr.Dropdown(label="选择检查点", choices=scan_diff_checkpoints(),
                                  interactive=True, scale=3)
            diff_refresh = gr.Button("刷新", scale=1)
        diff_info = gr.Textbox(label="详情", interactive=False, lines=4)
        with gr.Row():
            diff_del_btn = gr.Button("删除选中检查点")
        diff_status = gr.Textbox(label="操作结果", interactive=False)

    with gr.Accordion("导出模型", open=True):
        gr.Markdown("将训练检查点压缩（去除 optimizer 权重）并连同配置文件导出到指定目录，可直接用于推理。")
        with gr.Row():
            export_ckpt_dd = gr.Dropdown(label="主模型检查点", choices=scan_checkpoints(),
                                         interactive=True, scale=2)
            export_diff_dd = gr.Dropdown(label="扩散模型检查点 (可选)", choices=scan_diff_checkpoints(),
                                          interactive=True, scale=2)
        with gr.Row():
            export_dir_input = gr.Textbox(label="导出目录",
                                          placeholder="例如: D:\\my_models",
                                          interactive=True, scale=4)
            export_browse = gr.Button("浏览...", scale=1)
        export_btn = gr.Button("导出", variant="primary")
        export_output = gr.Textbox(label="导出结果", interactive=False, lines=5)

    with gr.Accordion("已导出模型 (trained/)", open=True):
        with gr.Row():
            exported_dd = gr.Dropdown(label="选择模型", choices=scan_exported_models(),
                                      interactive=True, scale=3)
            exported_refresh = gr.Button("刷新", scale=1)
        exported_info = gr.Textbox(label="详情", interactive=False, lines=5)
        with gr.Row():
            exported_del_btn = gr.Button("删除选中模型")
        exported_status = gr.Textbox(label="操作结果", interactive=False)

    # ── Events ───────────────────────────────────────────────────────
    ckpt_dd.change(get_ckpt_info, [ckpt_dd], [ckpt_info])
    ckpt_refresh.click(lambda: gr.Dropdown(choices=scan_checkpoints()), [], [ckpt_dd])
    ckpt_del_btn.click(delete_checkpoint, [ckpt_dd], [ckpt_status, ckpt_dd])

    diff_dd.change(get_diff_info, [diff_dd], [diff_info])
    diff_refresh.click(lambda: gr.Dropdown(choices=scan_diff_checkpoints()), [], [diff_dd])
    diff_del_btn.click(delete_diff_checkpoint, [diff_dd], [diff_status, diff_dd])

    export_browse.click(browse_export_dir, [], [export_dir_input])
    export_btn.click(export_model, [export_ckpt_dd, export_diff_dd, export_dir_input], [export_output])

    exported_dd.change(get_exported_info, [exported_dd], [exported_info])
    exported_refresh.click(lambda: gr.Dropdown(choices=scan_exported_models()), [], [exported_dd])
    exported_del_btn.click(delete_exported_model, [exported_dd], [exported_status, exported_dd])
