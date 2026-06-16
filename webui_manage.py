import glob
import json
import os
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).parent
DATASET_RAW_DIR = ROOT / "dataset_raw"
DATASET_44K_DIR = ROOT / "dataset" / "44k"
FILELIST_TRAIN = ROOT / "filelists" / "train.txt"
FILELIST_VAL = ROOT / "filelists" / "val.txt"
LOGS_DIR = ROOT / "logs" / "44k"
DIFF_DIR = LOGS_DIR / "diffusion"
TRAINED_DIR = ROOT / "trained"
CONFIG_PATH = ROOT / "configs" / "config.json"
DIFF_CONFIG_PATH = ROOT / "configs" / "diffusion.yaml"
FEATURE_INDEX_PATH = LOGS_DIR / "feature_and_index.pkl"
CLUSTER_MODEL_GLOB = "kmeans_*.pt"
RAW_WAV_SUFFIX = ".wav"
DATASET_CACHE_SUFFIXES = [
    (".soft.pt", "内容特征"),
    (".f0.npy", "F0"),
    (".spec.pt", "谱图"),
    (".vol.npy", "音量"),
    (".mel.npy", "mel"),
    (".aug_mel.npy", "增广mel"),
    (".aug_vol.npy", "增广音量"),
]


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


def scan_datasets() -> list[str]:
    if not DATASET_RAW_DIR.exists():
        return []
    return [d.name for d in sorted(DATASET_RAW_DIR.iterdir()) if d.is_dir()]


def _count_wavs(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() == RAW_WAV_SUFFIX)


def _load_filelist_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = line.strip().replace("\\", "/")
            if not item:
                continue
            parts = item.lstrip("./").split("/")
            if len(parts) >= 4 and parts[0] == "dataset" and parts[1] == "44k":
                speaker = parts[2]
                counts[speaker] = counts.get(speaker, 0) + 1
    return counts


def _speaker_cache_counts(speaker: str) -> dict[str, tuple[int, int]]:
    raw_dir = DATASET_RAW_DIR / speaker
    resampled_dir = DATASET_44K_DIR / speaker
    raw_total = _count_wavs(raw_dir)
    resampled_total = _count_wavs(resampled_dir)

    counts = {
        "原始 WAV": (raw_total, raw_total),
        "重采样 WAV": (resampled_total, raw_total or resampled_total),
    }
    wav_bases = [p.stem for p in resampled_dir.glob("*.wav")] if resampled_dir.exists() else []
    for suffix, label in DATASET_CACHE_SUFFIXES:
        hit = sum(1 for stem in wav_bases if (resampled_dir / f"{stem}{suffix}").exists())
        counts[label] = (hit, resampled_total)
    return counts


def _ratio_text(done: int, total: int) -> str:
    if total <= 0:
        return "0/0"
    return f"{done}/{total}"


def _safe_delete_path(path: Path):
    if path.is_dir():
        shutil.rmtree(str(path), ignore_errors=True)
    elif path.exists():
        path.unlink()


def _remove_filelist_speaker(path: Path, speaker: str):
    if not path.exists():
        return
    kept = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = line.strip()
            if not item:
                continue
            parts = item.replace("\\", "/").lstrip("./").split("/")
            if len(parts) >= 3 and parts[0] == "dataset" and parts[1] == "44k" and parts[2] == speaker:
                continue
            kept.append(item)
    with open(path, "w", encoding="utf-8") as f:
        if kept:
            f.write("\n".join(kept) + "\n")
        else:
            f.write("")


def _cleanup_dataset_artifacts(speaker: str):
    _safe_delete_path(DATASET_RAW_DIR / speaker)
    _safe_delete_path(DATASET_44K_DIR / speaker)
    _remove_filelist_speaker(FILELIST_TRAIN, speaker)
    _remove_filelist_speaker(FILELIST_VAL, speaker)

    if DATASET_44K_DIR.exists() and not any(DATASET_44K_DIR.iterdir()):
        DATASET_44K_DIR.rmdir()
    if DATASET_RAW_DIR.exists() and not any(DATASET_RAW_DIR.iterdir()):
        DATASET_RAW_DIR.rmdir()


def describe_datasets() -> str:
    if not DATASET_RAW_DIR.exists():
        return """
<div class="svc-card">
  <div class="svc-title">数据集管理</div>
  <div>dataset_raw/ 不存在。</div>
</div>
"""

    speakers = [d.name for d in sorted(DATASET_RAW_DIR.iterdir()) if d.is_dir()]
    train_counts = _load_filelist_counts(FILELIST_TRAIN)
    val_counts = _load_filelist_counts(FILELIST_VAL)

    if not speakers:
        return """
<div class="svc-card">
  <div class="svc-title">数据集管理</div>
  <div>dataset_raw/ 存在，但没有可管理的数据集目录。</div>
</div>
"""

    cards = []
    for speaker in speakers:
        counts = _speaker_cache_counts(speaker)
        raw_done, raw_total = counts["原始 WAV"]
        res_done, res_total = counts["重采样 WAV"]
        train_ref = train_counts.get(speaker, 0)
        val_ref = val_counts.get(speaker, 0)
        cache_html = "".join(
            f"<div class='svc-data-row'>"
            f"<span>{label}</span><span class='svc-num'>{_ratio_text(done, total)}</span></div>"
            for label, (done, total) in counts.items()
        )
        cards.append(f"""
<details class="svc-card">
  <summary class="svc-summary">
    <span class="svc-title">{speaker}</span>
    <span class="svc-muted">raw {raw_done} | 44k {res_done} | train {train_ref} | val {val_ref}</span>
  </summary>
  <div class="svc-stat-grid">
    <div class="svc-stat">原始 WAV: <b>{_ratio_text(raw_done, raw_total)}</b></div>
    <div class="svc-stat">重采样 WAV: <b>{_ratio_text(res_done, res_total)}</b></div>
    <div class="svc-stat">train 引用: <b>{train_ref}</b></div>
    <div class="svc-stat">val 引用: <b>{val_ref}</b></div>
  </div>
  <div class="svc-cache">
    <div class="svc-title">预处理缓存</div>
    {cache_html}
  </div>
</details>
""")

    global_html = f"""
<div class="svc-card">
  <div class="svc-title">数据集管理</div>
  <div class="svc-title">全局预处理文件</div>
  <div class="svc-file-grid">
    <div>filelists/train.txt: <b>{'存在' if FILELIST_TRAIN.exists() else '不存在'}</b></div>
    <div>filelists/val.txt: <b>{'存在' if FILELIST_VAL.exists() else '不存在'}</b></div>
    <div>configs/config.json: <b>{'存在' if CONFIG_PATH.exists() else '不存在'}</b></div>
    <div>configs/diffusion.yaml: <b>{'存在' if DIFF_CONFIG_PATH.exists() else '不存在'}</b></div>
  </div>
</div>
"""

    return global_html + "".join(cards)


def delete_dataset(selection: str):
    speaker = (selection or "").strip()
    if not speaker:
        return "请先选择一个数据集", gr.update(choices=scan_datasets())
    if not DATASET_RAW_DIR.exists() or not (DATASET_RAW_DIR / speaker).exists():
        return "数据集不存在", gr.update(choices=scan_datasets())

    _cleanup_dataset_artifacts(speaker)
    return f"✓ 已删除数据集及缓存: {speaker}", gr.update(choices=scan_datasets(), value=None)


FEATURE_PATTERNS = ["feature_and_index.pkl", "kmeans_*.pt"]


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


def export_model(ckpt_selection: str, diff_selection: str, feat_selection: str, export_dir: str):
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

    feat_name = _parse_selection(feat_selection) if feat_selection else ""
    if feat_name:
        feat_src = LOGS_DIR / feat_name
        if feat_src.exists():
            feat_out = out_dir / feat_name
            shutil.copy2(str(feat_src), str(feat_out))
            result_lines.append(f"✓ 特征检索模型已导出: {feat_out}")
        else:
            result_lines.append(f"⚠ 特征检索模型文件不存在: {feat_src}")

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


# ── Feature retrieval / cluster models ───────────────────────────────────────

def scan_feature_models() -> list[str]:
    if not LOGS_DIR.exists():
        return []
    choices = []
    for pattern in FEATURE_PATTERNS:
        for f in sorted(LOGS_DIR.glob(pattern)):
            size = _fmt_size(f.stat().st_size)
            mtime = _fmt_time(f.stat().st_mtime)
            kind = "FAISS 特征检索" if f.suffix == ".pkl" else "KMeans 聚类"
            extra = _get_feature_detail(f)
            label = f"{f.name} | {kind} | {size} | {mtime}"
            if extra:
                label += f" | {extra}"
            choices.append(label)
    return choices


def _get_feature_detail(path: Path) -> str:
    try:
        if path.suffix == ".pkl":
            with open(path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                spks = list(data.keys())
                total = sum(idx.ntotal for idx in data.values() if hasattr(idx, "ntotal"))
                return f"{len(spks)}个说话人, {total}向量"
        elif path.suffix == ".pt":
            import torch
            data = torch.load(str(path), map_location="cpu", weights_only=False)
            if isinstance(data, dict):
                spks = [k for k in data.keys() if k != "__model__"]
                return f"{len(spks)}个说话人"
    except Exception:
        pass
    return ""


def get_feature_info(selection: str) -> str:
    name = _parse_selection(selection)
    if not name:
        return ""
    f = LOGS_DIR / name
    lines = [f"文件: {f}"]
    if f.exists():
        lines.append(f"大小: {_fmt_size(f.stat().st_size)}")
        lines.append(f"修改时间: {_fmt_time(f.stat().st_mtime)}")
        kind = "FAISS 特征检索索引" if f.suffix == ".pkl" else "KMeans 聚类模型"
        lines.append(f"类型: {kind}")
        detail = _get_feature_detail(f)
        if detail:
            lines.append(f"详情: {detail}")
    return "\n".join(lines)


def delete_feature_model(selection: str):
    name = _parse_selection(selection)
    if not name:
        return "请先选择一个模型", gr.Dropdown(choices=scan_feature_models())
    f = LOGS_DIR / name
    if f.exists():
        os.remove(f)
        return f"✓ 已删除: {name}", gr.Dropdown(choices=scan_feature_models(), value=None)
    return "文件不存在", gr.Dropdown(choices=scan_feature_models())


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_management_tab():
    gr.Markdown("## 模型管理\n"
                "管理训练检查点和已导出的模型。")

    with gr.Accordion("数据集管理 (dataset_raw/ 与预处理缓存)", open=True):
        gr.Markdown("查看每个数据集的原始 WAV、重采样结果以及预处理阶段生成的缓存文件。")
        with gr.Row():
            dataset_dd = gr.Dropdown(label="选择数据集", choices=scan_datasets(), interactive=True, scale=3)
            dataset_refresh = gr.Button("刷新", variant="primary", scale=1)
        with gr.Row():
            dataset_delete_btn = gr.Button("删除选中数据集及缓存", variant="stop")
        dataset_status = gr.Textbox(label="操作结果", interactive=False)
        dataset_overview = gr.HTML(value=describe_datasets())

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

        gr.Markdown("---")
        gr.Markdown("**特征检索 / 聚类模型**")
        with gr.Row():
            feat_dd = gr.Dropdown(label="选择模型", choices=scan_feature_models(),
                                  interactive=True, scale=3)
            feat_refresh = gr.Button("刷新", scale=1)
        feat_info = gr.Textbox(label="详情", interactive=False, lines=5)
        with gr.Row():
            feat_del_btn = gr.Button("删除选中模型")
        feat_status = gr.Textbox(label="操作结果", interactive=False)

    with gr.Accordion("导出模型", open=True):
        gr.Markdown("将训练检查点压缩（去除 optimizer 权重）并连同配置文件导出到指定目录，可直接用于推理。")
        with gr.Row():
            export_ckpt_dd = gr.Dropdown(label="主模型检查点", choices=scan_checkpoints(),
                                         interactive=True, scale=2)
            export_diff_dd = gr.Dropdown(label="扩散模型检查点 (可选)", choices=scan_diff_checkpoints(),
                                          interactive=True, scale=2)
            export_feat_dd = gr.Dropdown(label="特征检索/聚类模型 (可选)", choices=scan_feature_models(),
                                          interactive=True, scale=2)
        export_dir_input = gr.Textbox(label="导出目录",
                                      placeholder="例如: /workspace/so-vits-svc-cu128/trained 或 /mnt/d/my_models",
                                      interactive=True)
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

    feat_dd.change(get_feature_info, [feat_dd], [feat_info])
    feat_refresh.click(lambda: gr.Dropdown(choices=scan_feature_models()), [], [feat_dd])
    feat_del_btn.click(delete_feature_model, [feat_dd], [feat_status, feat_dd])

    dataset_refresh.click(
        lambda: (gr.update(choices=scan_datasets()), describe_datasets()),
        [],
        [dataset_dd, dataset_overview],
    )
    dataset_delete_btn.click(delete_dataset, [dataset_dd], [dataset_status, dataset_dd]).then(
        describe_datasets, [], [dataset_overview]
    )
    export_btn.click(export_model, [export_ckpt_dd, export_diff_dd, export_feat_dd, export_dir_input], [export_output])

    exported_dd.change(get_exported_info, [exported_dd], [exported_info])
    exported_refresh.click(lambda: gr.Dropdown(choices=scan_exported_models()), [], [exported_dd])
    exported_del_btn.click(delete_exported_model, [exported_dd], [exported_status, exported_dd])
