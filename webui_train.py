import json
import os
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog
import urllib.request
import zipfile
from pathlib import Path

import gradio as gr
import torch

PYTHON = sys.executable
ROOT = Path(__file__).parent
WEBUI_CONFIG = ROOT / "webui_config.json"


def _load_webui_config() -> dict:
    if WEBUI_CONFIG.exists():
        try:
            with open(WEBUI_CONFIG, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_webui_config(cfg: dict):
    with open(WEBUI_CONFIG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def _get_saved_dataset_dir() -> str:
    return _load_webui_config().get("dataset_dir", "")


def _save_dataset_dir(path: str):
    cfg = _load_webui_config()
    cfg["dataset_dir"] = path
    _save_webui_config(cfg)


def _save_webui_config_key(key: str, value):
    cfg = _load_webui_config()
    cfg[key] = value
    _save_webui_config(cfg)


def _get_webui_config_key(key: str, default=None):
    return _load_webui_config().get(key, default)


def browse_dataset_dir():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title="选择数据集目录")
    root.destroy()
    if folder:
        _save_dataset_dir(folder)
        return folder
    return _get_saved_dataset_dir()

import re

_procs: dict = {
    "download": None,
    "resample": None,
    "flist": None,
    "hubert": None,
    "train": None,
    "diff": None,
    "index": None,
}
_log_buffers: dict = {k: [] for k in _procs}
_progress: dict = {k: 0.0 for k in _procs}
_lock = threading.Lock()
_MAX_LOG_LINES = 500

_PROGRESS_RE = re.compile(r'(\d+)%|(\d+)/(\d+)')

CONFIG_PATH = ROOT / "configs" / "config.json"

# ── Pretrained model registry ────────────────────────────────────────────────

PRETRAIN_MODELS = {
    "pretrain/checkpoint_best_legacy_500.pt": {
        "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt",
        "desc": "ContentVec 语音编码器 (vec768l12, 必需)",
        "size_mb": 181,
        "required": True,
    },
    "pretrain/rmvpe.pt": {
        "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
        "desc": "RMVPE F0 预测器 (推荐)",
        "size_mb": 181,
        "required": True,
    },
    "pretrain/nsf_hifigan": {
        "url": "https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip",
        "desc": "NSF-HiFiGAN 声码器 (推理增强, zip压缩包)",
        "size_mb": 50,
        "required": False,
        "zip": True,
        "zip_extract_to": "pretrain/nsf_hifigan",
    },
}

BASE_MODELS = {
    "logs/44k/G_0.pth": {
        "url": "https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/resolve/main/vec768l12/vol_emb/G_0.pth",
        "desc": "Generator 底模 vec768l12+vol_emb (强烈推荐)",
        "size_mb": 199,
    },
    "logs/44k/D_0.pth": {
        "url": "https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/resolve/main/vec768l12/vol_emb/D_0.pth",
        "desc": "Discriminator 底模 vec768l12+vol_emb",
        "size_mb": 178,
    },
    "logs/44k/diffusion/model_0.pt": {
        "url": "https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/resolve/main/diffusion/768l12/model_0.pt",
        "desc": "扩散模型底模 (可选)",
        "size_mb": 210,
    },
}


def check_environment(dataset_dir) -> str:
    lines = []
    lines.append("═══ 环境检查 ═══\n")

    # CUDA
    if torch.cuda.is_available():
        dev_name = torch.cuda.get_device_properties(0).name
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        lines.append(f"[OK] CUDA 可用: {dev_name} ({vram:.1f} GB)")
        lines.append(f"     PyTorch CUDA 版本: {torch.version.cuda}")
    else:
        lines.append("[!!] CUDA 不可用 — 将使用 CPU，训练会极慢")

    lines.append(f"[--] Python: {sys.version.split()[0]}")
    lines.append(f"[--] PyTorch: {torch.__version__}")

    # dataset_raw
    lines.append("\n═══ 数据目录 ═══\n")
    dataset_raw = Path(dataset_dir) if dataset_dir else ROOT / "dataset_raw"
    if dataset_raw.exists():
        speakers = [d.name for d in dataset_raw.iterdir() if d.is_dir()]
        if speakers:
            for spk in speakers:
                wavs = list((dataset_raw / spk).glob("*.wav"))
                lines.append(f"[OK] {dataset_raw}/{spk}/ — {len(wavs)} 个 WAV 文件")
        else:
            lines.append(f"[!!] {dataset_raw}/ 存在但没有说话人子目录")
    else:
        lines.append(f"[!!] {dataset_raw}/ 目录不存在")

    # Pretrained models
    lines.append("\n═══ 预训练模型 ═══\n")
    for path, info in PRETRAIN_MODELS.items():
        full = ROOT / path
        tag = "必需" if info["required"] else "可选"
        if info.get("zip"):
            if full.exists() and full.is_dir() and any(full.iterdir()):
                lines.append(f"[OK] {path}/ (已解压)")
            else:
                lines.append(f"[缺失] {path}/ — {info['desc']} [{tag}]")
        else:
            if full.exists():
                size = full.stat().st_size / 1024**2
                lines.append(f"[OK] {path} ({size:.0f} MB)")
            else:
                lines.append(f"[缺失] {path} — {info['desc']} [{tag}]")

    # Base models
    lines.append("\n═══ 训练底模 ═══\n")
    for path, info in BASE_MODELS.items():
        full = ROOT / path
        if full.exists():
            size = full.stat().st_size / 1024**2
            lines.append(f"[OK] {path} ({size:.0f} MB)")
        else:
            lines.append(f"[缺失] {path} — {info['desc']}")

    return "\n".join(lines)


_download_cancel = threading.Event()


def _download_file(url: str, dest: Path, key: str, zip_extract_to: str = None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    display_name = dest.name

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "so-vits-svc/4.1"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 1024 * 1024  # 1 MB

            with open(tmp, "wb") as f:
                while True:
                    if _download_cancel.is_set():
                        with _lock:
                            _log_buffers[key].append(f"[取消] 下载已取消: {display_name}")
                        tmp.unlink(missing_ok=True)
                        return
                    chunk = resp.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded * 100 // total
                        mb_done = downloaded / 1024**2
                        mb_total = total / 1024**2
                        with _lock:
                            progress_line = f"  下载中: {display_name} — {mb_done:.1f}/{mb_total:.1f} MB ({pct}%)"
                            if _log_buffers[key] and _log_buffers[key][-1].startswith("  下载中:"):
                                _log_buffers[key][-1] = progress_line
                            else:
                                _log_buffers[key].append(progress_line)

        if zip_extract_to:
            extract_dir = ROOT / zip_extract_to
            extract_dir.mkdir(parents=True, exist_ok=True)
            with _lock:
                _log_buffers[key].append(f"  解压中: {display_name} → {zip_extract_to}/")
            import shutil
            with zipfile.ZipFile(tmp, "r") as zf:
                zf.extractall(extract_dir)
            nested = extract_dir / extract_dir.name
            if nested.is_dir():
                for item in nested.iterdir():
                    dest_item = extract_dir / item.name
                    if item.is_dir():
                        shutil.copytree(str(item), str(dest_item), dirs_exist_ok=True)
                    else:
                        shutil.copy2(str(item), str(dest_item))
                shutil.rmtree(str(nested))
            tmp.unlink(missing_ok=True)
            with _lock:
                _log_buffers[key].append(f"[完成] {display_name} 下载并解压成功")
        else:
            tmp.rename(dest)
            with _lock:
                _log_buffers[key].append(f"[完成] {display_name} 下载成功")
    except Exception as e:
        tmp.unlink(missing_ok=True)
        with _lock:
            _log_buffers[key].append(f"[错误] 下载失败 {display_name}: {e}")


def start_download(dl_pretrain, dl_base):
    key = "download"
    with _lock:
        if _procs.get(key) == "running":
            return "下载任务已在进行中"
        _log_buffers[key] = []
        _procs[key] = "running"
    _download_cancel.clear()

    tasks = []
    if dl_pretrain:
        for path, info in PRETRAIN_MODELS.items():
            full = ROOT / path
            if info.get("zip"):
                if full.exists() and full.is_dir() and any(full.iterdir()):
                    continue
                zip_dest = ROOT / (path.rstrip("/") + ".zip")
                tasks.append((info["url"], zip_dest, info.get("zip_extract_to")))
            else:
                if not full.exists():
                    tasks.append((info["url"], full, None))
    if dl_base:
        for path, info in BASE_MODELS.items():
            full = ROOT / path
            if not full.exists():
                tasks.append((info["url"], full, None))

    if not tasks:
        with _lock:
            _log_buffers[key].append("[提示] 所有选中的文件都已存在，无需下载")
            _procs[key] = None
        return "所有文件已存在"

    def _worker():
        with _lock:
            _log_buffers[key].append(f"开始下载 {len(tasks)} 个文件...\n")
        for url, dest, zip_extract in tasks:
            if _download_cancel.is_set():
                break
            with _lock:
                _log_buffers[key].append(f"[开始] {dest.relative_to(ROOT)}")
            _download_file(url, dest, key, zip_extract_to=zip_extract)
        with _lock:
            _procs[key] = None
            if _download_cancel.is_set():
                _log_buffers[key].append("\n下载已被用户取消")
            else:
                _log_buffers[key].append("\n所有下载任务完成!")

    threading.Thread(target=_worker, daemon=True).start()
    return f"已启动下载 ({len(tasks)} 个文件)"


def stop_download():
    _download_cancel.set()
    return "正在取消下载..."


def get_download_log():
    return _get_log("download")


def clear_download_log():
    return _clear_log("download")


def get_download_status():
    with _lock:
        if _procs.get("download") == "running":
            return "下载中..."
    return "空闲"


def _launch(key: str, args: list, clear_log: bool = True) -> str:
    with _lock:
        if _procs[key] is not None and _procs[key].poll() is None:
            return f"[{key}] 已经在运行中，请先停止"
        if clear_log:
            _log_buffers[key].clear()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    cmd = [PYTHON] + args
    with _lock:
        _log_buffers[key].append(f"[命令] {' '.join(cmd)}")
        _log_buffers[key].append("")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        cwd=str(ROOT),
        env=env,
    )
    with _lock:
        _procs[key] = proc

    def _reader():
        with _lock:
            _progress[key] = 0.0
        for line in proc.stdout:
            stripped = line.rstrip()
            with _lock:
                # tqdm/rich 进度行用 \r 覆盖，可能含 \r
                clean = stripped.replace("\r", "")
                if clean:
                    _log_buffers[key].append(clean)
                    if len(_log_buffers[key]) > _MAX_LOG_LINES:
                        _log_buffers[key] = _log_buffers[key][-_MAX_LOG_LINES:]
                # 解析进度
                m = _PROGRESS_RE.search(stripped)
                if m:
                    if m.group(1):
                        _progress[key] = float(m.group(1)) / 100.0
                    elif m.group(2) and m.group(3):
                        total = int(m.group(3))
                        if total > 0:
                            _progress[key] = int(m.group(2)) / total
        proc.stdout.close()
        rc = proc.wait()
        with _lock:
            if rc == 0:
                _progress[key] = 1.0
                _log_buffers[key].append("")
                _log_buffers[key].append(f"[完成] 进程正常退出 (返回码 0)")
            else:
                _log_buffers[key].append("")
                _log_buffers[key].append(f"[异常] 进程退出，返回码: {rc}")

    threading.Thread(target=_reader, daemon=True).start()
    return f"[{key}] 已启动 (PID {proc.pid})"


def _stop(key: str) -> str:
    with _lock:
        proc = _procs[key]
    if proc is None or proc.poll() is not None:
        return f"[{key}] 没有正在运行的进程"
    pid = proc.pid
    try:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            capture_output=True, timeout=10
        )
    except Exception:
        proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    return f"[{key}] 已停止"


def _get_log(key: str) -> str:
    with _lock:
        return "\n".join(_log_buffers[key])


def _clear_log(key: str) -> str:
    with _lock:
        _log_buffers[key].clear()
    return ""


def _get_progress(key: str) -> float:
    with _lock:
        return _progress[key]


def _get_progress_pct(key: str) -> str:
    p = _get_progress(key)
    return f"{p * 100:.0f}%"


def _get_status(key: str) -> str:
    with _lock:
        proc = _procs[key]
    if proc is None:
        return "未启动"
    rc = proc.poll()
    if rc is None:
        return f"运行中 (PID {proc.pid})"
    if rc == 0:
        return "已完成"
    return f"已结束 (返回码 {rc})"


# ── Stage functions ──────────────────────────────────────────────────────────

def start_resample(dataset_dir, skip_loudnorm, num_processes):
    in_dir = dataset_dir.strip() if dataset_dir else ""
    if in_dir:
        _save_dataset_dir(in_dir)
    if not in_dir:
        in_dir = "./dataset_raw"

    in_path = Path(in_dir) if Path(in_dir).is_absolute() else ROOT / in_dir
    if not in_path.exists():
        return f"[错误] 输入目录不存在: {in_path}"
    if not in_path.is_dir():
        return f"[错误] 路径不是目录: {in_path}"

    speakers = [d for d in in_path.iterdir() if d.is_dir()]
    top_wavs = list(in_path.glob("*.wav"))

    if not speakers and not top_wavs:
        return f"[错误] 目录中既没有子文件夹也没有 .wav 文件: {in_path}"

    actual_in_path = in_path
    if not speakers and top_wavs:
        speaker_name = in_path.name
        dataset_raw = ROOT / "dataset_raw" / speaker_name
        dataset_raw.mkdir(parents=True, exist_ok=True)
        import shutil
        copied = 0
        for wav in top_wavs:
            shutil.copy2(str(wav), str(dataset_raw / wav.name))
            copied += 1
        actual_in_path = ROOT / "dataset_raw"
        with _lock:
            _log_buffers["resample"].clear()
            _log_buffers["resample"].append(f"[自动适配] 检测到目录内直接包含 wav 文件（无子文件夹）")
            _log_buffers["resample"].append(f"  已复制 {copied} 个文件 → dataset_raw/{speaker_name}/")
            _log_buffers["resample"].append(f"  说话人名称: {speaker_name}")
            _log_buffers["resample"].append("")
    else:
        total_wavs = 0
        for spk in speakers:
            total_wavs += len(list(spk.glob("*.wav")))
        if total_wavs == 0:
            return f"[错误] 子文件夹中没有 .wav 文件\n期望结构: {in_path}/<说话人名称>/*.wav"
        with _lock:
            _log_buffers["resample"].clear()
            _log_buffers["resample"].append(f"输入目录: {actual_in_path}")
            _log_buffers["resample"].append(f"发现 {len(speakers)} 个说话人, 共 {total_wavs} 个 WAV 文件")
            _log_buffers["resample"].append("")

    n_proc = int(num_processes) if num_processes else 0
    _save_webui_config_key("resample_num_processes", n_proc)
    _save_webui_config_key("resample_skip_loudnorm", bool(skip_loudnorm))

    args = ["resample.py", "--sr2", "44100",
            "--in_dir", str(actual_in_path),
            "--out_dir2", "./dataset/44k"]
    if skip_loudnorm:
        args.append("--skip_loudnorm")
    if n_proc > 0:
        args.extend(["--num_processes", str(n_proc)])
    return _launch("resample", args, clear_log=False)


def stop_resample():
    return _stop("resample")


def get_resample_log():
    return _get_log("resample")


def clear_resample_log():
    return _clear_log("resample")


def get_resample_progress():
    return _get_progress("resample")


def get_resample_status():
    status = _get_status("resample")
    if status == "已完成":
        out_dir = ROOT / "dataset" / "44k"
        if not out_dir.exists() or not any(out_dir.iterdir()):
            return "已完成 [警告: dataset/44k 为空，可能输入目录结构不正确]"
    if status == "未启动":
        out_dir = ROOT / "dataset" / "44k"
        if out_dir.exists():
            speakers = [d for d in out_dir.iterdir() if d.is_dir()]
            total_wavs = sum(len(list(s.glob("*.wav"))) for s in speakers)
            if total_wavs > 0:
                return f"未启动 [检测到已有重采样结果: {len(speakers)} 个说话人, {total_wavs} 个文件]"
    return status


def start_flist(speech_encoder, vol_aug, tiny):
    _save_webui_config_key("flist_encoder", speech_encoder)
    _save_webui_config_key("flist_vol_aug", bool(vol_aug))
    _save_webui_config_key("flist_tiny", bool(tiny))
    out_dir = ROOT / "dataset" / "44k"
    if not out_dir.exists():
        return "[错误] dataset/44k 目录不存在，请先完成第一步（音频重采样）"
    speakers = [d for d in out_dir.iterdir() if d.is_dir()]
    if not speakers:
        return "[错误] dataset/44k 中没有说话人子文件夹，请先完成第一步（音频重采样）"
    args = ["preprocess_flist_config.py",
            "--source_dir", "./dataset/44k",
            "--speech_encoder", speech_encoder]
    if vol_aug:
        args.append("--vol_aug")
    if tiny:
        args.append("--tiny")
    return _launch("flist", args)


def stop_flist():
    return _stop("flist")


def get_flist_log():
    return _get_log("flist")


def clear_flist_log():
    return _clear_log("flist")


def get_flist_progress():
    return _get_progress("flist")


def get_flist_status():
    return _get_status("flist")


def start_hubert(f0_predictor, num_processes, use_diff, device):
    _save_webui_config_key("hubert_f0", f0_predictor)
    _save_webui_config_key("hubert_num_processes", int(num_processes))
    _save_webui_config_key("hubert_use_diff", bool(use_diff))
    _save_webui_config_key("hubert_device", device)
    args = ["preprocess_hubert_f0.py",
            "--in_dir", "dataset/44k",
            "--f0_predictor", f0_predictor,
            "--num_processes", str(int(num_processes)),
            "--device", device]
    if use_diff:
        args.append("--use_diff")
    return _launch("hubert", args)


def stop_hubert():
    return _stop("hubert")


def get_hubert_log():
    return _get_log("hubert")


def clear_hubert_log():
    return _clear_log("hubert")


def get_hubert_progress():
    return _get_progress("hubert")


def get_hubert_status():
    return _get_status("hubert")


def start_train():
    return _launch("train", ["train.py", "-c", "configs/config.json", "-m", "44k"])


def stop_train():
    return _stop("train")


def get_train_log():
    return _get_log("train")


def clear_train_log():
    return _clear_log("train")


def get_train_progress():
    return _get_progress("train")


def get_train_status():
    return _get_status("train")


def start_train_diff():
    return _launch("diff", ["train_diff.py", "-c", "configs/diffusion.yaml"])


def stop_train_diff():
    return _stop("diff")


def get_train_diff_log():
    return _get_log("diff")


def clear_train_diff_log():
    return _clear_log("diff")


def get_train_diff_progress():
    return _get_progress("diff")


def get_train_diff_status():
    return _get_status("diff")


def start_index():
    return _launch("index", ["train_index.py",
                              "--root_dir", "dataset/44k",
                              "--output_dir", "logs/44k",
                              "-c", "configs/config.json"])


def stop_index():
    return _stop("index")


def get_index_log():
    return _get_log("index")


def clear_index_log():
    return _clear_log("index")


def get_index_progress():
    return _get_progress("index")


def get_index_status():
    return _get_status("index")


# ── Config helpers ───────────────────────────────────────────────────────────

def read_train_config():
    if not CONFIG_PATH.exists():
        return 6, 10000, 3, False
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    t = cfg.get("train", {})
    return (
        t.get("batch_size", 6),
        t.get("epochs", 10000),
        t.get("keep_ckpts", 3),
        t.get("fp16_run", False),
    )


def write_train_config(batch_size, epochs, keep_ckpts, fp16_run):
    if not CONFIG_PATH.exists():
        return "configs/config.json 不存在，请先运行预处理第二步"
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["train"]["batch_size"] = int(batch_size)
    cfg["train"]["epochs"] = int(epochs)
    cfg["train"]["keep_ckpts"] = int(keep_ckpts)
    cfg["train"]["fp16_run"] = bool(fp16_run)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return "配置已保存到 configs/config.json"


# ── Gradio UI factory ────────────────────────────────────────────────────────

_AUTOSCROLL_JS = """
<script>
(function() {
    const observer = new MutationObserver(function(mutations) {
        document.querySelectorAll('textarea[readonly]').forEach(function(el) {
            el.scrollTop = el.scrollHeight;
        });
    });
    observer.observe(document.body, {childList: true, subtree: true, characterData: true});
})();
</script>
"""


def build_training_tab():
    gr.HTML(value=_AUTOSCROLL_JS, visible=False)
    gr.Markdown("## So-VITS-SVC 训练流程\n"
                "按顺序完成以下各步骤。\n\n"
                "训练进程在WebUI重启后会继续在后台运行，可通过 `logs/44k/train.log` 查看进度。")

    with gr.Row():
        dataset_dir = gr.Textbox(
            label="数据集目录 (包含说话人子文件夹或直接包含wav的目录，留空则使用默认 dataset_raw/)",
            placeholder="例如: D:\\my_audio\\singer1_dataset  或留空使用 dataset_raw/",
            value=_get_saved_dataset_dir(),
            interactive=True,
            scale=4,
        )
        browse_btn = gr.Button("浏览...", scale=1)

    browse_btn.click(browse_dataset_dir, [], [dataset_dir])

    # ── Step 0: Environment check & download ─────────────────────────
    with gr.Accordion("前置步骤：环境检查与模型下载", open=True):
        gr.Markdown("检查 CUDA 环境、训练数据目录、预训练模型是否就绪。缺失的模型可一键从 HuggingFace 下载。")
        with gr.Row():
            env_check_btn = gr.Button("检查环境", variant="primary")
        env_check_output = gr.Textbox(label="检查结果", lines=18, max_lines=30, interactive=False)
        env_check_btn.click(check_environment, [dataset_dir], [env_check_output])

        gr.Markdown("---")
        gr.Markdown("**下载缺失的模型文件** (从 HuggingFace 下载，需要网络连接)")
        with gr.Row():
            dl_pretrain = gr.Checkbox(label="下载预训练模型 (ContentVec, RMVPE, NSF-HiFiGAN)", value=True)
            dl_base = gr.Checkbox(label="下载训练底模 (G_0.pth, D_0.pth)", value=True)
        with gr.Row():
            dl_start_btn = gr.Button("开始下载", variant="primary")
            dl_stop_btn = gr.Button("取消下载")
            dl_clear_btn = gr.Button("清除日志")
            dl_status = gr.Textbox(label="状态", value=get_download_status, every=2, interactive=False, scale=2)
        dl_log = gr.Textbox(label="下载日志", value=get_download_log, every=2, lines=10, max_lines=20, interactive=False)

        dl_start_btn.click(start_download, [dl_pretrain, dl_base], [dl_status])
        dl_stop_btn.click(stop_download, [], [dl_status])
        dl_clear_btn.click(clear_download_log, [], [dl_log])

    # ── Step 1: Resample ─────────────────────────────────────────────
    with gr.Accordion("第一步：音频重采样 (resample.py)", open=False):
        gr.Markdown("将数据集目录下的原始音频重采样至 44100Hz mono，输出到 `dataset/44k/`")
        with gr.Row():
            resample_skip_loudnorm = gr.Checkbox(
                label="跳过响度归一化 (--skip_loudnorm)",
                value=_get_webui_config_key("resample_skip_loudnorm", False),
            )
            resample_procs = gr.Slider(
                label="CPU 核心数 (0=自动)",
                minimum=0,
                maximum=max(os.cpu_count() or 1, 1),
                value=_get_webui_config_key("resample_num_processes", 0),
                step=1,
            )
        with gr.Row():
            resample_start_btn = gr.Button("开始重采样", variant="primary")
            resample_stop_btn = gr.Button("停止")
            resample_clear_btn = gr.Button("清除日志")
            resample_status = gr.Textbox(label="状态", value=get_resample_status, every=2, interactive=False, scale=2)
        resample_log = gr.Textbox(label="日志", value=get_resample_log, every=3, lines=8, max_lines=15, interactive=False)

        resample_start_btn.click(start_resample, [dataset_dir, resample_skip_loudnorm, resample_procs], [resample_status])
        resample_stop_btn.click(stop_resample, [], [resample_status])
        resample_clear_btn.click(clear_resample_log, [], [resample_log])

    # ── Step 2: flist + config ───────────────────────────────────────
    with gr.Accordion("第二步：生成文件列表和配置 (preprocess_flist_config.py)", open=False):
        gr.Markdown("扫描 `dataset/44k/`，生成训练/验证文件列表以及 `configs/config.json`、`configs/diffusion.yaml`")
        with gr.Row():
            flist_encoder = gr.Dropdown(
                label="语音编码器 (speech_encoder)",
                choices=["vec768l12", "vec256l9", "hubertsoft", "whisper-ppg",
                         "cnhubertlarge", "dphubert", "whisper-ppg-large", "wavlmbase+"],
                value=_get_webui_config_key("flist_encoder", "vec768l12"),
            )
            flist_vol_aug = gr.Checkbox(label="音量增强 (--vol_aug)",
                                        value=_get_webui_config_key("flist_vol_aug", True))
            flist_tiny = gr.Checkbox(label="Tiny模型 (--tiny)",
                                     value=_get_webui_config_key("flist_tiny", False))
        with gr.Row():
            flist_start_btn = gr.Button("开始生成", variant="primary")
            flist_stop_btn = gr.Button("停止")
            flist_clear_btn = gr.Button("清除日志")
            flist_status = gr.Textbox(label="状态", value=get_flist_status, every=2, interactive=False, scale=2)
        flist_log = gr.Textbox(label="日志", value=get_flist_log, every=3, lines=8, max_lines=15, interactive=False)

        flist_start_btn.click(start_flist, [flist_encoder, flist_vol_aug, flist_tiny], [flist_status])
        flist_stop_btn.click(stop_flist, [], [flist_status])
        flist_clear_btn.click(clear_flist_log, [], [flist_log])

    # ── Step 3: Hubert + F0 ──────────────────────────────────────────
    with gr.Accordion("第三步：提取特征和F0 (preprocess_hubert_f0.py)", open=False):
        gr.Markdown("提取语音内容编码和基频。已处理的文件会自动跳过，支持断点续跑。")
        with gr.Row():
            hubert_f0 = gr.Dropdown(label="F0预测器",
                                    choices=["rmvpe", "crepe", "pm", "dio", "harvest", "fcpe"],
                                    value=_get_webui_config_key("hubert_f0", "rmvpe"))
            hubert_procs = gr.Slider(label="并行进程数", minimum=1, maximum=max(os.cpu_count() or 1, 1),
                                     value=_get_webui_config_key("hubert_num_processes", 1), step=1)
            hubert_diff = gr.Checkbox(label="同时提取扩散模型特征 (--use_diff)",
                                      value=_get_webui_config_key("hubert_use_diff", True))
            hubert_dev = gr.Dropdown(label="设备", choices=["cuda:0", "cpu"],
                                     value=_get_webui_config_key("hubert_device", "cuda:0"))
        with gr.Row():
            hubert_start_btn = gr.Button("开始提取", variant="primary")
            hubert_stop_btn = gr.Button("停止")
            hubert_clear_btn = gr.Button("清除日志")
            hubert_status = gr.Textbox(label="状态", value=get_hubert_status, every=2, interactive=False, scale=2)
        hubert_log = gr.Textbox(label="日志", value=get_hubert_log, every=3, lines=8, max_lines=15, interactive=False)

        hubert_start_btn.click(start_hubert, [hubert_f0, hubert_procs, hubert_diff, hubert_dev], [hubert_status])
        hubert_stop_btn.click(stop_hubert, [], [hubert_status])
        hubert_clear_btn.click(clear_hubert_log, [], [hubert_log])

    # ── Training config editor ───────────────────────────────────────
    with gr.Accordion("训练参数配置 (configs/config.json)", open=False):
        gr.Markdown("在启动训练前可在此调整关键参数。修改后点击「保存配置」。")
        with gr.Row():
            cfg_batch = gr.Number(label="batch_size (每批样本数)", value=6, precision=0)
            cfg_epochs = gr.Number(label="epochs (总训练轮次)", value=10000, precision=0)
            cfg_keep = gr.Number(label="keep_ckpts (保留检查点数)", value=3, precision=0)
            cfg_fp16 = gr.Checkbox(label="fp16_run (混合精度训练)", value=False)
        with gr.Row():
            cfg_load_btn = gr.Button("从文件加载")
            cfg_save_btn = gr.Button("保存配置", variant="primary")
            cfg_msg = gr.Textbox(label="提示", interactive=False, scale=2)

        cfg_load_btn.click(read_train_config, [], [cfg_batch, cfg_epochs, cfg_keep, cfg_fp16])
        cfg_save_btn.click(write_train_config, [cfg_batch, cfg_epochs, cfg_keep, cfg_fp16], [cfg_msg])

    # ── Step 4: Main training ────────────────────────────────────────
    with gr.Accordion("第四步：训练主模型 (train.py)", open=False):
        gr.Markdown("启动后自动从最新检查点续训。检查点保存在 `logs/44k/`。\n\n"
                    "Tensorboard 监控: 在命令行执行 `python -m tensorboard.main --logdir=logs/44k`")
        with gr.Row():
            train_start_btn = gr.Button("开始训练", variant="primary")
            train_stop_btn = gr.Button("停止训练")
            train_clear_btn = gr.Button("清除日志")
            train_status = gr.Textbox(label="状态", value=get_train_status, every=2, interactive=False, scale=2)
        train_log = gr.Textbox(label="训练日志", value=get_train_log, every=3, lines=15, max_lines=30, interactive=False)

        train_start_btn.click(start_train, [], [train_status])
        train_stop_btn.click(stop_train, [], [train_status])
        train_clear_btn.click(clear_train_log, [], [train_log])

    # ── Step 5: Diffusion training ───────────────────────────────────
    with gr.Accordion("第五步（可选）：训练扩散模型 (train_diff.py)", open=False):
        gr.Markdown("扩散模型为可选增强项。注意：与主模型训练会争抢 GPU 显存，建议分开运行。\n\n"
                    "检查点保存到 `logs/44k/diffusion/`")
        with gr.Row():
            diff_start_btn = gr.Button("开始训练扩散模型", variant="primary")
            diff_stop_btn = gr.Button("停止")
            diff_clear_btn = gr.Button("清除日志")
            diff_status = gr.Textbox(label="状态", value=get_train_diff_status, every=2, interactive=False, scale=2)
        diff_log = gr.Textbox(label="扩散模型训练日志", value=get_train_diff_log, every=3, lines=12, max_lines=25, interactive=False)

        diff_start_btn.click(start_train_diff, [], [diff_status])
        diff_stop_btn.click(stop_train_diff, [], [diff_status])
        diff_clear_btn.click(clear_train_diff_log, [], [diff_log])

    # ── Step 6: Index ────────────────────────────────────────────────
    with gr.Accordion("第六步：构建特征检索索引 (train_index.py)", open=False):
        gr.Markdown("构建 FAISS 检索索引用于推理时音色增强。运行很快（通常数分钟内完成）。\n\n"
                    "输出文件: `logs/44k/feature_and_index.pkl`")
        with gr.Row():
            index_start_btn = gr.Button("开始构建索引", variant="primary")
            index_stop_btn = gr.Button("停止")
            index_clear_btn = gr.Button("清除日志")
            index_status = gr.Textbox(label="状态", value=get_index_status, every=2, interactive=False, scale=2)
        index_log = gr.Textbox(label="日志", value=get_index_log, every=3, lines=8, max_lines=15, interactive=False)

        index_start_btn.click(start_index, [], [index_status])
        index_stop_btn.click(stop_index, [], [index_status])
        index_clear_btn.click(clear_index_log, [], [index_log])
