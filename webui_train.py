import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import gradio as gr

PYTHON = sys.executable
ROOT = Path(__file__).parent

_procs: dict = {
    "resample": None,
    "flist": None,
    "hubert": None,
    "train": None,
    "diff": None,
    "index": None,
}
_log_buffers: dict = {k: [] for k in _procs}
_lock = threading.Lock()
_MAX_LOG_LINES = 500

CONFIG_PATH = ROOT / "configs" / "config.json"


def _launch(key: str, args: list) -> str:
    with _lock:
        if _procs[key] is not None and _procs[key].poll() is None:
            return f"[{key}] 已经在运行中，请先停止"
        _log_buffers[key].clear()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        [PYTHON] + args,
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
        for line in proc.stdout:
            with _lock:
                _log_buffers[key].append(line.rstrip())
                if len(_log_buffers[key]) > _MAX_LOG_LINES:
                    _log_buffers[key] = _log_buffers[key][-_MAX_LOG_LINES:]
        proc.stdout.close()

    threading.Thread(target=_reader, daemon=True).start()
    return f"[{key}] 已启动 (PID {proc.pid})"


def _stop(key: str) -> str:
    with _lock:
        proc = _procs[key]
    if proc is None or proc.poll() is not None:
        return f"[{key}] 没有正在运行的进程"
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    return f"[{key}] 已停止"


def _get_log(key: str) -> str:
    with _lock:
        return "\n".join(_log_buffers[key])


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

def start_resample(skip_loudnorm):
    args = ["resample.py", "--sr2", "44100",
            "--in_dir", "./dataset_raw",
            "--out_dir2", "./dataset/44k"]
    if skip_loudnorm:
        args.append("--skip_loudnorm")
    return _launch("resample", args)


def stop_resample():
    return _stop("resample")


def get_resample_log():
    return _get_log("resample")


def get_resample_status():
    return _get_status("resample")


def start_flist(speech_encoder, vol_aug, tiny):
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


def get_flist_status():
    return _get_status("flist")


def start_hubert(f0_predictor, num_processes, use_diff, device):
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


def get_hubert_status():
    return _get_status("hubert")


def start_train():
    return _launch("train", ["train.py", "-c", "configs/config.json", "-m", "44k"])


def stop_train():
    return _stop("train")


def get_train_log():
    return _get_log("train")


def get_train_status():
    return _get_status("train")


def start_train_diff():
    return _launch("diff", ["train_diff.py", "-c", "configs/diffusion.yaml"])


def stop_train_diff():
    return _stop("diff")


def get_train_diff_log():
    return _get_log("diff")


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

def build_training_tab():
    gr.Markdown("## So-VITS-SVC 训练流程\n"
                "按顺序完成以下各步骤。请将训练音频放入 `dataset_raw/<说话人名>/` 目录后开始。\n\n"
                "训练进程在WebUI重启后会继续在后台运行，可通过 `logs/44k/train.log` 查看进度。")

    # ── Step 1: Resample ─────────────────────────────────────────────
    with gr.Accordion("第一步：音频重采样 (resample.py)", open=True):
        gr.Markdown("将 `dataset_raw/<说话人名>/` 下的原始音频重采样至 44100Hz mono，输出到 `dataset/44k/`")
        with gr.Row():
            resample_skip_loudnorm = gr.Checkbox(label="跳过响度归一化 (--skip_loudnorm)", value=False)
        with gr.Row():
            resample_start_btn = gr.Button("开始重采样", variant="primary")
            resample_stop_btn = gr.Button("停止")
            resample_status = gr.Textbox(label="状态", value=get_resample_status, every=2, interactive=False, scale=2)
        resample_log = gr.Textbox(label="日志", value=get_resample_log, every=3, lines=8, max_lines=15, interactive=False)

        resample_start_btn.click(start_resample, [resample_skip_loudnorm], [resample_status])
        resample_stop_btn.click(stop_resample, [], [resample_status])

    # ── Step 2: flist + config ───────────────────────────────────────
    with gr.Accordion("第二步：生成文件列表和配置 (preprocess_flist_config.py)", open=False):
        gr.Markdown("扫描 `dataset/44k/`，生成训练/验证文件列表以及 `configs/config.json`、`configs/diffusion.yaml`")
        with gr.Row():
            flist_encoder = gr.Dropdown(
                label="语音编码器 (speech_encoder)",
                choices=["vec768l12", "vec256l9", "hubertsoft", "whisper-ppg",
                         "cnhubertlarge", "dphubert", "whisper-ppg-large", "wavlmbase+"],
                value="vec768l12"
            )
            flist_vol_aug = gr.Checkbox(label="音量增强 (--vol_aug)", value=True)
            flist_tiny = gr.Checkbox(label="Tiny模型 (--tiny)", value=False)
        with gr.Row():
            flist_start_btn = gr.Button("开始生成", variant="primary")
            flist_stop_btn = gr.Button("停止")
            flist_status = gr.Textbox(label="状态", value=get_flist_status, every=2, interactive=False, scale=2)
        flist_log = gr.Textbox(label="日志", value=get_flist_log, every=3, lines=8, max_lines=15, interactive=False)

        flist_start_btn.click(start_flist, [flist_encoder, flist_vol_aug, flist_tiny], [flist_status])
        flist_stop_btn.click(stop_flist, [], [flist_status])

    # ── Step 3: Hubert + F0 ──────────────────────────────────────────
    with gr.Accordion("第三步：提取特征和F0 (preprocess_hubert_f0.py)", open=False):
        gr.Markdown("提取语音内容编码和基频。已处理的文件会自动跳过，支持断点续跑。")
        with gr.Row():
            hubert_f0 = gr.Dropdown(label="F0预测器", choices=["rmvpe", "crepe", "pm", "dio", "harvest", "fcpe"], value="rmvpe")
            hubert_procs = gr.Slider(label="并行进程数", minimum=1, maximum=max(os.cpu_count() or 1, 1), value=1, step=1)
            hubert_diff = gr.Checkbox(label="同时提取扩散模型特征 (--use_diff)", value=True)
            hubert_dev = gr.Dropdown(label="设备", choices=["cuda:0", "cpu"], value="cuda:0")
        with gr.Row():
            hubert_start_btn = gr.Button("开始提取", variant="primary")
            hubert_stop_btn = gr.Button("停止")
            hubert_status = gr.Textbox(label="状态", value=get_hubert_status, every=2, interactive=False, scale=2)
        hubert_log = gr.Textbox(label="日志", value=get_hubert_log, every=3, lines=8, max_lines=15, interactive=False)

        hubert_start_btn.click(start_hubert, [hubert_f0, hubert_procs, hubert_diff, hubert_dev], [hubert_status])
        hubert_stop_btn.click(stop_hubert, [], [hubert_status])

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
            train_status = gr.Textbox(label="状态", value=get_train_status, every=2, interactive=False, scale=2)
        train_log = gr.Textbox(label="训练日志", value=get_train_log, every=3, lines=15, max_lines=30, interactive=False)

        train_start_btn.click(start_train, [], [train_status])
        train_stop_btn.click(stop_train, [], [train_status])

    # ── Step 5: Diffusion training ───────────────────────────────────
    with gr.Accordion("第五步（可选）：训练扩散模型 (train_diff.py)", open=False):
        gr.Markdown("扩散模型为可选增强项。注意：与主模型训练会争抢 GPU 显存，建议分开运行。\n\n"
                    "检查点保存到 `logs/44k/diffusion/`")
        with gr.Row():
            diff_start_btn = gr.Button("开始训练扩散模型", variant="primary")
            diff_stop_btn = gr.Button("停止")
            diff_status = gr.Textbox(label="状态", value=get_train_diff_status, every=2, interactive=False, scale=2)
        diff_log = gr.Textbox(label="扩散模型训练日志", value=get_train_diff_log, every=3, lines=12, max_lines=25, interactive=False)

        diff_start_btn.click(start_train_diff, [], [diff_status])
        diff_stop_btn.click(stop_train_diff, [], [diff_status])

    # ── Step 6: Index ────────────────────────────────────────────────
    with gr.Accordion("第六步：构建特征检索索引 (train_index.py)", open=False):
        gr.Markdown("构建 FAISS 检索索引用于推理时音色增强。运行很快（通常数分钟内完成）。\n\n"
                    "输出文件: `logs/44k/feature_and_index.pkl`")
        with gr.Row():
            index_start_btn = gr.Button("开始构建索引", variant="primary")
            index_stop_btn = gr.Button("停止")
            index_status = gr.Textbox(label="状态", value=get_index_status, every=2, interactive=False, scale=2)
        index_log = gr.Textbox(label="日志", value=get_index_log, every=3, lines=8, max_lines=15, interactive=False)

        index_start_btn.click(start_index, [], [index_status])
        index_stop_btn.click(stop_index, [], [index_status])
