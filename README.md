# so-vits-svc-cu128

This is a fork of [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc) (SoftVC VITS Singing Voice Conversion), featuring a **Gradio WebUI** for training, inference, and model management. It targets **CUDA 12.8** and includes fixes for newer PyTorch versions on Windows.

## Differences from Upstream

### GUI

The original project is CLI-only. This fork provides a full Gradio-based WebUI with the following pages:

- **Inference** — Load models, convert voice, adjust parameters visually. Supports local model selection with memory of the last used model.
- **Training** — 7-step guided workflow from dataset preprocessing through SoVITS training, diffusion training, and clustering model training.
- **Management** — Delete/export checkpoints, manage exported models, manage feature retrieval and clustering models.

Launch via `so-vits-svc.bat` or directly:

```shell
python webUI.py
```

### Windows & Encoding Fixes

- All file I/O enforces `encoding='utf-8'` — fixes garbled text (mojibake) on Windows (GBK locale).
- `train.py` reading `config.json` now handles GBK-encoded files gracefully.
- `filelists` are always written as UTF-8, preventing training crashes caused by CJK filenames.
- Gradio upgraded from 3.36 to 4.44 for Windows compatibility.

### PyTorch Compatibility

- Tested with **CUDA 12.8** and PyTorch 2.6.
- Fixed `UnpicklingError` when loading clustering models under PyTorch 2.6 (removed `weights_only=True`).
- Clustering training uses `MiniBatchKMeans` to avoid memory exhaustion on large datasets.
- KMeans parameters auto-adapt based on dataset size and available system memory.
- Feature index building memory footprint reduced to prevent OOM crashes.

### Logging & UX

- Logs auto-scroll to the bottom using native Gradio autoscroll.
- Dedicated clear-log button below the log area.
- Fixed button visibility and process termination issues.
- 14 independent polling timers merged into one, fixing long-running disconnect issues.
- Gradio queue concurrency increased, fixing log freezing after tab switches.
- Polling skips updates when nothing changed, eliminating UI flicker.

### Training & Config

- Training parameters in the WebUI are initialized from `config.json` on startup.
- Clustering model training is integrated as step 7 in the training workflow.
- Preprocessing pipeline received multiple fixes for edge cases.

### Auto-Download

Pre-trained models can be automatically downloaded, reducing manual setup steps.

## Disclaimer

This project is open-source and offline. It does not collect user data. Users are responsible for ensuring they have the rights to use their training data and the audio they process.

## License

AGPL 3.0 — same as upstream.

## Original README

For detailed documentation on model architecture, dataset preparation, preprocessing, training, and inference parameters, see the [upstream repository](https://github.com/svc-develop-team/so-vits-svc).
