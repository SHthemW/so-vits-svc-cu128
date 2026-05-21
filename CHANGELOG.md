# Changelog

All notable changes to this fork of [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc).

## 2026-05-20

### Launcher & Distribution
- **feat:** Python launcher — one-click startup via bundled Python runtime.
- **docs:** Added pre-built environment package download link.
- **docs:** Updated README with environment version info and pip/cmake notes.

## 2026-05-19

### Startup & Path Fixes
- **fix:** Fixed launcher script path resolution and added bundled Python integrity check.
- **fix:** Startup script improvements.

### Inference
- **fix:** Auto-create `raw/` directory during inference to prevent audio write failure.
- **feat:** Inference page remembers last uploaded model file.

### Docs
- **docs:** Added Python version requirements (3.9–3.10) to README.

## 2026-05-18

### README
- **docs:** Updated README.

### Clustering Model Fixes
- **fix:** Fixed PyTorch 2.6 `UnpicklingError` when loading clustering models (removed `weights_only=True`).

## 2026-05-15

### Clustering Model Training
- **feat:** Added step 7 — clustering model training — to the training workflow.
- **feat:** Management page now supports feature retrieval / clustering model management.
- **feat:** KMeans parameters auto-adapt based on dataset size and available system memory.
- **fix:** KMeans `batch_size` adjusted from 256 to 4096 to avoid premature convergence.
- **fix:** Switched clustering training to `MiniBatchKMeans` to avoid memory exhaustion on large datasets.
- **fix:** Fixed import error when running clustering training as a package module.
- **fix:** Reduced feature index building memory footprint to prevent OOM crashes.

## 2026-05-14

### Management Page
- **feat:** New "Management" page — delete/export checkpoints and manage exported models.

### Gradio Upgrade & UI
- **upgrade:** Gradio 3.36 → 4.44, log auto-scroll now uses native `autoscroll`.
- **fix:** Merged 14 independent polling timers into one, fixing disconnect issues after long runs.
- **fix:** Increased Gradio queue concurrency, fixing log freeze after tab switches.
- **fix:** Polling skips updates when nothing changed, eliminating log area flicker.
- **feat:** Inference page remembers last selected local model.

## 2026-05-13

### Encoding & File I/O (Windows)
- **fix:** Global file read/write now enforces `encoding='utf-8'` — fixes garbled text on Windows (GBK locale).
- **fix:** `train.py` reading `config.json` now handles GBK-encoded files gracefully.
- **fix:** `filelists` are always written as UTF-8, preventing training crashes from CJK filenames.

### Training Config
- **feat:** Training parameters in WebUI are initialized from `config.json` on startup.

### Logging UX
- **feat:** Logs auto-scroll to bottom.
- **feat:** Clear-log button added.
- **fix:** Fixed clear-log button visibility and process termination issues.

### Preprocessing
- **fix:** Multiple preprocessing pipeline fixes for edge cases.

## 2026-05-12

### Initial Release
- **feat:** Base project files (forked from upstream so-vits-svc).
- **feat:** Gradio WebUI — inference, training, and management pages.
- **feat:** Windows launch script (`so-vits-svc.bat`).
- **feat:** Auto-download of pre-trained models.
