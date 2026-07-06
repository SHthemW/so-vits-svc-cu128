import glob
import json
import logging
import os
import re
import subprocess
import sys
import time
import traceback
import warnings
import webbrowser
from importlib.util import find_spec
from itertools import chain
from pathlib import Path
from typing import Optional


def _prepare_torchcodec_ffmpeg_libraries():
    av_spec = find_spec("av")
    if av_spec is None or not av_spec.submodule_search_locations:
        return

    av_libs = Path(av_spec.submodule_search_locations[0]).parent / "av.libs"
    if not av_libs.is_dir():
        return

    library_patterns = {
        "libavutil.so.60": "libavutil-*.so.60.*",
        "libavcodec.so.62": "libavcodec-*.so.62.*",
        "libavformat.so.62": "libavformat-*.so.62.*",
        "libavdevice.so.62": "libavdevice-*.so.62.*",
        "libavfilter.so.11": "libavfilter-*.so.11.*",
        "libswscale.so.9": "libswscale-*.so.9.*",
        "libswresample.so.6": "libswresample-*.so.6.*",
    }

    for soname, pattern in library_patterns.items():
        matches = sorted(av_libs.glob(pattern))
        if not matches:
            continue
        link_path = av_libs / soname
        target_name = matches[0].name
        if link_path.is_symlink() and os.readlink(link_path) == target_name:
            continue
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(target_name)

    av_libs_path = str(av_libs.resolve())
    library_paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    if av_libs_path not in library_paths and os.environ.get("SVC_FFMPEG_LIBS_READY") != "1":
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = (
            av_libs_path
            if not env.get("LD_LIBRARY_PATH")
            else f"{av_libs_path}:{env['LD_LIBRARY_PATH']}"
        )
        env["SVC_FFMPEG_LIBS_READY"] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


_prepare_torchcodec_ffmpeg_libraries()

# os.system("wget -P cvec/ https://huggingface.co/spaces/innnky/nanami/resolve/main/checkpoint_best_legacy_500.pt")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
warnings.filterwarnings(
    "ignore",
    message="The pynvml package is deprecated.*",
    category=FutureWarning,
)

import gradio as gr
import gradio.routes
import librosa
import numpy as np
import soundfile
import torch

from compress_model import removeOptimizer
from edgetts.tts_voices import SUPPORTED_LANGUAGES
from inference.infer_tool import Svc
from utils import mix_model
from webui_manage import (
    CONFIG_PATH,
    DIFF_CONFIG_PATH,
    DIFF_DIR,
    LOGS_DIR,
    TRAINED_DIR,
    _fmt_size,
    _fmt_time,
    build_management_tab,
    scan_exported_models,
)
from webui_train import build_training_tab, register_dataset_transfer_routes, _get_webui_config_key, _save_webui_config_key
from startup_banner import emit_startup_banner

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('markdown_it').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('multipart').setLevel(logging.WARNING)

model = None
spk = None
debug = False

local_model_root = str(TRAINED_DIR)
project_root = Path(__file__).parent

cuda = {}
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_properties(i).name
        cuda[f"CUDA:{i} {device_name}"] = f"cuda:{i}"


def _install_dataset_transfer_routes():
    original_create_app = gradio.routes.App.create_app
    if getattr(original_create_app, "_svc_dataset_transfer_patched", False):
        return

    def create_app_with_dataset_transfer(*args, **kwargs):
        fastapi_app = original_create_app(*args, **kwargs)
        register_dataset_transfer_routes(fastapi_app)
        return fastapi_app

    create_app_with_dataset_transfer._svc_dataset_transfer_patched = True
    gradio.routes.App.create_app = staticmethod(create_app_with_dataset_transfer)


_install_dataset_transfer_routes()


def _gradio_share_enabled() -> bool:
    value = os.environ.get("GRADIO_SHARE")
    if value is None:
        return True
    return value.strip().lower() in {"1", "true", "yes", "on"}


SVC_UI_CSS = """
.svc-card,
.svc-alert {
    box-sizing: border-box;
    padding: 14px;
    border: 1px solid var(--border-color-primary, #d0d7de);
    border-radius: 8px;
    background: var(--block-background-fill, #ffffff);
    color: var(--body-text-color, #1f2328);
    margin-bottom: 12px;
}

.svc-alert {
    border-left-width: 4px;
}

.svc-alert--warning {
    border-left-color: #d89614;
    background: color-mix(in srgb, #d89614 10%, var(--block-background-fill, #ffffff));
}

.svc-alert--error {
    border-left-color: #cf222e;
    background: color-mix(in srgb, #cf222e 10%, var(--block-background-fill, #ffffff));
}

.svc-alert--success {
    border-left-color: #2da44e;
    background: color-mix(in srgb, #2da44e 10%, var(--block-background-fill, #ffffff));
}

.svc-title {
    font-weight: 700;
    margin-bottom: 6px;
}

.svc-heading-row,
.svc-data-row,
.svc-summary {
    display: flex;
    justify-content: space-between;
    gap: 16px;
    align-items: center;
}

.svc-heading-row {
    margin-bottom: 8px;
}

.svc-data-row {
    padding: 6px 0;
    border-top: 1px solid var(--border-color-primary, #d0d7de);
}

.svc-summary {
    cursor: pointer;
    flex-wrap: wrap;
    gap: 12px;
}

.svc-muted {
    color: var(--body-text-color-subdued, #57606a);
    font-size: 13px;
}

.svc-total {
    margin-top: 10px;
    color: var(--body-text-color, #1f2328);
    font-weight: 600;
}

.svc-stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 8px;
    margin-top: 10px;
}

.svc-file-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 8px;
}

.svc-stat {
    padding: 6px 0;
    border-top: 1px solid var(--border-color-primary, #d0d7de);
}

.svc-cache {
    margin-top: 10px;
}

.svc-num {
    font-variant-numeric: tabular-nums;
}

.svc-note {
    display: grid;
    gap: 8px;
}

.svc-note-line {
    line-height: 1.55;
}

.svc-alert .svc-title,
.svc-card .svc-title {
    color: var(--body-text-color, #1f2328);
}

.svc-alert code,
.svc-card code {
    color: var(--body-text-color, #1f2328);
    background: var(--input-background-fill, #f6f8fa);
    border: 1px solid var(--border-color-primary, #d0d7de);
    border-radius: 4px;
    padding: 0 4px;
}
"""

SVC_UI_JS = r"""
() => {
  const originalFetch = window.fetch.bind(window);
  const gradioApiPath = /\/(run|queue\/join|queue\/data)(\/|\?|$)/;

  function requestUrl(input) {
    if (typeof input === "string") return input;
    if (input && typeof input.url === "string") return input.url;
    return "";
  }

  function requestMethod(input, init) {
    return (init && init.method) || (input && input.method) || "GET";
  }

  function shouldWrap(input, init) {
    return requestMethod(input, init).toUpperCase() === "POST" && gradioApiPath.test(requestUrl(input));
  }

  function cloneInit(init) {
    if (!init || !init.signal) return init;
    return { ...init, signal: undefined };
  }

  async function fetchWithRetry(input, init) {
    let lastResponse = null;
    let lastError = null;
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        const response = await originalFetch(input, attempt === 0 ? init : cloneInit(init));
        const contentType = response.headers.get("content-type") || "";
        if (contentType.toLowerCase().includes("application/json")) {
          return response;
        }
        lastResponse = response.clone();
      } catch (error) {
        lastError = error;
      }
      await new Promise((resolve) => setTimeout(resolve, 700 * (attempt + 1)));
    }

    if (lastResponse) {
      const text = await lastResponse.text().catch(() => "");
      const preview = text.replace(/\s+/g, " ").trim().slice(0, 240);
      return new Response(
        JSON.stringify({
          error: "公网 .live 返回了非 JSON 响应，请重试或使用本地地址。"
            + (preview ? " 响应摘要: " + preview : "")
        }),
        {
          status: lastResponse.ok ? 502 : lastResponse.status || 502,
          statusText: lastResponse.statusText || "Bad Gateway",
          headers: { "content-type": "application/json" }
        }
      );
    }

    return new Response(
      JSON.stringify({
        error: "公网 .live 请求失败，请重试或使用本地地址。"
          + (lastError ? " " + lastError.message : "")
      }),
      { status: 502, statusText: "Bad Gateway", headers: { "content-type": "application/json" } }
    );
  }

  const compressionState = {
    initialized: false,
    uploadInFlight: false,
    compressedUploadBytes: 0,
    uploadProgressTimer: null,
    uploadCompleteLogged: false,
    parseWaveBaseline: "",
    parseObserver: null,
    parseTimer: null,
  };

  function uploadProgressMessage(loaded, total) {
    const expected = compressionState.compressedUploadBytes || total || 0;
    const denominator = total || expected;
    const percent = denominator ? Math.min(100, Math.round((loaded / denominator) * 100)) : 0;
    const loadedText = denominator ? `${formatBytes(Math.min(loaded, denominator))} / ${formatBytes(denominator)}` : formatBytes(loaded);
    return `上传中: ${percent}% (${loadedText}，压缩后 ${formatBytes(expected)})`;
  }

  function nativeUploadPercent() {
    const cssValue = document.documentElement.style.getPropertyValue("--upload-progress-width");
    const cssPercent = Number.parseFloat(cssValue);
    if (Number.isFinite(cssPercent)) return clamp(cssPercent, 0, 100);
    const progress = audioComponentRoot()?.querySelector("progress");
    if (progress && Number.isFinite(progress.value)) return clamp(progress.value, 0, 100);
    return null;
  }

  function startUploadProgressWatch() {
    stopUploadProgressWatch();
    document.documentElement.style.setProperty("--upload-progress-width", "0%");
    compressionState.uploadInFlight = true;
    compressionState.uploadCompleteLogged = false;
    let lastPercent = -1;
    const startedAt = Date.now();
    logCompressionStep(uploadProgressMessage(0, compressionState.compressedUploadBytes));
    compressionState.uploadProgressTimer = window.setInterval(() => {
      const currentWaveformSignature = waveformSignature();
      if (currentWaveformSignature && currentWaveformSignature !== compressionState.parseWaveBaseline) {
        markUploadComplete();
        return;
      }
      if (Date.now() - startedAt > 120000) {
        compressionState.uploadInFlight = false;
        stopUploadProgressWatch();
        logCompressionStep("上传状态未知，请查看上传控件");
        return;
      }
      const percent = nativeUploadPercent();
      if (percent === null || Math.round(percent) === Math.round(lastPercent)) return;
      lastPercent = percent;
      const total = compressionState.compressedUploadBytes;
      const loaded = total ? Math.round(total * percent / 100) : 0;
      logCompressionStep(uploadProgressMessage(loaded, total));
      if (percent >= 99.5 && Date.now() - startedAt > 300) {
        markUploadComplete();
      }
    }, 200);
  }

  function stopUploadProgressWatch() {
    if (compressionState.uploadProgressTimer) {
      window.clearInterval(compressionState.uploadProgressTimer);
      compressionState.uploadProgressTimer = null;
    }
  }

  function markUploadComplete() {
    if (compressionState.uploadCompleteLogged) return;
    compressionState.uploadCompleteLogged = true;
    compressionState.uploadInFlight = false;
    stopUploadProgressWatch();
    logCompressionStep(`上传完成: 压缩后 ${formatBytes(compressionState.compressedUploadBytes)}`);
    beginAudioParseWatch();
  }

  window.fetch = (input, init) => {
    if (shouldWrap(input, init)) return fetchWithRetry(input, init);
    return originalFetch(input, init);
  };

  function componentContainsEvent(componentId, event) {
    const path = typeof event.composedPath === "function" ? event.composedPath() : [];
    return path.some((node) => node && node.id === componentId);
  }

  function controlRoot(id) {
    return document.getElementById(id);
  }

  function checkboxValue(id) {
    const root = controlRoot(id);
    const input = root && root.querySelector('input[type="checkbox"]');
    return Boolean(input && input.checked);
  }

  function numericValue(id, fallback) {
    const root = controlRoot(id);
    if (!root) return fallback;
    const input = root.querySelector('input[type="number"], input[type="range"]');
    const value = input ? Number(input.value) : Number(root.textContent);
    return Number.isFinite(value) ? value : fallback;
  }

  function setNativeValue(element, value) {
    const prototype = element instanceof HTMLTextAreaElement
      ? HTMLTextAreaElement.prototype
      : HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(prototype, "value")?.set;
    if (setter) setter.call(element, value);
    else element.value = value;
    element.dispatchEvent(new Event("input", { bubbles: true }));
    element.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function updateOutputMessage(message) {
    const root = document.getElementById("svc-output-message");
    const field = root && root.querySelector("textarea, input");
    if (field) setNativeValue(field, message || "");
  }

  function logCompressionStep(message) {
    updateOutputMessage(message);
    console.info(`[SVC音频上传] ${message}`);
  }

  function formatBytes(bytes) {
    if (!Number.isFinite(bytes)) return "0 B";
    if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
    if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${bytes} B`;
  }

  function audioComponentRoot() {
    return document.getElementById("svc-vc-audio-input");
  }

  function audioComponentLooksParsed() {
    const signature = waveformSignature();
    return Boolean(signature && signature !== compressionState.parseWaveBaseline);
  }

  function collectDomTreeNodes(root, selector) {
    const nodes = [];
    const visit = (node) => {
      if (!node) return;
      if (node.querySelectorAll) nodes.push(...node.querySelectorAll(selector));
      const descendants = node.querySelectorAll ? Array.from(node.querySelectorAll("*")) : [];
      for (const descendant of descendants) {
        if (descendant.shadowRoot) visit(descendant.shadowRoot);
      }
    };
    visit(root);
    return nodes;
  }

  function waveformSignature() {
    const root = audioComponentRoot();
    if (!root) return "";
    const waveformHosts = collectDomTreeNodes(root, "#waveform, [data-testid^='waveform-'], .waveform-container");
    const scanRoots = waveformHosts.length ? waveformHosts : [root];
    const canvasSignatures = scanRoots.flatMap((scanRoot) => collectDomTreeNodes(scanRoot, "canvas"))
      .map(canvasRenderSignature)
      .filter(Boolean);
    const svgSignatures = scanRoots.flatMap((scanRoot) => collectDomTreeNodes(scanRoot, "svg"))
      .map(svgWaveSignature)
      .filter(Boolean);
    const durationSignature = waveformDurationSignature(root);
    return [...canvasSignatures, ...svgSignatures, durationSignature].filter(Boolean).join("|");
  }

  function waveformDurationSignature(root) {
    const duration = root.querySelector("#duration");
    const durationText = duration ? duration.textContent.trim() : "";
    if (!durationText || durationText === "0:00") return "";
    const waveformHost = root.querySelector("#waveform");
    const hostBox = waveformHost && waveformHost.getBoundingClientRect();
    if (!hostBox || hostBox.width < 40 || hostBox.height < 16) return "";
    const shadowSize = waveformHost.shadowRoot ? waveformHost.shadowRoot.textContent.length : 0;
    return `duration:${durationText}:${Math.round(hostBox.width)}x${Math.round(hostBox.height)}:${shadowSize}`;
  }

  function canvasRenderSignature(canvas) {
    const box = canvas.getBoundingClientRect();
    if (!canvas.width || !canvas.height || box.width < 40 || box.height < 16) return "";
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) return "";
    try {
      const width = canvas.width;
      const height = canvas.height;
      const sampleWidth = Math.min(width, 240);
      const sampleHeight = Math.min(height, 80);
      const image = context.getImageData(0, 0, sampleWidth, sampleHeight).data;
      let alphaPixels = 0;
      let coloredPixels = 0;
      let hash = 2166136261;
      for (let i = 0; i < image.length; i += 16) {
        const alpha = image[i + 3];
        if (!alpha) continue;
        alphaPixels += 1;
        const r = image[i];
        const g = image[i + 1];
        const b = image[i + 2];
        if (r !== 255 || g !== 255 || b !== 255) {
          coloredPixels += 1;
          hash ^= r + (g << 8) + (b << 16) + (alpha << 24);
          hash = Math.imul(hash, 16777619);
        }
      }
      if (coloredPixels < 8) return "";
      return `canvas:${width}x${height}:${alphaPixels}:${coloredPixels}:${hash >>> 0}`;
    } catch (error) {
      return "";
    }
  }

  function svgWaveSignature(svg) {
    const box = svg.getBoundingClientRect();
    if (!box || box.width < 40 || box.height < 16) return "";
    const shapes = Array.from(svg.querySelectorAll("path[d], polyline[points], polygon[points]"))
      .map((node) => node.getAttribute("d") || node.getAttribute("points") || "")
      .filter((value) => value.length > 12);
    if (!shapes.length) return "";
    let hash = 2166136261;
    const joined = shapes.join("|");
    for (let i = 0; i < joined.length; i++) {
      hash ^= joined.charCodeAt(i);
      hash = Math.imul(hash, 16777619);
    }
    return `svg:${Math.round(box.width)}x${Math.round(box.height)}:${shapes.length}:${hash >>> 0}`;
  }

  function stopAudioParseWatch() {
    if (compressionState.parseObserver) {
      compressionState.parseObserver.disconnect();
      compressionState.parseObserver = null;
    }
    if (compressionState.parseTimer) {
      window.clearInterval(compressionState.parseTimer);
      compressionState.parseTimer = null;
    }
  }

  function beginAudioParseWatch() {
    stopAudioParseWatch();
    logCompressionStep("音频解析中...");
    const startedAt = Date.now();
    const markDoneIfReady = () => {
      if (audioComponentLooksParsed()) {
        stopAudioParseWatch();
        logCompressionStep("解析完成");
        return true;
      }
      if (Date.now() - startedAt > 120000) {
        stopAudioParseWatch();
        logCompressionStep("音频解析状态未知，请查看上传控件");
        return true;
      }
      return false;
    };

    const root = audioComponentRoot();
    if (root) {
      compressionState.parseObserver = new MutationObserver(markDoneIfReady);
      compressionState.parseObserver.observe(root, {
        childList: true,
        subtree: true,
        attributes: true,
        characterData: true,
      });
    }
    compressionState.parseTimer = window.setInterval(markDoneIfReady, 250);
    markDoneIfReady();
  }

  function clamp(number, min, max) {
    return Math.min(max, Math.max(min, number));
  }

  function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  function audioBufferToWav(channelData, sampleRate) {
    const bytesPerSample = 2;
    const dataSize = channelData.length * bytesPerSample;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    writeString(view, 0, "RIFF");
    view.setUint32(4, 36 + dataSize, true);
    writeString(view, 8, "WAVE");
    writeString(view, 12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * bytesPerSample, true);
    view.setUint16(32, bytesPerSample, true);
    view.setUint16(34, 8 * bytesPerSample, true);
    writeString(view, 36, "data");
    view.setUint32(40, dataSize, true);

    let offset = 44;
    for (let i = 0; i < channelData.length; i++, offset += 2) {
      const sample = clamp(channelData[i], -1, 1);
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    }

    return new Blob([view], { type: "audio/wav" });
  }

  function mixToMono(audioBuffer) {
    const length = audioBuffer.length;
    const channels = audioBuffer.numberOfChannels;
    const mono = new Float32Array(length);
    for (let channel = 0; channel < channels; channel++) {
      const data = audioBuffer.getChannelData(channel);
      for (let i = 0; i < length; i++) mono[i] += data[i] / channels;
    }
    return mono;
  }

  function besselI0(x) {
    let sum = 1;
    let term = 1;
    const half = x / 2;
    for (let k = 1; k <= 24; k++) {
      term *= (half * half) / (k * k);
      sum += term;
      if (term < sum * 1e-12) break;
    }
    return sum;
  }

  function sinc(x) {
    if (Math.abs(x) < 1e-8) return 1;
    const pix = Math.PI * x;
    return Math.sin(pix) / pix;
  }

  function highQualityResample(input, sourceRate, targetRate) {
    if (sourceRate === targetRate) return input.slice();
    const ratio = targetRate / sourceRate;
    const outputLength = Math.max(1, Math.round(input.length * ratio));
    const output = new Float32Array(outputLength);
    const filterScale = Math.min(1, ratio);
    const radius = 32;
    const beta = 8.6;
    const support = Math.ceil(radius / filterScale);
    const phaseCount = 2048;
    const betaI0 = besselI0(beta);
    const tables = Array.from({ length: phaseCount }, (_, phaseIndex) => {
      const fraction = phaseIndex / phaseCount;
      const weights = new Float32Array(support * 2 + 1);
      for (let offset = -support; offset <= support; offset++) {
        const x = (fraction - offset) * filterScale;
        const r = x / radius;
        const window = Math.abs(r) >= 1
          ? 0
          : besselI0(beta * Math.sqrt(1 - r * r)) / betaI0;
        weights[offset + support] = filterScale * sinc(x) * window;
      }
      return weights;
    });

    for (let i = 0; i < outputLength; i++) {
      const center = i / ratio;
      const base = Math.floor(center);
      const phaseIndex = Math.min(phaseCount - 1, Math.max(0, Math.round((center - base) * phaseCount)));
      const weights = tables[phaseIndex];
      let sum = 0;
      let weightSum = 0;

      for (let offset = -support; offset <= support; offset++) {
        const j = base + offset;
        if (j < 0 || j >= input.length) continue;
        const weight = weights[offset + support];
        sum += input[j] * weight;
        weightSum += weight;
      }

      output[i] = weightSum ? sum / weightSum : 0;
    }

    return output;
  }

  async function decodeAudioFile(file) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) throw new Error("当前浏览器不支持 AudioContext");
    const context = new AudioContextClass();
    try {
      const arrayBuffer = await file.arrayBuffer();
      const decoded = await context.decodeAudioData(arrayBuffer.slice(0));
      return decoded;
    } finally {
      if (typeof context.close === "function") {
        context.close().catch(() => {});
      }
    }
  }

  async function compressAudioFile(file, ratioPercent) {
    const decoded = await decodeAudioFile(file);
    const ratio = clamp(Number(ratioPercent) || 20, 1, 100) / 100;
    const qualityFloorRate = decoded.sampleRate >= 32000 ? 24000 : decoded.sampleRate;
    const targetSampleRate = clamp(
      Math.round(Math.max(decoded.sampleRate * ratio, qualityFloorRate)),
      1000,
      decoded.sampleRate
    );
    const mono = mixToMono(decoded);
    const resampled = highQualityResample(mono, decoded.sampleRate, targetSampleRate);
    const wavBlob = audioBufferToWav(resampled, targetSampleRate);
    const originalName = file.name || "audio";
    const stem = originalName.replace(/\.[^/.]+$/, "") || "audio";
    const compressedName = `${stem}_compressed_${Math.round(ratio * 100)}p.wav`;
    const compressedFile = new File([wavBlob], compressedName, {
      type: "audio/wav",
      lastModified: Date.now(),
    });
    Object.defineProperty(compressedFile, "svcTargetSampleRate", { value: targetSampleRate });
    return compressedFile;
  }

  function replaceInputFiles(input, files) {
    const transfer = new DataTransfer();
    for (const file of files) transfer.items.add(file);
    input.files = transfer.files;
  }

  async function handleAudioUploadChange(event) {
    const input = event.target;
    if (!(input instanceof HTMLInputElement) || input.type !== "file") return;
    if (!componentContainsEvent("svc-vc-audio-input", event)) return;
    if (input.dataset.svcCompressedUpload === "1") {
      delete input.dataset.svcCompressedUpload;
      return;
    }
    if (input.dataset.svcCompressionBypass === "1") {
      delete input.dataset.svcCompressionBypass;
      return;
    }
    if (!checkboxValue("svc-audio-compress-enabled")) {
      return;
    }

    const files = Array.from(input.files || []);
    if (!files.length) return;

    event.preventDefault();
    event.stopImmediatePropagation();

    const ratio = clamp(numericValue("svc-audio-compress-ratio", 20), 1, 100);
    stopAudioParseWatch();
    stopUploadProgressWatch();
    compressionState.uploadInFlight = false;
    compressionState.uploadCompleteLogged = false;
    logCompressionStep(`压缩中: 目标比例 ${ratio}%`);

    try {
      const compressedFiles = [];
      for (const file of files) {
        compressedFiles.push(await compressAudioFile(file, ratio));
      }
      const beforeBytes = files.reduce((sum, file) => sum + file.size, 0);
      const afterBytes = compressedFiles.reduce((sum, file) => sum + file.size, 0);
      const sampleRates = Array.from(new Set(
        compressedFiles.map((file) => file.svcTargetSampleRate).filter(Boolean)
      ));
      const sampleRateText = sampleRates.length
        ? `，采样率 ${sampleRates.map((rate) => `${(rate / 1000).toFixed(rate % 1000 ? 1 : 0)}kHz`).join("/")}`
        : "";
      logCompressionStep(`压缩完成: ${formatBytes(beforeBytes)} -> ${formatBytes(afterBytes)}${sampleRateText}`);

      compressionState.parseWaveBaseline = waveformSignature();
      compressionState.compressedUploadBytes = afterBytes;
      replaceInputFiles(input, compressedFiles);
      input.dataset.svcCompressedUpload = "1";
      startUploadProgressWatch();
      input.dispatchEvent(new Event("change", { bubbles: true, composed: true }));
    } catch (error) {
      input.dataset.svcCompressionBypass = "1";
      input.dispatchEvent(new Event("change", { bubbles: true, composed: true }));
      logCompressionStep(`浏览器压缩失败，已按原文件上传: ${error.message || error}`);
    }
  }

  function installAudioCompressionHook() {
    if (compressionState.initialized) return;
    compressionState.initialized = true;
    document.addEventListener("change", handleAudioUploadChange, true);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", installAudioCompressionHook, { once: true });
  } else {
    installAudioCompressionHook();
  }
}
"""

def upload_mix_append_file(files,sfiles):
    try:
        if(sfiles is None):
            file_paths = [file.name for file in files]
        else:
            file_paths = [file.name for file in chain(files,sfiles)]
        p = {file:100 for file in file_paths}
        return file_paths, gr.Textbox(value=json.dumps(p, indent=2))
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)

def mix_submit_click(js,mode):
    try:
        assert js.lstrip()!=""
        modes = {"凸组合":0, "线性组合":1}
        mode = modes[mode]
        data = json.loads(js)
        data = list(data.items())
        model_path,mix_rate = zip(*data)
        path = mix_model(model_path,mix_rate,mode)
        return f"成功，文件被保存在了{path}"
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)

def updata_mix_info(files):
    try:
        if files is None:
            return gr.Textbox(value="")
        p = {file.name:100 for file in files}
        return gr.Textbox(value=json.dumps(p, indent=2))
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)

def _parse_dropdown_selection(selection: str) -> str:
    return selection.split("|")[0].strip() if selection else ""


def _local_model_dir_from_selection(selection: str) -> str:
    rel_dir = _parse_dropdown_selection(selection)
    if not rel_dir:
        return ""
    return str(TRAINED_DIR / rel_dir)


def _server_path_from_selection(selection: str) -> Optional[Path]:
    rel_path = _parse_dropdown_selection(selection)
    if not rel_path:
        return None
    path = Path(rel_path)
    if path.is_absolute():
        return path
    if rel_path.startswith(("logs/", "configs/")):
        return project_root / rel_path
    return TRAINED_DIR / rel_path


def _local_model_paths_from_selection(selection: str) -> tuple[str, str]:
    path = _server_path_from_selection(selection)
    if path is None:
        return "", ""
    if path.is_file():
        config_path = path.parent / "config.json"
        if not config_path.exists():
            config_path = CONFIG_PATH
        return str(path), str(config_path)
    return _pick_model_file(str(path)), _pick_single_file(str(path), "*.json", "配置文件")


def _local_file_path_from_selection(selection: str) -> str:
    path = _server_path_from_selection(selection)
    if path is None:
        return ""
    return str(path)


def _pick_single_file(folder: str, pattern: str, label: str) -> str:
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if not files:
        raise gr.Error(f"本地模型目录中未找到{label}: {folder}")
    return files[0]


def _pick_model_file(folder: str) -> str:
    files = sorted(glob.glob(os.path.join(folder, "*.pth")))
    candidates = [f for f in files if not Path(f).name.startswith("D_")]
    if not candidates:
        raise gr.Error(f"本地模型目录中未找到模型文件: {folder}")
    return candidates[0]


def _last_local_model_choice(choices: list[str]) -> Optional[str]:
    last = _get_webui_config_key("last_local_model", None)
    if not last:
        return None
    for choice in choices:
        if _parse_dropdown_selection(choice) == last:
            return choice
    return None


def _last_local_file_choice(config_key: str, choices: list[str]) -> Optional[str]:
    last = _get_webui_config_key(config_key, None)
    if not last:
        return None
    last_path = Path(last)
    if last_path.is_absolute():
        try:
            last = str(last_path.relative_to(TRAINED_DIR))
        except ValueError:
            return None
    for choice in choices:
        if _parse_dropdown_selection(choice) == last:
            return choice
    return None


def modelAnalysis(model_path,config_path,cluster_model_path,device,enhance,diff_model_path,diff_config_path,only_diffusion,use_spk_mix,local_model_enabled,local_model_selection,local_diff_model_selection,local_diff_config_selection,local_cluster_model_selection):
    global model
    try:
        device = cuda[device] if "CUDA" in device else device
        # get model and config path
        if (local_model_enabled):
            # local path
            model_path, config_path = _local_model_paths_from_selection(local_model_selection)
            if not model_path:
                raise gr.Error("请先选择服务器本地模型")
            diff_model_path = _local_file_path_from_selection(local_diff_model_selection)
            diff_config_path = _local_file_path_from_selection(local_diff_config_selection)
            cluster_model_path = _local_file_path_from_selection(local_cluster_model_selection)
        else:
            # upload from webpage
            if model_path is None:
                raise gr.Error("请先上传模型文件，并等待上传完成")
            if config_path is None:
                raise gr.Error("请先上传配置文件，并等待上传完成")
            model_path = model_path.name
            config_path = config_path.name
            diff_model_path = diff_model_path.name if diff_model_path is not None else ""
            diff_config_path = diff_config_path.name if diff_config_path is not None else ""
            cluster_model_path = cluster_model_path.name if cluster_model_path is not None else ""
        cluster_filepath = os.path.split(cluster_model_path) if cluster_model_path else ("", "no_cluster")
        fr = ".pkl" in cluster_filepath[1]
        model = Svc(model_path,
                config_path,
                device=device if device != "Auto" else None,
                cluster_model_path = cluster_model_path,
                nsf_hifigan_enhance=enhance,
                diffusion_model_path = diff_model_path,
                diffusion_config_path = diff_config_path,
                shallow_diffusion = True if diff_model_path else False,
                only_diffusion = only_diffusion,
                spk_mix_enable = use_spk_mix,
                feature_retrieval = fr
                )
        spks = list(model.spk2id.keys())
        device_name = torch.cuda.get_device_properties(model.dev).name if "cuda" in str(model.dev) else str(model.dev)
        if local_model_enabled and local_model_selection:
            _save_webui_config_key("last_local_model", _parse_dropdown_selection(local_model_selection))
            if local_diff_model_selection:
                _save_webui_config_key("last_local_diff_model", _parse_dropdown_selection(local_diff_model_selection))
            if local_diff_config_selection:
                _save_webui_config_key("last_local_diff_config", _parse_dropdown_selection(local_diff_config_selection))
            if local_cluster_model_selection:
                _save_webui_config_key("last_local_cluster_model", _parse_dropdown_selection(local_cluster_model_selection))
        else:
            _save_webui_config_key("last_upload_model", model_path)
            _save_webui_config_key("last_upload_config", config_path)
            if cluster_model_path:
                _save_webui_config_key("last_cluster_model", cluster_model_path)
            if diff_model_path:
                _save_webui_config_key("last_diff_model", diff_model_path)
            if diff_config_path:
                _save_webui_config_key("last_diff_config", diff_config_path)
        msg = f"成功加载模型到设备{device_name}上\n"
        if not cluster_model_path:
            msg += "未加载聚类模型或特征检索模型\n"
        elif fr:
            msg += f"特征检索模型{cluster_filepath[1]}加载成功\n"
        else:
            msg += f"聚类模型{cluster_filepath[1]}加载成功\n"
        if not diff_model_path:
            msg += "未加载扩散模型\n"
        else:
            msg += f"扩散模型{os.path.basename(diff_model_path)}加载成功\n"
        msg += "当前模型的可用音色：\n"
        for i in spks:
            msg += i + " "
        return gr.Dropdown(choices=spks, value=spks[0]), msg
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)

    
def modelUnload():
    global model
    if model is None:
        return gr.Dropdown(choices=[], value=""), "没有模型需要卸载!"
    else:
        model.unload_model()
        model = None
        torch.cuda.empty_cache()
        return gr.Dropdown(choices=[], value=""), "模型卸载完毕!"
    
def vc_infer(output_format, sid, audio_path, truncated_basename, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment):
    global model
    _audio = model.slice_inference(
        audio_path,
        sid,
        vc_transform,
        slice_db,
        cluster_ratio,
        auto_f0,
        noise_scale,
        pad_seconds,
        cl_num,
        lg_num,
        lgr_num,
        f0_predictor,
        enhancer_adaptive_key,
        cr_threshold,
        k_step,
        use_spk_mix,
        second_encoding,
        loudness_envelope_adjustment
    )  
    model.clear_empty()
    #构建保存文件的路径，并保存到results文件夹内
    str(int(time.time()))
    if not os.path.exists("results"):
        os.makedirs("results")
    key = "auto" if auto_f0 else f"{int(vc_transform)}key"
    cluster = "_" if cluster_ratio == 0 else f"_{cluster_ratio}_"
    isdiffusion = "sovits"
    if model.shallow_diffusion:
        isdiffusion = "sovdiff"

    if model.only_diffusion:
        isdiffusion = "diff"
    
    output_file_name = 'result_'+truncated_basename+f'_{sid}_{key}{cluster}{isdiffusion}.{output_format}'
    output_file = os.path.join("results", output_file_name)
    soundfile.write(output_file, _audio, model.target_sample, format=output_format)
    return output_file

def vc_fn(sid, input_audio, output_format, vc_transform, auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,f0_predictor,enhancer_adaptive_key,cr_threshold,k_step,use_spk_mix,second_encoding,loudness_envelope_adjustment):
    global model
    try:
        if input_audio is None:
            return "You need to upload an audio", None
        if model is None:
            return "You need to upload an model", None
        if getattr(model, 'cluster_model', None) is None and model.feature_retrieval is False:
            if cluster_ratio != 0:
                return "You need to upload an cluster model or feature retrieval model before assigning cluster ratio!", None
        #print(input_audio)    
        audio, sampling_rate = soundfile.read(input_audio)
        #print(audio.shape,sampling_rate)
        if np.issubdtype(audio.dtype, np.integer):
            audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        #print(audio.dtype)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        # 从Gradio上传的临时文件路径中提取basename（去掉可能存在的随机后缀）
        raw_stem = Path(input_audio).stem
        # 旧版Gradio会在文件名末尾追加6位随机串，新版不再追加。仅当stem足够长时才尝试截断
        if len(raw_stem) > 10:
            truncated_basename = raw_stem[:-6]
        else:
            truncated_basename = raw_stem
        if not truncated_basename:
            truncated_basename = "audio"
        processed_audio = os.path.join("raw", f"{truncated_basename}.wav")
        os.makedirs("raw", exist_ok=True)
        soundfile.write(processed_audio, audio, sampling_rate, format="wav")
        output_file = vc_infer(output_format, sid, processed_audio, truncated_basename, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment)

        return "Success", output_file
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)

def text_clear(text):
    return re.sub(r"[\n\,\(\) ]", "", text)

def vc_fn2(_text, _lang, _gender, _rate, _volume, sid, output_format, vc_transform, auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,f0_predictor,enhancer_adaptive_key,cr_threshold, k_step,use_spk_mix,second_encoding,loudness_envelope_adjustment):
    global model
    try:
        if model is None:
            return "You need to upload an model", None
        if getattr(model, 'cluster_model', None) is None and model.feature_retrieval is False:
            if cluster_ratio != 0:
                return "You need to upload an cluster model or feature retrieval model before assigning cluster ratio!", None
        _rate = f"+{int(_rate*100)}%" if _rate >= 0 else f"{int(_rate*100)}%"
        _volume = f"+{int(_volume*100)}%" if _volume >= 0 else f"{int(_volume*100)}%"
        if _lang == "Auto":
            _gender = "Male" if _gender == "男" else "Female"
            subprocess.run([sys.executable, "edgetts/tts.py", _text, _lang, _rate, _volume, _gender])
        else:
            subprocess.run([sys.executable, "edgetts/tts.py", _text, _lang, _rate, _volume])
        target_sr = 44100
        y, sr = librosa.load("tts.wav")
        resampled_y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        soundfile.write("tts.wav", resampled_y, target_sr, subtype = "PCM_16")
        input_audio = "tts.wav"
        #audio, _ = soundfile.read(input_audio)
        output_file_path = vc_infer(output_format, sid, input_audio, "tts", vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment)
        os.remove("tts.wav")
        return "Success", output_file_path
    except Exception as e:
        if debug: traceback.print_exc()  # noqa: E701
        raise gr.Error(e)

def model_compression(_model):
    if _model == "":
        return "请先选择要压缩的模型"
    else:
        model_path = os.path.split(_model.name)
        filename, extension = os.path.splitext(model_path[1])
        output_model_name = f"{filename}_compressed{extension}"
        output_path = os.path.join(os.getcwd(), output_model_name)
        removeOptimizer(_model.name, output_path)
        return f"模型已成功被保存在了{output_path}"

def scan_local_models():
    choices = list(scan_exported_models())
    if LOGS_DIR.exists():
        for f in sorted(LOGS_DIR.glob("G_*.pth")):
            rel = f.relative_to(project_root)
            choices.append(f"{rel} | 检查点 | {_fmt_size(f.stat().st_size)} | {_fmt_time(f.stat().st_mtime)}")
    return choices


def _scan_files(roots: list[Path], patterns: list[str]) -> list[str]:
    seen = set()
    choices = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for f in sorted(root.rglob(pattern)):
                if not f.is_file() or f in seen:
                    continue
                seen.add(f)
                if f.is_relative_to(TRAINED_DIR):
                    rel = f.relative_to(TRAINED_DIR)
                else:
                    rel = f.relative_to(project_root)
                choices.append(f"{rel} | {_fmt_size(f.stat().st_size)} | {_fmt_time(f.stat().st_mtime)}")
    return choices


def scan_local_diffusion_models():
    return _scan_files([TRAINED_DIR, DIFF_DIR], ["diffusion_*.pt", "model_*.pt"])


def scan_local_diffusion_configs():
    choices = _scan_files([TRAINED_DIR, DIFF_DIR], ["diffusion.yaml", "diffusion.yml", "config.yaml", "config.yml"])
    if DIFF_CONFIG_PATH.exists():
        rel = DIFF_CONFIG_PATH.relative_to(project_root)
        choices.append(f"{rel} | {_fmt_size(DIFF_CONFIG_PATH.stat().st_size)} | {_fmt_time(DIFF_CONFIG_PATH.stat().st_mtime)}")
    return choices


def scan_local_cluster_models():
    return _scan_files([TRAINED_DIR, LOGS_DIR], ["feature_and_index.pkl", "kmeans_*.pt"])


def local_model_refresh_fn():
    model_choices = scan_local_models()
    diff_model_choices = scan_local_diffusion_models()
    diff_config_choices = scan_local_diffusion_configs()
    cluster_choices = scan_local_cluster_models()
    return (
        gr.Dropdown(choices=model_choices, value=_last_local_model_choice(model_choices)),
        gr.Dropdown(choices=diff_model_choices, value=_last_local_file_choice("last_local_diff_model", diff_model_choices)),
        gr.Dropdown(choices=diff_config_choices, value=_last_local_file_choice("last_local_diff_config", diff_config_choices)),
        gr.Dropdown(choices=cluster_choices, value=_last_local_file_choice("last_local_cluster_model", cluster_choices)),
    )

def debug_change():
    global debug
    debug = debug_button.value

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue = gr.themes.colors.green,
        font=["Source Sans Pro", "Arial", "sans-serif"],
        font_mono=['JetBrains mono', "Consolas", 'Courier New']
    ),
    analytics_enabled=False,
    css=SVC_UI_CSS,
    js=SVC_UI_JS,
) as app:
    with gr.Tabs():
        with gr.TabItem("训练"):
            build_training_tab()
        with gr.TabItem("管理"):
            build_management_tab()
        with gr.TabItem("推理"):
            gr.Markdown(value="""
                So-vits-svc 4.0 推理 webui
                """)
            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="""
                        <font size=2> 模型设置</font>
                        """)
                    with gr.Tabs():
                        # invisible checkbox that tracks tab status
                        local_model_enabled = gr.Checkbox(value=False, visible=False)
                        with gr.TabItem('上传') as local_model_tab_upload:
                            _last_upload_model = _get_webui_config_key("last_upload_model", None)
                            _last_upload_config = _get_webui_config_key("last_upload_config", None)
                            _last_diff_model = _get_webui_config_key("last_diff_model", None)
                            _last_diff_config = _get_webui_config_key("last_diff_config", None)
                            _last_cluster = _get_webui_config_key("last_cluster_model", None)
                            with gr.Row():
                                model_path = gr.File(label="选择模型文件",
                                                     value=_last_upload_model if _last_upload_model and os.path.exists(_last_upload_model) else None)
                                config_path = gr.File(label="选择配置文件",
                                                      value=_last_upload_config if _last_upload_config and os.path.exists(_last_upload_config) else None)
                            with gr.Row():
                                diff_model_path = gr.File(label="选择扩散模型文件",
                                                           value=_last_diff_model if _last_diff_model and os.path.exists(_last_diff_model) else None)
                                diff_config_path = gr.File(label="选择扩散模型配置文件",
                                                            value=_last_diff_config if _last_diff_config and os.path.exists(_last_diff_config) else None)
                            cluster_model_path = gr.File(label="选择聚类模型或特征检索文件（没有可以不选）",
                                                          value=_last_cluster if _last_cluster and os.path.exists(_last_cluster) else None)
                        with gr.TabItem('本地') as local_model_tab_local:
                            gr.Markdown(f'可选择 {local_model_root} 下的已导出模型，或 logs/44k 下的训练检查点')
                            local_model_refresh_btn = gr.Button('刷新本地模型列表')
                            _local_models = scan_local_models()
                            _last_model_val = _last_local_model_choice(_local_models)
                            local_model_selection = gr.Dropdown(label='选择模型或检查点', choices=_local_models, value=_last_model_val, interactive=True)
                            _local_diff_models = scan_local_diffusion_models()
                            _local_diff_configs = scan_local_diffusion_configs()
                            _local_clusters = scan_local_cluster_models()
                            with gr.Row():
                                local_diff_model_selection = gr.Dropdown(
                                    label='选择扩散模型文件（没有可以不选）',
                                    choices=_local_diff_models,
                                    value=_last_local_file_choice("last_local_diff_model", _local_diff_models),
                                    interactive=True,
                                )
                                local_diff_config_selection = gr.Dropdown(
                                    label='选择扩散模型配置文件（没有可以不选）',
                                    choices=_local_diff_configs,
                                    value=_last_local_file_choice("last_local_diff_config", _local_diff_configs),
                                    interactive=True,
                                )
                            local_cluster_model_selection = gr.Dropdown(
                                label='选择聚类模型或特征检索文件（没有可以不选）',
                                choices=_local_clusters,
                                value=_last_local_file_choice("last_local_cluster_model", _local_clusters),
                                interactive=True,
                            )
                    device = gr.Dropdown(label="推理设备，默认为自动选择CPU和GPU", choices=["Auto",*cuda.keys(),"cpu"], value="Auto")
                    enhance = gr.Checkbox(label="是否使用NSF_HIFIGAN增强,该选项对部分训练集少的模型有一定的音质增强效果，但是对训练好的模型有反面效果，默认关闭", value=False)
                    only_diffusion = gr.Checkbox(label="是否使用全扩散推理，开启后将不使用So-VITS模型，仅使用扩散模型进行完整扩散推理，默认关闭", value=False)
                with gr.Column():
                    gr.Markdown(value="""
                        <font size=3>左侧文件全部选择完毕后(全部文件模块显示download)，点击“加载模型”进行解析：</font>
                        """)
                    model_load_button = gr.Button(value="加载模型", variant="primary")
                    model_unload_button = gr.Button(value="卸载模型", variant="primary")
                    sid = gr.Dropdown(label="音色（说话人）")
                    sid_output = gr.Textbox(label="Output Message")


            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="""
                        <font size=2> 推理设置</font>
                        """)
                    auto_f0 = gr.Checkbox(label="自动f0预测，配合聚类模型f0预测效果更好,会导致变调功能失效（仅限转换语音，歌声勾选此项会究极跑调）", value=False)
                    f0_predictor = gr.Dropdown(label="选择F0预测器,可选择crepe,pm,dio,harvest,rmvpe,默认为pm(注意：crepe为原F0使用均值滤波器)", choices=["pm","dio","harvest","crepe","rmvpe"], value="pm")
                    vc_transform = gr.Number(label="变调（整数，可以正负，半音数量，升高八度就是12）", value=0)
                    cluster_ratio = gr.Number(label="聚类模型/特征检索混合比例，0-1之间，0即不启用聚类/特征检索。使用聚类/特征检索能提升音色相似度，但会导致咬字下降（如果使用建议0.5左右）", value=0)
                    slice_db = gr.Number(label="切片阈值", value=-40)
                    output_format = gr.Radio(label="音频输出格式", choices=["wav", "flac", "mp3"], value = "wav")
                    noise_scale = gr.Number(label="noise_scale 建议不要动，会影响音质，玄学参数", value=0.4)
                    k_step = gr.Slider(label="浅扩散步数，只有使用了扩散模型才有效，步数越大越接近扩散模型的结果", value=100, minimum = 1, maximum = 1000)
                with gr.Column():
                    pad_seconds = gr.Number(label="推理音频pad秒数，由于未知原因开头结尾会有异响，pad一小段静音段后就不会出现", value=0.5)
                    cl_num = gr.Number(label="音频自动切片，0为不切片，单位为秒(s)", value=0)
                    lg_num = gr.Number(label="两端音频切片的交叉淡入长度，如果自动切片后出现人声不连贯可调整该数值，如果连贯建议采用默认值0，注意，该设置会影响推理速度，单位为秒/s", value=0)
                    lgr_num = gr.Number(label="自动音频切片后，需要舍弃每段切片的头尾。该参数设置交叉长度保留的比例，范围0-1,左开右闭", value=0.75)
                    enhancer_adaptive_key = gr.Number(label="使增强器适应更高的音域(单位为半音数)|默认为0", value=0)
                    cr_threshold = gr.Number(label="F0过滤阈值，只有启动crepe时有效. 数值范围从0-1. 降低该值可减少跑调概率，但会增加哑音", value=0.05)
                    loudness_envelope_adjustment = gr.Number(label="输入源响度包络替换输出响度包络融合比例，越靠近1越使用输出响度包络", value = 0)
                    second_encoding = gr.Checkbox(label = "二次编码，浅扩散前会对原始音频进行二次编码，玄学选项，效果时好时差，默认关闭", value=False)
                    use_spk_mix = gr.Checkbox(label = "动态声线融合", value = False, interactive = False)
            with gr.Tabs():
                with gr.TabItem("音频转音频"):
                    vc_input3 = gr.Audio(label="选择音频", type="filepath", elem_id="svc-vc-audio-input")
                    with gr.Row():
                        with gr.Column():
                            audio_compress_enabled = gr.Checkbox(
                                label="上传前在浏览器内高质量压缩音频",
                                value=False,
                                elem_id="svc-audio-compress-enabled",
                            )
                        with gr.Column():
                            audio_compress_ratio = gr.Slider(
                                label="音频上传目标压缩比例（%，质量优先）",
                                minimum=1,
                                maximum=100,
                                value=20,
                                step=1,
                                elem_id="svc-audio-compress-ratio",
                            )
                    vc_submit = gr.Button("音频转换", variant="primary")
                with gr.TabItem("文字转音频"):
                    text2tts=gr.Textbox(label="在此输入要转译的文字。注意，使用该功能建议打开F0预测，不然会很怪")
                    with gr.Row():
                        tts_gender = gr.Radio(label = "说话人性别", choices = ["男","女"], value = "男")
                        tts_lang = gr.Dropdown(label = "选择语言，Auto为根据输入文字自动识别", choices=SUPPORTED_LANGUAGES, value = "Auto")
                        tts_rate = gr.Slider(label = "TTS语音变速（倍速相对值）", minimum = -1, maximum = 3, value = 0, step = 0.1)
                        tts_volume = gr.Slider(label = "TTS语音音量（相对值）", minimum = -1, maximum = 1.5, value = 0, step = 0.1)
                    vc_submit2 = gr.Button("文字转换", variant="primary")
            with gr.Row():
                with gr.Column():
                    vc_output1 = gr.Textbox(label="Output Message", elem_id="svc-output-message")
                with gr.Column():
                    vc_output2 = gr.Audio(label="Output Audio", interactive=False)

        with gr.TabItem("小工具/实验室特性"):
            gr.Markdown(value="""
                        <font size=2> So-vits-svc 4.0 小工具/实验室特性</font>
                        """)
            with gr.Tabs():
                with gr.TabItem("静态声线融合"):
                    gr.Markdown(value="""
                        <font size=2> 介绍:该功能可以将多个声音模型合成为一个声音模型(多个模型参数的凸组合或线性组合)，从而制造出现实中不存在的声线 
                                          注意：
                                          1.该功能仅支持单说话人的模型
                                          2.如果强行使用多说话人模型，需要保证多个模型的说话人数量相同，这样可以混合同一个SpaekerID下的声音
                                          3.保证所有待混合模型的config.json中的model字段是相同的
                                          4.输出的混合模型可以使用待合成模型的任意一个config.json，但聚类模型将不能使用
                                          5.批量上传模型的时候最好把模型放到一个文件夹选中后一起上传
                                          6.混合比例调整建议大小在0-100之间，也可以调为其他数字，但在线性组合模式下会出现未知的效果
                                          7.混合完毕后，文件将会保存在项目根目录中，文件名为output.pth
                                          8.凸组合模式会将混合比例执行Softmax使混合比例相加为1，而线性组合模式不会
                        </font>
                        """)
                    mix_model_path = gr.Files(label="选择需要混合模型文件")
                    mix_model_upload_button = gr.UploadButton("选择/追加需要混合模型文件", file_count="multiple")
                    mix_model_output1 = gr.Textbox(
                                            label="混合比例调整，单位/%",
                                            interactive = True
                                         )
                    mix_mode = gr.Radio(choices=["凸组合", "线性组合"], label="融合模式",value="凸组合",interactive = True)
                    mix_submit = gr.Button("声线融合启动", variant="primary")
                    mix_model_output2 = gr.Textbox(
                                            label="Output Message"
                                         )
                    mix_model_path.change(updata_mix_info,[mix_model_path],[mix_model_output1], queue=False, show_api=False)
                    mix_model_upload_button.upload(upload_mix_append_file, [mix_model_upload_button,mix_model_path], [mix_model_path,mix_model_output1], queue=False, show_api=False)
                    mix_submit.click(mix_submit_click, [mix_model_output1,mix_mode], [mix_model_output2])
                
                with gr.TabItem("模型压缩工具"):
                    gr.Markdown(value="""
                        该工具可以实现对模型的体积压缩，在**不影响模型推理功能**的情况下，将原本约600M的So-VITS模型压缩至约200M, 大大减少了硬盘的压力。
                        **注意：压缩后的模型将无法继续训练，请在确认封炉后再压缩。**
                    """)
                    model_to_compress = gr.File(label="模型上传")
                    compress_model_btn = gr.Button("压缩模型", variant="primary")
                    compress_model_output = gr.Textbox(label="输出信息", value="")

                    compress_model_btn.click(model_compression, [model_to_compress], [compress_model_output])

    with gr.Tabs():
        with gr.Row(variant="panel"):
            with gr.Column():
                gr.Markdown(value="""
                    <font size=2> WebUI设置</font>
                    """)
                debug_button = gr.Checkbox(label="Debug模式，如果向社区反馈BUG需要打开，打开后控制台可以显示具体错误提示", value=debug)
        # refresh local model list
        local_model_refresh_btn.click(
            local_model_refresh_fn,
            outputs=[
                local_model_selection,
                local_diff_model_selection,
                local_diff_config_selection,
                local_cluster_model_selection,
            ],
            queue=False,
            show_api=False,
        )
        # set local enabled/disabled on tab switch
        local_model_tab_upload.select(lambda: False, outputs=local_model_enabled, queue=False, show_api=False)
        local_model_tab_local.select(lambda: True, outputs=local_model_enabled, queue=False, show_api=False)
        
        vc_submit.click(vc_fn, [sid, vc_input3, output_format, vc_transform,auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,f0_predictor,enhancer_adaptive_key,cr_threshold,k_step,use_spk_mix,second_encoding,loudness_envelope_adjustment], [vc_output1, vc_output2])
        vc_submit2.click(vc_fn2, [text2tts, tts_lang, tts_gender, tts_rate, tts_volume, sid, output_format, vc_transform,auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,f0_predictor,enhancer_adaptive_key,cr_threshold,k_step,use_spk_mix,second_encoding,loudness_envelope_adjustment], [vc_output1, vc_output2])

        debug_button.change(debug_change,[],[], queue=False, show_api=False)
        model_load_button.click(
            modelAnalysis,
            [
                model_path,
                config_path,
                cluster_model_path,
                device,
                enhance,
                diff_model_path,
                diff_config_path,
                only_diffusion,
                use_spk_mix,
                local_model_enabled,
                local_model_selection,
                local_diff_model_selection,
                local_diff_config_selection,
                local_cluster_model_selection,
            ],
            [sid,sid_output],
            show_api=False,
        )
        model_unload_button.click(modelUnload,[],[sid,sid_output], queue=False, show_api=False)
    app.queue(default_concurrency_limit=8)
    webbrowser.open("http://127.0.0.1:7860")
    emit_startup_banner("# WebUI")
    app.launch(
        share=_gradio_share_enabled(),
        max_file_size=os.environ.get("SVC_MAX_FILE_SIZE", "20gb"),
    )


 
