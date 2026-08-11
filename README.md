# So-VITS-SVC-Cu12.8-GUI Fork

<img src="docs/screenshots/webui-main.png" alt="So-VITS-SVC GUI 主界面" width="75%">

[简体中文](README.md) | [English](README_en.md)

本项目是 [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)（SoftVC VITS 歌声转换）的 fork，在原版基础上增加了 **Gradio WebUI**，支持可视化训练、推理和模型管理。

目标环境为 **CUDA 12.8** (支持RTX50系显卡, 原版不支持)，并修复了较新 PyTorch 版本在 Windows 上的各类兼容性问题。

本 CUDA 12.8 整合分支由 **SHW（GitHub：[SHthemW](https://github.com/SHthemW)）** 整合与维护，当前项目仓库为 [SHthemW/so-vits-svc-cu128](https://github.com/SHthemW/so-vits-svc-cu128)，完整归属说明见 [AUTHORS.md](AUTHORS.md)。

完整更新历史见 [CHANGELOG_zh_CN.md](CHANGELOG_zh_CN.md)。

整合包QQ交流群: 1104444127

## 快速开始

本程序以源代码格式分发, 需要额外安装运行环境才能运行.

如果你不想手动安装依赖, 可以使用我已经部署好的云端镜像, 或者下载已安装环境的整合压缩包.

这里只会介绍项目的部署方式. 有关具体使用方法, 可以看语雀文档: https://www.yuque.com/shenhanwen-ozfty/oogl43/dim2na4llgo3quz9?singleDoc# 《So-VITS-SVC 用户使用手册 (简体中文)》

### 使用预配置的云端镜像(推荐)

- 优云智算站: https://www.compshare.cn/images/bYafh6fXRTsK?referral_code=n2qzZuyGlyDPbOYHPvUGy

### 从整合包运行

- 夸克网盘: https://pan.quark.cn/s/b6ec45c084e8?pwd=Ahhv

下载并解压后，运行 `so-vits-svc-start_gui/so-vits-svc-start_gui.exe` 即可打开 WebUI。

### 从源代码运行

git clone 源代码到本地, 根据[环境要求](#环境要求)板块安装依赖.

完成安装后运行 `python start_gui.py` 即可，脚本会自动选择当前平台可用的 Python 环境。

## 环境要求

### Python 版本

- 需要 **Python 3.9 ~ 3.10**。
- 实测 **Python 3.9.8** 可正常运行。
- Python 3.8 不受支持（PyTorch 2.7 已不再支持）。Python 3.11+ 不受支持（`fairseq==0.12.2` 不兼容）。

### pip 与依赖编译

- **pip 版本必须为 24.0**，不能使用更高版本（高版本 pip 在解析部分旧依赖时会出现兼容性问题，导致安装失败）。
- 部分依赖已不再提供预编译 wheel 分发，安装时需要 **cmake** 从源码自行编译。请确保系统已安装 cmake 并加入 PATH。

### PyTorch

- 测试环境为 **CUDA 12.8** + PyTorch 2.7.0.dev20250309+cu128。

## 启动命令

如果想快速使用，可以运行 `python install_sovits_command.py` 安装 `sovits` 命令。整合包用户也可以运行 `so-vits-svc-install_sovits_command` 目录内的同名可执行程序。

后续可以打开新终端并运行 `sovits start webui`来一键从任何目录启动程序.

## 构建发布包

Windows 下运行 `_build.bat` 会使用隔离的 PyInstaller 环境构建两个目录式可执行程序，并将程序源码、内置运行环境、预训练模型和可执行程序打包到 `dist/so-vits-svc-cu128-yyyyMMdd-HHmmss.zip`。

构建采用 `--onedir` 且禁用 UPX，以减少单文件自解压和二进制压缩导致的杀毒软件误报。若本机证书存储中有代码签名证书，可先设置 `SOVITS_SIGN_CERT_SHA1` 为证书指纹，构建脚本将使用 `signtool.exe` 对两个主程序签名。

- `_build.bat --dry-run`：仅验证发布文件清单，不构建可执行程序。
- `_build.bat --build-only`：仅构建并自检两个可执行程序，不生成大型压缩包。

## 与原版的主要区别

### GUI 界面

原版仅有命令行接口。本 fork 提供完整的 Gradio WebUI，包含以下页面：

- **推理页面** — 加载模型、转换声音、可视化调参。支持从本地模型列表选择，会记住上次使用的模型。
- **训练页面** — 7 步引导式工作流，从数据集预处理到 SoVITS 训练、扩散模型训练、聚类模型训练。
- **管理页面** — 删除/导出检查点、管理已导出模型、管理特征检索和聚类模型。

### Windows 编码修复

- 全部文件读写强制指定 `encoding='utf-8'`，修复 Windows 中文环境下出现乱码的问题。
- `train.py` 读取 `config.json` 时兼容 GBK 编码文件。
- `filelists` 始终以 UTF-8 写入，避免中文文件名导致训练中断。
- Gradio 从 3.36 升级到 4.44.1，改善 Windows 兼容性。

### 日志与UX

- 日志自动滚动到底部（使用 Gradio 原生 autoscroll）。
- 日志框下方有独立的清空日志按钮。
- 修复清空日志按钮不显示、停止进程不彻底等问题。
- 14 个独立轮询合并为单一计时器，修复长时间运行后断连的问题。
- 提高 Gradio 队列并发数，修复切换 Tab 后日志不更新的问题。
- 轮询无变化时跳过更新，消除 UI 闪烁。

### 训练与配置

- 支持自动下载预训练模型，减少手动配置步骤。
- WebUI 训练参数在启动时从 `config.json` 读取已保存的值。
- 聚类模型训练作为训练流程的第 7 步集成。
- 修复预处理流程的多个边界问题。
- 修复 PyTorch 2.6+ 下加载聚类模型时的 `UnpicklingError`。
- 聚类训练改用 `MiniBatchKMeans`，避免大数据集时内存耗尽或卡死。
- KMeans 参数根据数据集大小和可用内存自动适配。
- 降低特征索引构建的内存占用，避免 OOM 崩溃。

## 免责声明

本项目为开源、离线项目，不收集用户数据。

使用者应对其训练数据和转换的音频拥有合法权利。

## 许可证

AGPL 3.0，与原项目一致。

## 原始文档

关于模型架构、数据集准备、预处理、训练、推理参数等详细文档，请参阅[原仓库](https://github.com/svc-develop-team/so-vits-svc)。
