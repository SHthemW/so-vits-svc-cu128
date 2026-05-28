# So-VITS-SVC-CU128 使用文档

## 项目简介

本项目是 [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc) 的 fork，基于 SoftVC VITS 歌声转换模型，提供 **Gradio WebUI** 可视化界面，支持训练、推理和模型管理。目标环境为 **CUDA 12.8** + PyTorch 2.7。

## 快速开始

### 方式一：使用预构建环境包（推荐）

如果你只是想运行项目，可以直接下载预构建的 Python 环境包，解压后即可使用：

> 下载链接：<https://pan.quark.cn/s/28b5ef9da0c4>

下载后解压到项目目录下的 `python_env/` 文件夹中。

### 方式二：手动配置环境

1. 安装 **Python 3.9 ~ 3.10**（推荐 3.9.8）
2. 安装 **pip 24.0**（不能使用更高版本）
3. 安装 **cmake** 并确保已加入 PATH（部分依赖需从源码编译）
4. 安装 PyTorch（CUDA 12.8 版本）
5. 安装依赖：

```shell
pip install -r requirements.txt
```

### 启动 WebUI

双击 `so-vits-svc.bat`，或执行：

```shell
python webUI.py
```

启动后会自动打开浏览器访问 `http://127.0.0.1:7860`。

如果只需要命令行环境（不启动 WebUI），可以运行 `so-vits-svc-cmd.bat` 来激活虚拟环境。

---

## 界面概览

WebUI 包含以下四个页面：

| 页面 | 功能 |
|------|------|
| **训练** | 7 步引导式训练流程 |
| **管理** | 检查点管理、模型导出、特征/聚类模型管理 |
| **推理** | 加载模型进行声音转换（音频转音频 / 文字转音频） |
| **小工具/实验室特性** | 静态声线融合、模型压缩 |

---

## 推理

### 1. 加载模型

进入「推理」页面，在左侧「模型设置」区域加载所需的模型文件：

**必选：**

- **模型文件**（`.pth`）— So-VITS 主模型
- **配置文件**（`.json`）— 与模型对应的 config

**可选：**

- **扩散模型文件**（`.pt`）+ 扩散配置文件（`.yaml`）— 启用浅扩散，提升音质
- **聚类模型**（`.pt`）或**特征检索文件**（`.pkl`）— 提升音色相似度

模型提供两种加载方式：

- **上传**：从本地选择文件
- **本地**：将包含 `G_xxx.pth` + `config.json` 的文件夹放入 `trained/` 目录，刷新列表后选择。WebUI 会记住上次使用的模型。

全部文件选择完毕后，选择推理设备（默认为 Auto），点击「**加载模型**」按钮。

### 2. 推理参数

加载成功后，右侧会显示可用音色列表。调整推理参数：

| 参数 | 说明 | 建议值 |
|------|------|--------|
| 音色（说话人） | 选择目标音色 | — |
| 变调 | 半音数量，正数升调，负数降调 | 0 |
| 自动 F0 预测 | 语音推荐开启；歌声勾选会跑调 | 语音开/歌声关 |
| F0 预测器 | pm/dio/harvest/crepe/rmvpe | rmvpe 效果较好 |
| 聚类/特征检索比例 | 0-1，越大音色越像但咬字下降 | 0.5 |
| 切片阈值 | 静音检测阈值，单位 dB | -40 |
| noise_scale | 玄学参数，影响音质 | 0.4 |
| 浅扩散步数 | 需加载扩散模型，步数越大越接近扩散结果 | 100 |
| 推理音频 pad 秒数 | 开头结尾静音填充，防止异响 | 0.5 |

### 3. 执行转换

**音频转音频**：在「音频转音频」标签页上传音频文件，点击「音频转换」。

**文字转音频**：在「文字转音频」标签页输入文字，选择 TTS 语言和性别，调整语速/音量，点击「文字转换」。此功能通过 Edge TTS 先将文字合成为语音，再进行声音转换。

转换结果会保存在 `results/` 目录下，文件名格式为 `result_{音色}_{key}_{参数}.wav`。

---

## 训练

训练页面提供 7 步引导式工作流，按顺序完成即可。每个步骤可独立启动/停止，进度和日志实时显示。

### 数据集准备

数据集需要放在 `dataset_raw/` 目录下（也可在页面指定其他目录），结构如下：

```
dataset_raw/
  speaker_A/
    audio001.wav
    audio002.wav
    ...
  speaker_B/
    audio001.wav
    ...
```

也可以直接放在一个目录下（如 `dataset_raw/my_singer/*.wav`），程序会自动适配。

### 前置步骤：环境检查与模型下载

点击「**检查环境**」确认 CUDA、PyTorch、数据目录、预训练模型是否就绪。

缺失的模型可一键下载：
- **预训练模型**：ContentVec 语音编码器（必需）、RMVPE F0 预测器（推荐）、NSF-HiFiGAN 声码器
- **训练底模**：Generator 底模 G_0.pth（强烈推荐）、Discriminator 底模 D_0.pth、扩散模型底模

### 第一步：音频重采样

将所有原始音频重采样到 **44100Hz mono**，输出到 `dataset/44k/`。

参数：
- **跳过响度归一化**：如数据已做响度处理可勾选
- **CPU 核心数**：0 为自动检测

### 第二步：生成文件列表和配置

扫描 `dataset/44k/`，自动生成：

- `filelists/train.txt` — 训练文件列表
- `filelists/val.txt` — 验证文件列表
- `configs/config.json` — 主模型配置文件
- `configs/diffusion.yaml` — 扩散模型配置文件

参数：
- **语音编码器**：推荐 `vec768l12`
- **音量增强**：建议开启，增加数据多样性
- **Tiny 模型**：使用更小的模型结构，参数更少、更快但质量略低

### 第三步：提取特征和 F0

提取语音内容编码（HuBERT/ContentVec）和基频（F0）。已处理的文件会自动跳过（断点续跑）。

参数：
- **F0 预测器**：推荐 `rmvpe`
- **并行进程数**：根据 CPU 核心数设置
- **同时提取扩散特征**：如计划训练扩散模型则勾选

### 训练参数配置

在启动训练前可调整关键参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| batch_size | 每批样本数 | 6 |
| epochs | 总训练轮次 | 10000 |
| keep_ckpts | 保留最近 N 个检查点 | 3 |
| fp16_run | 混合精度训练（省显存） | 关闭 |

### 第四步：训练主模型

启动 So-VITS 主模型训练，会自动从最新检查点续训。检查点保存在 `logs/44k/`。

可用 TensorBoard 监控训练进度：

```shell
python -m tensorboard.main --logdir=logs/44k
```

### 第五步（可选）：训练扩散模型

训练浅扩散模型以增强音质。扩散模型与主模型分开训练，会争抢 GPU 显存，建议分开运行。

检查点保存在 `logs/44k/diffusion/`。

### 第六步：构建特征检索索引

基于训练数据构建 FAISS 检索索引，用于推理时的音色增强（通常数分钟完成）。

输出文件：`logs/44k/feature_and_index.pkl`

### 第七步（可选）：训练聚类模型

训练 KMeans 聚类模型，可作为特征检索的替代方案。输出文件：`logs/44k/kmeans_10000.pt`

---

## 模型管理

进入「管理」页面，可进行以下操作：

- **删除检查点**：清理训练过程中产生的旧检查点（包含 Generator 和 Discriminator）
- **导出模型**：将指定步数的检查点导出到 `trained/` 目录，自动打包 G + D + config，方便用于推理
- **管理已导出模型**：删除、查看信息
- **特征检索模型管理**：删除特征检索索引文件
- **聚类模型管理**：删除聚类模型文件

---

## 小工具/实验室特性

### 静态声线融合

将多个单说话人模型合成为一个新声音模型（凸组合或线性组合），制造现实不存在的声线。

要求：
- 仅支持单说话人模型
- 多说话人模型需说话人数量相同
- 所有模型的 `config.json` 中 `model` 字段需一致
- 输出文件为项目根目录的 `output.pth`

### 模型压缩

压缩 So-VITS 模型文件（约 600MB → 约 200MB），不影响推理。**压缩后的模型无法继续训练**。

---

## 命令行工具

本项目也保留了原项目的命令行入口：

| 命令 | 用途 |
|------|------|
| `python inference_main.py` | 命令行推理 |
| `python train.py -c configs/config.json -m 44k` | 训练主模型 |
| `python train_diff.py -c configs/diffusion.yaml` | 训练扩散模型 |
| `python flask_api.py` | 启动 Flask API 服务 |
| `python onnx_export.py` | 导出 ONNX 模型 |

---

## 目录结构

```
├── webUI.py              # WebUI 主入口
├── webui_train.py        # 训练页面逻辑
├── webui_manage.py       # 管理页面逻辑
├── so-vits-svc.bat       # Windows 启动脚本（WebUI）
├── so-vits-svc-cmd.bat   # Windows 命令行环境脚本
├── configs/              # 运行时配置文件
│   ├── config.json       # 主模型配置（由训练流程自动生成）
│   └── diffusion.yaml    # 扩散模型配置（由训练流程自动生成）
├── configs_template/     # 配置文件模板
│   ├── config_template.json
│   ├── config_tiny_template.json
│   └── diffusion_template.yaml
├── dataset_raw/          # 原始训练数据（需自行准备）
├── dataset/              # 预处理后的数据（自动生成）
│   └── 44k/
├── logs/                 # 训练日志和检查点
│   └── 44k/
│       ├── G_xxx.pth     # Generator 检查点
│       ├── D_xxx.pth     # Discriminator 检查点
│       └── diffusion/    # 扩散模型检查点
├── trained/              # 已导出/可推理的模型
├── pretrain/             # 预训练模型
│   ├── checkpoint_best_legacy_500.pt  # ContentVec
│   ├── rmvpe.pt          # RMVPE F0 预测器
│   └── nsf_hifigan/      # NSF-HiFiGAN 声码器
├── results/              # 推理输出音频
├── raw/                  # 推理临时文件
├── filelists/            # 训练/验证文件列表（自动生成）
├── requirements.txt      # Python 依赖
└── python_env/           # 捆绑的 Python 虚拟环境（需下载或自行配置）
```

---

## 常见问题

**Q: 启动时报错 "找不到捆绑的 Python 环境"？**

A: 检查 `python_env/Python/python.exe` 是否存在。如果使用手动配置方式，直接运行 `python webUI.py` 即可。

**Q: pip 安装依赖失败？**

A: 确保 pip 版本为 24.0（`pip install pip==24.0`），并已安装 cmake。部分旧依赖不提供预编译 wheel，需要从源码编译。

**Q: 推理时提示需要聚类模型？**

A: 将聚类/特征检索比例设为 0 即可不使用。或者先将模型加载上再设非零值。

**Q: 训练时显存不足？**

A: 减小 `batch_size`，启用 `fp16_run` 混合精度训练，或使用 Tiny 模型配置。
