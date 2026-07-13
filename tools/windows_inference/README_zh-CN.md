# Blendit Windows STEP 面分类推理程序

本程序扫描给定路径中的 `.step` / `.stp` 文件，为每个 STEP 面预测类别，并生成同名 `.seg` 文件。

输出标签固定为：

- 非过渡面：`0`
- EBF：`4`
- VBF：`6`

`.seg` 第 N 行对应 OpenCascade 读取的第 N 个面，每行只有一个整数标签。

## 1. 系统要求

- Windows 10 或 Windows 11，64 位
- Miniconda 或 Anaconda
- 首次安装依赖时需要联网
- 推理默认使用 CPU，不要求 NVIDIA 显卡

建议将压缩包解压到不含特殊符号的短路径，例如 `D:\Blendit-Inference`。输入路径可以包含中文；程序会自动兼容部分 OpenCascade 对中文或超长路径的限制。

## 2. 首次安装

双击 `install_env.bat`，或在 Anaconda Prompt 中运行：

```bat
cd /d D:\Blendit-Inference
install_env.bat
```

脚本会创建名为 `blendit-infer` 的独立 Conda 环境。以后推理不需要重复安装。

## 3. 推理

最简单的方法是双击 `run_inference.bat`，然后粘贴包含 STEP 文件的目录。

也可以从 Anaconda Prompt 传入路径：

```bat
run_inference.bat "D:\data\step_models"
```

默认递归扫描子目录，并将结果写到输入目录旁边的 `step_models_seg`：

```text
D:\data\step_models\part_a.step
D:\data\step_models\subdir\part_b.stp

D:\data\step_models_seg\part_a.seg
D:\data\step_models_seg\subdir\part_b.seg
D:\data\step_models_seg\prediction_manifest.json
```

如需指定输出目录，使用第二个参数：

```bat
run_inference.bat "D:\data\step_models" "D:\data\predicted_seg"
```

也可以直接给一个 STEP 文件：

```bat
run_inference.bat "D:\data\part_a.step"
```

## 4. 输出说明

- 每个输入 STEP 产生一个同名 `.seg`。
- 子目录结构会保留，避免不同目录中的同名模型互相覆盖。
- `prediction_manifest.json` 记录输入、输出、面数、三类数量、平均置信度和失败原因。
- 某一个 STEP 解析失败时，程序会继续处理其他文件，并返回非零退出码。
- 默认会覆盖已有预测结果。需要保留旧文件时，可直接运行高级命令并添加 `--skip-existing`。

## 5. 高级命令

```bat
conda run --no-capture-output -n blendit-infer python app\launcher.py "D:\data\step_models" --batch-size 2 --skip-existing
```

常用选项：

- `--output-dir PATH`：指定输出目录。
- `--batch-size N`：一次推理 N 个模型；默认 `1` 最稳妥。
- `--no-recursive`：只扫描输入目录第一层。
- `--skip-existing`：跳过已经存在的 `.seg`。
- `--fail-fast`：遇到第一个错误就停止。
- `--device cpu`：强制 CPU 推理。

## 6. 常见问题

如果提示“未找到 conda”，请从“Anaconda Prompt”运行脚本，或先安装 Miniconda。

如果安装时出现下载或依赖求解错误，请检查网络后重新运行 `install_env.bat`；脚本会更新已有环境。

如果某个 STEP 失败，请打开输出目录中的 `prediction_manifest.json` 查看该文件对应的 `error`。常见原因包括 STEP 损坏、模型中没有面，或 OpenCascade 无法读取该 STEP 变体。
