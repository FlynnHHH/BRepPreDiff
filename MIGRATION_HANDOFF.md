# BRepPreDiff 服务器迁移交接说明

生成日期：2026-08-28

## 迁移包内容

本迁移包只包含 Git 仓库之外、在新服务器直接复现实验或继续训练所需的数据：

- `cache/features/`：预处理后的 B-Rep NPZ 特征缓存。
- `data/splits/`：训练、验证、测试划分及数据审计文件。
- `data/labels/`：FabWave 等分类任务使用的标签文件。
- `runs/**/checkpoints/best.pt`：按验证指标选出的最佳 checkpoint。
- `runs/**/checkpoints/last.pt`：训练结束或中断时的最新 checkpoint，可用于续训。
- `MIGRATION_HANDOFF.md`：本说明文档。

本包不包含项目源码。请先在目标服务器拉取对应的 Git 仓库和提交，再解压本包。

## 当前数据规模

- `best.pt`/`last.pt`：551 个文件，约 8.71 GiB。
- `cache/features/`：约 17 GB。
- `data/splits/`：约 8.5 MB。
- `data/labels/`：约 52 MB。

## 解压

建议在已经拉取好的 BRepPreDiff 仓库根目录执行：

```bash
tar -xf /path/to/brepprediff_transfer_20260828.tar
```

解压后，仓库中将出现或补充：

```text
cache/features/
data/splits/
data/labels/
runs/**/checkpoints/{best,last}.pt
```

## 修改数据路径

当前 `data/*.yaml` 中存在旧服务器绝对路径，例如：

```text
/data/hhfeng/brepprediff/cache/features
/data/hhfeng/TMCAD
/data/hhfeng/FabWave
/data/hhfeng/MFCAD++
/data/hhfeng/fusion360seg
/data/hhfeng/fusion360rec
/data/hhfeng/fusion360ass
```

本迁移包把缓存放在仓库内的 `cache/features/`。在目标服务器上应将所用数据配置的 `cache_dir`、`cache_dirs` 和 `invalid_log` 改为新位置。可以使用绝对路径，也可以使用相对仓库根目录的路径。

例如：

```yaml
data:
  cache_dir: cache/features/tmcad
```

如果只复用缓存且保持 `prepare_on_start: false`，训练不会重新解析 STEP 文件。此时通常不需要上传原始 STEP/SEG 数据，但 split 中引用的每个样本都必须能在对应缓存目录中找到。

如需重建缓存，则还要单独迁移原始 STEP、SEG/JSON/CLS 数据，并更新 `steps_dir`、`segs_dir` 或 `labels_dir`。这些原始 CAD 数据不在本迁移包中。

## 环境安装

在仓库根目录执行：

```bash
conda env create -f environment.yml
conda activate brepprediff
python -m pip install --no-deps -e .
pytest -q
```

## Checkpoint 使用方式

微调时加载预训练 checkpoint：

```bash
brepprediff-finetune --config configs/finetune.yaml \
  --override train.pretrain_checkpoint=runs/pretrain/<run>/checkpoints/best.pt
```

断点续训：

```bash
brepprediff-pretrain --config configs/pretrain.yaml \
  --override train.resume=runs/pretrain/<run>/checkpoints/last.pt
```

多卡运行示例：

```bash
torchrun --standalone --nproc_per_node=4 \
  -m brepprediff.training.pretrain \
  --config configs/pretrain.yaml
```

根据目标服务器 GPU 数量调整 `--nproc_per_node`、每卡 `train.batch_size` 和 `train.num_workers`。

## 验证迁移结果

检查缓存和 checkpoint 数量：

```bash
find cache/features -type f -name '*.npz' | wc -l
find runs -type f \( -name best.pt -o -name last.pt \) | wc -l
```

检查所有配置中仍然存在的旧服务器路径：

```bash
rg -n '/data/hhfeng|/home/hhfeng' configs data
```

正式训练前建议先运行：

```bash
pytest -q
```

如果要进行完整缓存校验，需要确保原始 STEP/标签数据也已迁移，然后运行：

```bash
brepprediff-prepare-data --config <所用数据配置> --workers 16
```

不要在缺少原始 STEP 数据时启用 `prepare_on_start: true` 或执行缓存重建。
