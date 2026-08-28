# BRepPreDiff finetune 推理可视化

此网页读取 `results/` 目录下的 PLY 文件。默认 BRepPreDiff expanded-data MLP 区域使用官方
`data/splits/finetune_test.txt`（不是 `testset_lxy`），展示
NonTransition/VBF/EBF 三分类结果与 Macro 指标：

- `*_instance_pred_rgb.ply`：SEG GT 高亮；找不到 SEG 时为灰色原始模型。
- `*_semantic_pred.ply`：BRepPreDiff finetune 预测结果，VBF 为粉色，EBF 为黄色。

当前扩充数据集按 seed 42 和 80%/10%/10% 划分；移除 1 个无法完成 OCC 提取的
train 样本后，有效 train/val/test 为 14,295/1,787/1,787。官方 test split 页面结果来自：

- checkpoint：`runs/finetune/20260817-153907_brepprediff_seg_expanded_default_20260817/checkpoints/best.pt`
- epoch：14（训练预算 200 epochs，按 validation accuracy 选择）
- 样本 / 面：1,787 / 102,980
- Accuracy / Macro-F1 / Macro-IoU：0.987833 / 0.968617 / 0.939800
- 配置：Edge Update Attention encoder + MLP head；全局 batch 512

在 BRepPreDiff 仓库内解压本目录后，可运行：

```bash
PYTHONPATH=../../src python run_finetune_inference.py \
  --config ../../configs/finetune.yaml \
  --checkpoint ../../runs/finetune/<run>/checkpoints/last.pt \
  --step /path/to/model.step \
  --seg /path/to/model.seg
```

也可以传入 STEP 列表或目录，不需要预先准备 `.npz`：

```bash
PYTHONPATH=../../src python run_finetune_inference.py \
  --config ../../configs/finetune.yaml \
  --checkpoint ../../runs/finetune/<run>/checkpoints/last.pt \
  --step-list /path/to/test_steps.txt \
  --seg-list /path/to/test_segs.txt \
  --step-root /path/to/test_step_root \
  --seg-root /path/to/test_seg_root
```

```bash
PYTHONPATH=../../src python run_finetune_inference.py \
  --config ../../configs/finetune.yaml \
  --checkpoint ../../runs/finetune/<run>/checkpoints/last.pt \
  --step-dir /path/to/test_steps \
  --seg-dir /path/to/test_segs
```

如果要继续使用 split 文件，也可以运行：

```bash
PYTHONPATH=../../src python run_finetune_inference.py \
  --config ../../configs/finetune.yaml \
  --checkpoint ../../runs/finetune/<run>/checkpoints/last.pt \
  --split test \
  --split-file ../../data/abc_splits/<test_split>.txt
```

生成 PLY 后启动网页：

```bash
python viewer_server.py
```

浏览器打开 `http://localhost:8061/`。

## testset_lxy 最优模型三分类结果

网页中的 `BRepPreDiff Best · testset_lxy` 独立分区使用扩充数据官方
test split Macro-F1 最高的 Edge Update Attention + MLP 权重，左栏展示
SEG GT，右栏展示模型预测。该模型在官方 test split 上的 Macro-F1 为
`0.968617`，在 `testset_lxy` 上的 Accuracy / Macro-F1 / Macro-IoU 为
`0.895305 / 0.842671 / 0.735020`。

本次网页产物来自：

```bash
PYTHONPATH=src python -m brepprediff.inference.finetune_visualize \
  --checkpoint runs/finetune/20260817-153907_brepprediff_seg_expanded_default_20260817/checkpoints/best.pt \
  --step-dir /data/hhfeng/testset_lxy/step \
  --step-root /data/hhfeng/testset_lxy/step \
  --seg-dir /data/hhfeng/testset_lxy/seg \
  --seg-root /data/hhfeng/testset_lxy/seg \
  --output-dir tools/visualize/results \
  --manifest-name brepprediff_best_lxy_manifest.json \
  --sample-prefix brepprediff_best_lxy__
```

对应的逐面预测 SEG 单独输出到 `results/brepprediff_best_lxy_seg/`：

```bash
PYTHONPATH=src python -m brepprediff.inference.step_to_seg \
  /data/hhfeng/testset_lxy/step \
  --checkpoint runs/finetune/20260817-153907_brepprediff_seg_expanded_default_20260817/checkpoints/best.pt \
  --config runs/finetune/20260817-153907_brepprediff_seg_expanded_default_20260817/config.yaml \
  --data-config data/finetune.yaml \
  --output-dir tools/visualize/results/brepprediff_best_lxy_seg
```

## cjq step_numeric 无标注推理

`/data/hhfeng/cjq/step_numeric/` 的 3,993 个 STEP 使用同一最优微调模型推理。该数据无
GT，因此网页左栏显示灰色原模型、右栏显示三分类预测，逐面预测另存为
`results/cjq_step_numeric_seg/`：

```bash
PYTHONPATH=src python -m brepprediff.inference.finetune_visualize \
  --checkpoint runs/finetune/20260817-153907_brepprediff_seg_expanded_default_20260817/checkpoints/best.pt \
  --step-dir /data/hhfeng/cjq/step_numeric \
  --step-root /data/hhfeng/cjq/step_numeric \
  --cache-dir tools/visualize/cache/cjq_step_numeric \
  --output-dir tools/visualize/results \
  --seg-output-dir tools/visualize/results/cjq_step_numeric_seg \
  --manifest-name cjq_step_numeric_manifest.json \
  --sample-prefix cjq_step_numeric__
```

## FilletRec 二分类测试

FilletRec JSON 标签按 OCC face 顺序存放，0 表示非过渡面，1 表示过渡面。加入
`--binary-transition` 后，BRepPreDiff 的 VBF/EBF 都映射为 1，并在 manifest 和网页中
显示面级 Accuracy、Precision、Recall、F1：

```bash
PYTHONPATH=../../src python run_finetune_inference.py \
  --config ../../configs/finetune.yaml \
  --data-config ../../data/filletrec.yaml \
  --checkpoint ../../runs/finetune/<run>/checkpoints/best.pt \
  --split test \
  --split-file ../../data/filletrec/filletRec/test.txt \
  --cache-dir /data/hhfeng/brepprediff/cache/filletrec/test \
  --binary-transition \
  --sample-prefix filletrec__
```

原生二分类微调结果使用 `configs/finetune_filletrec_diffloss.yaml`。此时
`--binary-transition` 会直接保留模型的 0/1 预测，不再执行 VBF/EBF 合并：

```bash
cd ../..
PYTHONPATH=src python -m brepprediff.inference.finetune_visualize \
  --config configs/finetune_filletrec_diffloss.yaml \
  --checkpoint runs/finetune/<run>/checkpoints/best.pt \
  --split test --cache-dir /data/hhfeng/brepprediff/cache/filletrec/test \
  --output-dir tools/visualize/results \
  --binary-transition --sample-prefix filletrec__
```
