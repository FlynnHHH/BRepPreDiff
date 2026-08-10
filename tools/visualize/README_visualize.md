# Blendit finetune 推理可视化

此网页读取 `results/` 目录下的 PLY 文件。默认 Blendit baseline 区域使用
`data/splits/finetune_test.txt`，展示 NonTransition/VBF/EBF 三分类结果与 Macro 指标：

- `*_instance_pred_rgb.ply`：SEG GT 高亮；找不到 SEG 时为灰色原始模型。
- `*_semantic_pred.ply`：Blendit finetune 预测结果，VBF 为粉色，EBF 为黄色。

在 Blendit 仓库内解压本目录后，可运行：

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

网页中的 `Blendit Best · testset_lxy` 独立分区使用现有三分类测试
Macro-F1 最高的 MLP 权重，左栏展示 SEG GT，右栏展示模型预测：

```bash
PYTHONPATH=src python -m blendit.inference.finetune_visualize \
  --config configs/finetune.yaml \
  --checkpoint runs/finetune/20260721-000045_suite_20260720_ablation_head_mlp_full/checkpoints/best.pt \
  --step-dir /data/hhfeng/testset_lxy/step \
  --step-root /data/hhfeng/testset_lxy/step \
  --seg-dir /data/hhfeng/testset_lxy/seg \
  --seg-root /data/hhfeng/testset_lxy/seg \
  --output-dir tools/visualize/results \
  --manifest-name blendit_best_lxy_manifest.json \
  --sample-prefix blendit_best_lxy__
```

对应的逐面预测 SEG 单独输出到 `results/blendit_best_lxy_seg/`：

```bash
PYTHONPATH=src python -m blendit.inference.step_to_seg \
  /data/hhfeng/testset_lxy/step \
  --checkpoint runs/finetune/20260721-000045_suite_20260720_ablation_head_mlp_full/checkpoints/best.pt \
  --config configs/finetune.yaml \
  --output-dir tools/visualize/results/blendit_best_lxy_seg
```

## FilletRec 二分类测试

FilletRec JSON 标签按 OCC face 顺序存放，0 表示非过渡面，1 表示过渡面。加入
`--binary-transition` 后，Blendit 的 VBF/EBF 都映射为 1，并在 manifest 和网页中
显示面级 Accuracy、Precision、Recall、F1：

```bash
PYTHONPATH=../../src python run_finetune_inference.py \
  --config ../../configs/finetune.yaml \
  --data-config ../../data/filletrec.yaml \
  --checkpoint ../../runs/finetune/<run>/checkpoints/best.pt \
  --split test \
  --split-file ../../data/filletrec/filletRec/test.txt \
  --cache-dir ../../data/cache/filletrec/test \
  --binary-transition \
  --sample-prefix filletrec__
```

原生二分类微调结果使用 `configs/finetune_filletrec_diffloss.yaml`。此时
`--binary-transition` 会直接保留模型的 0/1 预测，不再执行 VBF/EBF 合并：

```bash
cd ../..
PYTHONPATH=src python -m blendit.inference.finetune_visualize \
  --config configs/finetune_filletrec_diffloss.yaml \
  --checkpoint runs/finetune/<run>/checkpoints/best.pt \
  --split test --cache-dir data/cache/filletrec/test \
  --output-dir tools/visualize/results \
  --binary-transition --sample-prefix filletrec__
```
