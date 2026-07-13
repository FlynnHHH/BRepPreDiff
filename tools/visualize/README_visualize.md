# Blendit finetune 推理可视化

此网页读取 `results/` 目录下的 PLY 文件：

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

## FilletRec 二分类测试

FilletRec JSON 标签按 OCC face 顺序存放，0 表示非过渡面，1 表示过渡面。加入
`--binary-transition` 后，Blendit 的 VBF/EBF 都映射为 1，并在 manifest 和网页中
显示面级 Accuracy、Precision、Recall、F1：

```bash
PYTHONPATH=../../src python run_finetune_inference.py \
  --config ../../configs/finetune.yaml \
  --checkpoint ../../runs/finetune/<run>/checkpoints/best.pt \
  --split test \
  --split-file ../../data/filletrec/filletRec/test.txt \
  --cache-dir ../../data/cache/filletrec/test \
  --binary-transition \
  --sample-prefix filletrec__ \
  --override data.root=../../data/filletrec/filletRec \
  --override data.steps_dir=steps \
  --override data.segs_dir=labels
```
