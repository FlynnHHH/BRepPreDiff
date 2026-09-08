# Fusion360Seg 97% 准确率优化报告

本轮实验始于 2026-09-07，最终在官方 Fusion360Seg 测试集上取得
**97.0974% face accuracy**，达到预定的 97% 目标。

## 1. 实验协议与数据隔离

- 仅使用 Fusion360Seg s2.0.0；原始数据位于
  `/home/nvme03/hhfeng/fusion360segmentationdataset/s2.0.0/`。
- Train/Val/Test 分别包含 24,964 / 5,350 / 5,366 个 CAD，内部无重复且两两
  交集为 0。Train/Val 来自官方 train，Test 来自官方 test。
- 预训练严格执行 50 epochs，仅读取训练几何、移除 face labels，并禁用
  validation/test 数据源。
- 下游实验严格执行 100 epochs，仅使用 Fusion360Seg 训练标签；最佳 checkpoint
  只按 validation face accuracy 选择。该显式预算覆盖项目默认的 200 epochs。
- 搜索只使用 validation。候选成员与权重冻结后才执行独立 test；没有利用 test
  拟合参数或反向选择模型。
- 用户后续限定只使用 GPU 4，因此最终阶段全部在 GPU 4 串行完成。
- 每次实验保存源码快照及 SHA-256、命令、嵌入 checkpoint 的配置、split hashes、
  日志和评估 JSON。

下文“增长”均为相对 baseline 的**绝对百分点（pp）**，不是相对百分比。

## 2. 模块化代码改动

### 2.1 编码器：局部网格与全局上下文

文件：`src/brepprediff/models/encoder.py`

- 增加可选 face-grid 2D encoder 和 edge-grid 1D encoder，独立提取 UV face grid
  与 edge sampling grid 的局部几何模式。
- 增加 multi-scale context，可学习地融合输入层和各 GNN 中间层，而非只用末层。
- 对每个 CAD 计算 face embedding 的 mean/max context 并广播回所有 faces，让
  face 分类同时利用实体级形状和操作上下文。
- 支持 face/edge categorical mask，防止预训练时被遮蔽类别通过 embedding 泄漏。

Multi-scale context 将 validation 从 96.1974% 提升至 96.3889%，即
**+0.1914 pp**；单独 grid encoder 没有收益。

### 2.2 扩散预训练：类别属性遮蔽

文件：`src/brepprediff/models/diffusion.py`

- 新增 `attribute_mask_ratio`，遮蔽 face surface type、edge type 和 edge relation。
- 同一拓扑边的 reverse/parallel records 使用一致 mask，避免从反向记录恢复答案。
- 类别重建 loss 只在真正被遮蔽的位置计算。

50% masking 的结果为 96.0786%，相对 baseline **-0.1189 pp**，表明当前
50-epoch 预算下额外预训练难度没有转化为下游收益。

### 2.3 分割头：边界、操作族与结构化标签扩散

文件：

- `src/brepprediff/models/segmentation.py`
- `src/brepprediff/models/downstream.py`
- `src/brepprediff/training/finetune.py`

改动包括：

1. Boundary auxiliary head，预测相邻 faces 是否跨语义边界。
2. Boundary-gated refinement，以预测边界控制邻域信息聚合。
3. Operation-family auxiliary head，将 8 类映射为 Extrude、Cut、Fillet、Chamfer、
   Revolve 五个操作族；辅助 CE 权重为 0.2，推理仍输出原始 8 类。
4. Structured graph-label diffusion：同一 CAD 共享 timestep，支持完整 graph label
   noise 与多步采样。
5. 训练循环统一处理主输出、辅助 loss 和学习率 scheduler。

引入 operation supervision 的依据是 context 模型的 validation confusion matrix：
CutSide→ExtrudeSide 有 693 个错误，ExtrudeSide→CutSide 有 419 个，两者合计占
全部错误的 38.53%。但它与 rotation 组合后只比普通 rotation 高 0.0025 pp，
说明对 aggregate accuracy 基本中性。

### 2.4 几何预处理与数据增强

文件：

- `src/brepprediff/data/graph.py`
- `src/brepprediff/data/dataset.py`

- 增加按 CAD 的各向同性物理归一化，统一处理坐标、面积、长度、曲率，同时保持
  normals/tangents 为单位向量、trim mask 为精确 0/1。
- 增加 label-preserving SO(3) 刚体旋转；同步旋转 face center/normal、face-grid
  position/normal、edge-grid position/tangent。
- 增加 UV-grid 旋转与翻转重参数化。
- 增加可配置 rotation probability。最终策略保留 50% canonical CAD，另 50%
  随机旋转，兼顾方向不变性和 canonical 评估分布。
- 增加确定性评估旋转。TTA 在原始几何上旋转后再做 per-graph normalization，
  避免旋转已逐通道标准化 tensor 的错误捷径。

物理归一化为 95.6619%，相对 baseline **-0.5355 pp**；100% rotation 为
96.9057%，提升 **+0.7082 pp**；50/50 mixed rotation 达到 97.2410%，提升
**+1.0435 pp**，是最有效的单模型改动。

### 2.5 原始 B-Rep 局部共享边几何

文件：

- `src/brepprediff/brep/occ_extractor.py`
- `scripts/fusion97_local_edges.py`

- 从 STEP pcurve 计算共享边两侧的局部 face normals，更新 angle/dot/relation。
- 使用独立 cache `cache/features/fusion360seg_localedge_v1`，不覆盖 baseline；
  face、label、edge order、length 和 edge sampling grid 保持不变。
- 完成 train+val 共 30,314 个 CAD 的构建，零失败；新处理部分有 990,196 个成功
  edge pairs 和 14,917 个 fallback，并复用 100-CAD pilot。

该方案为 96.1699%，相对 baseline **-0.0275 pp**。当前 unsigned 表达仍不能
区分 convex/concave，可能限制收益。

### 2.6 评估：TTA、拓扑平滑与多模型融合

文件：

- `src/brepprediff/training/evaluate.py`
- `scripts/fusion97_validate_candidates.py`

- 增加 deterministic rotation TTA，对 identity 和固定 seeded SO(3) views 的
  概率求平均，并在 JSON 中记录旋转矩阵。
- 增加多 checkpoint probability ensemble。每个成员从自身 embedded config
  重建，并检查 task、feature dimensions 和 class count。
- 增加 confidence-gated graph smoothing：主要让低置信度 face 吸收邻接概率，
  高置信度 face、孤立 face 和清晰边界基本不变；alpha 被检查并记录。
- Validation search 使用预声明固定候选集，保存源码快照和 hashes，并明确记录
  `selection_split: val` 与 test 访问状态。

四视角 rotation TTA 为 97.0458%，未超过最佳单模型。三模型等概率 ensemble
达到 97.4087%，比最佳单模型再高 **+0.1677 pp**，相对 baseline 累计
**+1.2112 pp**。

### 2.7 实验基础设施与测试

文件：

- `scripts/fusion97_experiment.py`
- `scripts/fusion97_status.py`
- `tests/test_fusion97_variants.py`
- `tests/test_evaluate.py`

- 自动审计 split 唯一性、官方归属、互斥性和 cache 完整性。
- 强制本 campaign 只接受 GPU 4，保存 snapshot、hashes、manifest、command 和结果。
- 安全复用 50-epoch encoder，并检查 epoch、label stripping、数据来源和禁用
  val/test 等约束。
- 新增 masking 隔离、几何不变性、boundary/operation gradients、structured
  diffusion、rotation、TTA、ensemble 和 topology smoothing 回归测试。

最终完整测试结果为 **144 passed**。

## 3. 所有实验相对 baseline 的增长

以下结果均来自相同的 5,350-CAD、79,920-face validation split；Baseline 为
**96.1974%**。

| Variant | 核心变化 | 最佳 epoch | Val accuracy (%) | 相对 baseline (pp) |
|---|---|---:|---:|---:|
| baseline | 128 维、4 层 edge-update attention | 99 | 96.1974 | +0.0000 |
| masked | 预训练遮蔽 50% 类别属性 | 78 | 96.0786 | -0.1189 |
| wide | Hidden dimension 128 → 256 | 97 | 96.2588 | +0.0613 |
| grid | 独立 face/edge grid encoders | 89 | 96.0861 | -0.1114 |
| context | 多层融合 + graph mean/max context | 97 | 96.3889 | +0.1914 |
| boundary | 边界辅助监督 | 88 | 96.2112 | +0.0138 |
| boundary_refine | 边界监督 + gated refinement | 89 | 96.1324 | -0.0651 |
| geometric | 各向同性物理归一化 | 70 | 95.6619 | -0.5355 |
| geometric_grid | 物理归一化 + grid encoders | 91 | 95.9735 | -0.2240 |
| localedge | STEP pcurve 局部共享边 normals | 80 | 96.1699 | -0.0275 |
| context_cosine | Context + cosine LR | 63 | 96.2400 | +0.0425 |
| context_operation | Context + 操作族辅助监督 | 82 | 96.2988 | +0.1014 |
| context_rotation | Context + 100% 随机旋转 | 92 | 96.9057 | +0.7082 |
| context_rotation_seed43 | 独立 seed-43 rotation | 98 | 96.8143 | +0.6169 |
| context_rotation_operation | Rotation + 操作族监督 | 97 | 96.9082 | +0.7107 |
| context_rotation_mix | 50% canonical + 50% rotation | 90 | **97.2410** | **+1.0435** |
| final ensemble | Rotation + mix + rotation/operation | — | **97.4087** | **+1.2112** |

## 4. Validation-only 融合搜索

| 排名 | 候选 | Val accuracy (%) |
|---:|---|---:|
| 1 | rotation + rotation_mix + rotation_operation | **97.4087** |
| 2 | 上述三者 + seed43 | 97.4012 |
| 3 | rotation + rotation_mix | 97.3986 |
| 4 | rotation + rotation_mix + seed43 | 97.3811 |
| 5 | rotation + seed43 | 97.2447 |
| 6 | rotation + rotation_operation | 97.2147 |
| 7 | rotation + context | 97.1271 |
| 8 | rotation best + epoch100 + seed43 | 97.1234 |
| 9 | rotation TTA ×4 | 97.0458 |
| 10 | rotation + context_operation | 96.9494 |
| 11 | rotation best + epoch100 | 96.9494 |

候选集在结果产生前已经固定。最终选择第一名，不根据 test 调整成员或权重。

## 5. 最终结果

最佳单模型 `context_rotation_mix`：

- Validation：**97.2410%**，epoch 90；
- 官方 Test：**96.9197%**。

最终冻结的等概率 ensemble：

1. `context_rotation`，epoch 92；
2. `context_rotation_mix`，epoch 90；
3. `context_rotation_operation`，epoch 97。

最终独立官方 Test：

- Face accuracy：**97.0974%**；
- 正确 faces：**74,833 / 77,070**；
- CAD 数：5,366；
- Weighted F1：**97.0802%**；
- Macro F1：**90.9602%**；
- Rotation TTA：1，即未使用 TTA；
- Graph smoothing alpha：0.0，即未使用平滑后处理。

Baseline 没有执行官方 test，因此报告只给出最终 test 绝对值，不声明缺少证据的
baseline test 增长。

## 6. 核心结论

1. **旋转增强是最主要的有效因素。** 100% rotation 提升 +0.7082 pp，说明原始
   坐标方向存在明显过拟合。
2. **增强强度需要匹配评估分布。** 50% canonical + 50% rotation 比 100%
   rotation 再高 0.3353 pp，是最强单模型。
3. **全局 CAD context 有稳定收益。** 单独提升 +0.1914 pp，并成为后续强方案的
   公共基础。
4. **模型多样性比单纯增加容量更有效。** 三种训练策略的 probability ensemble
   比最佳单模型再高 0.1677 pp。
5. **复杂度不等于准确率。** Masking、grid encoder、物理归一化、local-edge
   normals 和 boundary refinement 均未超过 baseline；保留这些负结果可避免
   重复尝试。
6. **测试集隔离得到保持。** 所有候选与 ensemble 成员均由 validation 选择，
   最终组合冻结后才进行独立 test。
