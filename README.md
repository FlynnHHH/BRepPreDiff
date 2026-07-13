## 1. 总体结论

我建议可以采用下面这个 pipeline：

```text
20w 粗标签 CAD 模型
        ↓
B-Rep 图构建
        ↓
对 face / edge 几何特征加噪
        ↓
B-Rep Encoder 编码 noisy graph
        ↓
Diffusion Denoising Head 预测噪声或预测原始几何
        ↓
得到预训练 Encoder
        ↓
3w 精标签数据微调非过渡面 / VBF / EBF 任务头
```

更正式地说：

```text
Pretraining:
    noisy B-Rep graph → encoder → denoising head → reconstruct clean geometry/topology

Fine-tuning:
    clean B-Rep graph → pretrained encoder → segmentation head → non-transition/VBF/EBF
```

这个方向在方法论上有依据。CAD 表示学习中已有自监督预训练工作强调利用大规模无标签 CAD/B-Rep 几何来改善下游任务；B-Rep 领域也已经出现 masked BRep autoencoder、BRep-BERT、BRep 生成式 diffusion 等相关方向。Masked BRep Autoencoder 明确通过重建 masked faces/edges 的原始几何和属性来学习 B-Rep 几何/拓扑结构；LearnCAD 也表明用无标签 CAD 几何进行预训练可以改善监督任务表现。([arXiv][1])

---

# 2. 推荐方案：B-Rep Denoising Diffusion Pretraining

你的预训练不要扩散标签，而是扩散 **几何特征、拓扑关系特征和局部 UV-grid 几何**。

## 2.1 输入图

每个 CAD 模型构造成 B-Rep 属性图：

```text
G = (V, E)
```

其中：

```text
V = faces
E = face-face adjacency
```

每个 face 的特征：

```text
x_i^f = [
    surface_type,
    area,
    centroid,
    normal,
    k1, k2,
    mean_curvature,
    gaussian_curvature,
    UV-grid sampled points,
    UV-grid sampled normals
]
```

每条 edge / face-face relation 的特征：

```text
x_ij^e = [
    edge_type,
    edge_length,
    dihedral_angle,
    convexity,
    smoothness,
    continuity
]
```

你的 diffusion 不一定要对整个 B-Rep 拓扑结构加噪。**更稳妥的是固定拓扑图，只对连续几何属性和关系属性加噪。**

---

# 3. Diffusion 预训练具体怎么做

## 3.1 对连续几何特征加噪

对于每个 face 的连续特征，例如：

```text
centroid
area
normal
curvature
UV-grid xyz
UV-grid normal
dihedral angle
edge length
```

使用标准 diffusion forward process：

```text
x_t = sqrt(α_t) x_0 + sqrt(1 - α_t) ε
```

其中：

```text
x_0：原始干净几何特征
x_t：第 t 步加噪后的几何特征
ε ~ N(0, I)：高斯噪声
t：随机采样的 diffusion timestep
```

模型输入：

```text
G_t = (X_f^t, X_e^t, A)
```

也就是：

```text
加噪后的 face feature
加噪后的 edge feature
原始 adjacency topology
```

然后让模型预测噪声：

```text
ε_pred = DenoisingHead(Encoder(G_t), t)
```

损失为：

```text
L_diff = || ε - ε_pred ||_2^2
```

这就是最标准、最容易实现的 diffusion-style denoising objective。

---

## 3.2 也可以预测 clean geometry

除了预测噪声，也可以让模型直接预测干净特征：

```text
x_0_pred = DenoisingHead(Encoder(G_t), t)
```

损失为：

```text
L_recon = || x_0 - x_0_pred ||_1
```

两种方式都可以。实际实现中，我更推荐：

```text
预测噪声 ε + 辅助重建 x_0
```

即：

```text
L_denoise = || ε - ε_pred ||_2^2 + λ || x_0 - x_0_pred ||_1
```

这样训练更稳定，而且更容易保证学到的 embedding 对几何重建有用。

---

# 4. 哪些特征适合 diffusion，哪些不适合？

## 4.1 适合加高斯噪声的特征

这些特征是连续值，适合直接 diffusion：

```text
face area
face centroid
face normal
principal curvature k1, k2
mean curvature
gaussian curvature
UV-grid xyz coordinates
UV-grid normals
edge length
dihedral angle
```

尤其推荐对这几类做 diffusion：

```text
UV-grid sampled points
UV-grid sampled normals
curvature statistics
dihedral angle
```

因为它们和过渡面识别最相关。

EBF 通常表现为曲率连续变化的边过渡区域，VBF 通常出现在多个过渡面汇聚的顶点区域。因此，让 encoder 学会从 noisy curvature / noisy normals / noisy dihedral angles 中恢复干净几何，会增强它对过渡面结构的感知能力。

---

## 4.2 不建议直接加高斯噪声的特征

下面这些是离散或类别属性：

```text
surface_type
edge_type
convex / concave / smooth
face degree
loop count
```

这些不适合直接加高斯噪声。

可以采用两种处理方式：

### 方式 A：不扩散，只做辅助分类重建

```text
h_i → surface_type
h_ij → edge_type / convexity
```

损失：

```text
L_cat = CE(surface_type_pred, surface_type_gt)
      + CE(edge_type_pred, edge_type_gt)
      + CE(convexity_pred, convexity_gt)
```

### 方式 B：离散 diffusion

可以对类别 token 做 random replacement / categorical corruption：

```text
surface_type → random surface_type
edge_type → random edge_type
```

然后预测原始类别。

但我建议你初期不要上离散 diffusion，先用 **连续几何 diffusion + 离散属性分类重建**，实现成本更低，也更稳定。

---

# 5. 推荐的预训练结构

可以设计成下面这样：

```text
Noisy B-Rep Graph G_t
        ↓
Face / Edge Feature Embedding
        ↓
B-Rep Encoder
        ↓
Face Embedding h_i
Edge / Relation Embedding h_ij
        ↓
────────────────────────────────────
        ↓              ↓            ↓
Noise Prediction   Clean Recon.   Aux. Heads
Head               Head           surface / edge / relation
        ↓              ↓            ↓
L_noise            L_recon        L_cat + L_rel
```

也就是：

```text
B-Rep Encoder:
    输入 noisy graph
    输出 face-level embedding

Diffusion Head:
    预测被加到几何特征上的噪声

Reconstruction Head:
    辅助重建干净几何特征

Topology / Relation Head:
    预测 surface type、edge type、convexity、dihedral relation 等
```

预训练总损失：

```text
L_pretrain =
    L_noise
  + λ1 L_recon
  + λ2 L_cat
  + λ3 L_rel
  + λ4 L_noisy_label
```

其中：

```text
L_noise：预测 diffusion noise
L_recon：重建 clean geometry
L_cat：重建 surface type / edge type 等离散属性
L_rel：预测邻接面关系
L_noisy_label：粗标签辅助监督
```

推荐初始权重：

```text
L_pretrain =
    L_noise
  + 0.5 L_recon
  + 0.5 L_cat
  + 0.3 L_rel
  + 0.2 L_noisy_label
```

这里 `L_noisy_label` 仍然可以使用，但权重不要高。它只提供粗语义先验，不作为主监督。

---

# 6. 粗标签在 diffusion 预训练中怎么用？

你的 20w 数据有粗标签，但粗标签混淆。因此我建议：

```text
diffusion 主任务：几何/拓扑去噪
粗标签：辅助分类任务
```

结构如下：

```text
Noisy geometry + topology
        ↓
Encoder
        ↓
h_i
        ├── Denoising head: 预测干净几何
        └── Noisy label head: 预测 coarse non-transition/VBF/EBF
```

粗标签损失：

```text
L_noisy_label = CE(p_i, y_i^coarse_soft)
```

使用 soft label：

```text
coarse non-transition:
    [0.90, 0.05, 0.05]

coarse VBF:
    [0.05, 0.80, 0.15]

coarse EBF:
    [0.05, 0.15, 0.80]
```

这样可以让模型利用粗标签，但不会过度相信它。

---

# 7. 微调阶段

预训练结束后：

```text
保留：
    B-Rep Encoder

丢弃：
    Diffusion Head
    Reconstruction Head
    Categorical Reconstruction Head
    Noisy Label Head

新增：
    Fine Segmentation Head
```

微调结构：

```text
Clean B-Rep Graph
        ↓
Pretrained B-Rep Encoder
        ↓
Face Embedding h_i
        ↓
Segmentation Head
        ↓
non-transition / VBF / EBF
```

损失：

```text
L_finetune = CE(p_i, y_i^fine)
```

如果类别不平衡明显：

```text
L_finetune = WeightedCE(p_i, y_i^fine) + β DiceLoss(p_i, y_i^fine)
```

推荐：

```text
β = 0.3 ~ 0.5
```

---

# 8. 你可以写成这样的 pipeline

```text
Input:
    20w coarsely labeled CAD models
    3w accurately labeled CAD models

Step 1:
    Parse CAD models into B-Rep attributed graphs.

Step 2:
    Add diffusion noise to continuous face and edge attributes:
        - UV-grid points
        - UV-grid normals
        - curvature
        - area
        - dihedral angle
        - edge length

Step 3:
    Pretrain a B-Rep encoder through denoising objectives:
        - noise prediction
        - clean geometry reconstruction
        - surface / edge type reconstruction
        - relation prediction
        - noisy coarse-label auxiliary supervision

Step 4:
    Remove all pretraining heads.

Step 5:
    Add a face-level segmentation head.

Step 6:
    Fine-tune on 3w accurately labeled CAD models.

Output:
    Face-level non-transition / VBF / EBF predictions.
```

---

# 9. 论文中可以这样描述

```text
本文进一步设计了一种基于扩散去噪的 B-Rep 表征预训练策略。给定由 CAD 模型解析得到的 B-Rep 属性图，本文固定其拓扑邻接结构，并对面节点和边关系中的连续几何属性施加逐步高斯噪声，包括曲率、法向、UV-grid 采样点、边长和二面角等。随后，模型以加噪后的 B-Rep 属性图作为输入，通过 B-Rep 编码器提取上下文相关的面级表示，并利用扩散去噪头预测加入的噪声或恢复原始几何属性。该过程使编码器能够学习面片局部几何、相邻面关系以及曲率连续结构等通用 B-Rep 表征。

考虑到粗标注数据中存在 EBF 与 VBF 混淆，本文不将粗标签作为主要重建目标，而是将其作为辅助监督信号引入预训练过程。预训练完成后，移除扩散去噪头和辅助任务头，仅保留 B-Rep 编码器，并在精确标注数据上添加面级分类头进行微调，从而实现 EBF、VBF 和非过渡面的精确识别。
```

---

# 10. 和普通 MAE 预训练相比的优缺点

| 方法                  | 优点                         | 缺点                |
| ------------------- | -------------------------- | ----------------- |
| Masked Autoencoder  | 简单、稳定、容易实现                 | mask 比例和重建目标设计很关键 |
| Diffusion Denoising | 学到更细粒度的几何扰动鲁棒性，适合曲率/法向/二面角 | 训练更慢，调参更复杂        |
| 粗标签监督预训练            | 直接利用 20w 标签                | 容易学习标签噪声          |
| Diffusion + 粗标签辅助   | 同时利用几何自监督和粗语义先验            | 实现复杂度最高           |

对你的任务来说，我认为排序大概是：

```text
Diffusion + 几何/拓扑重建 + 粗标签辅助
    >
几何/拓扑 MAE + 粗标签辅助
    >
纯几何/拓扑 MAE
    >
粗标签监督预训练
    >
mask label reconstruction
```

---

# 11. 需要注意的风险

## 风险 1：不要做完整 B-Rep 生成

现在已有一些 B-Rep diffusion / CAD diffusion 工作尝试直接生成 B-Rep 模型，例如 BrepDiff、BrepGen、HoLa 等，它们说明 diffusion 可以用于 CAD/B-Rep 生成，但你的目标是**过渡面识别**，不是生成 CAD。完整生成式 B-Rep diffusion 会引入曲面合法性、拓扑一致性、边界闭合等复杂问题，工程代价很高。([ACM Digital Library][2])

所以建议你做：

```text
B-Rep feature denoising diffusion
```

而不是：

```text
full B-Rep generative diffusion
```

---

## 风险 2：图结构 diffusion 不要太激进

图数据上的 diffusion representation learning 是可行方向，但图结构本身有离散性和信噪比衰减问题。有工作指出，在图学习中直接使用各向同性白噪声 forward diffusion 可能导致信噪比快速下降，因此图上的 diffusion 需要更谨慎地设计噪声过程。([开放评审][3])

对你的 B-Rep 任务，最稳妥的做法是：

```text
固定拓扑 A
只对连续几何属性加噪
离散拓扑/类别属性用辅助分类重建
```

不要一开始就对 adjacency matrix、face-edge incidence、loop structure 做 diffusion。

---

## 风险 3：确保 encoder 真正学到表示

如果 denoising head 太强，模型可能把主要能力放在 decoder/head 上，encoder 学到的表示反而不强。这个问题在生成式预训练中很常见。

所以建议：

```text
encoder：相对强
denoising head：相对轻量
```

例如：

```text
Encoder: 6~12 层 Graph Transformer / BRepFormer-style blocks
Denoising Head: 2~3 层 MLP
```

这样去噪能力主要来自 encoder，而不是 decoder。

---

# 12. 最推荐的最终版本

我建议你的最终方法可以叫：

```text
Diffusion-based B-Rep Geometric Pretraining for Blend Face Recognition
```

核心流程：

```text
20w 粗标签模型
        ↓
构建 B-Rep 属性图
        ↓
对连续几何属性加 diffusion noise
        ↓
B-Rep encoder 学习去噪表示
        ↓
粗标签作为弱监督辅助任务
        ↓
获得 pretrained encoder
        ↓
3w 精标签微调
        ↓
面级非过渡面 / VBF / EBF 识别
```

最终预训练目标：

```text
L_pretrain =
    L_diff_noise
  + 0.5 L_geometry_recon
  + 0.5 L_topology_attr_recon
  + 0.3 L_relation
  + 0.2 L_coarse_label
```

最终微调目标：

```text
L_finetune =
    WeightedCE(y_pred, y_fine)
  + 0.3 DiceLoss(y_pred, y_fine)
```

我的建议是：**可以采用 diffusion，但先做“特征去噪 diffusion”，不要做完整 B-Rep 生成 diffusion；固定拓扑，对曲率、法向、UV-grid、二面角等连续几何属性加噪，训练 encoder 去噪，再用 3w 精标签微调。**

[1]: https://arxiv.org/html/2603.14927v1?utm_source=chatgpt.com "Masked BRep Autoencoder via Hierarchical Graph ..."
[2]: https://dl.acm.org/doi/10.1145/3721238.3730698?utm_source=chatgpt.com "BrepDiff: Single-Stage B-rep Diffusion Model"
[3]: https://openreview.net/forum?id=oDtyJt5JLk&utm_source=chatgpt.com "Directional diffusion models for graph representation learning"

---

# 13. PyTorch 代码实现

本仓库已经按上面的 idea 提供了一个可运行的 PyTorch 项目骨架。实现重点是：

```text
STEP + SEG
    ↓
pythonocc-core 解析 B-Rep
    ↓
缓存 face / edge 图特征到 npz
    ↓
Diffusion feature denoising pretrain
    ↓
加载 encoder 做 face segmentation fine-tune
```

## 13.1 代码框架

```text
configs/default.yaml
    控制数据路径、OCC 特征抽取参数、模型大小、diffusion 参数、训练参数。

src/blendit/brep/occ_extractor.py
    使用 pythonocc-core 读取 STEP。
    OCC 解析出的 face 顺序作为内部 face id 顺序。
    .seg 文件第 N 行标签对应 OCC 第 N 个 face。

src/blendit/data/
    StepSegDataset：匹配 STEP/SEG 文件，按需抽取并缓存 npz。
    graph.py：BRepGraph / GraphBatch，负责不同模型图的 batch 拼接。

src/blendit/models/
    encoder.py：纯 PyTorch B-Rep message passing encoder。
    diffusion.py：预训练模型、diffusion schedule、预训练 loss。
    segmentation.py：微调分割模型、Weighted CE + Dice loss。

src/blendit/training/
    pretrain.py：diffusion denoising 预训练入口。
    finetune.py：面级非过渡面/VBF/EBF 微调入口。
    smoke.py：CPU 合成图 smoke test，不依赖真实 STEP/OCC。

tests/test_cpu_smoke.py
    pytest smoke test。
```

### OCC STEP 特征抽取明细

`src/blendit/brep/occ_extractor.py` 会把一个 STEP 模型解析成面级图，最终缓存到 npz 的字段如下：

```text
face_cont           [num_faces, 11 + uv_grid_size^2 * 6]
face_surface_type   [num_faces]
edge_index          [2, num_directed_edges]
edge_cont           [num_directed_edges, 3]
edge_type           [num_directed_edges]
edge_relation       [num_directed_edges]
labels              [num_faces]，仅当存在 .seg 时写入
```

默认 `uv_grid_size=4`，所以 `face_cont` 是 `11 + 4*4*6 = 107` 维，`edge_cont` 是 3 维。

每个 face 的连续特征 `face_cont` 按下面顺序拼接：

```text
log1p(face_area)                         1 维
face centroid xyz                        3 维
UV 参数域中心点处的 unit normal xyz       3 维
中心点曲率: min/max/mean/Gaussian         4 维
UV-grid 采样点: point xyz + normal xyz    uv_grid_size^2 * 6 维
```

其中：

```text
face_area / centroid
    来自 BRepGProp.SurfaceProperties。

surface_type
    来自 BRepAdaptor_Surface.GetType()，做 +1 后 clamp 到 surface_type_vocab 范围内。

UV-grid
    在 face 的 UV bounds 内均匀采样。无法获得有效 UV bounds 时使用 [-1, 1] x [-1, 1]。

normal / curvature
    来自 BRepLProp_SLProps。若 face orientation 是 TopAbs_REVERSED，会翻转 normal。
```

face-face 边由 OCC 拓扑关系构造：

```text
TopExp.MapShapesAndAncestors(shape, EDGE, FACE)
```

每条拓扑 edge 找到相邻 face。如果同一条 edge 连接多个 face，会对相邻 face 做两两组合，并写成双向有向边。每条有向边的 `edge_cont` 是：

```text
log1p(edge_length)       来自 BRepGProp.LinearProperties
dihedral_angle / pi      angle = acos(dot(face_normal_i, face_normal_j))
normal_dot               dot(face_normal_i, face_normal_j)
```

离散 edge 属性：

```text
edge_type
    来自 BRepAdaptor_Curve.GetType()，做 +1 后 clamp 到 edge_type_vocab 范围内。

edge_relation
    angle <= smooth_angle_degrees 时为 1，否则为 2。
```

`labels` 不是 OCC 几何特征，而是从 `.seg` 读取的监督标签。第 N 行标签对应 OCC 解析出的第 N 个 face，默认按 ABC/BrepDit 的原始标签规则映射到三分类。

## 13.2 数据目录约定

默认配置读取：

```text
data/
    steps/
        part_0001.step
        part_0002.stp
    segs/
        part_0001.seg
        part_0002.seg
    splits/
        train.txt
        val.txt
        test.txt
```

`.seg` 文件每一行放一个整数标签。默认按 ABC/BrepDit 的原始标签映射成三分类：

```text
class 0 = NonTransition / 非过渡面
class 1 = VBF
class 2 = EBF

raw SEG 0/1/2/3/5/7 -> class 0
raw SEG 4             -> class 2 EBF
raw SEG 6             -> class 1 VBF
```

对应默认配置：

```yaml
labels:
  raw_to_class_map:
    4: 2
    6: 1
  default_class: 0
```

如果你的 `.seg` 已经是三分类 class id，可以关闭原始标签映射：

```bash
--override labels.raw_to_class_map=null
```

split 文件每行可以写 step 文件名、相对路径或不带后缀的 stem。如果 `configs/default.yaml` 里 `data.train_split` 为空，则会扫描 `data/steps` 下全部 STEP/STP 文件。

如果训练时只有已经抽取好的 `.npz` cache，没有 STEP/SEG 原始文件，可以开启
cache-only 模式：

```bash
--override data.cache_only=true
```

cache-only 模式不会读取 STEP/SEG，也不会重新抽取特征。训练集会直接扫描
`data.cache_dir` 下的全部 `.npz` 文件，不使用 `data.train_split`。验证/测试集
如果配置了 `data.val_split` / `data.test_split`，则仍会用 split 直接定位 cache，
split 可以写：

```text
00067352_6baf10385f.npz        # 相对 data.cache_dir 的 cache 文件名
finetune/00067352_6baf10385f.npz
/abs/path/to/00067352_6baf10385f.npz
00067352.step                  # 自动在 data.cache_dir 下匹配 00067352_*.npz
```

如果需要严格的 train/val 隔离，cache-only 训练时应把训练 cache 和验证 cache 放到
不同目录，或者不要配置验证集。

## 13.3 conda 环境

当前项目已经提供 `environment.yml`，默认用于创建 CPU 环境 `blendit`，包含：

```text
python=3.10
pytorch CPU
pythonocc-core
numpy / pyyaml / tqdm / pytest
```

创建环境：

```bash
conda env create -f environment.yml
conda activate blendit
```

如果环境已经存在，更新环境：

```bash
conda env update -n blendit -f environment.yml --prune
```

如果你想复用已有环境，例如当前机器上的 `meshpred` 环境，先验证核心依赖：

```bash
conda activate meshpred
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "from OCC.Core.STEPControl import STEPControl_Reader; print('pythonocc-core ok')"
```

运行本项目时直接使用 `PYTHONPATH=src python -m ...`，不强制做 editable install。若你仍然希望安装 `blendit-*` 命令行入口，可以在环境里执行：

```bash
python -m pip install --no-deps --no-build-isolation -e .
```

这里 `torch.cuda.is_available()` 在 CPU 环境里应输出 `False`。

## 13.4 CPU smoke test

这个测试只用合成图，不会读取 STEP，也不会开始训练真实数据：

```bash
conda activate meshpred
PYTHONPATH=src python -m blendit.training.smoke --config configs/default.yaml --override train.device=cpu
```

如果环境里安装了 `pytest`，再运行：

```bash
PYTHONPATH=src pytest -q
```

## 13.5 抽取并缓存 B-Rep 特征

先按需要修改 `configs/default.yaml`：

```yaml
data:
  root: data
  steps_dir: steps
  segs_dir: segs
  cache_dir: data/cache/features
  cache_only: false
  train_split: data/splits/train.txt
  val_split: data/splits/val.txt
```

然后手动执行缓存：

```bash
conda activate meshpred
PYTHONPATH=src python -m blendit.data.dataset --config configs/default.yaml --split train
PYTHONPATH=src python -m blendit.data.dataset --config configs/default.yaml --split val
```

缓存文件会写到：

```text
data/cache/features/*.npz
```

如果缓存阶段产生 invalid jsonl，需要先过滤 split，再用于训练：

```bash
PYTHONPATH=src python -m blendit.data.filter_split \
  --split data/abc_splits/pretrain_train.txt \
  --invalid-log data/cache/features/invalid/pretrain_train_invalid.jsonl \
  --output data/abc_splits/pretrain_train_clean.txt \
  --removed-output data/abc_splits/pretrain_train_invalid_removed.txt
```

`scripts/prepare_abc_data.sh` 会自动生成对应的
`pretrain_train_clean.txt`、`finetune_train_clean.txt` 和
`finetune_val_clean.txt`。训练时优先使用这些 clean split，避免命中缓存失败的样本。

如果训练中出现 `nan` loss，先扫描 cache 是否含有 NaN/Inf：

```bash
PYTHONPATH=src python -m blendit.data.scan_cache \
  --cache-dir data/cache/features/pretrain/first10000 \
  --invalid-log data/cache/features/invalid/pretrain_first10000_nonfinite.jsonl
```

训练脚本也会在 batch/loss/gradient 出现非有限值时直接报错，并打印触发问题的
`sample_ids`，避免继续保存无效 checkpoint。

## 13.6 启动 diffusion 预训练

手动启动，默认 CPU：

```bash
conda activate meshpred
PYTHONPATH=src python -m blendit.training.pretrain \
  --config configs/default.yaml \
  --override train.device=cpu \
  --override train.batch_size=2 \
  --override train.epochs=20
```

多卡 GPU 训练使用 `torchrun`。这里 `train.batch_size` 是每张 GPU 上的
batch size，总 batch size 为 `train.batch_size * nproc_per_node`：

```bash
conda activate meshpred
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src torchrun --standalone --nproc_per_node=4 \
  -m blendit.training.pretrain \
  --config configs/default.yaml \
  --override train.device=cuda \
  --override train.batch_size=2 \
  --override train.epochs=20
```

每次运行都会创建时间戳目录：

```text
runs/pretrain/YYYYmmdd-HHMMSS_baseline/
    config.yaml
    logs/pretrain.log
    checkpoints/epoch_0001.pt
    checkpoints/last.pt
```

从 pretrain checkpoint 继续预训练时，使用同阶段恢复入口 `train.resume`，并把
`train.epochs` 设为要训练到的总 epoch：

```bash
conda activate meshpred
PYTHONPATH=src python -m blendit.training.pretrain \
  --config configs/default.yaml \
  --override train.device=cpu \
  --override train.resume=runs/pretrain/YYYYmmdd-HHMMSS_baseline/checkpoints/last.pt \
  --override train.epochs=40
```

## 13.7 启动精标数据微调

把 config 的数据路径切到精标数据，或用 override 指定。从 pretrain ckpt 初始化
finetune 训练：

```bash
conda activate meshpred
PYTHONPATH=src python -m blendit.training.finetune \
  --config configs/default.yaml \
  --override train.device=cpu \
  --override train.pretrain_checkpoint=runs/pretrain/YYYYmmdd-HHMMSS_baseline/checkpoints/last.pt \
  --override train.batch_size=2 \
  --override train.epochs=30
```

多卡微调同样使用 `torchrun`：

```bash
conda activate meshpred
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src torchrun --standalone --nproc_per_node=4 \
  -m blendit.training.finetune \
  --config configs/default.yaml \
  --override train.device=cuda \
  --override train.pretrain_checkpoint=runs/pretrain/YYYYmmdd-HHMMSS_baseline/checkpoints/last.pt \
  --override train.batch_size=2 \
  --override train.epochs=30
```

`train.pretrain_checkpoint` 会加载 diffusion pretrain ckpt 中兼容的 `encoder.*`
权重，并在类别数匹配时把 `coarse_label_head.*` 初始化到 finetune 的
`seg_head.*`。旧入口 `train.pretrained_encoder` 仍可用，但只加载 encoder。

输出目录：

```text
runs/finetune/YYYYmmdd-HHMMSS_baseline/
    config.yaml
    logs/finetune.log
    checkpoints/best.pt
    checkpoints/last.pt
```

## 13.8 Finetune checkpoint 推理可视化

使用 finetune 后的 `last.pt` 或 `best.pt` 对 test split 推理，并把 STEP 网格
导出为网页可直接加载的 PLY：

```bash
PYTHONPATH=src python -m blendit.inference.finetune_visualize \
  --config configs/finetune.yaml \
  --checkpoint runs/finetune/YYYYmmdd-HHMMSS_baseline/checkpoints/best.pt \
  --step /path/to/model.step \
  --seg /path/to/model.seg \
  --output-dir tools/visualize/results
```

也可以直接传入 STEP 列表或目录，不需要预先准备 `.npz`：

```bash
PYTHONPATH=src python -m blendit.inference.finetune_visualize \
  --config configs/finetune.yaml \
  --checkpoint runs/finetune/YYYYmmdd-HHMMSS_baseline/checkpoints/best.pt \
  --step-list /path/to/test_steps.txt \
  --seg-list /path/to/test_segs.txt \
  --step-root /path/to/test_step_root \
  --seg-root /path/to/test_seg_root \
  --output-dir tools/visualize/results
```

```bash
PYTHONPATH=src python -m blendit.inference.finetune_visualize \
  --config configs/finetune.yaml \
  --checkpoint runs/finetune/YYYYmmdd-HHMMSS_baseline/checkpoints/best.pt \
  --step-dir /path/to/test_steps \
  --seg-dir /path/to/test_segs \
  --output-dir tools/visualize/results
```

如果仍想用 split 文件，也支持旧方式：

```bash
PYTHONPATH=src python -m blendit.inference.finetune_visualize \
  --config configs/finetune.yaml \
  --checkpoint runs/finetune/YYYYmmdd-HHMMSS_baseline/checkpoints/best.pt \
  --split test \
  --split-file data/abc_splits/<test_split>.txt \
  --output-dir tools/visualize/results
```

输出文件：

```text
tools/visualize/results/
    <sample>_instance_pred_rgb.ply   # SEG GT 高亮；找不到 SEG 时为灰色原始模型
    <sample>_semantic_pred.ply       # VBF 粉色，EBF 黄色
    prediction_manifest.json
```

然后启动网页：

```bash
cd tools/visualize
python viewer_server.py
```

浏览器打开 `http://localhost:8061/`。`tools/visualize/run_finetune_inference.py`
会默认把 PLY 写到网页目录下的 `results/`。

## 13.9 Windows STEP 到 SEG 推理包

面分类推理入口会递归扫描给定目录中的 STEP/STP，并输出原始 SEG 标签：

```text
非过渡面 = 0
EBF       = 4
VBF       = 6
```

在源码环境中运行：

```bash
PYTHONPATH=src python -m blendit.inference.step_to_seg /path/to/steps \
  --checkpoint runs/finetune/YYYYmmdd-HHMMSS_finetune/checkpoints/best.pt
```

默认结果写到输入目录旁的 `<输入目录名>_seg`，并保留子目录结构。构建 Windows
分发压缩包：

```bash
python scripts/build_windows_inference_package.py \
  --checkpoint runs/finetune/YYYYmmdd-HHMMSS_finetune/checkpoints/best.pt
```

产物为 `dist/Blendit-Windows-Inference.zip`。Windows 用户解压后先运行
`install_env.bat`，之后用 `run_inference.bat "D:\path\to\steps"` 推理。完整说明见
压缩包内的 `README_zh-CN.md`。

## 13.10 关键 config 参数

```yaml
brep:
  uv_grid_size: 4
  smooth_angle_degrees: 5.0

model:
  hidden_dim: 128
  num_layers: 4
  num_classes: 3

diffusion:
  timesteps: 1000
  noise_loss_weight: 1.0
  recon_loss_weight: 0.5
  categorical_loss_weight: 0.5
  relation_loss_weight: 0.3
  coarse_label_loss_weight: 0.2

train:
  device: cpu
  batch_size: 2
  lr: 1.0e-3
```

暂时没有空闲 GPU 时保持：

```yaml
train:
  device: cpu
```

后续有 GPU 后改成：

```bash
--override train.device=cuda
```

单机多卡时不要直接用 `python -m ...`，改用 `torchrun --standalone
--nproc_per_node=<GPU数量> -m ...`。训练脚本会自动按 `LOCAL_RANK` 选择
GPU，并且只在 rank 0 写日志和 checkpoint。
