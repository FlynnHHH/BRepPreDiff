# CAD 数据集论文所需输入信息调研与 BRepPreDiff 对比

> 调研对象：`reports/CAD数据集论文准确率调研与BRepPreDiff对比_2026-07-23.md` 中出现的全部 38 篇独立论文（按 URL 去重）
> 检索与复核截止：2026-08-17  
> 目标：回答每种方法从 CAD/STEP/B-Rep 中实际需要哪些信息，而不是再次比较 Accuracy  
> `明确` 表示论文正文、补充材料或官方实现给出了字段；`继承` 表示论文明确复用某个公开表示；`有限披露` 表示只能确认表示族，不能可靠确认全部通道。

## 1. 结论摘要

- 38 篇论文并不都直接读取 STEP。其输入大致分为五族：**B-Rep 属性/拓扑图、UV-grid 图、精确参数原语、点云/网格/多视图、工程知识或下游派生信息**。
- 最常见的原生 B-Rep 信息是：面邻接、surface type、face area、面坐标/normal、edge/curve type、edge length、convexity/dihedral angle。UV-Net 族还需要 face UV-grid 的 `XYZ + normal + trimming mask`，以及 edge U-grid 的 `XYZ + tangent`。
- **“使用 UV-grid”不是一个充分描述。** BRepPreDiff 的面网格是 `XYZ + normal`，没有 trimming mask；边也没有 U-grid/tangent。因此它与 UV-Net/AAGNet/BrepMFR/Masked HGT 的输入并不等价。
- BRepNet 是少数完全不需要坐标、normal 或 UV-grid 的代表：它依赖 surface/curve 类型、face area、edge length、edge convexity、闭合标志、coedge 方向和完整 coedge 拓扑游走。
- BRT、Brep2Shape 不走规则 UV 栅格路线，而需要可分解为 Bézier 曲线/曲面原语的参数几何及 face-edge-vertex 拓扑；这是 BRepPreDiff 当前缓存无法直接提供的。
- TopoGNN、Topo-Geom DualGNN、PP-Brep 在普通面邻接之外显式加入 vertex/loop 次序、平行/垂直/同轴/相切/共面、hyperedge 或长程平行关系；BRepPreDiff 当前也没有这些关系。
- KDH-CAD 与 MVCNN++ 只需要从 CAD 渲染出的多视图；KDH-CAD 还需要类别级文本、示意图片和教程视频。它们不需要 UV-grid、face normal 或 face area 作为网络显式输入。
- 若目标是扩大 BRepPreDiff 的输入覆盖，最有价值且改动最小的顺序是：**trimming mask → edge U-grid/tangent → loop count/face degree/bounding box → edge convexity → 全局几何关系**。Bézier/NURBS 原语编码属于另一条、改动明显更大的路线。

## 2. 统一字段定义

| 字段组 | 本报告含义 |
|---|---|
| Topology | face adjacency，以及更细的 face-edge/coedge/vertex incidence、loop order、edge adjacency 或 hyperedge |
| Face UV-grid | 在曲面参数域规则采样；通常每点含 XYZ、surface normal、trim mask |
| Edge U-grid | 沿参数曲线规则采样；通常每点含 XYZ、tangent；有些方法还附两侧 face normals |
| Face attributes | surface type、area、centroid、AABB/BBox、loop count、degree、curvature 等非栅格属性 |
| Edge attributes | curve type、length、convex/concave/smooth、dihedral angle、closed/seam 等 |
| Exact primitives | NURBS/B-spline/Bézier 的控制点、节点或分片，而不是采样点阵 |
| Derived relations | parallel、perpendicular、coaxial、tangent、coplanar、graph distance、feature openness 等由 CAD kernel 推导的信息 |
| External modalities | point cloud、mesh/facets、rendered views、尺寸元数据、文本/图片/视频知识 |

注意：face normal 可能指面中心法向，也可能指 UV-grid 每个采样点的法向；二者不可互换。edge convexity 也不等于简单的两法向夹角：严谨的凹凸判定通常还需要边切向和面定向。

## 3. BRepPreDiff 当前实际输入

以下字段直接由 `src/brepprediff/brep/occ_extractor.py` 和 `src/brepprediff/data/graph.py` 核实，而不是从 README 的概括反推。

### 3.1 面字段

每个 face 的连续向量为：

1. `log1p(area)`：1 维；
2. centroid XYZ：3 维；
3. UV 包围区间中心点处的 outward normal：3 维；
4. 同一点的 `min / max / mean / Gaussian curvature`：4 维；
5. `g × g` 面参数网格：每点 `XYZ + normal` 共 6 维；当前常用 `g=4`。

另有离散 `surface_type`。因此面连续维数为 `11 + 6g²`，`g=4` 时为 107。

### 3.2 边与拓扑字段

- 图节点是 faces；共享同一 B-Rep edge 的 face pair 形成双向边。
- 连续边属性：`log1p(edge length)`、`dihedral angle / π`、两面中心 normal 的 dot product。
- 离散边属性：CAD kernel 的 `edge_type`；`edge_relation` 仅按夹角阈值区分 smooth 与 non-smooth。

### 3.3 当前明确没有的字段

- face UV trimming mask；
- edge U-grid、edge sample coordinates、edge tangent；
- loop/inner-loop count、face degree、face AABB；
- 显式 concave/convex/smooth 三分类；
- coedge、coedge orientation、next/previous/mate、vertex 与 loop order；
- NURBS/Bézier control points、knots、weights；
- parallel/perpendicular/coaxial/tangent/coplanar 等非局部关系；
- point cloud/mesh/multi-view/rendering、construction history 或外部工程知识。

## 4. 全部论文总表

### 4.1 Fusion360Seg 与通用 B-Rep 表征论文

| 论文 / 方法 | 网络实际需要的 CAD 信息 | 与 BRepPreDiff 的关键差异 | 证据 |
|---|---|---|:---:|
| [BRepNet](https://openaccess.thecvf.com/content/CVPR2021/html/Lambourne_BRepNet_A_Topological_Message_Passing_System_for_Solid_Models_CVPR_2021_paper.html) | 完整 face/edge/coedge 拓扑及 `next/previous/mate/face/edge` 游走；face：surface type、rational-NURBS flag、area；edge：curve type、convexity、closed flag、length；coedge：与底层 curve 同向标志。**不需要 XYZ、normal、UV-grid。** | BRepPreDiff 有坐标/normal/curvature，但没有 coedge 拓扑、coedge 方向、closed flag 和严格 convexity。 | 明确 |
| [UV-Net](https://openaccess.thecvf.com/content/CVPR2021/html/Jayaraman_UV-Net_Learning_From_Boundary_Representations_CVPR_2021_paper.html) | face-adjacency graph；face 2D UV-grid：XYZ、outward normal、trim mask；edge 1D U-grid：XYZ、可选 tangent。整体中心化并缩放。 | BRepPreDiff 面网格无 trim mask，边无 U-grid/tangent；BRepPreDiff 额外有 area、centroid、curvature、类型和 dihedral。 | 明确 |
| [CADOps-Net](https://arxiv.org/abs/2208.10555) | 输入 B-Rep 首先经 BRepNet-style encoder；需要 BRepNet 的 face/edge/coedge 属性与拓扑。联合监督还需要每面 operation type 和 operation-step ID；`JL+` 使用增强的 step aggregation/联合学习。 | 几何输入比 BRepPreDiff 少连续坐标，多 coedge 信息；训练标签多出 construction step。 | 明确/继承 |
| [Self-Supervised Representation Learning for CAD / SSL4CAD](https://openaccess.thecvf.com/content/CVPR2023/html/Jones_Self-Supervised_Representation_Learning_for_CAD_CVPR_2023_paper.html) | face-edge-vertex 层级图；解析曲面/曲线参数及 orientation；限定 plane/cylinder/cone/sphere/torus 和 line/circle/ellipse 等固定参数原语。自监督 target 还要在 face UV 域查询 XYZ 与 trimming-boundary signed distance。 | BRepPreDiff 不输入解析原语参数、vertex 层级或 trimming SDF；它用固定网格和显式属性。 | 明确 |
| [BRep-BERT](https://doi.org/10.1145/3583780.3615237) | face-adjacency/topological sequence；face `10×10×7` UV-grid（XYZ、normal、trim mask）和 curve U-grid，经 2D/1D CNN tokenization；图相对位置/几何关系。星号版本还需要 CAD construction temporal supervision。 | BRepPreDiff 无 trim mask、edge U-grid、Transformer 相对位置；也不使用 construction time。 | 明确 |
| [Bringing Attention to CAD / BRT](https://doi.org/10.1016/j.cad.2025.103940) | vertex coordinates；B-spline/NURBS curve 与 surface 的 Bézier 分解/控制点；trimmed surface 的 Bézier triangles；face-edge-vertex topology 被序列化为 topology-aware tokens。 | 不是规则 UV-grid。BRepPreDiff 缓存没有精确控制点、Bézier 分片、vertex tokens。 | 明确 |
| [Brep2Shape](https://arxiv.org/abs/2602.07429) | 沿用 BRT 的 Bézier primitive tokenization：face/surface 与 edge/curve 两路控制点原语；face-edge incidence、face adjacency、edge line graph；预训练 target 是从参数原语产生的 dense spatial points。 | BRepPreDiff 的采样点是输入，Brep2Shape 的 Bézier 控制点是输入、密集点是自监督 target。 | 明确 |
| [Masked BRep Autoencoder via Hierarchical Graph Transformer](https://arxiv.org/abs/2603.14927) | face 两级 UV-grid `3×3` 与 `13×13`，每点 XYZ+normal+trim；face attrs：6 类 type、area、centroid、BBox；edge 13 点，每点 XYZ+tangent+两侧 normals；edge attrs：type、length、convexity；face adjacency。 | 相比 BRepPreDiff 多 trim、BBox、edge samples/tangent/incident normals、convexity与双分辨率；BRepPreDiff 多中心 curvature。 | 明确 |
| [BRepMAE](https://arxiv.org/abs/2602.22701) | gAAG；face UV geometry，face type、area、centroid、BBox；edge sampled geometry 为 XYZ+tangent+左右面 normals，另有 edge type、length、convexity；拓扑邻接。 | 与 Masked HGT 接近；BRepPreDiff 缺 trim/BBox/edge sampling/convexity。 | 明确 |
| [FoV-Net](https://arxiv.org/abs/2602.24084) | face-adjacency graph；每面 local-reference-frame UV-grid；outward/inward/其他 FoV ray-casting grids，记录周围 face 的交点/深度上下文；surface type 与 area。 | BRepPreDiff 使用全局坐标网格且无 ray casting/LRF，因此不具其旋转不变描述符。 | 明确 |
| [TopoGNN](https://doi.org/10.2139/ssrn.6604901) | STEP 的 heterogeneous face-edge-vertex graph；orientation-aware incidence、adjacency、loop-order relations；NURBS/UV descriptors；face、edge、vertex 各自的 VAE geometry embeddings；也比较 compact handcrafted descriptors。 | BRepPreDiff 是 face-only homogeneous graph，缺 vertex、loop order、NURBS 参数与 entity-specific VAE。 | 明确到表示族；逐通道有限披露 |
| [Modified PointNet++](https://doi.org/10.1016/j.cad.2023.103629) | 从 B-Rep surface 采样 point cloud；XYZ、unit surface normal，并将 face-level shape descriptors/归一化 face index传播到点。训练时要保留 point-to-BRep-face mapping，以计算 face loss并把点预测聚合回面。 | BRepPreDiff不需要点云或点面映射；该法通常不使用精确 face adjacency/edge features。 | 明确 |
| [Two-level feature reconstruction network](https://www.sciencedirect.com/science/article/pii/S095219762601050X) | 输入现有 B-Rep；一级预测 face type、edge type 与 virtual links，二级用这些关系重建 feature hierarchy。可确认需要 face/edge topology 与几何编码，但公开页面未完整列出底层几何通道。额外监督包括 edge creation precedence、同一 feature 的 virtual-link labels 和 instance annotations。 | 最大差异是额外的 edge/virtual-link/instance 监督；底层通道不宜凭基线反推。 | 有限披露 |
| [Segmentation of CAD models using hybrid representation](https://doi.org/10.1016/j.vrih.2025.01.001) | 可确认融合 B-Rep/topology 与离散几何表示；可访问正文信息不足以逐项核实 UV、normal、area 等通道。 | 不作逐字段断言；复现前必须取得正文/代码。 | 有限披露 |
| [Uncertainty estimation review](https://doi.org/10.1007/978-3-031-96196-0_4) | 这是评估/不确定性与人工复核流程，不是新的 CAD encoder。其 CAD 输入需求继承被评估的 segmentation backbone；另需 predictive uncertainty 和人工 review decisions。 | 不应把它当成独立几何表示与 BRepPreDiff 比较。 | 明确到研究角色 |

### 4.2 MFCAD++ 加工特征论文

| 论文 / 方法 | 网络实际需要的 CAD 信息 | 与 BRepPreDiff 的关键差异 | 证据 |
|---|---|---|:---:|
| [Hierarchical CADNet](https://doi.org/10.1016/j.cad.2022.103226) | 两层图：上层 face-adjacency graph 表示 B-Rep topology；下层 facet graph/三角网格表示每个面的 surface geometry；Edge 版本加入相邻面 edge convexity。 | BRepPreDiff无需 tessellation/facet graph；它有 UV samples，但缺严格 convexity。 | 明确 |
| [BRepGAT](https://doi.org/10.1093/jcde/qwad100) | Face Attribute Adjacency Graph；node 至少含 surface type、area、normal、BBox ratio、outer/inner-loop 信息；edge 含 curve type 及相邻关系/边描述符；GAT使用面邻接。 | BRepPreDiff缺 loop、BBox ratio与明确 edge convexity字段，但有更密的面采样和 curvature。 | 明确 |
| [AAGNet](https://doi.org/10.1016/j.rcim.2023.102661) | gAAG/face adjacency；face UV-grid `XYZ+normal+trim` 与 edge U-grid `XYZ+tangent`；face global attrs（type、area、centroid等）和 edge attrs（type、length、convexity、dihedral等）；多任务还需 semantic、instance adjacency、bottom-face labels。 | BRepPreDiff只覆盖其中部分面属性和 dihedral；缺 trim、edge grid/tangent、instance/bottom labels。 | 明确/官方实现 |
| [BrepMFR](https://doi.org/10.1016/j.cagd.2024.102318) | face UV-grid 7 通道；face attrs：8 类 surface type、area、loop count、adjacent-face count；edge attrs：8 类 curve type、length、convexity、dihedral angle；还使用 face-pair 最短 graph distance embedding。 | BRepPreDiff缺 trim、loop/degree、convexity、graph distance；BrepMFR论文主干没有把 edge U-grid列为最终 edge geometry。 | 明确 |
| [MFTReNet](https://doi.org/10.1016/j.aei.2024.102721) | STEP → attributed face graph，融合 face/edge geometry 与 topology；除逐面 semantic 外还需要 instance grouping 与 feature-feature topological relationship labels。公开页未足以逐项确认全部 node/edge 通道。 | 任务监督显著多于 BRepPreDiff；不能把 AAGNet 字段未经核实地全部套用。 | 有限披露 |
| [Sheet-metalNet](https://doi.org/10.1038/s41598-024-61443-2) | multidimensional attributed face-edge graph：face surface type/area 等 node attributes，edge relation/convexity等 edge attributes与 adjacency；增量学习还需旧类 exemplars/类别阶段信息。 | 与 BRepPreDiff同属属性面图，但不依赖 UV-grid；显式多维 AAG 字段与训练协议不同。 | 明确到属性组 |
| [SCUT B-Rep GNN](https://zrb.bjb.scut.edu.cn/EN/10.12141/j.issn.1000-565X.230497) | B-Rep adjacency graph 直接输入高效 GNN；公开英文摘要未列出 node/edge descriptor 的完整通道。 | 只能确认 topology graph，不足以确认是否含 UV-grid、area或normal。 | 有限披露 |
| [SFRGNN-DA](https://doi.org/10.1016/j.jmsy.2025.05.005) | B-Rep face graph 的 geometry/topology features，加 domain-adaptation 所需 source/target domain 样本与 domain labels/对抗信号。可访问页面未给完整原始字段表。 | 除几何输入外还需跨域数据；逐字段需正文/代码复核。 | 有限披露 |
| [Semantic Direct Modeling](https://www.researchgate.net/publication/390989999_Semantic_Direct_Modeling) | MFCAD++ 派生 B-Rep semantic graph/face features用于识别，再把 semantic result 提供给 direct-modeling generation；公开稿信息不足以核实所有原始通道。 | 它还需要生成/编辑语义目标，不是单纯逐面分类器。 | 有限披露 |
| [EMD-GNN](https://doi.org/10.1016/j.aei.2026.104609) | 双图/edge-modulated B-Rep graph，明确目标是同时编码 faces、edges与交互关系；可访问索引未给完整 attribute table。 | 不可靠地声称其含某个具体 UV/area字段；需正式全文或代码。 | 有限披露 |
| [TEGNet](https://www.researchgate.net/publication/401656636) | 新 face attributed graph；face/edge geometric attributes + topology；论文片段明确 edge U-grid 含 coordinates（通常还含 curve differential information），face 分支含 UV geometry。 | BRepPreDiff无 edge U-grid；其他精确通道因正式公开正文不足而不展开。 | 部分明确 |
| [MMNet](https://doi.org/10.32604/cmes.2026.078073) | face adjacency；face UV-grid：XYZ、normal、trim；edge U-grid；face global attrs：surface type、area、centroid等；edge global attrs：curve type、length、convexity、dihedral；cross/channel attention融合。另需 semantic/instance labels。 | 是比 BRepPreDiff更完整的 UV+AAG输入；BRepPreDiff缺 trim、edge grid与多个 global attrs。 | 明确 |
| [Topo-Geom DualGNN](https://doi.org/10.3390/machines14040362) | TAAG：face `N×N×7` UV-grid，edge `N×6` XYZ+tangent，face adjacency；GRG node：area、loop count、shape index、PCA axes、mean inertia、eccentricity、surface type；GRG edge：parallel/perpendicular/coaxial/tangent/coplanar。 | BRepPreDiff缺 trim、edge grid以及全部 GRG/非局部工程关系。 | 明确 |
| [MSFRNet](https://doi.org/10.1016/j.aei.2026.104365) | B-Rep graph 的 local/global features；此外显式使用 machining-feature openness 与 concavity/convexity rule sets 区分近似特征。公开页面未列完整基础 tensor。 | 比 BRepPreDiff多规则推理与 feature-level openness；底层通道不能全部确认。 | 部分明确 |
| [Shaft process-planning model](https://doi.org/10.3390/app16020828) | 先获得 machining feature instances，再构造扩展 AAG：feature node/edge attributes，用于每个 feature 的加工方案分类；因此依赖上游 feature recognition、feature geometry/topological relations及工艺标签。 | 它不是直接 STEP→face label 的同层级模型；输入已含高层 feature semantics。 | 明确 |
| [AgentsCAD](https://arxiv.org/abs/2607.02448) | STEP/B-Rep face adjacency；每面几何用于 overhang angle（相对 build direction，阈值 45°）；可选 GraphSAGE 注入 MFCAD++ semantic face labels；LLM还接收制造约束/文本上下文。 | 比 BRepPreDiff多 build direction、overhang与LLM语义；GraphSAGE底层字段披露有限。 | 明确到系统接口 |

### 4.3 FabWave/TMCAD 分类、检索与数据集论文

| 论文 / 方法 | 网络实际需要的 CAD 信息 | 与 BRepPreDiff 的关键差异 | 证据 |
|---|---|---|:---:|
| [FabSearch / FabWave 数据集](https://doi.org/10.1115/1.4043211) | 这是数据集/检索系统论文，不定义一个与上述 GNN 同构的固定 encoder 输入。数据侧提供 CAD models、part categories及可用于几何搜索/制造检索的信息。 | 应作为数据来源而非独立 UV/图模型；下游论文各自重新提取表示。 | 明确到研究角色 |
| [MVCNN++](https://doi.org/10.1115/1.4047486) | 从 CAD 渲染的多个 2D views；另拼接整体 part dimensions/size metadata；ImageNet-pretrained CNN。 | 不需要 B-Rep topology、UV-grid、face normal、area或edge信息。 | 明确 |
| [VGNet](https://doi.org/10.1109/TMM.2024.3521706) | 双模态：多个 rendered views + B-Rep attribute graph。后者含 face node 与 shared-edge attributes/adjacency，论文 Table I 给出属性；两路注意力融合。 | BRepPreDiff只有 B-Rep一路，无 rendered views；VGNet面向 whole-part retrieval/classification。 | 明确到模态/属性图 |
| [CADGCL](https://doi.org/10.1007/s00371-025-03949-y) | STEP→B-Rep attributed graph：node geometric attributes + graph edges/topology；训练时还要 edge-betweenness centrality，用于结构保持的 edge perturbation，以及 feature masks。公开 HTML 未逐项列全 node channels。 | BRepPreDiff不计算 EBC/contrastive graph views；精确属性需代码/全文表格。 | 部分明确 |
| [PP-Brep](https://openaccess.thecvf.com/content/CVPR2026/html/Hao_PP-Brep_Few-Shot_B-rep_Classification_with_Hybrid_Graph_Representation_CVPR_2026_paper.html) | face `32×32×8` nonparametric grid + 17D attrs：surface type、area、degree/loop/adjacent-face count、Gaussian/mean/principal curvatures等；adjacency edge `32×12` samples + convexity/length/curve type；parallel edge 的 angle、plane distance、area ratio、centroid distance、direction flag、offset；另构造 hyperedges。 | BRepPreDiff缺第8个网格通道、edge grid、parallel graph、hypergraph和多数结构属性。 | 明确 |
| [BRepCLIP](https://arxiv.org/abs/2606.05515) | CAD object 被序列化为 face/edge tokens；surface 与 curve 使用独立离散 vocab，附 surface/curve type、spatial/semantic descriptors及 topology；对齐 frozen CLIP text/image embeddings。公开摘要未给每个 descriptor 数值定义。 | BRepPreDiff是连续属性图，不是离散 primitive vocabulary；也无图文对齐数据。 | 明确到 token 组 |
| [KDH-CAD](https://arxiv.org/abs/2606.01702) | 每个 CAD part 的 `M` 个 rendered views；类别级工程知识 triplet：canonical terminology、教材/网页文本+示意图、教程视频（论文实验视频采 12 frames）；冻结 Qwen3-VL vision/language encoder。 | 不读取 UV、normal、area、topology；额外依赖外部知识材料和 VLM。 | 明确 |

## 5. 按信息字段反查论文

该表用于回答“如果缓存里增加某字段，哪些论文路线可以更接近复现”。`✓` 是明确使用，`~` 是表示族确认但逐通道有限，空白为不需要或未证实。

| 方法族/代表 | Face adjacency | Face UV XYZ | Face normal | Trim mask | Face area/type | Curvature | Edge U-grid/tangent | Edge type/length | Convexity/dihedral | Loop/vertex/coedge | 精确 Bézier/NURBS | 外部模态/知识 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| BRepPreDiff | ✓ | ✓ | ✓ |  | ✓ | ✓ |  | ✓ | dihedral |  |  |  |
| BRepNet/CADOps | ✓ |  |  |  | ✓ |  |  | ✓ | ✓ | coedge ✓ |  | step labels（CADOps） |
| UV-Net | ✓ | ✓ | ✓ | ✓ |  |  | ✓ |  |  |  |  |  |
| BRep-BERT | ✓ | ✓ | ✓ | ✓ |  |  | ✓ |  |  | relative position |  | temporal labels（*） |
| AAGNet/MMNet | ✓ | ✓ | ✓ | ✓ | ✓ | 部分 | ✓ | ✓ | ✓ | loops/degree |  | instance/bottom labels |
| BrepMFR | ✓ | ✓ | ✓ | ✓ | ✓ |  |  | ✓ | ✓ | loops/degree |  | graph distance |
| Masked HGT/BRepMAE | ✓ | ✓ | ✓ | ✓ | ✓ |  | ✓ | ✓ | ✓ |  |  |  |
| Topo-Geom DualGNN | ✓ | ✓ | ✓ | ✓ | ✓ | derived descriptors | ✓ |  |  | loops |  | 5类几何关系 |
| PP-Brep | ✓ | ✓ | ✓ | ~ | ✓ | ✓ | ✓ | ✓ | ✓ | loops/hyperedge |  | parallel graph |
| BRT/Brep2Shape | ✓ | target only | target only | via trimmed primitives | primitive-derived |  | target only | primitive-derived |  | vertex ✓ | ✓ |  |
| TopoGNN | ✓ | ✓/~ | ✓/~ | ~ | compact descriptors | ~ | ✓/~ | ~ | ~ | vertex/loop-order ✓ | NURBS ✓ |  |
| Hierarchical CADNet | ✓ |  | facet normals/geometry |  | ~ |  |  |  | ✓ | facet graph |  | mesh/facets |
| Modified PointNet++ | point-face mapping | point XYZ | point normal |  | propagated descriptors | ~ |  |  |  | face ID |  | point cloud |
| FoV-Net | ✓ | LRF ✓ | LRF ✓ | ✓ | ✓ |  |  |  |  |  |  | ray-cast FoV grids |
| MVCNN++ |  |  |  |  |  |  |  |  |  |  |  | views + dimensions |
| VGNet | ✓ |  | ~ |  | attrs ✓ |  |  | attrs ✓ | ~ |  |  | rendered views |
| KDH-CAD |  |  |  |  |  |  |  |  |  |  |  | views + text/image/video |

## 6. 对 BRepPreDiff 的可执行建议

### 6.1 低成本、高覆盖

1. **加入 trimming mask。** 当前网格在 UV bounding rectangle 全部采样，即使点位于 trimmed face 外仍被当作有效曲面点。增加 1D mask 后，面网格从 6 通道变为 7 通道，可直接靠近 UV-Net、BRep-BERT、AAGNet、BrepMFR、MMNet、Masked HGT 和 Topo-Geom DualGNN。
2. **加入 edge U-grid。** 建议每条 B-Rep edge 均匀采样 `XYZ + unit tangent`；若面向 Masked HGT/BRepMAE，再追加两侧 face normals。当前单个 length/type/dihedral 无法表达曲线局部形状。
3. **补充 cheap scalar attributes。** face loop count、inner-loop count、degree、BBox center/half-size；edge strict convexity。它们都可由 OCC 稳定提取，不需改变图的实体类型。

### 6.2 中成本、用于关系建模

4. 增加 parallel/perpendicular/coaxial/tangent/coplanar 五类 relation graph，并与当前物理 adjacency 分开存储，避免把非邻接关系混入同一 edge type。
5. 增加 face-edge incidence 或显式 edge nodes。这样才能自然支持 multiple shared edges、edge update、line graph和 PP-Brep/Brep2Shape 式双流，而不把所有共享边压成 face pair 属性。

### 6.3 高成本、应单独立项

6. coedge/loop/vertex heterogeneous graph；需要稳定保存 orientation、next/previous/mate 和 loop order。
7. NURBS/B-spline→Bézier primitive decomposition、控制点 tokenization 与 trimmed Bézier triangulation。该路线接近 BRT/Brep2Shape，不应被当作“把 UV-grid 加密”处理。
8. 多视图或知识增强只适合 whole-model classification/retrieval；对当前 face segmentation 主线并非同一输入改造。

## 7. 复现检查清单

复现任意论文前，至少记录：

- STEP kernel 与版本；是否修复/跳过 invalid solid、seam edge、degenerate edge；
- 坐标中心化、尺度归一化、旋转对齐与 normal orientation；
- face/edge 的参数区间、UV/U 分辨率、端点采样策略和 trim 判定；
- 一个 face pair 多条 shared edges 时是保留、合并、选最长还是构造 multigraph；
- edge convexity 的符号约定与 smooth tolerance；
- surface/curve type vocabulary 对 NURBS、B-spline、offset、intersection curve 的映射；
- topology 是 face adjacency、edge nodes、coedges还是完整 face-edge-vertex heterogeneous graph；
- 数据增强、预训练语料、construction history、instance/relationship labels与外部知识是否属于必要输入。

## 8. 证据边界

- 优先级为原论文正文/补充材料、正式出版 HTML/PDF、arXiv/OpenReview 和官方代码。38 篇论文均保留原始链接。
- SCUT B-Rep GNN、SFRGNN-DA、Semantic Direct Modeling、EMD-GNN、hybrid representation 等论文的可访问页面没有完整 attribute table；本报告明确标为“有限披露”，没有用相邻论文的字段替代。
- AAGNet、UV-Net 等公开实现被多个后续方法复用；只有原文明确写出“沿用/基于”时才记为继承。
- “论文用到某数据集”不等于“它直接消费数据集中预计算的同一特征”。多数工作从 STEP 重新提取自己的图、栅格、点云或视图。
- 本报告只列模型输入和构造输入必需的 CAD 信息；训练标签单独注明，不把标签误写成推理时的 CAD 几何字段。
