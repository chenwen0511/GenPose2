# GenPose2 模型结构 · 文字版解读

对应图纸：[`genpose2_model.xml`](./genpose2_model.xml)（draw.io 可打开编辑）  
相关代码：`networks/posenet.py`、`networks/gf_algorithms/scorenet.py`、`networks/gf_algorithms/energynet.py`、`networks/scalenet.py`、`networks/gf_algorithms/samplers.py`  
原理展开：[`learning/模型讲解.md`](../learning/模型讲解.md)

---

## 0. 总览：图在画什么

图纸按 **信号处理 / 数据流** 从左到右展开，四段主链路与图标题一致：

```text
Observation  →  Score Diffusion  →  Energy Aggregate  →  Scale
   观测编码         扩散采样位姿           能量排序聚合         尺寸回归
```

| 图中虚线框 | 角色一句话 |
|------------|------------|
| Observation Encoder | 把 RGB + 点云压成条件特征 `pts_feat` |
| PoseScoreNet Φθ | 在噪声位姿上预测 score，ODE 循环去噪，采出 K 个候选 |
| PoseEnergyNet Ψφ | 给候选打能量分，裁剪离群后聚合成一个 6D |
| ScaleNet | 已知朝向后回归 `size_3d = [l,w,h]`（可跳过） |

配色与 Transformer 示意图同级约定：粉=输入/输出，绿=编码，蓝=MLP 头，黄=融合，橙=线性/score 输出。

---

## 1. INPUTS（最左列）

### 1.1 ROI RGB · `H×W×3`

实例裁剪后的彩色图，送入 DINOv2。  
网络**不负责分割**；分割 / mask 在进模型前完成。

### 1.2 Instance Points · `[B, 1024, 3]`

实例点云（默认 1024 点）。由深度 + mask + 相机内参反投影得到。

### 1.3 Mask + Camera K

图中紫色块：不直接进 Score 头，而是 **反投影出点云**（虚线箭头指向 Instance Points）。

### 1.4 Noise pose `p(t)` · `[B, 9]` + 时间 `t`

扩散过程中的「当前糊位姿」与噪声水平：

- 位姿默认 `rot_matrix`：**9 维 = Rx(3) + Ry(3) + T(3)**（连续 6D 旋转 + 平移）
- 训练：真值加噪得到 `p(t)`
- 推理：从大噪声起步（或 tracking 时用上一帧初值）

---

## 2. Observation Encoder（`GFObjectPose`）

对应代码：`networks/posenet.py` → `extract_pts_feature`。

### 2.1 DINOv2 ViT-S/14（绿）

- **冻结**（`requires_grad=False`），`dino_dim=384`
- `pointwise`：按点取 patch 特征，拼到 xyz 上再进 PointNet++
- `global`：可另出全局 `rgb_feat` 再拼进 Score 头

### 2.2 PointNet++ MSG Fus（蓝）

几何（+ DINO 贴附特征）→ 集合抽象。

### 2.3 `pts_feat` · `[B, 1024]`（黄）

观测条件的全局向量。之后：

- 进 Score / Energy 的 Concat
- **虚线复用**到 Energy、Scale（图中绿色虚线）

**读图要点**：外观若只走 `pointwise`，会先被绑在点上再经 PointNet++；深度很差时，这一段就是瓶颈所在。

---

## 3. PoseScoreNet Φθ（中间大框）

对应：`networks/gf_algorithms/scorenet.py` + `samplers.py`。

### 3.1 条件编码

| 模块 | 作用 |
|------|------|
| `t_encoder` | Gaussian Fourier 把时间 `t` → 128 维 |
| `pose_encoder` | 当前噪声位姿 9 → 256 → 256 |
| **Concat** | `pts_feat ⊕ t_feat ⊕ pose_feat`（global 时再加 `rgb_feat`） |

### 3.2 回归头 `Rx_Ry_and_T`（蓝）

- RotHead **Rx** → 3（旋转矩阵第一列）
- RotHead **Ry** → 3（第二列）
- TransHead **T** → 3（平移）

合成原始输出 `fθ ∈ R⁹`。

### 3.3 score 输出（橙）

```text
score = fθ / σ(t)     形状 [B, 9]
```

含义：位姿空间里「往哪走更像真值」的坡度（分数场），不是分类置信度。

### 3.4 ODE / PC Sampler + 红色虚线环

```text
p ← p + score · dt     （示意；实际为 PF-ODE / PC）
循环直到 t → 0
```

- 红虚线 **update p(t)**：采样器把更新后的位姿送回 `pose_encoder`，形成去噪环
- 对每个实例独立重复 **K 次**（默认 K=50，不同随机种子）→ 候选集
- 结束时平移会 **加回 `pts_center`**，回到相机坐标系

**读图要点**：Score 框右侧标注 `ODE / PC · ×K steps`——一次前向只出一步 score；完整候选靠外环多次采样。

---

## 4. PoseEnergyNet Ψφ（右中框）

结构与 Score **同构**，训练目标不同（能量 / 排序）。图注「同构异训」。

### 4.1 数据流

1. **Candidates** `[B, K, 9]`：来自 Score ODE×K  
2. **Energy** `Ψ = ⟨p, sθ⟩`（默认 IP 模式，旋转/平移可分开）  
3. **Rank & Cut**：按能量排序，`retain_ratio=0.4` 丢掉后 60%  
4. **Aggregate**：四元数平均 + 平移平均（可选 DBSCAN 聚主峰）

本仓库约定：**能量越高越优先保留**（排序方向以代码为准）。

### 4.2 绿色虚线 `pts_feat`

Energy 也要吃同一套观测特征，故从 Encoder 旁路接入。

---

## 5. ScaleNet + 最终输出（最右列）

### 5.1 ScaleNet

1. 输入聚合后的旋转 **axes `R̂`**，`encode_axes` → 256  
2. 与复用的 **`pts_feat`** 融合（Fusion MLP）  
3. 输出 **`size_3d [l, w, h]`**（约 5 MB；料盘尺寸固定时可写死并跳过）

### 5.2 Final 6D Pose（粉）

```text
R ∈ SO(3),  T ∈ R³   （含 pts_center 回加后的相机系平移）
+ size_3d（若启用）
```

---

## 6. 整图信号流（文字版）

```text
ROI RGB ──► DINOv2 ──┐
                     ├─► PointNet++ ──► pts_feat ──┬──► Score Concat ──► Rx/Ry/T ──► score
实例点云 ────────────┘                             │                      │
Mask/K ─(反投影)─► 点云                             │                      ▼
                                                   │              ODE 环更新 p(t)  ×K
噪声 p(t), t ──► t_enc / pose_enc ─────────────────┘                      │
                                                                          ▼
                                                              Candidates [B,K,9]
                                                                          │
                                                          Energy → Rank → Aggregate
                                                                │              │
                                                                │              ├──► Final 6D
                                                         pts_feat (复用)        │
                                                                │              ▼
                                                                └──► ScaleNet ──► size_3d
```

---

## 7. 图例与底部说明（对应 XML 底部）

| 颜色 | 含义 |
|------|------|
| 粉 | Input / Output |
| 绿 | Encoder / 类注意力编码 |
| 蓝 | MLP / FFN 头 |
| 黄 | Fusion / Concat |
| 橙 | Linear / score 输出 |

线型：

- **实线箭头**：主数据流  
- **红色虚线**：ODE 残差环（更新 `p(t)`）  
- **绿色虚线**：`pts_feat` 旁路复用  

底部紫框摘要：

```text
Train: DSM on Φθ → Energy DSM (IP Ψ=⟨p,Φ⟩) → Scale MSE
Infer: noise → ODE×K → Energy rank/aggregate → Scale（或写死 size_3d）
```

---

## 8. 怎么对照图纸阅读

1. 先跟 **粉块输入 → 黄 `pts_feat`**：建立「观测条件」  
2. 再进 **Score 框**：看 Concat 吃哪些量，Rx/Ry/T 如何出 score  
3. 盯住 **红环**：理解扩散不是单次前向，而是采样器反复调 Φθ  
4. 再看 **Energy**：K 候选如何变成一个位姿  
5. 最后 **Scale / Final**：尺寸与 6D 如何落地  

编辑结构时请改 [`genpose2_model.xml`](./genpose2_model.xml)；本文只做文字锚定，便于评审与培训时口头讲解。
