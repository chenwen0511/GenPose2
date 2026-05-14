# GenPose2 推理调用说明

本文说明仓库内**离线文件推理**与**相机实时推理**两条路径，以及如何把你已有的 `learning/inputs/`、`segment/`、`results/ckpts/` 串起来。

---

## 1. 推理在代码里怎么走

核心类在 `runners/infer.py` 与 `runners/infer_camera.py` 中（逻辑基本一致），对外用法可以概括为：

1. **加载三个权重**：Score（扩散采样）、Energy（能量排序）、Scale（尺度/轴向细化）。
2. **构造 `InferDataset`**：内部需要同分辨率的 **RGB、深度、实例/语义 mask、相机内参 meta**。
3. **调用 `GenPose2.inference(data, ...)`**：
   - Score 网络多次采样得到候选位姿；
   - Energy 网络对候选打分并聚合（四元数平均 + 可选 DBSCAN）；
   - Scale 网络在点云与特征上细化旋转与 3D 尺寸；
4. **可视化**：`visualize_pose(...)` 用 `cutoop` 的 `DetectMatch._draw_image` 在 RGB 上画坐标系与包围盒。

创建模型：

```python
from runners.infer import create_genpose2, visualize_pose
from datasets.datasets_infer import InferDataset

GenPose2 = create_genpose2(
    score_model_path="results/ckpts/ScoreNet/scorenet.pth",
    energy_model_path="results/ckpts/EnergyNet/energynet.pth",
    scale_model_path="results/ckpts/ScaleNet/scalenet.pth",
)
```

单帧（与 `runners/infer.py` 里 `main()` 一致）：用**统一前缀**加载四件套，再推理：

```python
prefix = "learning/inputs/frame000_"  # 对应 frame000_color.png 等
data = InferDataset.alternetive_init(
    prefix,
    img_size=GenPose2.cfg.img_size,
    device=GenPose2.cfg.device,
    n_pts=GenPose2.cfg.num_points,
)
pose, length = GenPose2.inference(data, prev_pose=None, tracking=False)
img = visualize_pose(data, pose, length, visualize_image=False)
```

`InferDataset.alternetive_init` 会读取（见 `datasets/datasets_infer.py`）：

| 文件 | 说明 |
|------|------|
| `{prefix}color.png` | RGB，与深度、mask 同高宽 |
| `{prefix}depth.exr` | 深度图（由 `cutoop.data_loader.Dataset.load_depth` 读取） |
| `{prefix}mask.exr` | 与 Omni6D 数据类似的 mask（由 `Dataset.load_mask` 读取） |
| `{prefix}meta.json` | 内参；若为 dict，需含 `camera.intrinsics`（见下节） |

`runners/infer.py` 的 `main()` 用 `glob(DATA_PATH + '/*_color.png')`，对每张图把路径里的 `color.png` 换成前缀，再 `alternetive_init`，因此**命名必须能拼出上述四个文件名**。

---

## 2. `meta.json` 与 `learning/inputs/camera.json` 的关系

推理用的 **不是** 当前仓库里那份简化 `camera.json`（只有 `cam_K` 与 `depth_scale`）的格式，而是 `InferDataset` 在代码里解析的嵌套结构（与 `datasets_infer.py` / `datasets_infer_camera.py` 中一致）：

```json
{
  "camera": {
    "intrinsics": {
      "fx": 393.1242370605469,
      "fy": 392.81097412109375,
      "cx": 322.4469909667969,
      "cy": 242.9351806640625,
      "width": 640,
      "height": 480
    }
  }
}
```

请根据你的 RGB 实际分辨率填写 `width` / `height`（与 `color.png` 一致）。`cam_K` 若为 3×3 行主序展平，通常有 `fx=K[0], fy=K[4], cx=K[2], cy=K[5]`。

**深度单位**：`datasets_infer` 里会把过大的深度置 0（如 `> 4.0` 视为无效）。深度需与标定一致（常见为 **米**；若相机是毫米，需先乘 `depth_scale` 转成米再写入 `depth.exr`，或与训练数据一致）。

---

## 3. 前置分割：`segment/` 与 GenPose2 输入的衔接

`segment/yolo_seg_backend.py` 中的 `run_yolo_segmentation(...)` 会：

- 用 Ultralytics YOLO **分割**权重对 RGB 推理；
- 在 `output_dir` 下写出 **`{rgb 文件名}_mask.exr`**（与 `cutoop.Dataset.load_mask` 约定一致，供 `InferDataset.alternetive_init` 使用）。

**GenPose2 的 `InferDataset` 只读同前缀的 `mask.exr`。** Mask 中像素值为 **物体 id**：单物体为 `1`；多实例为 `1,2,3,...`（`255` 会被忽略）。

---

## 4. 两条运行入口（仓库已有）

| 脚本 | 数据从哪来 | 说明 |
|------|------------|------|
| `python runners/infer.py` | 磁盘：`DATA_PATH/*_color.png` + 同前缀 `depth.exr` / `mask.exr` / `meta.json` | 纯文件；`InferDataset` 来自 `datasets_infer.py` |
| `python runners/infer_camera.py` | `USE_CAM=True`：RealSense + SAM2 点选 mask；`USE_CAM=False`：与上表相同的文件序列 | 依赖 `pyrealsense2`、相机序列号、`segment-anything-2-real-time` 等（见该文件顶部参数） |

权重路径在示例里为（可按你实际文件名调整）：

- `results/ckpts/ScoreNet/scorenet.pth`
- `results/ckpts/EnergyNet/energynet.pth`
- `results/ckpts/ScaleNet/scalenet.pth`

---

## 5. 已有 ckpt + segment + `learning/inputs/` 时，还缺什么？

对照下面清单即可。

**必备（否则 `alternetive_init` 无法跑通）**

1. **按前缀命名的四文件**（可放在 `learning/inputs/` 或任意目录，并把 `DATA_PATH` / `prefix` 指过去）  
   - `*_color.png`  
   - 同前缀 `*_depth.exr`  
   - 同前缀 `*_mask.exr`  
   - 同前缀 `*_meta.json`（嵌套 `camera.intrinsics`，不能只有平铺的 `cam_K`）

2. **深度 + RGB + mask 空间对齐、同分辨率**；深度在物体区域有足够有效值（否则会在 `get_per_object` 中断言失败）。

3. **运行环境**：CUDA、与编译一致的 PyTorch；若走 C++/CUDA 扩展，需已编译 `pointnet2` 等（见项目其它安装说明）。

**你已具备时可略过**

- `results/ckpts/` 下三个网络权重路径正确。  
- `segment/` 下 YOLO `.pt`：用 `run_yolo_segmentation` 可直接得到与推理约定一致的 **`mask.exr`**（再与 `color` / `depth` / `meta` 同前缀放置即可）。

**若走 `infer_camera.py` 的相机分支**

- RealSense 串口、SAM2 权重路径、`pyrealsense2` 等；与纯文件推理无关。

**若只走文件推理**

- 不需要 RealSense / SAM2；把 `runners/infer.py` 里 `DATA_PATH` 改成你的序列目录即可。

---

## 6. 小结

- **调用顺序**：准备 `color.png` + `depth.exr` + `mask.exr` + `meta.json` → `InferDataset.alternetive_init` → `create_genpose2(...).inference` → `visualize_pose`。  
- **`learning/inputs/camera.json`** 不能直接替代 **`meta.json`**，需按第 2 节展开为 `camera.intrinsics`。  
- **`segment/`**：`run_yolo_segmentation` 写出 **`mask.exr`**；需与 RGB/深度同目录同前缀，并保证对齐。

把上述四件套凑齐并确认深度单位与内参分辨率一致后，即可用 `runners/infer.py` 做离线推理。

---

## 7. 单张推理：`runners/smt_infer.py`（分割 + GenPose2 一键）

仓库提供 **`runners/smt_infer.py`**：先调用 `segment/yolo_seg_backend.py` 写 `mask.exr`，再与 `runners/infer.py` 相同管线推理。命令行参数名使用 **`--score-ckpt` / `--energy-ckpt` / `--scale-ckpt`**（勿写 `--scale`，会与 `configs/config.py` 里 `--scale_embedding` 等缩写冲突）；导入前会暂存 `sys.argv`，避免 `get_config()` 误解析本脚本参数。

示例：

```bash
python runners/smt_infer.py \
  --rgb learning/inputs/1_.png \
  --depth learning/inputs/1_depth.png \
  --meta learning/inputs/1_meta.json \
  --yolo-weights segment/yolo_seg.pt \
  --score-ckpt results/ckpts/ScoreNet/scorenet.pth \
  --energy-ckpt results/ckpts/EnergyNet/energynet.pth \
  --scale-ckpt results/ckpts/ScaleNet/scalenet.pth \
  --save-vis learning/outputs/1_pose_vis.png
```

**只关心「第一个」物体（单目标）时**：保持默认 **`--yolo-max-instances 1`**，且 mask 中前景为实例 id **`1`** 即可；此时推理与可视化对应**单个**位姿/包围盒。若 mask 里出现多个实例 id（1、2、3…），`InferDataset` 会对每个 id 各估一套位姿，可视化可能叠多个框；单目标场景应避免多 id 或把 YOLO 只保留最高分实例。

---

## 8. 首次运行：DINOv2 预训练权重下载

配置里使用 DINO（如 `cfg.dino = 'pointwise'`）时，**第一次**走推理可能由 PyTorch Hub 拉取 ViT-S/14 预训练，终端会出现类似日志（路径以本机 `$HOME` 为准）：

```text
Downloading: "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth" to /home/ubuntu/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth
```

下载完成后会缓存在上述目录；**离线或内网环境**需事先把同名文件放到该路径，或设置可写的 `TORCH_HOME` / 使用已缓存的机器拷贝整个 `checkpoints` 目录，避免运行时访问外网。

