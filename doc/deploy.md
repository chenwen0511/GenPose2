# GenPose2 部署指南（HTTP 推理服务）

本文说明如何在 Linux + NVIDIA GPU 机器上部署 `http_server.py`，提供 `POST /infer` 6D 位姿推理接口。

**适用场景**：已拿到代码仓库；权重从企业微盘下载（不依赖公网 Dropbox）。

---

## 1. 环境要求

| 项目 | 建议 |
|------|------|
| 系统 | Ubuntu 20.04 / 22.04 |
| GPU | NVIDIA，显存 ≥ 8 GB（推荐 12 GB+） |
| CUDA | 11.8（与 PyTorch 2.1 cu118 匹配） |
| Python | 3.10.x |
| 磁盘 | 代码 ~2 GB + 权重 ~250 MB + 运行缓存 |

---

## 2. 获取代码

```bash
# 示例：按你们实际 Git 地址克隆
git clone <你们的 GenPose2 仓库地址> GenPose2
cd GenPose2
```

仓库根目录下文称 `$ROOT`（后文命令均在此目录执行）。

---

## 3. 下载并放置权重（企业微盘）

权重 **不在公网**，请从 **企业微盘** 下载：

| 微盘文件名 | 说明 |
|------------|------|
| `genpose2_weights_results.zip` | `results/` 目录下全部推理权重 |

### 3.1 解压到仓库

```bash
cd $ROOT

# 将 zip 放到当前目录或指定路径后解压
unzip /path/to/genpose2_weights_results.zip -d .

# 或先解压到临时目录再移动
# unzip /path/to/genpose2_weights_results.zip -d /tmp/genpose2_weights
# cp -r /tmp/genpose2_weights/results ./
```

### 3.2 解压后目录结构（必须一致）

```text
GenPose2/
└── results/
    └── ckpts/
        ├── ScoreNet/
        │   └── scorenet.pth      # 约 114 MB
        ├── EnergyNet/
        │   └── energynet.pth     # 约 114 MB
        └── ScaleNet/
            └── scalenet.pth      # 约 5 MB
```

合计约 **233 MB**。

### 3.3 校验权重是否就位

```bash
ls -lh results/ckpts/ScoreNet/scorenet.pth
ls -lh results/ckpts/EnergyNet/energynet.pth
ls -lh results/ckpts/ScaleNet/scalenet.pth
```

三个文件均存在且大小非 0 即可。

> **说明**：GenPose2 推理 **不需要 CAD**；上述三个 `.pth` 为 HTTP 服务启动时必载权重。  
> 若使用默认 **YOLO 分割**，还需 `segment/yolo_seg.pt`（约 6.4 MB，一般在代码仓内；若缺失需另行获取）。

---

## 4. 创建 Python 环境

```bash
conda create -n genpose2 python=3.10.14 -y
conda activate genpose2
```

### 4.1 安装 PyTorch（CUDA 11.8）

```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu118
```

### 4.2 安装项目依赖

```bash
cd $ROOT
pip install -r requirements.txt --upgrade-strategy only-if-needed
```

### 4.3 编译 pointnet2

```bash
cd networks/pts_encoder/pointnet2_utils/pointnet2
python setup.py install
cd $ROOT
```

### 4.4 安装 cutoop（读 depth.exr / mask.exr）

```bash
sudo apt-get update
sudo apt-get install -y libopenexr-dev
pip install cutoop
```

### 4.5 验证 GPU

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

应输出 `cuda: True` 及 GPU 名称。

---

## 5. 配置环境变量（可选）

默认会读取 `results/ckpts/` 下路径，一般 **无需设置**。需要覆盖时使用：

```bash
export GENPOSE2_SCORE_CKPT=$ROOT/results/ckpts/ScoreNet/scorenet.pth
export GENPOSE2_ENERGY_CKPT=$ROOT/results/ckpts/EnergyNet/energynet.pth
export GENPOSE2_SCALE_CKPT=$ROOT/results/ckpts/ScaleNet/scalenet.pth

# 推理结果输出目录（默认 service_outputs/）
export GENPOSE2_OUTPUT_ROOT=$ROOT/service_outputs

# 分割后端：yolo（默认）或 sam3
export GENPOSE2_SEG_BACKEND=yolo
```

---

## 6. 启动 HTTP 服务

### 6.1 前台启动（调试）

```bash
cd $ROOT
conda activate genpose2

python http_server.py --host 0.0.0.0 --port 8002
```

成功日志应包含：

```text
[http_server] GenPose2 models loaded
Application startup complete
```

### 6.2 后台启动

```bash
cd $ROOT
conda activate genpose2
mkdir -p logs

nohup python http_server.py --host 0.0.0.0 --port 8002 > logs/http_server.log 2>&1 &
tail -f logs/http_server.log
```

### 6.3 Gradio UI（SAM3 分割 / SAM3+GenPose2 / 缺货）

需先启动外部 SAM3 HTTP 服务（默认 `http://127.0.0.1:18003/infer`，见 `config/conf.json`）。

```bash
cd $ROOT
conda activate genpose2
bash start.sh start    # 默认 http://0.0.0.0:18090/
bash start.sh status
bash start.sh stop
# 日志: logs/ui.log
```

页签：**SAM3 分割**；**SAM3 + GenPose2**（含抓取位姿 `xyzrxryrz` mm/° + 目标正方体 / `grasp_pose.json`）；**缺货商品位姿估计**（M3 识别缺货 → qwen 生成 SAM3 提示词 → SAM3 → GenPose2 → M3+空间先验估计目的位姿）。`conf.json` → `vlm.sam3_prompt` / `vlm.reason` / `vlm.missing_prompt`；MiniMax Key 放 `ANTHROPIC_API_KEY` 或 `config/secrets.local.json`（勿提交仓库）。默认开启 Depth→RGB 对齐。启动时在主线程预加载三网权重。
### 6.4 systemd（生产环境，可选）

创建 `/etc/systemd/system/genpose2.service`（按实际用户与路径修改）：

```ini
[Unit]
Description=GenPose2 HTTP Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/path/to/GenPose2
Environment=PATH=/path/to/miniconda3/envs/genpose2/bin:/usr/bin
ExecStart=/path/to/miniconda3/envs/genpose2/bin/python http_server.py --host 0.0.0.0 --port 8002
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable genpose2
sudo systemctl start genpose2
sudo systemctl status genpose2
```

---

## 7. 部署验证

### 7.1 健康检查

```bash
curl -s http://127.0.0.1:8002/health | python -m json.tool
```

关注字段：

| 字段 | 期望 |
|------|------|
| `status` | `"ok"` |
| `genpose_loaded` | `true` |
| `seg_backend` | `"yolo"` 或 `"sam3"` |

若 `genpose_loaded` 为 `false`，检查权重路径与启动日志。

### 7.2 推理冒烟测试

准备一帧 `rgb.png`、`depth.png`、`camera.json`；若已有 mask 可跳过分割：

```bash
curl -X POST "http://127.0.0.1:8002/infer" \
  -F "rgb=@/path/to/rgb.png" \
  -F "depth=@/path/to/depth.png" \
  -F "camera=@/path/to/camera.json" \
  -F "mask=@/path/to/mask.png" \
  --max-time 600 \
  | python -m json.tool
```

成功响应应包含 `xyz_mm`、`detections_pem`、`depth_colormap_path` 等字段。

接口字段说明见仓库内 API 文档（若已维护 `doc/API.md`）。

---

## 8. 两种推理模式

### 模式 A：自带 mask（推荐，跳过分割）

上传 `rgb` + `depth` + `camera` + **`mask`**，不跑 YOLO/SAM3。

- mask 须与 rgb/depth 同分辨率、对齐  
- mask 内有效深度占比建议 **> 50%**（见 `learning/如何补救.md`）

### 模式 B：自动分割（YOLO 默认）

只上传 `rgb` + `depth` + `camera`，服务内 YOLO 生成 mask 再推理。

- 需 `segment/yolo_seg.pt`  
- 需能检出目标类别

### 模式 C：SAM3 文本分割（可选）

```bash
export GENPOSE2_SEG_BACKEND=sam3
export SAM6D_SAM3_ROOT=/path/to/sam3
export SAM6D_SAM3_PROMPT="your object"

python http_server.py --host 0.0.0.0 --port 8002 --seg-backend sam3
```

需单独部署 SAM3 环境与脚本，复杂度较高，一般优先用 **自带 mask** 或 **YOLO**。

---

## 9. 常见问题

| 现象 | 处理 |
|------|------|
| `GenPose2 model not loaded` | 检查 `results/ckpts/` 三个 `.pth` 是否存在；看 `logs/http_server.log` 加载报错 |
| `No module named 'cutoop'` | `pip install cutoop`，并安装 `libopenexr-dev` |
| pointnet2 编译失败 | 确认 CUDA、gcc 与 PyTorch 版本匹配；在 pointnet2 目录重新 `python setup.py install` |
| `YOLO returned no segmentation masks` | 未上传 mask 且 YOLO 未检出目标；改上传 mask 或换图/调 YOLO |
| 位姿明显不准 | 多为 **mask 内 depth 大量为 0**；看返回的 `depth_colormap_path`，目标框内应有彩色有效深度 |
| 首次推理很慢 | DINO 等可能从 PyTorch Hub 拉预训练；内网需提前缓存或配置 `TORCH_HOME` |
| 端口被占用 | 换 `--port 8003` 或 `lsof -i :8002` 查占用 |

---

## 10. 部署检查清单

- [ ] 代码已克隆到目标机  
- [ ] 企业微盘 `genpose2_weights_results.zip` 已解压到 `results/`  
- [ ] 三个 `.pth` 校验通过（约 233 MB）  
- [ ] conda 环境 `genpose2` 已创建  
- [ ] PyTorch CUDA 可用  
- [ ] `requirements.txt` 已安装  
- [ ] pointnet2、cutoop 已安装  
- [ ] `python http_server.py` 启动且 `genpose_loaded=true`  
- [ ] `/health` 与一次 `/infer` 冒烟通过  
- [ ] （可选）`bash start.sh start` 后 Gradio UI `18090` 可访问；日志在 `logs/`  
- [ ] 防火墙已放行 `8002`（若远程访问；UI 为 `18090`）

---

## 11. 相关文档

| 文档 | 内容 |
|------|------|
| `README.md` | 项目总览、Gradio UI、训练环境说明 |
| `config/conf.json` | UI / SAM3 API / 双 VLM profile（`sam3_prompt`+`reason`）/ GenPose2 权重默认配置 |
| `learning/infer.md` | 离线推理与 HTTP 说明 |
| `learning/如何补救.md` | depth / mask 质量与补救 |
| `learning/案例分析1.md` | 深度丢失案例 |
