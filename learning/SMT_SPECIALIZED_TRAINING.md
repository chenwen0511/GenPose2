# GenPose2 SMT 专项训练工作区

此目录用于固定型号 SMT 料盘的单类别、多实例 GenPose2 训练。

## 目录

```text
/data/quinn/smt/
  GenPose2/                 独立 git worktree
  datasets/                 训练数据集入口
  scripts/                  预检和训练脚本
  logs/                     nohup/tmux 日志
  GenPose2/results/ckpts/   checkpoint
  GenPose2/results/logs/    TensorBoard 日志
  validation/               数据和模型验证报告
```

## 默认训练方案

- DINOv2 保持冻结；
- 微调 PointNet2 和 ScoreNet；
- ScoreNet 完成后训练 EnergyNet；
- 固定型号料盘不训练 ScaleNet，部署时返回 `obj_meta.json` 的固定尺寸。

## 使用顺序

1. 将整理后的 SOPE 数据放到或软链接到：

   `/data/quinn/smt/datasets/genpose2_smt_train_v1`

2. 将该数据集的标准 `obj_meta.json` 放到：

   `/data/quinn/smt/GenPose2/configs/obj_meta.json`

3. 运行预检：

   `python /data/quinn/smt/scripts/preflight_dataset.py /data/quinn/smt/datasets/genpose2_smt_train_v1`

4. 先做 loader smoke test，再启动 ScoreNet。长训练必须使用 tmux + nohup；当前 GPU 有其他服务，启动前要求至少 14 GiB 空闲显存。

5. ScoreNet 选出最终 checkpoint 后设置 `SCORE_CKPT`，再启动 EnergyNet。

训练准备完成后使用标准持久化启动器。4090-3 当前未安装 `tmux`，正式启动前需要先经用户确认安装；启动脚本会在缺少 `tmux` 时拒绝运行。

```bash
bash /data/quinn/smt/scripts/launch.sh score
bash /data/quinn/smt/scripts/status.sh score

SCORE_CKPT=/data/quinn/smt/GenPose2/results/ckpts/ScoreNet_smt_v1/ckpt_epochN.pth \
  bash /data/quinn/smt/scripts/launch.sh energy
```

不要在 GPU 剩余显存低于 14 GiB 时绕过门禁启动训练。

## 重要限制

- 数据必须只有一个统一类别和一个统一尺寸定义；
- 使用 `--load_per_object` 展开每张图的全部可见实例；不要同时传 `--per_obj`；
- val/test 必须是独立目录和独立场景组；
- 当前只准备代码，不代表训练已开始。
