#!/usr/bin/env python3
"""
多料盘密排 / 遮挡 SOPE 数据合成。

覆盖：
  - 料盘数量 n = 1..10
  - 面间距 gap = 6 / 26 / 46 mm（中心距 = 料盘厚度 + gap）
  - 每种 (n, gap) 组合 50 帧（默认 40 train + 10 test）

几何约定（贴合「竖插料架、多见上沿」）：
  - CAD 中心化后薄方向为局部 Z
  - 先直立（薄轴→料架排列方向 Y），再整体随机/边缘视角变换到相机系
  - 多实例 z-buffer 互遮挡；多数帧用挡板裁掉画面下半，模拟只见上沿

输出与单物体合成相同的 SOPE 目录结构；每帧 mask_id=1..n，共享同一 oid。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

# 复用同目录单物体合成工具
_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_factory.generate_sope_dataset import (  # noqa: E402
    CLASS_LABEL,
    CLASS_NAME,
    CX,
    CY,
    FX,
    FY,
    IMG_HEIGHT,
    IMG_WIDTH,
    OID,
    center_mesh,
    load_mesh,
    make_obj_meta,
    save_exr,
)

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

# 默认组合
TRAY_COUNTS = list(range(1, 11))  # 1..10
GAPS_MM = (6, 26, 46)
FRAMES_PER_COMBO = 50
TRAIN_PER_COMBO = 40
TEST_PER_COMBO = 10


def rot_x(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 → 四元数 (w,x,y,z)。"""
    m = R
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def upright_rotation() -> np.ndarray:
    """
    薄轴原为 +Z（硬币平放）→ 直立后薄轴沿 +Y（料架槽排列方向），
    盘面落在 XZ（竖直）。等价于绕 X 轴 +90°。
    """
    return rot_x(90.0)


def sample_pack_extrinsic(edge_view: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    采样整排料盘相对相机的姿态。
    edge_view=True：略俯视、正对上沿，利于「只见上沿」。
    """
    if edge_view:
        # 俯视 15°~45°，偏航 ±25°，横滚小抖动
        pitch = np.random.uniform(15.0, 45.0)   # 绕 X：让上沿更靠画面上方/更可见
        yaw = np.random.uniform(-25.0, 25.0)
        roll = np.random.uniform(-8.0, 8.0)
        R = rot_z(yaw) @ rot_x(pitch) @ rot_z(roll)
        t = np.array(
            [
                np.random.uniform(-0.05, 0.05),
                np.random.uniform(-0.04, 0.02),
                np.random.uniform(0.55, 0.85),
            ],
            dtype=np.float64,
        )
    else:
        # 更自由的视角（仍以前半球为主）
        pitch = np.random.uniform(-20.0, 50.0)
        yaw = np.random.uniform(-40.0, 40.0)
        roll = np.random.uniform(-15.0, 15.0)
        R = rot_z(yaw) @ rot_x(pitch) @ rot_z(roll)
        t = np.array(
            [
                np.random.uniform(-0.08, 0.08),
                np.random.uniform(-0.06, 0.06),
                np.random.uniform(0.50, 0.95),
            ],
            dtype=np.float64,
        )
    return R, t


def build_instance_poses(
    n_trays: int,
    gap_m: float,
    thickness_m: float,
    R_pack: np.ndarray,
    t_pack: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """返回每片料盘在相机系下的 (R, t)。"""
    R_up = upright_rotation()
    pitch = thickness_m + gap_m
    poses = []
    for i in range(n_trays):
        # 直立后沿 Y 排列；n=1 时 gap 不影响位置
        y = (i - (n_trays - 1) / 2.0) * pitch if n_trays > 1 else 0.0
        t_local = np.array([0.0, y, 0.0], dtype=np.float64)
        R_i = R_pack @ R_up
        t_i = R_pack @ t_local + t_pack
        poses.append((R_i, t_i))
    return poses


def render_multitray(
    mesh,
    poses: List[Tuple[np.ndarray, np.ndarray]],
    n_samples_per_tray: int,
    baffle_ratio: float | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    多实例点采样 z-buffer。
    baffle_ratio: 若为 0~1，清空图像下方该比例区域（模拟挡板挡住下半）。
    """
    import trimesh

    H, W = IMG_HEIGHT, IMG_WIDTH
    depth = np.full((H, W), np.inf, dtype=np.float32)
    mask = np.zeros((H, W), dtype=np.uint8)
    color = np.full((H, W, 3), 64, dtype=np.uint8)

    base_color = np.array([225.0, 225.0, 225.0], dtype=np.float32)
    light_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # 模板表面采样一次，各实例共用
    pts_local, face_idx_tmpl = trimesh.sample.sample_surface(mesh, n_samples_per_tray)
    normals_local = mesh.face_normals[face_idx_tmpl]

    for inst_id, (R, t) in enumerate(poses, start=1):
        pts_cam = pts_local @ R.T + t
        z = pts_cam[:, 2].astype(np.float32)
        z_safe = np.maximum(z, 1e-6)
        u = (FX * pts_cam[:, 0] / z_safe + CX).astype(np.int32)
        v = (FY * pts_cam[:, 1] / z_safe + CY).astype(np.int32)
        valid = (z > 0.0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not np.any(valid):
            continue
        u, v, z = u[valid], v[valid], z[valid]
        nrm = normals_local[valid] @ R.T

        d_inst = np.full((H, W), np.inf, dtype=np.float32)
        np.minimum.at(d_inst, (v, u), z)
        better = d_inst < depth
        depth[better] = d_inst[better]
        mask[better] = np.uint8(inst_id)

        # 着色：仅本实例贡献且最终胜出的像素
        front = np.isclose(z, depth[v, u], rtol=0.0, atol=1e-5)
        if np.any(front):
            ndotl = np.maximum(np.sum(nrm[front] * light_dir, axis=1), 0.0)
            shading = 0.35 + 0.65 * ndotl
            obj_color = np.clip(base_color[None, :] * shading[:, None], 0, 255).astype(
                np.uint8
            )
            # 只写当前仍属于本实例的像素
            uu, vv = u[front], v[front]
            own = mask[vv, uu] == inst_id
            color[vv[own], uu[own]] = obj_color[own]

    valid_px = np.isfinite(depth)
    depth[~valid_px] = 0.0

    if baffle_ratio is not None and 0.0 < baffle_ratio < 1.0:
        v_cut = int(H * (1.0 - baffle_ratio))  # 保留上方 (1-ratio)，清空下方
        depth[v_cut:, :] = 0.0
        mask[v_cut:, :] = 0
        color[v_cut:, :] = 64

    return color, depth, mask


def make_multitray_meta(
    poses: List[Tuple[np.ndarray, np.ndarray]],
    bbox_side_len: List[float],
    *,
    n_trays: int,
    gap_mm: int,
    edge_view: bool,
    baffle_ratio: float | None,
) -> Dict:
    objects = {}
    for inst_id, (R, t) in enumerate(poses, start=1):
        q = mat_to_quat_wxyz(R).tolist()
        tt = t.tolist()
        objects[str(inst_id)] = {
            "meta": {
                "oid": OID,
                "class_name": CLASS_NAME,
                "class_label": CLASS_LABEL,
                "instance_path": "",
                "scale": [1.0, 1.0, 1.0],
                "is_background": False,
                "bbox_side_len": bbox_side_len,
            },
            "quaternion_wxyz": q,
            "translation": tt,
            "is_valid": True,
            "id": inst_id,
            "material": [],
            "world_quaternion_wxyz": q,
            "world_translation": tt,
        }

    return {
        "camera": {
            "intrinsics": {
                "fx": FX,
                "fy": FY,
                "cx": CX,
                "cy": CY,
                "width": IMG_WIDTH,
                "height": IMG_HEIGHT,
            },
            "quaternion": [1.0, 0.0, 0.0, 0.0],
            "translation": [0.0, 0.0, 0.0],
            "scene_obj_path": "",
            "background_image_path": "",
            "background_depth_path": "",
            "distances": [],
            "kind": "multitray_shelf",
        },
        "objects": objects,
        "scene_dataset": "Omni6DPose",
        "env_param": {
            "n_trays": n_trays,
            "gap_mm": gap_mm,
            "edge_view": edge_view,
            "baffle_ratio": baffle_ratio,
        },
        "face_up": True,
        "concentrated": True,
        "comments": (
            f"multitray n={n_trays} gap_mm={gap_mm} "
            f"edge_view={edge_view} baffle={baffle_ratio}"
        ),
        "runtime_seed": 0,
        "baseline_dis": 0,
        "emitter_dist_l": 0,
    }


def write_frame(
    out_dir: Path,
    frame_id: int,
    color: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    meta: Dict,
) -> None:
    prefix = out_dir / f"{frame_id:06d}"
    Image.fromarray(color).save(str(prefix) + "_color.png")
    save_exr(Path(str(prefix) + "_depth.exr"), depth, is_mask=False)
    save_exr(Path(str(prefix) + "_mask.exr"), mask, is_mask=True)
    Path(str(prefix) + "_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def generate_combos(
    mesh,
    bbox_side_len: List[float],
    thickness_m: float,
    training_dir: Path,
    *,
    n_samples_per_tray: int,
    seed: int,
) -> Dict:
    np.random.seed(seed)
    train_dir = training_dir / "SOPE" / "000000" / "train" / "Omni6DPose" / "000000"
    test_dir = training_dir / "SOPE" / "000000" / "test" / "Omni6DPose" / "000000"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # 清空旧帧，避免与上一版单物体数据混杂
    for d in (train_dir, test_dir):
        for p in d.glob("*"):
            p.unlink()

    manifest = []
    train_id = 0
    test_id = 0
    total = len(TRAY_COUNTS) * len(GAPS_MM) * FRAMES_PER_COMBO
    done = 0

    print(
        f"[plan] counts={TRAY_COUNTS} gaps_mm={GAPS_MM} "
        f"per_combo={FRAMES_PER_COMBO} (train {TRAIN_PER_COMBO}/test {TEST_PER_COMBO}) "
        f"total={total}"
    )
    print(f"[mesh] thickness={thickness_m*1000:.2f}mm  bbox={bbox_side_len}")

    for n_trays in TRAY_COUNTS:
        for gap_mm in GAPS_MM:
            gap_m = gap_mm / 1000.0
            for k in range(FRAMES_PER_COMBO):
                edge_view = np.random.rand() < 0.75
                # 70% 帧加挡板，裁掉下半 35%~55%
                if np.random.rand() < 0.70:
                    baffle = float(np.random.uniform(0.35, 0.55))
                else:
                    baffle = None

                R_pack, t_pack = sample_pack_extrinsic(edge_view)
                poses = build_instance_poses(
                    n_trays, gap_m, thickness_m, R_pack, t_pack
                )
                color, depth, mask = render_multitray(
                    mesh, poses, n_samples_per_tray, baffle
                )

                # 若完全看不见则重采样几次
                retries = 0
                while int((mask > 0).sum()) < 80 and retries < 8:
                    R_pack, t_pack = sample_pack_extrinsic(True)
                    poses = build_instance_poses(
                        n_trays, gap_m, thickness_m, R_pack, t_pack
                    )
                    color, depth, mask = render_multitray(
                        mesh, poses, n_samples_per_tray, baffle
                    )
                    retries += 1

                meta = make_multitray_meta(
                    poses,
                    bbox_side_len,
                    n_trays=n_trays,
                    gap_mm=gap_mm,
                    edge_view=edge_view,
                    baffle_ratio=baffle,
                )

                is_train = k < TRAIN_PER_COMBO
                if is_train:
                    write_frame(train_dir, train_id, color, depth, mask, meta)
                    fid = train_id
                    split = "train"
                    train_id += 1
                else:
                    write_frame(test_dir, test_id, color, depth, mask, meta)
                    fid = test_id
                    split = "test"
                    test_id += 1

                manifest.append(
                    {
                        "split": split,
                        "frame_id": f"{fid:06d}",
                        "n_trays": n_trays,
                        "gap_mm": gap_mm,
                        "edge_view": edge_view,
                        "baffle_ratio": baffle,
                        "mask_pixels": int((mask > 0).sum()),
                        "visible_ids": sorted(
                            {int(x) for x in np.unique(mask) if int(x) > 0}
                        ),
                    }
                )
                done += 1
                if done % 50 == 0 or done == total:
                    print(f"  [{done}/{total}] last=n{n_trays}_gap{gap_mm}mm → {split}/{fid:06d}")

    return {
        "train_frames": train_id,
        "test_frames": test_id,
        "train_dir": str(train_dir),
        "test_dir": str(test_dir),
        "items": manifest,
    }


def main():
    parser = argparse.ArgumentParser(description="多料盘密排遮挡 SOPE 合成")
    parser.add_argument(
        "--training_dir",
        default=str(_ROOT / "datasets" / "train_set_1st_0810"),
    )
    parser.add_argument("--cad_dir", default=str(_THIS))
    parser.add_argument(
        "--n_samples_per_tray",
        type=int,
        default=40000,
        help="每片料盘表面采样点数",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    training_dir = Path(args.training_dir).resolve()
    training_dir.mkdir(parents=True, exist_ok=True)

    mesh = center_mesh(load_mesh(args.cad_dir))
    bbox = (mesh.bounds[1] - mesh.bounds[0]).tolist()
    # 薄方向：最短边
    extents = mesh.bounds[1] - mesh.bounds[0]
    thickness_m = float(np.min(extents))
    print(f"[mesh] verts={len(mesh.vertices)} faces={len(mesh.faces)} extents={extents.tolist()}")

    summary = generate_combos(
        mesh,
        bbox,
        thickness_m,
        training_dir,
        n_samples_per_tray=args.n_samples_per_tray,
        seed=args.seed,
    )

    obj_meta = make_obj_meta(bbox)
    meta_dir = training_dir / "Meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "obj_meta.json").write_text(
        json.dumps(obj_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    # 同步到训练配置
    cfg_meta = _ROOT / "configs" / "obj_meta.json"
    cfg_meta.parent.mkdir(parents=True, exist_ok=True)
    cfg_meta.write_text(
        json.dumps(obj_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    out_summary = {
        "training_dir": str(training_dir),
        "tray_counts": TRAY_COUNTS,
        "gaps_mm": list(GAPS_MM),
        "frames_per_combo": FRAMES_PER_COMBO,
        "train_per_combo": TRAIN_PER_COMBO,
        "test_per_combo": TEST_PER_COMBO,
        "thickness_m": thickness_m,
        "bbox_side_len": bbox,
        "n_samples_per_tray": args.n_samples_per_tray,
        "seed": args.seed,
        **{k: summary[k] for k in ("train_frames", "test_frames", "train_dir", "test_dir")},
        "items": summary["items"],
    }
    (training_dir / "manifest_multitray.json").write_text(
        json.dumps(out_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("多料盘密排数据生成完成")
    print(f"  train: {summary['train_frames']} → {summary['train_dir']}")
    print(f"  test : {summary['test_frames']} → {summary['test_dir']}")
    print(f"  obj_meta → {meta_dir / 'obj_meta.json'}")
    print(f"  configs  → {cfg_meta}")
    print(f"  manifest → {training_dir / 'manifest_multitray.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
