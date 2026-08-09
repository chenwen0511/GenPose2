"""
SOPE 格式的 GenPose2 训练/测评数据生成脚本。

输入:
    data_factory/smt_tray.PLY  或  data_factory/零件1.STEP

输出:
    ./trainging/SOPE/000000/{train,test}/Omni6DPose/000000/
        - {frame_id:06d}_color.png
        - {frame_id:06d}_depth.exr   (float32, 米)
        - {frame_id:06d}_mask.exr    (uint8, 实例 ID)
        - {frame_id:06d}_meta.json

    ./trainging/Meta/obj_meta.json   (需拷贝到 GenPose2/configs/obj_meta.json)

使用方法:
    python data_factory/generate_sope_dataset.py                 # 默认 600+200
    python data_factory/generate_sope_dataset.py --n_train 100   # 自定义数量
    python data_factory/generate_sope_dataset.py --training_dir /path/to/out

依赖: trimesh, numpy, imageio, pillow (已在 requirements.txt 中)
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image

# ============================ 配置 ============================
# 图像尺寸与相机内参（与 meta.json 对齐）
IMG_WIDTH = 640
IMG_HEIGHT = 480
FX = 600.0
FY = 600.0
CX = IMG_WIDTH / 2.0
CY = IMG_HEIGHT / 2.0

# 物体元数据（在 obj_meta.json 中以 oid 形式出现）
OID = "smt_factory-smt_tray-000001"
CLASS_NAME = "smt_tray"
CLASS_LABEL = 0

# 每帧表面采样点数（越大越密但越慢；80K 已能让 640x480 大部分被覆盖）
N_SURFACE_SAMPLES = 80000

# 随机平移范围（物体在相机前方的工作区域，单位米）
T_XY_RANGE = (-0.08, 0.08)
T_Z_RANGE = (0.55, 0.95)


# ============================ Mesh 加载 ============================

def load_mesh(cad_dir="data_factory"):
    """从 data_factory 加载 CAD 模型（PLY 或 STEP），自动合并 Scene 子几何"""
    import trimesh
    candidates = [
        os.path.join(cad_dir, "smt_tray.PLY"),
        os.path.join(cad_dir, "零件1.STEP"),
    ]
    for path in candidates:
        if os.path.exists(path):
            print(f"[mesh] 加载 CAD: {path}")
            mesh = trimesh.load(path, force='mesh')
            if not isinstance(mesh, trimesh.Trimesh):
                # STEP 等格式可能返回 Scene，需要合并所有子几何
                if hasattr(mesh, 'geometry'):
                    geometries = [g for g in mesh.geometry.values()
                                  if isinstance(g, trimesh.Trimesh)]
                    if not geometries:
                        raise RuntimeError(f"{path} 中没有可用的 Trimesh 几何")
                    mesh = trimesh.util.concatenate(geometries)
                else:
                    raise RuntimeError(f"无法将 {path} 加载为 Trimesh")
            return mesh
    raise FileNotFoundError(
        "未在 data_factory 下找到 CAD 文件 (smt_tray.PLY / 零件1.STEP)"
    )


def center_mesh(mesh):
    """将 mesh 中心化到原点（便于姿态估计时以几何中心为基准）"""
    centroid = np.mean(mesh.vertices, axis=0)
    mesh.vertices = mesh.vertices - centroid
    return mesh


# ============================ 随机姿态 ============================

def random_quaternion_wxyz():
    """在 SO(3) 上均匀采样的四元数 (w, x, y, z 顺序)"""
    u1, u2, u3 = np.random.uniform(0.0, 1.0, 3)
    w = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
    x = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
    y = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
    z = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
    return np.array([w, x, y, z], dtype=np.float64)


def random_translation():
    """随机平移（米）：物体放在相机前方工作区域"""
    return np.array([
        np.random.uniform(*T_XY_RANGE),
        np.random.uniform(*T_XY_RANGE),
        np.random.uniform(*T_Z_RANGE),
    ], dtype=np.float64)


# ============================ 渲染 ============================

def render_frame(mesh, quat_wxyz, translation, n_samples=N_SURFACE_SAMPLES):
    """
    基于表面点采样的 z-buffer 渲染（无需 OpenGL/显卡）。

    返回:
        color: (H, W, 3) uint8 RGB
        depth: (H, W) float32 深度（米），0 表示无效
        mask:  (H, W) uint8 实例 ID
    """
    import trimesh

    H, W = IMG_HEIGHT, IMG_WIDTH

    # 1. 在 mesh 表面均匀采样
    points, face_idx = trimesh.sample.sample_surface(mesh, n_samples)

    # 2. 旋转矩阵 (wxyz 四元数 → 3x3)
    R = trimesh.transformations.quaternion_matrix(quat_wxyz)[:3, :3]

    # 3. 物体坐标系 → 相机坐标系:  P_cam = R @ P_obj + t
    points_cam = points @ R.T + translation

    # 4. 透视投影（针孔模型）
    z = points_cam[:, 2].astype(np.float32)
    z_safe = np.maximum(z, 1e-6)
    u_f = FX * points_cam[:, 0] / z_safe + CX
    v_f = FY * points_cam[:, 1] / z_safe + CY
    u = u_f.astype(np.int32)
    v = v_f.astype(np.int32)

    # 5. 过滤（深度 > 0 且落在图像内）
    valid = (z > 0.0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[valid]
    v = v[valid]
    z = z[valid]
    face_idx = face_idx[valid]

    # 6. Z-buffer: 每像素保留最小深度 (向量化)
    depth = np.zeros((H, W), dtype=np.float32)
    mask = np.zeros((H, W), dtype=np.uint8)
    np.minimum.at(depth, (v, u), z)
    mask[v, u] = 1  # 实例 ID = 1

    # 7. RGB: 基于法线的 Lambertian 着色（白色塑料外观）
    base_color = np.array([225.0, 225.0, 225.0], dtype=np.float32)  # 白色
    light_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)         # 相机正前方打光

    normals = mesh.face_normals[face_idx]
    normals_cam = normals @ R.T
    ndotl = np.maximum(np.sum(normals_cam * light_dir, axis=1), 0.0)
    shading = 0.35 + 0.65 * ndotl  # 环境光 + 漫反射

    color = np.full((H, W, 3), 64, dtype=np.uint8)            # 深灰色背景
    obj_color = np.clip(base_color[None, :] * shading[:, None], 0, 255).astype(np.uint8)
    color[v, u] = obj_color

    return color, depth, mask


# ============================ 文件保存 ============================

def save_exr(path, array):
    """保存单通道 EXR（优先 imageio，回退 OpenCV）"""
    try:
        import imageio
        imageio.imwrite(path, array)
        return
    except ImportError:
        pass
    try:
        import cv2
        cv2.imwrite(path, array)
    except Exception as e:
        raise RuntimeError(f"保存 EXR 失败 {path}: {e} (需安装 imageio 或 opencv-python)")


def make_meta(quat_wxyz, translation, bbox_side_len):
    """生成 meta.json 内容"""
    return {
        "camera": {
            "intrinsics": {
                "fx": FX, "fy": FY, "cx": CX, "cy": CY,
                "width": IMG_WIDTH, "height": IMG_HEIGHT,
            },
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "translation": [0.0, 0.0, 0.0],
            "scene_obj_path": "",
            "background_image_path": "",
            "background_depth_path": "",
            "distances": [],
            "kind": "",
        },
        "objects": [
            {
                "mask_id": 1,
                "meta": {
                    "oid": OID,
                    "class_name": CLASS_NAME,
                    "class_label": CLASS_LABEL,
                    "instance_path": "",
                    "scale": [1.0, 1.0, 1.0],
                    "is_background": False,
                    "bbox_side_len": bbox_side_len,
                },
                "quaternion_wxyz": quat_wxyz.tolist(),
                "translation": translation.tolist(),
                "is_valid": True,
                "id": 1,
                "material": [],
                "world_quaternion_wxyz": quat_wxyz.tolist(),
                "world_translation": translation.tolist(),
            }
        ],
        "scene_dataset": "Omni6DPose",
        "env_param": {},
        "face_up": True,
        "concentrated": False,
        "comments": "",
        "runtime_seed": 0,
        "baseline_dis": 0,
        "emitter_dist_l": 0,
    }


def make_obj_meta(bbox_side_len):
    """生成 obj_meta.json 内容（料盘为圆柱体：x/y 轴任意旋转对称）"""
    return {
        "objects": [
            {
                "oid": OID,
                "class_name": CLASS_NAME,
                "class_label": CLASS_LABEL,
                "bbox_side_len": bbox_side_len,
                "symmetry": {
                    "any": False,
                    "x": "any",   # 圆柱体：绕 X 轴任意旋转对称
                    "y": "any",   # 圆柱体：绕 Y 轴任意旋转对称
                    "z": "none",
                },
            }
        ]
    }


# ============================ 数据生成 ============================

def generate_split(mesh, bbox_side_len, split_name, num_frames, training_dir):
    """生成一个 split (train / test) 的所有帧"""
    split_dir = os.path.join(
        training_dir, "SOPE", "000000", split_name, "Omni6DPose", "000000"
    )
    os.makedirs(split_dir, exist_ok=True)
    print(f"\n[{split_name}] 生成 {num_frames} 帧 → {split_dir}")

    for i in range(num_frames):
        quat = random_quaternion_wxyz()
        trans = random_translation()

        color, depth, mask = render_frame(mesh, quat, trans)

        prefix = f"{i:06d}"
        Image.fromarray(color).save(os.path.join(split_dir, f"{prefix}_color.png"))
        save_exr(os.path.join(split_dir, f"{prefix}_depth.exr"), depth)
        save_exr(os.path.join(split_dir, f"{prefix}_mask.exr"), mask)

        meta = make_meta(quat, trans, bbox_side_len)
        with open(os.path.join(split_dir, f"{prefix}_meta.json"),
                  "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        if (i + 1) % 50 == 0 or (i + 1) == num_frames:
            print(f"  [{split_name}] {i + 1}/{num_frames}")

    return split_dir


def main():
    parser = argparse.ArgumentParser(
        description="生成 SOPE 格式的 GenPose2 训练/测评数据"
    )
    parser.add_argument("--training_dir", default="./trainging",
                        help="输出根目录 (默认 ./trainging)")
    parser.add_argument("--cad_dir", default="data_factory",
                        help="CAD 文件所在目录 (默认 data_factory)")
    parser.add_argument("--n_train", type=int, default=600,
                        help="训练样本数 (默认 600)")
    parser.add_argument("--n_test", type=int, default=200,
                        help="测评样本数 (默认 200)")
    parser.add_argument("--n_samples", type=int, default=N_SURFACE_SAMPLES,
                        help="每帧表面采样点数 (默认 80000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认 42)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    global N_SURFACE_SAMPLES
    N_SURFACE_SAMPLES = args.n_samples

    # 1. 加载并中心化 mesh
    mesh = load_mesh(args.cad_dir)
    mesh = center_mesh(mesh)
    print(f"[mesh] vertices={len(mesh.vertices)}, faces={len(mesh.faces)}")
    print(f"[mesh] 包围盒 min={mesh.bounds[0].tolist()}, max={mesh.bounds[1].tolist()}")

    bbox_side_len = (mesh.bounds[1] - mesh.bounds[0]).tolist()
    print(f"[mesh] bbox_side_len = {bbox_side_len}")

    # 2. 生成 train / test
    train_dir = generate_split(
        mesh, bbox_side_len, "train", args.n_train, args.training_dir
    )
    test_dir = generate_split(
        mesh, bbox_side_len, "test", args.n_test, args.training_dir
    )

    # 3. 生成 obj_meta.json
    obj_meta = make_obj_meta(bbox_side_len)
    meta_dir = os.path.join(args.training_dir, "Meta")
    os.makedirs(meta_dir, exist_ok=True)
    obj_meta_path = os.path.join(meta_dir, "obj_meta.json")
    with open(obj_meta_path, "w", encoding="utf-8") as f:
        json.dump(obj_meta, f, indent=2, ensure_ascii=False)
    print(f"\n[meta] obj_meta.json → {obj_meta_path}")

    # 4. 完成总结
    print("\n" + "=" * 60)
    print("所有数据生成完成！")
    print("=" * 60)
    print(f"训练: {args.n_train} 帧 → {train_dir}")
    print(f"测试: {args.n_test} 帧 → {test_dir}")
    print(f"物体元数据: {obj_meta_path}")
    print("\n下一步:")
    print(f"  1) 把 obj_meta.json 拷贝到训练配置目录:")
    print(f"     copy \"{obj_meta_path}\" \"configs/obj_meta.json\"")
    print(f"  2) 用 --data_path 指向 SOPE 根目录训练:")
    print(f"     --data_path \"{os.path.join(args.training_dir, 'SOPE')}\"")


if __name__ == "__main__":
    main()
