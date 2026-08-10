import os
import numpy as np

# ==============================================================================
# 电子料盘 (SMT Tray) 参数
# ==============================================================================
# 料盘主体（盘体）直径，单位毫米。脚本以此识别主体点云 vs 侧面白色塑料噪点。
TRAY_MAIN_DIAMETER_MM = 158.0


def farthest_point_sampling(pts, num_samples=1024):
    """
    最远点采样 (Farthest Point Sampling, FPS)
    确保采样的点在点云空间中分布尽可能均匀。

    参数:
        pts (numpy.ndarray): 原始输入点云，形状为 (N, 3)
        num_samples (int): 需要采样的目标点数，默认 1024
    返回:
        numpy.ndarray: 采样后的点云，形状为 (num_samples, 3)
    """
    N = len(pts)
    if N < num_samples:
        # 如果原始点数不足，随机重复点以补齐
        indices = np.random.choice(N, num_samples - N, replace=True)
        pts = np.vstack([pts, pts[indices]])
        N = len(pts)

    farthest_pts = np.zeros((num_samples, 3))
    # 随机选择第一个点
    farthest_pts[0] = pts[np.random.randint(N)]
    # 初始化每个点到已选择点集的最短距离
    distances = np.sum((pts - farthest_pts[0])**2, axis=1)

    for i in range(1, num_samples):
        # 选择距离当前已选点集最远的点
        farthest_idx = np.argmax(distances)
        farthest_pts[i] = pts[farthest_idx]
        # 更新剩余点到新加入点的距离
        dist_to_new_point = np.sum((pts - farthest_pts[i])**2, axis=1)
        distances = np.minimum(distances, dist_to_new_point)

    return farthest_pts


def _classify_tray_points(points, main_radius_mm, radial_tolerance=0.05):
    """
    按电子料盘的几何特征对采样点进行分类：

      - main_points: 料盘主体顶面（径向在 158mm 盘体内 + 顶面一侧）
                     这是相机正面视角下主要观察到的表面。
      - side_points: 圆柱侧面白色塑料噪点（径向超出主体半径 + 顶面一侧）
                     因白色塑料反射而产生，需稀疏采样并抖动。

    假设料盘主轴对齐 Z 轴（盘体水平摆放），即 XY 平面为盘面。
    若模型方向不一致，请先旋转到 Z-up 再喂入本脚本。
    """
    radial_distance = np.linalg.norm(points[:, :2], axis=1)
    z_coords = points[:, 2]
    z_mid = np.median(z_coords)

    # 主体半径阈值（留一点容差，避免被 CAD 微小误差误判为侧面）
    radius_threshold = main_radius_mm * (1.0 + radial_tolerance)
    is_main = radial_distance < radius_threshold
    is_top = z_coords > z_mid

    main_mask = is_main & is_top            # 主体顶面：相机正面看到的核心
    side_mask = (~is_main) & is_top         # 圆柱侧壁上半圈：白色塑料反射噪点

    return points[main_mask], points[side_mask]


def process_cad_to_genpose_input(mesh_file, output_ply=None, num_points=1024,
                                  main_diameter_mm=TRAY_MAIN_DIAMETER_MM,
                                  side_noise_ratio=0.18,
                                  side_position_jitter=0.0008):
    """
    加载电子料盘等 CAD 模型（STL/OBJ/PLY/STEP），下采样并模拟生成符合 GenPose 输入要求的 1024 个点云数据。

    ▸ 电子料盘的特殊处理（与普通零件的关键区别）:
      1. 料盘主体（盘体）直径 158mm（可通过 main_diameter_mm 调整）。
      2. 侧面是白色塑料材质，会产生噪点云返回：
           - 这些点稀疏采样（占总数约 side_noise_ratio）
           - 并叠加微小位置扰动（side_position_jitter, 单位米）模拟反射噪点
      3. 正面视角主要看到料盘主体顶面（圆形 158mm），因此主体顶面点云被完整保留，
           侧面噪点只作为辅助信息。

    ▸ 假设条件:
      - CAD 模型的主轴对齐 Z 轴（盘体水平摆放），盘面在 XY 平面。
      - 若模型方向不一致，请先在外部旋转到 Z-up 再喂入本脚本。

    参数:
        mesh_file (str): 3D 模型路径，支持 STL/OBJ/PLY/STEP 格式
        output_ply (str): 处理后的点云保存路径 (可选)
        num_points (int): 目标点数，GenPose 默认 1024
        main_diameter_mm (float): 料盘主体直径 (mm)，默认 158
        side_noise_ratio (float): 侧面噪点占总点数的比例，默认 0.18
        side_position_jitter (float): 侧面噪点的位置扰动幅度 (米)，默认 0.0008
    """
    try:
        import trimesh
    except ImportError:
        print("未安装 trimesh 库。请在您的本地环境中运行: pip install trimesh")
        return None

    print(f"正在加载 CAD 模型: {mesh_file}...")
    mesh = trimesh.load(mesh_file, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        # STEP 等格式可能返回 Scene，需要合并所有子几何
        if hasattr(mesh, 'geometry'):
            geometries = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not geometries:
                print(f"错误: {mesh_file} 中没有可用的 Trimesh 几何。")
                return None
            mesh = trimesh.util.concatenate(geometries)
        else:
            print(f"错误: 无法将 {mesh_file} 加载为 Trimesh 类型。")
            return None

    print(f"模型信息: 顶点数={len(mesh.vertices)}, 面数={len(mesh.faces)}")
    print(f"包围盒: min={mesh.bounds[0]}, max={mesh.bounds[1]}")

    # 1. 表面高密度均匀采样（为后续模拟部分观测提供基础）
    print("在 Mesh 表面进行高密度均匀采样...")
    full_sampled_points, _ = trimesh.sample.sample_surface(mesh, 20000)

    # 2. 按 158mm 主体直径识别主体 vs 侧面
    main_radius_mm = main_diameter_mm / 2.0
    print(f"按主体直径 {main_diameter_mm}mm（半径 {main_radius_mm}mm）分类点云...")
    main_points, side_candidates = _classify_tray_points(
        full_sampled_points, main_radius_mm=main_radius_mm
    )
    print(f"  主体候选点(顶面): {len(main_points)}")
    print(f"  侧面候选点(白色塑料噪点): {len(side_candidates)}")

    if len(main_points) == 0:
        print("警告: 未识别到主体点云，将回退到全部采样点。")
        main_points = full_sampled_points
        side_candidates = np.empty((0, 3))

    # 3. 侧面噪点：稀疏采样 + 抖动（模拟白色塑料反射噪点）
    if len(side_candidates) > 0:
        n_side_points = max(20, int(num_points * side_noise_ratio))
        n_side_points = min(n_side_points, len(side_candidates))
        side_indices = np.random.choice(len(side_candidates), n_side_points, replace=False)
        side_points = side_candidates[side_indices].copy()
        if side_position_jitter > 0:
            side_points += np.random.normal(0.0, side_position_jitter, side_points.shape)
        print(f"侧面噪点采样: {len(side_points)} 个 (目标占比 {side_noise_ratio*100:.1f}%)")
    else:
        side_points = np.empty((0, 3))
        print("无侧面噪点候选。")

    # 4. 合并主点云与侧面噪点
    combined_points = (
        np.vstack([main_points, side_points])
        if len(side_points) > 0 else main_points
    )

    # 如果合并后仍不足 1024，随机补齐
    if len(combined_points) < num_points:
        deficit = num_points - len(combined_points)
        extra = combined_points[
            np.random.choice(len(combined_points), deficit, replace=True)
        ]
        combined_points = np.vstack([combined_points, extra])
        print(f"补齐 {deficit} 个点（合并点不足 {num_points}）。")

    # 5. FPS 下采样到精准的 1024 个点
    print(f"使用 FPS（最远点采样）下采样至 {num_points} 个点...")
    final_points = farthest_point_sampling(combined_points, num_points)

    # 6. 中心化（Subtract Centroid）
    # GenPose2 内部会将坐标中心化以提高特征提取的稳定性
    centroid = np.mean(final_points, axis=0)
    normalized_points = final_points - centroid

    print(f"处理完成！输出点云形状: {normalized_points.shape}")
    print(f"点云中心位置 (Centroid): {centroid}")

    if output_ply:
        # 将处理后的点云保存为 PLY 格式便于三维可视化
        pc = trimesh.points.PointCloud(normalized_points)
        pc.export(output_ply)
        print(f"处理后的点云已保存至: {output_ply}")

    return normalized_points, centroid


if __name__ == "__main__":
    # 处理 data_factory 目录下的电子料盘 CAD 文件
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cad_candidates = [
        os.path.join(base_dir, "smt_tray_01.STEP"),
        os.path.join(base_dir, "smt_tray.PLY"),
        os.path.join(base_dir, "零件1.STEP"),
    ]

    for cad_file in cad_candidates:
        if os.path.exists(cad_file):
            print("=" * 64)
            print(f"开始处理: {cad_file}")
            print("=" * 64)
            out_ply = os.path.splitext(cad_file)[0] + "_genpose.ply"
            process_cad_to_genpose_input(cad_file, output_ply=out_ply)
            print()
