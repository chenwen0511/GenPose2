#!/usr/bin/env python3
"""SOPE 帧 → GLB 场景（点云 + 相机视锥 + 各料盘姿态），仅用于预览，不改训练数据。"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import trimesh

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ui.pointcloud import (  # noqa: E402
    camera_points_m_to_glb,
    camera_pose_m_to_glb,
    create_pose_axes_mesh,
    create_pose_marker_sphere,
    depth_rgb_instance_to_pointcloud,
    export_scene_glb,
    inject_axis_points,
    points_to_colored_mesh,
)

CACHE_DIR = Path(__file__).resolve().parent / "_glb_cache"
INSTANCE_PALETTE = (
    (0, 220, 120),
    (255, 140, 40),
    (80, 160, 255),
    (255, 80, 160),
    (220, 220, 60),
    (180, 100, 255),
    (255, 100, 100),
    (100, 255, 220),
    (200, 180, 120),
    (120, 200, 255),
)


def _quat_wxyz_to_R(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    n = np.sqrt(w * w + x * x + y * y + z * z) + 1e-12
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _iter_objects(meta: Dict[str, Any]) -> List[Tuple[int, Dict[str, Any]]]:
    raw = meta.get("objects", {})
    items: List[Tuple[int, Dict[str, Any]]] = []
    if isinstance(raw, dict):
        for k, v in raw.items():
            if not isinstance(v, dict):
                continue
            mid = int(k) if str(k).isdigit() else int(v.get("id", 0) or 0)
            items.append((mid, v))
    elif isinstance(raw, list):
        for i, v in enumerate(raw):
            if isinstance(v, dict):
                mid = int(v.get("mask_id", i + 1))
                items.append((mid, v))
    items.sort(key=lambda x: x[0])
    return items


def create_camera_frustum_mesh(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: float,
    height: float,
    *,
    near_m: float = 0.05,
    far_m: float = 0.25,
    radius_m: float = 0.0012,
    color_rgba: Sequence[int] = (0, 220, 255, 255),
) -> trimesh.Trimesh:
    """相机坐标系视锥线框（光心原点，看向 +Z），再转到 GLB 预览系。"""

    def _corner(u: float, v: float, z: float) -> np.ndarray:
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.array([x, y, z], dtype=np.float64)

    near_corners = [
        _corner(0, 0, near_m),
        _corner(width, 0, near_m),
        _corner(width, height, near_m),
        _corner(0, height, near_m),
    ]
    far_corners = [
        _corner(0, 0, far_m),
        _corner(width, 0, far_m),
        _corner(width, height, far_m),
        _corner(0, height, far_m),
    ]
    origin = np.zeros(3, dtype=np.float64)

    segs: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(4):
        segs.append((near_corners[i], near_corners[(i + 1) % 4]))
        segs.append((far_corners[i], far_corners[(i + 1) % 4]))
        segs.append((near_corners[i], far_corners[i]))
        segs.append((origin, far_corners[i]))

    segs_glb = [
        (camera_points_m_to_glb(a.reshape(1, 3))[0], camera_points_m_to_glb(b.reshape(1, 3))[0])
        for a, b in segs
    ]

    parts: List[trimesh.Trimesh] = []
    rgba = np.array(color_rgba, dtype=np.uint8)
    if rgba.shape[0] == 3:
        rgba = np.concatenate([rgba, np.array([255], dtype=np.uint8)])

    for p0, p1 in segs_glb:
        vec = p1 - p0
        length = float(np.linalg.norm(vec))
        if length < 1e-8:
            continue
        cyl = trimesh.creation.cylinder(radius=radius_m, height=length, sections=8)
        z_axis = np.array([0.0, 0.0, 1.0])
        d = vec / length
        if np.allclose(d, z_axis):
            T = np.eye(4)
        elif np.allclose(d, -z_axis):
            T = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
        else:
            axis = np.cross(z_axis, d)
            axis = axis / (np.linalg.norm(axis) + 1e-12)
            angle = float(np.arccos(np.clip(np.dot(z_axis, d), -1, 1)))
            T = trimesh.transformations.rotation_matrix(angle, axis)
        cyl.apply_transform(T)
        cyl.apply_translation((p0 + p1) * 0.5)
        cyl.visual.face_colors = np.tile(rgba, (len(cyl.faces), 1))
        parts.append(cyl)

    cam_ball = create_pose_marker_sphere(
        camera_points_m_to_glb(origin.reshape(1, 3))[0],
        radius_m=radius_m * 3.5,
        color_rgba=(0, 220, 255, 255),
    )
    parts.append(cam_ball)
    t0, r0 = camera_pose_m_to_glb([0, 0, 0], np.eye(3))
    parts.append(
        create_pose_axes_mesh(t0, r0, axis_length_m=0.06, radius_m=radius_m * 1.2)
    )
    return trimesh.util.concatenate(parts)


def create_bbox_wire_mesh(
    t_glb: np.ndarray,
    r_glb: np.ndarray,
    side_len_m: Sequence[float],
    *,
    radius_m: float = 0.0010,
    color_rgba: Sequence[int] = (255, 255, 80, 220),
) -> Optional[trimesh.Trimesh]:
    """物体包围盒线框。"""
    sx, sy, sz = [float(v) * 0.5 for v in side_len_m]
    corners_local = np.array(
        [
            [-sx, -sy, -sz],
            [sx, -sy, -sz],
            [sx, sy, -sz],
            [-sx, sy, -sz],
            [-sx, -sy, sz],
            [sx, -sy, sz],
            [sx, sy, sz],
            [-sx, sy, sz],
        ],
        dtype=np.float64,
    )
    corners = (corners_local @ r_glb.T) + np.asarray(t_glb, dtype=np.float64)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    rgba = np.array(color_rgba, dtype=np.uint8)
    if rgba.shape[0] == 3:
        rgba = np.concatenate([rgba, np.array([255], dtype=np.uint8)])
    parts = []
    for a, b in edges:
        p0, p1 = corners[a], corners[b]
        vec = p1 - p0
        length = float(np.linalg.norm(vec))
        if length < 1e-8:
            continue
        cyl = trimesh.creation.cylinder(radius=radius_m, height=length, sections=6)
        z_axis = np.array([0.0, 0.0, 1.0])
        d = vec / length
        if np.allclose(d, z_axis):
            T = np.eye(4)
        elif np.allclose(d, -z_axis):
            T = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
        else:
            axis = np.cross(z_axis, d)
            axis /= np.linalg.norm(axis) + 1e-12
            angle = float(np.arccos(np.clip(np.dot(z_axis, d), -1, 1)))
            T = trimesh.transformations.rotation_matrix(angle, axis)
        cyl.apply_transform(T)
        cyl.apply_translation((p0 + p1) * 0.5)
        cyl.visual.face_colors = np.tile(rgba, (len(cyl.faces), 1))
        parts.append(cyl)
    if not parts:
        return None
    return trimesh.util.concatenate(parts)


def build_frame_glb(
    *,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    mask_ids: np.ndarray,
    meta: Dict[str, Any],
    out_path: Path,
    max_points: int = 80000,
    show_bbox: bool = True,
    axis_length_m: float = 0.05,
) -> Path:
    """
    从一帧 SOPE 四件套生成 GLB：
      - RGB-D 点云（实例区保留彩色，背景变暗）
      - 相机视锥 + 相机坐标轴
      - 每个料盘：姿态轴 + 原点球 + 可选 bbox
    """
    cam = meta.get("camera", {}) or {}
    intr = cam.get("intrinsics", {}) or {}
    fx = float(intr.get("fx", 600.0))
    fy = float(intr.get("fy", 600.0))
    cx = float(intr.get("cx", rgb.shape[1] / 2.0))
    cy = float(intr.get("cy", rgb.shape[0] / 2.0))
    width = float(intr.get("width", rgb.shape[1]))
    height = float(intr.get("height", rgb.shape[0]))
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    depth_mm = np.asarray(depth_m, dtype=np.float32) * 1000.0
    points, colors, _stats = depth_rgb_instance_to_pointcloud(
        depth_mm,
        rgb,
        mask_ids.astype(np.uint8),
        K,
        factor_depth=1000.0,
        max_points=max_points,
        background_dim=0.35,
        prefer_instances=True,
    )

    cloud_mesh = points_to_colored_mesh(points, colors, scale_m=0.0016)
    geometries: List[Any] = [cloud_mesh]

    far = (
        float(np.clip(np.percentile(depth_m[depth_m > 0], 80), 0.15, 0.45))
        if np.any(depth_m > 0)
        else 0.25
    )
    geometries.append(
        create_camera_frustum_mesh(
            fx, fy, cx, cy, width, height, near_m=0.04, far_m=min(far, 0.35)
        )
    )

    for mid, obj in _iter_objects(meta):
        q = obj.get("quaternion_wxyz") or [1, 0, 0, 0]
        t = obj.get("translation") or [0, 0, 0]
        R = _quat_wxyz_to_R(q)
        t_glb, r_glb = camera_pose_m_to_glb(t, R)
        color = INSTANCE_PALETTE[(max(mid, 1) - 1) % len(INSTANCE_PALETTE)]
        geometries.append(
            create_pose_axes_mesh(
                t_glb, r_glb, axis_length_m=axis_length_m, radius_m=0.0018
            )
        )
        geometries.append(
            create_pose_marker_sphere(
                t_glb, radius_m=0.006, color_rgba=(*color, 255)
            )
        )
        ax_pts, ax_cols = inject_axis_points(
            t_glb, r_glb, axis_length_m=axis_length_m, core_color=color
        )
        geometries.append(points_to_colored_mesh(ax_pts, ax_cols, scale_m=0.0012))

        if show_bbox:
            m = obj.get("meta") or {}
            side = m.get("bbox_side_len") or [0.177, 0.177, 0.01]
            box = create_bbox_wire_mesh(
                t_glb, r_glb, side, color_rgba=(*color, 200)
            )
            if box is not None:
                geometries.append(box)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_scene_glb(geometries, out_path)
    return out_path


def cached_glb_path(prefix: Path, meta: Dict[str, Any]) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = (
        f"{prefix}|{json.dumps(meta.get('env_param', {}), sort_keys=True)}"
        f"|{len(meta.get('objects') or {})}"
    )
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{prefix.name}_{digest}.glb"


def build_or_reuse_frame_glb(
    *,
    prefix: Path,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    mask_ids: np.ndarray,
    meta: Dict[str, Any],
) -> str:
    out = cached_glb_path(prefix, meta)
    if out.is_file() and out.stat().st_size > 1000:
        return str(out)
    build_frame_glb(
        rgb=rgb,
        depth_m=depth_m,
        mask_ids=mask_ids,
        meta=meta,
        out_path=out,
    )
    return str(out)
