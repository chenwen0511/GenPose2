#!/usr/bin/env python3
"""
对 SOPE 风格评测集（四件套）批量跑 tray_0810 GenPose2，并写出：
  - {id}_pred_vis.png     2D 位姿叠加
  - {id}_pred_meta.json   预测位姿（覆盖 objects，保留 camera）
  - {id}_pred.glb         点云 + 相机视锥 + 预测姿态轴/bbox

示例::

  conda activate genpose2
  python datasets/infer_tray0810_on_sope.py \\
    --data_root datasets/train_set_1st_0810-eval \\
    --score_ckpt results/ckpts/ScoreNet_tray_0810/ckpt_epoch50.pth \\
    --energy_ckpt results/ckpts/EnergyNet_tray_0810/ckpt_epoch50.pth \\
    --scale_ckpt results/ckpts/ScaleNet_tray_0810/ckpt_epoch4.pth
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

# 避免 import 时 get_config() 吞掉 CLI
_SAVED_ARGV: Optional[List[str]] = None
if __name__ == "__main__":
    _SAVED_ARGV = sys.argv[:]
    sys.argv = [sys.argv[0]]

from cutoop.data_loader import Dataset  # noqa: E402
from datasets.datasets_infer import InferDataset  # noqa: E402
from runners.infer import create_genpose2, visualize_pose  # noqa: E402

# scene_glb 在 ui_display
_UI = _REPO / "datasets" / "ui_display"
if str(_UI) not in sys.path:
    sys.path.insert(0, str(_UI))
from scene_glb import build_frame_glb  # noqa: E402

if _SAVED_ARGV is not None:
    sys.argv = _SAVED_ARGV[:]


def _R_to_quat_wxyz(R: np.ndarray) -> List[float]:
    """旋转矩阵 → wxyz（与 cutoop / 合成 meta 一致）。"""
    R = np.asarray(R, dtype=np.float64)
    t = float(np.trace(R))
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = 2.0 * np.sqrt(max(1e-12, 1.0 + R[0, 0] - R[1, 1] - R[2, 2]))
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = 2.0 * np.sqrt(max(1e-12, 1.0 + R[1, 1] - R[0, 0] - R[2, 2]))
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(max(1e-12, 1.0 + R[2, 2] - R[0, 0] - R[1, 1]))
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    return [float(v) for v in q]


def _find_prefixes(root: Path) -> List[Path]:
    colors = sorted(root.rglob("*_color.png"))
    return [Path(str(p)[: -len("_color.png")]) for p in colors]


def _mask_ids_u8(mask: np.ndarray) -> np.ndarray:
    """Dataset.load_mask 可能是 float id/255；统一成 uint8 id。"""
    if np.issubdtype(mask.dtype, np.floating):
        return np.clip(np.rint(mask * 255.0), 0, 255).astype(np.uint8)
    return mask.astype(np.uint8)


def _build_pred_meta(
    base_meta: Dict[str, Any],
    poses_np: np.ndarray,
    lengths_np: np.ndarray,
    mask_ids: np.ndarray,
) -> Dict[str, Any]:
    """用预测位姿替换 objects；mask_id 按 mask 中非 0 id 顺序对齐 InferDataset。"""
    ids = [int(i) for i in np.unique(mask_ids) if int(i) not in (0, 255)]
    meta = json.loads(json.dumps(base_meta))  # deep copy via json
    objects: Dict[str, Any] = {}
    n = min(len(ids), int(poses_np.shape[0]))
    for k in range(n):
        mid = ids[k]
        T = poses_np[k]
        R = T[:3, :3]
        t = T[:3, 3]
        size = [float(x) for x in lengths_np[k].tolist()]
        objects[str(mid)] = {
            "meta": {
                "oid": "smt_tray",
                "class_name": "smt_tray",
                "class_label": 0,
                "bbox_side_len": size,
            },
            "quaternion_wxyz": _R_to_quat_wxyz(R),
            "translation": [float(x) for x in t.tolist()],
            "is_valid": True,
            "source": "genpose2_tray_0810",
        }
    meta["objects"] = objects
    meta["comments"] = (
        f"pred_by=ScoreNet/EnergyNet/ScaleNet_tray_0810; "
        f"n_pred={n}; base_comments={base_meta.get('comments', '')}"
    )
    return meta


def infer_one(
    prefix: Path,
    genpose: Any,
) -> Tuple[bool, str]:
    color_p = Path(str(prefix) + "_color.png")
    depth_p = Path(str(prefix) + "_depth.exr")
    mask_p = Path(str(prefix) + "_mask.exr")
    meta_p = Path(str(prefix) + "_meta.json")
    for p in (color_p, depth_p, mask_p, meta_p):
        if not p.is_file():
            return False, f"missing {p.name}"

    color = Dataset.load_color(str(color_p))
    depth = Dataset.load_depth(str(depth_p))
    mask_raw = Dataset.load_mask(str(mask_p))
    mask_u8 = _mask_ids_u8(np.asarray(mask_raw))
    meta = json.loads(meta_p.read_text(encoding="utf-8"))

    if not np.any(mask_u8 > 0):
        return False, "empty mask"

    argv_saved = sys.argv[:]
    sys.argv = [sys.argv[0] if sys.argv else "infer_tray0810"]
    try:
        data = InferDataset(
            {"depth": depth, "color": color, "mask": mask_u8, "meta": meta},
            img_size=genpose.cfg.img_size,
            device=genpose.cfg.device,
            n_pts=genpose.cfg.num_points,
        )
        pose, length = genpose.inference(data, prev_pose=None, tracking=False, tracking_T0=0.15)
        vis_bgr = visualize_pose(data, pose, length, visualize_pts=False, visualize_image=False)
    finally:
        sys.argv = argv_saved

    poses_np = pose[0].cpu().numpy()
    lengths_np = length[0].cpu().numpy()
    if poses_np.shape[0] == 0:
        return False, "no poses"

    pred_meta = _build_pred_meta(meta, poses_np, lengths_np, mask_u8)
    vis_path = Path(str(prefix) + "_pred_vis.png")
    meta_out = Path(str(prefix) + "_pred_meta.json")
    glb_out = Path(str(prefix) + "_pred.glb")

    cv2.imwrite(str(vis_path), vis_bgr)
    meta_out.write_text(json.dumps(pred_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # GLB：点云 + 相机 + 预测姿态（用 pred_meta）
    build_frame_glb(
        rgb=color,
        depth_m=np.asarray(depth, dtype=np.float32),
        mask_ids=mask_u8,
        meta=pred_meta,
        out_path=glb_out,
        max_points=60000,
        show_bbox=True,
        axis_length_m=0.06,
    )
    return True, f"n={poses_np.shape[0]} -> {vis_path.name}, {meta_out.name}, {glb_out.name}"


def main() -> None:
    parser = argparse.ArgumentParser(description="tray_0810 批量推理 SOPE 评测集")
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(_REPO / "datasets" / "train_set_1st_0810-eval"),
    )
    parser.add_argument(
        "--score_ckpt",
        type=str,
        default=str(_REPO / "results/ckpts/ScoreNet_tray_0810/ckpt_epoch50.pth"),
    )
    parser.add_argument(
        "--energy_ckpt",
        type=str,
        default=str(_REPO / "results/ckpts/EnergyNet_tray_0810/ckpt_epoch50.pth"),
    )
    parser.add_argument(
        "--scale_ckpt",
        type=str,
        default=str(_REPO / "results/ckpts/ScaleNet_tray_0810/ckpt_epoch4.pth"),
    )
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 帧（0=全部）")
    args = parser.parse_args()

    root = Path(args.data_root).expanduser().resolve()
    prefixes = _find_prefixes(root)
    if args.limit > 0:
        prefixes = prefixes[: args.limit]
    if not prefixes:
        raise SystemExit(f"未找到 *_color.png: {root}")

    for p in (args.score_ckpt, args.energy_ckpt, args.scale_ckpt):
        if not Path(p).is_file():
            raise SystemExit(f"缺少权重: {p}")

    print(f"[infer] data_root={root}  frames={len(prefixes)}")
    print(f"[infer] score={args.score_ckpt}")
    print(f"[infer] energy={args.energy_ckpt}")
    print(f"[infer] scale={args.scale_ckpt}")

    t0 = time.time()
    argv_saved = sys.argv[:]
    sys.argv = [sys.argv[0]]
    try:
        genpose = create_genpose2(args.score_ckpt, args.energy_ckpt, args.scale_ckpt)
    finally:
        sys.argv = argv_saved
    print(f"[infer] model loaded in {time.time() - t0:.1f}s")

    ok, fail = 0, 0
    records: List[Dict[str, Any]] = []
    for i, prefix in enumerate(prefixes):
        t1 = time.time()
        try:
            success, msg = infer_one(prefix, genpose)
        except Exception as exc:  # noqa: BLE001
            success, msg = False, f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        dt = time.time() - t1
        status = "ok" if success else "fail"
        if success:
            ok += 1
        else:
            fail += 1
        print(f"[{i+1}/{len(prefixes)}] {status} {prefix.name} ({dt:.2f}s) {msg}")
        records.append(
            {
                "prefix": str(prefix),
                "ok": success,
                "msg": msg,
                "seconds": round(dt, 3),
            }
        )

    manifest = {
        "data_root": str(root),
        "score_ckpt": args.score_ckpt,
        "energy_ckpt": args.energy_ckpt,
        "scale_ckpt": args.scale_ckpt,
        "n_total": len(prefixes),
        "n_ok": ok,
        "n_fail": fail,
        "elapsed_s": round(time.time() - t0, 2),
        "outputs": [
            "*_pred_vis.png",
            "*_pred_meta.json",
            "*_pred.glb",
        ],
        "records": records,
    }
    out_m = root / "manifest_tray0810_pred.json"
    out_m.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[infer] done ok={ok} fail={fail} manifest={out_m}")
    print(
        f"[infer] 可视化: python datasets/ui_display/app.py --root {root} --port 7862\n"
        f"         或直接打开各帧 *_pred_vis.png / *_pred.glb"
    )


if __name__ == "__main__":
    main()
