#!/usr/bin/env python3
"""
将无位姿标签的真实 RGB-D 样本，经 SAM3 TensorRT 分割服务切成 SOPE 格式评测集。

输入（每条一个目录）:
  /data/03-data/pose_training_data/<sample>/{rgb.png, depth.png, camera.json}

输出（与 train_set_1st_0810 同构）:
  datasets/eval_set_real_data_with_shelf_0810/
    Meta/obj_meta.json
    SOPE/000000/test/Omni6DPose/000000/
      {id:06d}_{color.png,depth.exr,mask.exr,meta.json}
    manifest.json   # 源样本 ↔ 输出帧 ↔ SAM3 分数映射

约定:
  - 每个 SAM3 检测实例 → 单独一帧（mask_id 恒为 1）
  - depth.exr 单位米；depth.png 按 mm（camera.depth_scale>=0.1 时 /1000）
  - 无真实 6D 标签：translation 用 mask 内深度反投影质心近似，旋转为单位四元数，
    comments 标明 pseudo_pose；is_valid=true 以便加载器可读
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

# 复用仓库内 RLE / EXR 工具
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from segment.sam3_seg_backend import _decode_rle_dict  # noqa: E402
from segment.yolo_seg_backend import save_genpose2_mask_exr  # noqa: E402

OID = "smt_factory-smt_tray-000001"
CLASS_NAME = "smt_tray"
CLASS_LABEL = 0
# 与合成训练集一致的包围盒（米）
DEFAULT_BBOX_SIDE_LEN = [0.177, 0.177, 0.01]


def save_depth_exr(path: Path, depth_m: np.ndarray) -> None:
    depth_m = np.asarray(depth_m, dtype=np.float32)
    if not cv2.imwrite(str(path), depth_m):
        raise RuntimeError(f"cv2.imwrite depth failed: {path}")


def load_camera(camera_json: Path) -> Tuple[Dict[str, float], float]:
    cam = json.loads(camera_json.read_text(encoding="utf-8"))
    K = cam.get("cam_K") or cam.get("intrinsic") or cam.get("K")
    if K is None:
        raise ValueError(f"camera.json 缺少 cam_K: {camera_json}")
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    depth_scale = float(cam.get("depth_scale", 1.0))
    # depth_scale>=0.1 → 深度图为毫米；否则 depth_m = raw * depth_scale
    if depth_scale >= 0.1:
        to_meters = 0.001
    else:
        to_meters = depth_scale
    intr = {
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
    }
    return intr, to_meters


def load_depth_meters(depth_path: Path, to_meters: float) -> np.ndarray:
    raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f"无法读取 depth: {depth_path}")
    if raw.ndim == 3:
        raw = raw[..., 0]
    depth = raw.astype(np.float32) * float(to_meters)
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0] = 0.0
    return depth


def estimate_translation(
    depth_m: np.ndarray, mask: np.ndarray, intr: Dict[str, float]
) -> List[float]:
    """mask 内有效深度中位数点的反投影，作为伪平移。"""
    ys, xs = np.where((mask > 0) & (depth_m > 1e-6))
    if len(xs) == 0:
        return [0.0, 0.0, 0.0]
    z = float(np.median(depth_m[ys, xs]))
    u = float(np.median(xs))
    v = float(np.median(ys))
    x = (u - intr["cx"]) * z / intr["fx"]
    y = (v - intr["cy"]) * z / intr["fy"]
    return [x, y, z]


def make_meta(
    *,
    intr: Dict[str, float],
    width: int,
    height: int,
    translation: List[float],
    bbox_side_len: List[float],
    source_name: str,
    sam3_score: float,
    sam3_bbox: List[float],
) -> Dict[str, Any]:
    quat = [1.0, 0.0, 0.0, 0.0]
    return {
        "camera": {
            "intrinsics": {
                "fx": intr["fx"],
                "fy": intr["fy"],
                "cx": intr["cx"],
                "cy": intr["cy"],
                "width": width,
                "height": height,
            },
            "quaternion": [1.0, 0.0, 0.0, 0.0],
            "translation": [0.0, 0.0, 0.0],
            "scene_obj_path": "",
            "background_image_path": "",
            "background_depth_path": "",
            "distances": [],
            "kind": "real",
        },
        "objects": {
            "1": {
                "meta": {
                    "oid": OID,
                    "class_name": CLASS_NAME,
                    "class_label": CLASS_LABEL,
                    "instance_path": "",
                    "scale": [1.0, 1.0, 1.0],
                    "is_background": False,
                    "bbox_side_len": bbox_side_len,
                },
                "quaternion_wxyz": quat,
                "translation": translation,
                "is_valid": True,
                "id": 1,
                "material": [],
                "world_quaternion_wxyz": quat,
                "world_translation": translation,
            }
        },
        "scene_dataset": "Omni6DPose",
        "env_param": {},
        "face_up": True,
        "concentrated": False,
        "comments": (
            f"real_unlabeled; source={source_name}; "
            f"pseudo_pose=depth_centroid+identity_quat; "
            f"sam3_score={sam3_score:.4f}; sam3_bbox={sam3_bbox}"
        ),
        "runtime_seed": 0,
        "baseline_dis": 0,
        "emitter_dist_l": 0,
    }


def make_obj_meta(bbox_side_len: List[float]) -> Dict[str, Any]:
    return {
        "class_list": [
            {
                "name": CLASS_NAME,
                "label": CLASS_LABEL,
                "instance_ids": [OID],
                "stat": {},
            }
        ],
        "instance_dict": {
            OID: {
                "object_id": OID,
                "source": "smt_factory",
                "name": CLASS_NAME,
                "obj_path": "",
                "tag": {
                    "datatype": "train",
                    "sceneChanger": False,
                    "symmetry": {
                        "any": False,
                        "x": "any",
                        "y": "any",
                        "z": "none",
                    },
                    "materialOptions": [],
                    "upAxis": ["z"],
                },
                "class_label": CLASS_LABEL,
                "class_name": CLASS_NAME,
                "dimensions": bbox_side_len,
            }
        },
    }


def call_sam3(
    api_url: str,
    rgb_path: Path,
    *,
    prompt: str,
    threshold: float,
    mask_threshold: float,
    timeout_s: float,
) -> Dict[str, Any]:
    b64 = base64.b64encode(rgb_path.read_bytes()).decode("ascii")
    payload = {
        "image_base64": b64,
        "prompt": prompt,
        "threshold": threshold,
        "mask_threshold": mask_threshold,
        "allow_empty": True,
        "return_vis_base64": False,
        "save_vis": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        api_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def decode_det_mask(det: Dict[str, Any], width: int, height: int) -> np.ndarray:
    seg = det.get("segmentation")
    if not isinstance(seg, dict) or "counts" not in seg:
        raise RuntimeError("detection 缺少 RLE segmentation")
    mask = _decode_rle_dict(seg)
    if mask.shape[0] != height or mask.shape[1] != width:
        mask = cv2.resize(
            mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    return mask.astype(bool)


def list_source_dirs(src_root: Path) -> List[Path]:
    dirs = []
    for p in sorted(src_root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "rgb.png").is_file() and (p / "depth.png").is_file() and (p / "camera.json").is_file():
            dirs.append(p)
    return dirs


def process_all(args: argparse.Namespace) -> None:
    src_root = Path(args.src_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    scene_dir = out_root / "SOPE" / "000000" / "test" / "Omni6DPose" / "000000"
    scene_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = out_root / "Meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    bbox_side_len = list(DEFAULT_BBOX_SIDE_LEN)
    # 若合成集 obj_meta 存在则对齐尺寸
    ref_meta = Path(args.ref_obj_meta)
    if ref_meta.is_file():
        ref = json.loads(ref_meta.read_text(encoding="utf-8"))
        objs = ref.get("objects") or []
        if objs and objs[0].get("bbox_side_len"):
            bbox_side_len = [float(x) for x in objs[0]["bbox_side_len"]]

    (meta_dir / "obj_meta.json").write_text(
        json.dumps(make_obj_meta(bbox_side_len), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    sources = list_source_dirs(src_root)
    if not sources:
        raise SystemExit(f"未在 {src_root} 找到含 rgb/depth/camera 的样本目录")

    print(f"[info] sources={len(sources)}  out={out_root}")
    print(f"[info] sam3={args.api_url}  prompt={args.prompt!r}")

    frame_id = 0
    manifest: List[Dict[str, Any]] = []
    n_skip_empty = 0
    n_skip_small = 0
    n_fail = 0

    for si, src in enumerate(sources):
        rgb_path = src / "rgb.png"
        depth_path = src / "depth.png"
        cam_path = src / "camera.json"
        print(f"\n[{si + 1}/{len(sources)}] {src.name}")

        try:
            intr, to_meters = load_camera(cam_path)
            rgb = np.array(Image.open(rgb_path).convert("RGB"))
            h, w = rgb.shape[:2]
            intr = {**intr}  # copy
            depth_m = load_depth_meters(depth_path, to_meters)
            if depth_m.shape[0] != h or depth_m.shape[1] != w:
                depth_m = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_NEAREST)

            t0 = time.perf_counter()
            resp = call_sam3(
                args.api_url,
                rgb_path,
                prompt=args.prompt,
                threshold=args.threshold,
                mask_threshold=args.mask_threshold,
                timeout_s=args.timeout,
            )
            wall = time.perf_counter() - t0
            dets = resp.get("detections") or []
            print(
                f"  sam3: n={len(dets)} wall={wall:.2f}s "
                f"svc_ms={resp.get('wall_ms')}"
            )
            if not dets:
                n_skip_empty += 1
                manifest.append(
                    {
                        "source": src.name,
                        "status": "no_detection",
                        "frames": [],
                    }
                )
                continue

            # 按分数降序，稳定输出
            dets = sorted(dets, key=lambda d: float(d.get("score", 0.0)), reverse=True)
            frames_written: List[Dict[str, Any]] = []

            for det in dets:
                score = float(det.get("score", 0.0))
                if score < args.threshold:
                    continue
                mask_bool = decode_det_mask(det, w, h)
                area = int(mask_bool.sum())
                if area < args.min_mask_pixels:
                    n_skip_small += 1
                    continue

                instance_u8 = np.zeros((h, w), dtype=np.uint8)
                instance_u8[mask_bool] = 1
                translation = estimate_translation(depth_m, instance_u8, intr)
                bbox = det.get("bbox") or []

                prefix = scene_dir / f"{frame_id:06d}"
                Image.fromarray(rgb).save(str(prefix) + "_color.png")
                save_depth_exr(Path(str(prefix) + "_depth.exr"), depth_m)
                save_genpose2_mask_exr(instance_u8, Path(str(prefix) + "_mask.exr"))
                meta = make_meta(
                    intr=intr,
                    width=w,
                    height=h,
                    translation=translation,
                    bbox_side_len=bbox_side_len,
                    source_name=src.name,
                    sam3_score=score,
                    sam3_bbox=[float(x) for x in bbox] if bbox else [],
                )
                Path(str(prefix) + "_meta.json").write_text(
                    json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )

                frames_written.append(
                    {
                        "frame_id": f"{frame_id:06d}",
                        "score": score,
                        "mask_pixels": area,
                        "bbox": bbox,
                        "translation": translation,
                    }
                )
                frame_id += 1

            manifest.append(
                {
                    "source": src.name,
                    "status": "ok" if frames_written else "all_filtered",
                    "num_detections": len(dets),
                    "frames": frames_written,
                }
            )
            print(f"  wrote {len(frames_written)} frames (total={frame_id})")

        except Exception as exc:
            n_fail += 1
            print(f"  FAIL: {exc}")
            manifest.append(
                {
                    "source": src.name,
                    "status": "error",
                    "error": str(exc),
                    "frames": [],
                }
            )

    summary = {
        "src_root": str(src_root),
        "out_root": str(out_root),
        "api_url": args.api_url,
        "prompt": args.prompt,
        "threshold": args.threshold,
        "mask_threshold": args.mask_threshold,
        "num_sources": len(sources),
        "num_frames": frame_id,
        "num_source_fail": n_fail,
        "num_source_empty": n_skip_empty,
        "num_instances_too_small": n_skip_small,
        "items": manifest,
    }
    (out_root / "manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("\n" + "=" * 60)
    print(f"完成: {frame_id} 帧 → {scene_dir}")
    print(f"失败源: {n_fail}  无检测源: {n_skip_empty}  过小实例: {n_skip_small}")
    print(f"manifest: {out_root / 'manifest.json'}")
    print(f"obj_meta: {meta_dir / 'obj_meta.json'}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="真实数据 + SAM3 → SOPE eval 集")
    parser.add_argument(
        "--src_root",
        default="/data/03-data/pose_training_data",
        help="真实数据根目录",
    )
    parser.add_argument(
        "--out_root",
        default=str(
            REPO_ROOT / "datasets" / "eval_set_real_data_with_shelf_0810"
        ),
        help="输出根目录",
    )
    parser.add_argument(
        "--api_url",
        default="http://127.0.0.1:18003/infer",
        help="SAM3 TensorRT HTTP /infer",
    )
    parser.add_argument(
        "--prompt",
        default="Plastic Reel Connected With Tape",
        help="SAM3 文本提示",
    )
    parser.add_argument("--threshold", type=float, default=0.41)
    parser.add_argument("--mask_threshold", type=float, default=0.50)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--min_mask_pixels",
        type=int,
        default=200,
        help="过小实例丢弃",
    )
    parser.add_argument(
        "--ref_obj_meta",
        default=str(
            REPO_ROOT / "datasets" / "train_set_1st_0810" / "Meta" / "obj_meta.json"
        ),
    )
    args = parser.parse_args()

    # 健康检查
    health = args.api_url.rsplit("/infer", 1)[0] + "/health"
    try:
        with urllib.request.urlopen(health, timeout=10) as r:
            h = json.loads(r.read().decode("utf-8"))
        print(f"[health] {h}")
        if not h.get("ok"):
            raise SystemExit("SAM3 /health 未就绪")
    except Exception as exc:
        raise SystemExit(f"无法连接 SAM3: {health} ({exc})") from exc

    process_all(args)


if __name__ == "__main__":
    main()
