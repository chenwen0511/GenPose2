"""Gradio Tab 2: SAM3 分割 + GenPose2 6D 位姿估计."""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import get_genpose2_conf, get_sam3_conf, resolve_repo_path  # noqa: E402
from ui.common import (  # noqa: E402
    depth_to_colormap,
    detection_summary,
    filepath_from_upload,
    load_camera_json_full,
    load_depth_mm,
    render_mask_bbox_previews,
    shift_rgb_xy,
)
from ui.genpose_runner import (  # noqa: E402
    build_poses_payload,
    camera_json_to_meta,
    export_pose_scene,
    load_depth_meters,
    load_mask_u8,
    run_genpose2_from_mask,
)
from ui.pointcloud import GRASP_CLOUD_MAX_POINTS  # noqa: E402

from scripts.sam3_seg import (  # noqa: E402
    DEFAULT_SAM3_API_URL,
    DEFAULT_SAM3_MASK_THRESHOLD,
    DEFAULT_SAM3_PROMPT,
    DEFAULT_SAM3_THRESHOLD,
    DEFAULT_SAM3_TIMEOUT_S,
    _decode_detection_mask,
    run_sam3_segmentation,
)
from scripts import sam3_seg  # noqa: E402

OUTPUT_ROOT = ROOT_DIR / "output" / "ui_runs"
logger = logging.getLogger("genpose_tab")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

# sensor, mask, bbox, pose_overlay,
# glb, glb_dl, ply_dl,
# poses_json, detail_json
GenPoseTabOut = Tuple[
    Optional[Image.Image],
    Optional[Image.Image],
    Optional[Image.Image],
    Optional[Image.Image],
    Optional[str],
    Optional[str],
    Optional[str],
    str,
    str,
]


def _empty_poses(message: str = "尚未计算出位姿") -> str:
    return json.dumps(
        {
            "success": False,
            "message": message,
            "xyzrxryrz": None,
            "frame": "camera",
            "preview_frame": "glb_y_up",
            "unit": {"xyz": "mm", "rx_ry_rz": "rad", "size_3d": "m"},
            "poses": [],
            "num_poses": 0,
        },
        ensure_ascii=False,
        indent=2,
    )


def _error(message: str, sensor_vis: Optional[Image.Image] = None) -> GenPoseTabOut:
    logger.error("genpose_tab failed: %s", message)
    err = json.dumps({"success": False, "message": message}, ensure_ascii=False, indent=2)
    return (
        sensor_vis,
        None,
        None,
        None,
        None,
        None,
        None,
        _empty_poses(message),
        err,
    )


def run_sam3_genpose_tab(
    rgb_img: Optional[Image.Image],
    depth_file: Any,
    camera_file: Any,
    prompt: str,
    api_url: str,
    threshold: float,
    mask_threshold: float,
    timeout_s: float,
    max_instances: int,
    score_ckpt: str,
    energy_ckpt: str,
    scale_ckpt: str,
    max_points: float,
    rgb_shift_x: float = -45.0,
    rgb_shift_y: float = 0.0,
) -> GenPoseTabOut:
    try:
        if rgb_img is None:
            return _error("请上传 RGB 图像")
        depth_path = filepath_from_upload(depth_file)
        camera_path = filepath_from_upload(camera_file)
        if depth_path is None or not depth_path.is_file():
            return _error("请上传深度文件（.npy / .png / .exr）")
        if camera_path is None or not camera_path.is_file():
            return _error("请上传 camera.json")

        for name, ckpt in (
            ("score", score_ckpt),
            ("energy", energy_ckpt),
            ("scale", scale_ckpt),
        ):
            path = resolve_repo_path((ckpt or "").strip())
            if not path.is_file():
                return _error(f"{name} checkpoint 不存在: {path}")

        prompt_text = (prompt or "").strip() or DEFAULT_SAM3_PROMPT
        depth_mm = load_depth_mm(depth_path)
        intrinsic, factor_depth, camera_meta = load_camera_json_full(camera_path)
        rgb = rgb_img.convert("RGB")
        if rgb.size != (depth_mm.shape[1], depth_mm.shape[0]):
            return _error(
                f"RGB 尺寸 {rgb.size} 与深度 {depth_mm.shape[1]}x{depth_mm.shape[0]} 不一致"
            )

        color_np = np.asarray(rgb, dtype=np.uint8)
        rgb_shift_meta: Dict[str, Any] = {"dx": 0, "dy": 0, "source": "none"}
        cam_rgb_shift = camera_meta.get("rgb_shift") or camera_meta.get("color_shift")
        if isinstance(cam_rgb_shift, (list, tuple)) and len(cam_rgb_shift) == 2:
            sx, sy = int(cam_rgb_shift[0]), int(cam_rgb_shift[1])
            rgb_shift_meta = {"dx": sx, "dy": sy, "source": "camera.json"}
        else:
            sx = int(round(float(rgb_shift_x or 0)))
            sy = int(round(float(rgb_shift_y or 0)))
            rgb_shift_meta = {"dx": sx, "dy": sy, "source": "ui"}
        color_for_cloud = color_np
        if rgb_shift_meta["dx"] != 0 or rgb_shift_meta["dy"] != 0:
            color_for_cloud = shift_rgb_xy(
                color_np, int(rgb_shift_meta["dx"]), int(rgb_shift_meta["dy"])
            )

        sensor_vis = depth_to_colormap(depth_mm)
        run_dir = OUTPUT_ROOT / time.strftime("%Y%m%d_%H%M%S") / f"pose_{uuid.uuid4().hex[:8]}"
        run_dir.mkdir(parents=True, exist_ok=True)
        rgb_path = run_dir / "rgb.png"
        rgb.save(rgb_path)

        sam3_seg.DEFAULT_SAM3_API_URL = (api_url or DEFAULT_SAM3_API_URL).strip()
        sam3_seg.DEFAULT_SAM3_TIMEOUT_S = float(timeout_s or DEFAULT_SAM3_TIMEOUT_S)

        t0 = time.perf_counter()
        sam3_result = run_sam3_segmentation(
            rgb_path,
            run_dir / "sam3_results",
            prompt=prompt_text,
            threshold=float(threshold),
            mask_threshold=float(mask_threshold),
            max_instances=int(max_instances) if max_instances else 0,
        )
        sam3_s = time.perf_counter() - t0

        dets = sam3_result.instance_dets or []
        if not dets:
            return _error("SAM3 未返回实例", sensor_vis)

        mask_vis, bbox_vis = render_mask_bbox_previews(
            rgb, dets, _decode_detection_mask, prompt=prompt_text
        )
        mask_u8 = load_mask_u8(sam3_result.mask_exr)
        if mask_u8.shape[:2] != (depth_mm.shape[0], depth_mm.shape[1]):
            mask_u8 = np.array(
                Image.fromarray(mask_u8).resize(
                    (depth_mm.shape[1], depth_mm.shape[0]), Image.NEAREST
                )
            )

        h, w = color_np.shape[:2]
        meta = camera_json_to_meta(camera_meta, width=w, height=h)
        depth_scale = float(meta.get("depth_scale", camera_meta.get("depth_scale", 0.001)))
        if depth_path.suffix.lower() == ".png" and depth_scale >= 0.1:
            # camera.json often stores depth_scale=1.0 meaning "raw is mm"
            depth_scale = 0.001
        depth_m = load_depth_meters(depth_path, depth_scale)
        if depth_m.shape[:2] != (h, w):
            return _error(
                f"深度尺寸 {depth_m.shape[:2]} 与 RGB {(h, w)} 不一致", sensor_vis
            )

        t1 = time.perf_counter()
        poses_np, lengths_np, pose_overlay = run_genpose2_from_mask(
            color_rgb=color_np,
            depth_m=depth_m,
            mask_u8=mask_u8,
            meta=meta,
            score_ckpt=score_ckpt,
            energy_ckpt=energy_ckpt,
            scale_ckpt=scale_ckpt,
        )
        pose_s = time.perf_counter() - t1
        elapsed = time.perf_counter() - t0

        pose_overlay_path = run_dir / "vis_pem.png"
        pose_overlay.save(pose_overlay_path)

        instance_scores = list(sam3_result.instance_scores or [])
        instance_bboxes = [d.get("bbox") for d in dets]
        poses_payload = build_poses_payload(
            poses_np,
            lengths_np,
            instance_scores=instance_scores,
            instance_bboxes=instance_bboxes,
        )
        files = export_pose_scene(
            depth_mm=depth_mm,
            color_rgb=color_for_cloud,
            intrinsic=intrinsic,
            factor_depth=factor_depth,
            poses_np=poses_np,
            lengths_np=lengths_np,
            run_dir=run_dir,
            stem="scene_sam3_genpose2",
            max_points=int(max_points) if max_points else GRASP_CLOUD_MAX_POINTS,
        )
        poses_payload["scene_glb"] = files["glb"]
        poses_payload["scene_ply"] = files["ply"]
        (run_dir / "poses.json").write_text(
            json.dumps(poses_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Also write detection_pem.json (HTTP 服务风格)
        detections_pem: List[Dict[str, Any]] = []
        for i, p in enumerate(poses_payload["poses"]):
            detections_pem.append(
                {
                    "scene_id": 0,
                    "image_id": 0,
                    "category_id": 1,
                    "instance_id": p["instance_id"],
                    "score": p["score"],
                    "t": p["xyz_mm"],
                    "R": p["rotation_matrix"],
                    "size_3d": p["size_3d"],
                    "bbox": p.get("bbox"),
                }
            )
        (run_dir / "detection_pem.json").write_text(
            json.dumps(detections_pem, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        detail = {
            "success": True,
            "pipeline": "SAM3 → GenPose2",
            "elapsed_s": round(elapsed, 3),
            "timing": {
                "sam3_s": round(sam3_s, 3),
                "pose_s": round(pose_s, 3),
            },
            "num_instances": int(sam3_result.num_instances),
            "num_poses": int(poses_np.shape[0]),
            "detections": detection_summary(dets),
            "poses": {
                "num": int(poses_np.shape[0]),
                "glb": files["glb"],
                "json": str(run_dir / "poses.json"),
                "detection_pem": str(run_dir / "detection_pem.json"),
            },
            "pose_overlay": str(pose_overlay_path),
            "rgb_shift": rgb_shift_meta,
            "sam3_api": sam3_seg.DEFAULT_SAM3_API_URL,
            "prompt": prompt_text,
            "score_ckpt": str(resolve_repo_path(score_ckpt)),
            "energy_ckpt": str(resolve_repo_path(energy_ckpt)),
            "scale_ckpt": str(resolve_repo_path(scale_ckpt)),
            "run_dir": str(run_dir),
        }

        return (
            sensor_vis,
            mask_vis,
            bbox_vis,
            pose_overlay,
            files["glb"],
            files["glb"],
            files["ply"],
            json.dumps(poses_payload, ensure_ascii=False, indent=2),
            json.dumps(detail, ensure_ascii=False, indent=2),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("genpose_tab exception:\n%s", traceback.format_exc())
        return _error(str(exc))


def build_sam3_genpose_tab() -> None:
    sam3_cfg = get_sam3_conf()
    gp_cfg = get_genpose2_conf()
    with gr.Tab("SAM3 + GenPose2"):
        gr.Markdown(
            "流水线：**SAM3 文本分割 → GenPose2 6D 位姿估计**。"
            "输出位姿叠加图、`poses.json`、点云 GLB（RGB 坐标轴）。"
        )
        with gr.Row():
            with gr.Column(scale=1):
                rgb = gr.Image(type="pil", label="上传 RGB", height=280, interactive=True)
                with gr.Row():
                    depth = gr.File(
                        label="深度（.npy / uint16 .png / .exr）",
                        type="filepath",
                        file_types=[".npy", ".png", ".tif", ".tiff", ".exr"],
                    )
                    camera = gr.File(
                        label="相机 camera.json",
                        type="filepath",
                        file_types=[".json"],
                    )
                with gr.Row():
                    rgb_shift_x = gr.Number(
                        label="点云上色：RGB 右移 dx（像素，+右）",
                        value=-45,
                        precision=0,
                    )
                    rgb_shift_y = gr.Number(
                        label="点云上色：RGB 下移 dy（像素，+下）",
                        value=0,
                        precision=0,
                    )
                prompt = gr.Textbox(
                    label="实例分割提示词",
                    value=str(sam3_cfg.get("default_prompt") or DEFAULT_SAM3_PROMPT),
                    lines=3,
                )
                with gr.Accordion("SAM3 推理参数", open=False):
                    api = gr.Textbox(
                        label="SAM3 API URL",
                        value=str(sam3_cfg.get("api_url") or DEFAULT_SAM3_API_URL),
                    )
                    thr = gr.Slider(
                        label="Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=float(sam3_cfg.get("threshold", DEFAULT_SAM3_THRESHOLD)),
                    )
                    mthr = gr.Slider(
                        label="Mask Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=float(
                            sam3_cfg.get("mask_threshold", DEFAULT_SAM3_MASK_THRESHOLD)
                        ),
                    )
                    timeout = gr.Number(
                        label="超时（秒）",
                        value=float(sam3_cfg.get("timeout_s", DEFAULT_SAM3_TIMEOUT_S)),
                        precision=0,
                    )
                    max_inst = gr.Number(label="max_instances（0=全部）", value=5, precision=0)
                with gr.Accordion("GenPose2 参数", open=True):
                    score = gr.Textbox(
                        label="score_ckpt",
                        value=str(gp_cfg.get("score_ckpt") or "results/ckpts/ScoreNet/scorenet.pth"),
                    )
                    energy = gr.Textbox(
                        label="energy_ckpt",
                        value=str(
                            gp_cfg.get("energy_ckpt") or "results/ckpts/EnergyNet/energynet.pth"
                        ),
                    )
                    scale = gr.Textbox(
                        label="scale_ckpt",
                        value=str(gp_cfg.get("scale_ckpt") or "results/ckpts/ScaleNet/scalenet.pth"),
                    )
                    max_pts = gr.Number(
                        label="点云最大点数（降采样）",
                        value=int(GRASP_CLOUD_MAX_POINTS),
                        precision=0,
                    )
                btn = gr.Button("运行 SAM3 + GenPose2", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("#### 快速预览")
                out_depth = gr.Image(type="pil", label="传感器深度（伪彩色）", height=200)
                out_mask = gr.Image(type="pil", label="SAM3 实例 mask", height=200)
                out_bbox = gr.Image(type="pil", label="SAM3 实例 bbox", height=200)
                out_pose_img = gr.Image(
                    type="pil",
                    label="位姿叠加（坐标轴 + 3D 尺寸框）",
                    height=280,
                )

        gr.Markdown("### 点云 + 位姿坐标轴")
        gr.Markdown(
            "> **全幅 RGB 彩色点云** + **每实例 RGB 坐标轴**（X红/Y绿/Z蓝）。"
            "数值坐标系 `frame=camera`；3D 预览翻 Y 与 RGB 同向。"
        )
        with gr.Row():
            with gr.Column(scale=4):
                out_glb = gr.Model3D(label="位姿 GLB", height=480, display_mode="solid")
            with gr.Column(scale=1):
                gr.Markdown("**下载**")
                out_glb_dl = gr.File(label="GLB", interactive=False)
                out_ply_dl = gr.File(label="PLY", interactive=False)
        out_poses = gr.Code(
            label="poses.json",
            language="json",
            lines=12,
            value=_empty_poses(),
        )
        out_json = gr.Code(label="详情", language="json", lines=14)

        btn.click(
            fn=run_sam3_genpose_tab,
            inputs=[
                rgb,
                depth,
                camera,
                prompt,
                api,
                thr,
                mthr,
                timeout,
                max_inst,
                score,
                energy,
                scale,
                max_pts,
                rgb_shift_x,
                rgb_shift_y,
            ],
            outputs=[
                out_depth,
                out_mask,
                out_bbox,
                out_pose_img,
                out_glb,
                out_glb_dl,
                out_ply_dl,
                out_poses,
                out_json,
            ],
        )

        gr.Markdown(
            """
            **说明**
            - **SAM3**：外部 HTTP `POST /infer`（`image_base64`），生成实例 mask
            - **GenPose2**：ScoreNet → EnergyNet → ScaleNet，估计 6D 位姿与 3D 尺寸（无需 CAD）
            - **poses.json**：`xyz_mm` + ZYX 欧拉角（rad）+ `size_3d`（米），相机系
            - **点云 RGB 偏移**：仅影响 GLB 上色（默认 dx=-45）；也可用 `camera.json` 的 `rgb_shift:[dx,dy]`。不影响 SAM3 / GenPose2 / 2D 叠加图
            - **运行产物**：`output/ui_runs/<时间戳>/pose_*/`
            - 首次运行会加载三网权重，耗时较长；之后复用缓存
            """
        )
