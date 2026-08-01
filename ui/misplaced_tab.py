"""Gradio Tab: 缺货商品位姿估计。

① 手填或 VLM 识别缺货商品中文名
② VLM 生成 SAM3 提示词
③ SAM3 分割 → GenPose2 6D 位姿（复用 SAM3+GenPose2 流水线）
"""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional, Tuple

import gradio as gr
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import get_genpose2_conf, get_sam3_conf, get_vlm_conf  # noqa: E402
from ui.genpose_tab import (  # noqa: E402
    _empty_poses,
    generate_prompt_ui,
    run_sam3_genpose_tab,
)
from ui.pointcloud import GRASP_CLOUD_MAX_POINTS  # noqa: E402
from scripts.sam3_seg import (  # noqa: E402
    DEFAULT_SAM3_API_URL,
    DEFAULT_SAM3_MASK_THRESHOLD,
    DEFAULT_SAM3_PROMPT,
    DEFAULT_SAM3_THRESHOLD,
    DEFAULT_SAM3_TIMEOUT_S,
)
from scripts.vlm_prompt import (  # noqa: E402
    DEFAULT_MISSING_PRODUCT_PROMPT,
    DEFAULT_VLM_API_URL,
    DEFAULT_VLM_MODEL,
    DEFAULT_VLM_TIMEOUT_S,
    identify_missing_product,
)

logger = logging.getLogger("misplaced_tab")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

IdentifyOut = Tuple[str, str, str]
# product_name, sam3_prompt, vlm_status + GenPoseTabOut (9)
PipelineOut = Tuple[
    str,
    str,
    str,
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


def _default_missing_prompt() -> str:
    cfg = get_vlm_conf()
    return str(
        cfg.get("missing_prompt")
        or cfg.get("misplaced_prompt")
        or DEFAULT_MISSING_PRODUCT_PROMPT
    )


def run_identify_missing(
    rgb_img: Optional[Image.Image],
    identify_prompt: str,
    vlm_api_url: str,
    vlm_model: str,
    vlm_timeout_s: float,
) -> IdentifyOut:
    """用大模型识别缺货商品名，回填可编辑的商品名文本框。"""
    try:
        if rgb_img is None:
            return "", "", json.dumps(
                {"success": False, "message": "请上传货架 RGB 图像"},
                ensure_ascii=False,
                indent=2,
            )
        t0 = time.perf_counter()
        result = identify_missing_product(
            rgb_img.convert("RGB"),
            identify_prompt,
            api_url=(vlm_api_url or DEFAULT_VLM_API_URL).strip(),
            model=(vlm_model or DEFAULT_VLM_MODEL).strip(),
            timeout_s=float(vlm_timeout_s or DEFAULT_VLM_TIMEOUT_S),
        )
        elapsed = time.perf_counter() - t0
        detail = {
            "success": True,
            "step": 1,
            "step_name": "识别缺货商品名",
            "product_name": result["product_name"],
            "raw_reply": result["raw_reply"],
            "prompt": (identify_prompt or "").strip() or DEFAULT_MISSING_PRODUCT_PROMPT,
            "vlm_api": (vlm_api_url or DEFAULT_VLM_API_URL).strip(),
            "vlm_model": (vlm_model or DEFAULT_VLM_MODEL).strip(),
            "elapsed_s": round(elapsed, 3),
            "next": "确认/修改商品名后，可生成 SAM3 提示词或直接运行完整流水线",
        }
        return (
            result["product_name"],
            result["raw_reply"],
            json.dumps(detail, ensure_ascii=False, indent=2),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("identify missing product failed:\n%s", traceback.format_exc())
        return "", "", json.dumps(
            {"success": False, "message": str(exc)},
            ensure_ascii=False,
            indent=2,
        )


def _pipeline_error(
    message: str,
    product_name: str = "",
    sam3_prompt: str = "",
) -> PipelineOut:
    err = json.dumps({"success": False, "message": message}, ensure_ascii=False, indent=2)
    return (
        product_name,
        sam3_prompt,
        message,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        _empty_poses(message),
        err,
    )


def run_missing_pose_pipeline(
    rgb_img: Optional[Image.Image],
    depth_file: Any,
    camera_file: Any,
    product_name: str,
    auto_identify: bool,
    identify_prompt: str,
    sam3_prompt: str,
    use_vlm_sam3_prompt: bool,
    vlm_api_url: str,
    vlm_model: str,
    vlm_timeout_s: float,
    api_url: str,
    threshold: float,
    mask_threshold: float,
    timeout_s: float,
    max_instances: int,
    score_ckpt: str,
    energy_ckpt: str,
    scale_ckpt: str,
    max_points: float,
    rgb_shift_x: float,
    rgb_shift_y: float,
    enable_depth_align: bool,
    enable_workspace_outlier: bool,
    max_depth_mm: float,
    min_depth_mm: float,
    depth_percentile_hi: float,
    enable_instance_outlier: bool,
    sor_nb_neighbors: float,
    sor_std_ratio: float,
    z_mad_ratio: float,
) -> PipelineOut:
    """完整流水线：商品名（手填/识别）→ SAM3 提示词 → SAM3 → GenPose2。"""
    name = (product_name or "").strip()
    identify_meta: dict = {"enabled": False}
    try:
        if rgb_img is None:
            return _pipeline_error("请上传货架 RGB 图像")

        # ① 商品名：勾选自动识别且未手填时，先跑 VLM
        if bool(auto_identify) and not name:
            t0 = time.perf_counter()
            result = identify_missing_product(
                rgb_img.convert("RGB"),
                identify_prompt,
                api_url=(vlm_api_url or DEFAULT_VLM_API_URL).strip(),
                model=(vlm_model or DEFAULT_VLM_MODEL).strip(),
                timeout_s=float(vlm_timeout_s or DEFAULT_VLM_TIMEOUT_S),
            )
            name = result["product_name"]
            identify_meta = {
                "enabled": True,
                "product_name": name,
                "raw_reply": result["raw_reply"],
                "elapsed_s": round(time.perf_counter() - t0, 3),
            }
        if not name:
            return _pipeline_error(
                "请手填缺货商品名，或勾选「运行前自动识别缺货商品名」",
                product_name=product_name or "",
                sam3_prompt=sam3_prompt or "",
            )

        # ②③ 复用 SAM3+GenPose2：默认由商品名生成 SAM3 提示词
        outs = run_sam3_genpose_tab(
            rgb_img,
            depth_file,
            camera_file,
            sam3_prompt,
            name,
            bool(use_vlm_sam3_prompt),
            vlm_api_url,
            vlm_model,
            vlm_timeout_s,
            api_url,
            threshold,
            mask_threshold,
            timeout_s,
            max_instances,
            score_ckpt,
            energy_ckpt,
            scale_ckpt,
            max_points,
            rgb_shift_x,
            rgb_shift_y,
            enable_depth_align,
            enable_workspace_outlier,
            max_depth_mm,
            min_depth_mm,
            depth_percentile_hi,
            enable_instance_outlier,
            sor_nb_neighbors,
            sor_std_ratio,
            z_mad_ratio,
        )

        # 回填实际使用的 SAM3 提示词到 UI
        used_prompt = (sam3_prompt or "").strip()
        vlm_status = "已运行完整流水线"
        try:
            detail_obj = json.loads(outs[-1])
            if identify_meta.get("enabled"):
                detail_obj["missing_identify"] = identify_meta
            detail_obj["product_name"] = name
            detail_obj["pipeline"] = "缺货识别 → SAM3 → GenPose2"
            if detail_obj.get("vlm_prompt", {}).get("prompt"):
                used_prompt = str(detail_obj["vlm_prompt"]["prompt"])
            elif detail_obj.get("prompt"):
                used_prompt = str(detail_obj["prompt"])
            outs = (*outs[:-1], json.dumps(detail_obj, ensure_ascii=False, indent=2))
            if detail_obj.get("success"):
                vlm_status = f"缺货商品「{name}」→ SAM3「{used_prompt}」→ GenPose2 完成"
            else:
                vlm_status = str(detail_obj.get("message") or "流水线失败")
        except Exception:  # noqa: BLE001
            pass

        return (name, used_prompt, vlm_status, *outs)
    except Exception as exc:  # noqa: BLE001
        logger.error("missing pose pipeline failed:\n%s", traceback.format_exc())
        return _pipeline_error(
            str(exc),
            product_name=name or (product_name or ""),
            sam3_prompt=sam3_prompt or "",
        )


def build_misplaced_tab() -> None:
    sam3_cfg = get_sam3_conf()
    gp_cfg = get_genpose2_conf()
    vlm_cfg = get_vlm_conf()
    with gr.Tab("缺货商品位姿估计"):
        gr.Markdown(
            "流水线：**① 缺货商品名（手填 / 大模型识别）→ ② SAM3 提示词 → "
            "③ SAM3 分割 → ④ GenPose2 位姿**。"
        )
        with gr.Row():
            with gr.Column(scale=1):
                rgb = gr.Image(type="pil", label="上传货架 RGB", height=260, interactive=True)
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
                enable_depth_align = gr.Checkbox(
                    label="对齐 Depth→RGB（推荐：修正像素/3D 偏差；默认开启）",
                    value=True,
                )
                with gr.Row():
                    rgb_shift_x = gr.Number(
                        label="RGB↔Depth 偏移 dx（默认 -45→Depth 右移 45 对齐到 RGB）",
                        value=-45,
                        precision=0,
                    )
                    rgb_shift_y = gr.Number(
                        label="RGB↔Depth 偏移 dy（+下）",
                        value=0,
                        precision=0,
                    )

                gr.Markdown("#### ① 缺货商品名")
                product_name = gr.Textbox(
                    label="缺货商品名字（可手填，也可由大模型识别后回填）",
                    placeholder="例如：可比克 / 怡宝",
                    lines=1,
                )
                identify_prompt = gr.Textbox(
                    label="缺货识别提示词",
                    value=_default_missing_prompt(),
                    lines=4,
                )
                with gr.Row():
                    auto_identify = gr.Checkbox(
                        label="运行前若未填商品名则自动识别",
                        value=True,
                    )
                    btn_identify = gr.Button("仅识别缺货商品名", variant="secondary")
                identify_raw = gr.Textbox(
                    label="识别模型原始回复",
                    interactive=False,
                    lines=3,
                )

                gr.Markdown("#### ② SAM3 提示词")
                vlm_api = gr.Textbox(
                    label="VLM API URL（chat/completions）",
                    value=str(vlm_cfg.get("api_url") or DEFAULT_VLM_API_URL),
                    lines=1,
                )
                with gr.Row():
                    vlm_model = gr.Textbox(
                        label="VLM 模型名",
                        value=str(vlm_cfg.get("model") or DEFAULT_VLM_MODEL),
                        scale=3,
                    )
                    vlm_timeout = gr.Number(
                        label="VLM 超时（秒）",
                        value=float(vlm_cfg.get("timeout_s", DEFAULT_VLM_TIMEOUT_S)),
                        precision=0,
                        scale=1,
                    )
                with gr.Row():
                    use_vlm_sam3_prompt = gr.Checkbox(
                        label="运行前由大模型根据商品名生成 SAM3 提示词",
                        value=True,
                    )
                    btn_gen_prompt = gr.Button("生成 SAM3 提示词", variant="secondary")
                sam3_prompt = gr.Textbox(
                    label="实例分割提示词（可手写 / 由上一步生成）",
                    value=str(sam3_cfg.get("default_prompt") or DEFAULT_SAM3_PROMPT),
                    lines=2,
                )
                vlm_status = gr.Textbox(label="状态", interactive=False, lines=2)

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
                    max_inst = gr.Number(
                        label="max_instances（0=全部）", value=5, precision=0
                    )
                with gr.Accordion("点云离群点剔除", open=False):
                    enable_workspace = gr.Checkbox(
                        label="① 全局：整幅深度点云离群点剔除（深度范围 + SOR）",
                        value=True,
                    )
                    with gr.Row():
                        max_depth = gr.Number(
                            label="max_depth_mm（0=不限制）", value=2500, precision=0
                        )
                        min_depth = gr.Number(label="min_depth_mm", value=50, precision=0)
                        depth_pct = gr.Number(label="depth_percentile_hi（0=关）", value=99.0)
                    enable_instance = gr.Checkbox(
                        label="② 按实例：各 SAM3 实例独立剔除后再送 GenPose2",
                        value=True,
                    )
                    sor_k = gr.Number(label="SOR 邻域点数 nb_neighbors", value=20, precision=0)
                    sor_std = gr.Number(label="SOR / 空间 std_ratio", value=1.5)
                    z_mad = gr.Number(label="实例深度 z_mad_ratio（MAD 倍数）", value=2.5)
                with gr.Accordion("GenPose2 参数", open=False):
                    score = gr.Textbox(
                        label="score_ckpt",
                        value=str(
                            gp_cfg.get("score_ckpt") or "results/ckpts/ScoreNet/scorenet.pth"
                        ),
                    )
                    energy = gr.Textbox(
                        label="energy_ckpt",
                        value=str(
                            gp_cfg.get("energy_ckpt")
                            or "results/ckpts/EnergyNet/energynet.pth"
                        ),
                    )
                    scale = gr.Textbox(
                        label="scale_ckpt",
                        value=str(
                            gp_cfg.get("scale_ckpt") or "results/ckpts/ScaleNet/scalenet.pth"
                        ),
                    )
                    max_pts = gr.Number(
                        label="点云最大点数（降采样）",
                        value=int(GRASP_CLOUD_MAX_POINTS),
                        precision=0,
                    )

                btn_run = gr.Button("运行缺货位姿估计（全流程）", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("#### 快速预览")
                out_depth = gr.Image(type="pil", label="传感器深度（伪彩色）", height=180)
                out_mask = gr.Image(type="pil", label="SAM3 实例 mask", height=180)
                out_bbox = gr.Image(type="pil", label="SAM3 实例 bbox", height=180)
                out_pose_img = gr.Image(
                    type="pil",
                    label="位姿叠加（坐标轴 + 3D 尺寸框）",
                    height=260,
                )

        gr.Markdown("### 点云 + 位姿坐标轴")
        with gr.Row():
            with gr.Column(scale=4):
                out_glb = gr.Model3D(label="位姿 GLB", height=440, display_mode="solid")
            with gr.Column(scale=1):
                gr.Markdown("**下载**")
                out_glb_dl = gr.File(label="GLB", interactive=False)
                out_ply_dl = gr.File(label="PLY", interactive=False)
        out_poses = gr.Code(
            label="poses.json",
            language="json",
            lines=10,
            value=_empty_poses(),
        )
        out_json = gr.Code(label="详情", language="json", lines=12)

        btn_identify.click(
            fn=run_identify_missing,
            inputs=[rgb, identify_prompt, vlm_api, vlm_model, vlm_timeout],
            outputs=[product_name, identify_raw, out_json],
        )
        btn_gen_prompt.click(
            fn=generate_prompt_ui,
            inputs=[rgb, product_name, vlm_api, vlm_model, vlm_timeout],
            outputs=[sam3_prompt, vlm_status],
        )
        btn_run.click(
            fn=run_missing_pose_pipeline,
            inputs=[
                rgb,
                depth,
                camera,
                product_name,
                auto_identify,
                identify_prompt,
                sam3_prompt,
                use_vlm_sam3_prompt,
                vlm_api,
                vlm_model,
                vlm_timeout,
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
                enable_depth_align,
                enable_workspace,
                max_depth,
                min_depth,
                depth_pct,
                enable_instance,
                sor_k,
                sor_std,
                z_mad,
            ],
            outputs=[
                product_name,
                sam3_prompt,
                vlm_status,
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
            - **商品名**：可直接手填；或点「仅识别缺货商品名」；或勾选「运行前若未填则自动识别」
            - **SAM3 提示词**：默认可由大模型根据商品名 + RGB 生成；也可手写
            - **全流程**：缺货名 → SAM3 提示词 → SAM3 分割 → GenPose2 位姿（与「SAM3 + GenPose2」页一致）
            - **Depth→RGB 对齐**默认开启，用 dx/dy（默认 -45）把 Depth 对齐到 RGB，修正 2D/3D 偏差
            - 需上传 RGB + 深度 + `camera.json`；产物在 `output/ui_runs/`
            """
        )
