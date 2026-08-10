#!/usr/bin/env python3
"""SOPE 训练样本浏览器：RGB / Depth / Mask + meta，← → 切换帧。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
from PIL import Image

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = (HERE.parent / "train_set_1st_0810").resolve()

ARROW_JS = """
() => {
  if (window.__sopeNavBound) return;
  window.__sopeNavBound = true;
  document.addEventListener('keydown', (e) => {
    const tag = (e.target && e.target.tagName) ? e.target.tagName.toLowerCase() : '';
    if (tag === 'input' || tag === 'textarea' || tag === 'select') return;
    if (e.key === 'ArrowLeft') {
      e.preventDefault();
      const btn = document.querySelector('#btn_prev button, #btn_prev');
      if (btn) btn.click();
    } else if (e.key === 'ArrowRight') {
      e.preventDefault();
      const btn = document.querySelector('#btn_next button, #btn_next');
      if (btn) btn.click();
    }
  });
}
"""


def find_scene_dirs(root: Path, split: str) -> List[Path]:
    """在 SOPE 根或数据集根下查找含 *_color.png 的场景目录。"""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        return []

    candidates: List[Path] = []
    sope = root / "SOPE" if (root / "SOPE").is_dir() else root

    # 典型: SOPE/*/train|test/**/**
    for scene in sorted(sope.glob(f"*/{split}/**/")):
        if scene.is_dir() and any(scene.glob("*_color.png")):
            candidates.append(scene)

    # 扁平目录: 直接放四件套
    if not candidates and any(root.glob("*_color.png")):
        candidates.append(root)

    # 去重、保序
    seen = set()
    out: List[Path] = []
    for p in candidates:
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def available_splits(root: Path) -> List[str]:
    """探测 root 下实际有数据的 split（优先 train → test → 其它）。"""
    found: List[str] = []
    for split in ("train", "test", "val", "eval"):
        if find_scene_dirs(root, split):
            found.append(split)
    return found


def list_frame_ids(scene_dir: Path) -> List[str]:
    ids = sorted({p.name[:6] for p in scene_dir.glob("*_color.png") if len(p.name) >= 10})
    return ids


def load_depth(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"无法读取 depth: {path}")
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth.astype(np.float32)


def load_mask_ids(path: Path) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f"无法读取 mask: {path}")
    if raw.ndim == 3:
        raw = raw[..., 0]
    if np.issubdtype(raw.dtype, np.floating):
        # GenPose2/cutoop: float = id/255
        return np.clip(np.rint(raw * 255.0), 0, 255).astype(np.uint8)
    return raw.astype(np.uint8)


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """有效深度 → turbo 伪彩；无效为深灰。"""
    out = np.full((*depth.shape, 3), 40, dtype=np.uint8)
    valid = depth > 0
    if not np.any(valid):
        return out
    vals = depth[valid]
    lo, hi = np.percentile(vals, [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.zeros_like(depth, dtype=np.float32)
    norm[valid] = np.clip((depth[valid] - lo) / (hi - lo), 0, 1)
    u8 = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    out[valid] = colored[valid]
    return out


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    vis = rgb.astype(np.float32).copy()
    ids = [i for i in np.unique(mask) if i > 0]
    palette = [
        (0, 255, 128),
        (255, 128, 0),
        (64, 160, 255),
        (255, 64, 160),
        (180, 255, 64),
        (255, 64, 64),
    ]
    for k, mid in enumerate(ids):
        color = np.array(palette[k % len(palette)], dtype=np.float32)
        m = mask == mid
        vis[m] = (1.0 - alpha) * vis[m] + alpha * color
    # 轮廓
    for mid in ids:
        m = (mask == mid).astype(np.uint8) * 255
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (255, 255, 0), 1)
    return np.clip(vis, 0, 255).astype(np.uint8)


def format_meta(meta: Dict[str, Any]) -> str:
    cam = meta.get("camera", {}).get("intrinsics", {})
    lines = [
        f"intrinsics: fx={cam.get('fx')} fy={cam.get('fy')} "
        f"cx={cam.get('cx')} cy={cam.get('cy')} "
        f"{cam.get('width')}x{cam.get('height')}",
        "",
    ]
    for i, obj in enumerate(meta.get("objects", [])):
        m = obj.get("meta", {})
        q = obj.get("quaternion_wxyz", [])
        t = obj.get("translation", [])
        lines.append(f"[{i}] mask_id={obj.get('mask_id')} oid={m.get('oid')}")
        lines.append(f"    class={m.get('class_name')} label={m.get('class_label')}")
        lines.append(f"    bbox_side_len={m.get('bbox_side_len')}")
        lines.append(f"    quat_wxyz={[round(float(x), 4) for x in q]}")
        lines.append(f"    translation_m={[round(float(x), 4) for x in t]}")
        lines.append(f"    is_valid={obj.get('is_valid')}")
        lines.append("")
    lines.append("--- raw json ---")
    lines.append(json.dumps(meta, indent=2, ensure_ascii=False))
    return "\n".join(lines)


def scan_dataset(root: str, split: str) -> Tuple[List[str], str, int, str]:
    """
    返回 (keys, info, max_index, resolved_split)。
    若所选 split 无数据，自动切到该目录下第一个有数据的 split。
    """
    root_p = Path(root)
    splits = available_splits(root_p)
    note = ""
    resolved = split
    scenes = find_scene_dirs(root_p, resolved)
    if not scenes and splits:
        resolved = splits[0]
        scenes = find_scene_dirs(root_p, resolved)
        note = f"（已自动从 {split} 切换到 {resolved}）"
    if not scenes:
        hint = f"可用 split: {splits}" if splits else "未探测到 train/test 样本"
        return [], f"未找到 split={split} 样本，请检查数据根目录。{hint}", 0, split

    keys: List[str] = []
    for si, scene in enumerate(scenes):
        for fid in list_frame_ids(scene):
            keys.append(f"{si}|{fid}|{scene}")
    info = f"找到 {len(scenes)} 个场景目录，共 {len(keys)} 帧（{resolved}）{note}"
    return keys, info, max(len(keys) - 1, 0), resolved


def load_frame(
    keys: List[str], index: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], str, str, int]:
    if not keys:
        empty = np.zeros((240, 320, 3), dtype=np.uint8)
        return empty, empty, empty, "无样本", "0 / 0", 0

    index = int(np.clip(index, 0, len(keys) - 1))
    _, fid, scene_s = keys[index].split("|", 2)
    scene = Path(scene_s)
    prefix = scene / fid

    rgb = np.array(Image.open(str(prefix) + "_color.png").convert("RGB"))
    depth = load_depth(Path(str(prefix) + "_depth.exr"))
    mask = load_mask_ids(Path(str(prefix) + "_mask.exr"))
    meta = json.loads(Path(str(prefix) + "_meta.json").read_text(encoding="utf-8"))

    depth_vis = colorize_depth(depth)
    mask_vis = overlay_mask(rgb, mask)

    fg = int((mask > 0).sum())
    d_fg = depth[mask > 0]
    d_stats = (
        f"depth_fg mean={float(d_fg.mean()):.4f}m "
        f"min={float(d_fg.min()):.4f} max={float(d_fg.max()):.4f}"
        if d_fg.size
        else "depth_fg: empty"
    )
    header = (
        f"frame={fid}  scene={scene.name}\n"
        f"path={prefix}_*\n"
        f"mask_fg_pixels={fg}  {d_stats}\n\n"
        f"{format_meta(meta)}"
    )
    status = f"{index + 1} / {len(keys)}   ({fid})"
    return rgb, depth_vis, mask_vis, header, status, index


def step_index(keys: List[str], index: int, delta: int) -> int:
    if not keys:
        return 0
    return int((int(index) + delta) % len(keys))


def build_ui(default_root: Path) -> gr.Blocks:
    with gr.Blocks(title="SOPE Sample Viewer") as demo:
        gr.Markdown(
            "## SOPE 样本浏览器\n"
            "查看 `color / depth / mask` 与 `meta`。"
            "**← / →** 切换上一张 / 下一张（勿在输入框内按）。"
        )

        with gr.Row():
            root_in = gr.Textbox(
                label="数据根目录（含 SOPE/ 或直接为 SOPE）",
                value=str(default_root),
                scale=4,
            )
            split_in = gr.Dropdown(
                label="split",
                choices=["train", "test"],
                value="train",
                scale=1,
            )
            scan_btn = gr.Button("扫描 / 刷新", variant="primary", scale=1)

        info = gr.Textbox(label="数据集信息", interactive=False)
        status = gr.Textbox(label="当前帧", interactive=False)

        with gr.Row():
            prev_btn = gr.Button("← 上一张", elem_id="btn_prev")
            next_btn = gr.Button("下一张 →", elem_id="btn_next")

        index_slider = gr.Slider(
            label="帧序号（从 0 开始）",
            minimum=0,
            maximum=1,
            step=1,
            value=0,
        )

        with gr.Row():
            rgb_img = gr.Image(label="RGB (color.png)", type="numpy")
            depth_img = gr.Image(label="Depth (伪彩)", type="numpy")
            mask_img = gr.Image(label="Mask 叠加", type="numpy")

        meta_box = gr.Textbox(label="Meta", lines=22, max_lines=40)

        keys_state = gr.State([])
        index_state = gr.State(0)

        def do_scan(root: str, split: str):
            keys, msg, max_i, resolved = scan_dataset(root, split)
            rgb, depth_vis, mask_vis, meta_s, st, idx = load_frame(keys, 0)
            # Gradio Slider 要求 maximum > minimum
            slider_max = max(int(max_i), 1)
            avail = available_splits(Path(root)) or ["train", "test"]
            # 仅在实际切换 split 时改 value，避免 Dropdown.change 递归
            if resolved != split:
                split_upd = gr.update(choices=avail, value=resolved)
            else:
                split_upd = gr.update(choices=avail)
            return (
                keys,
                idx,
                gr.update(maximum=slider_max, value=0),
                split_upd,
                msg,
                st,
                rgb,
                depth_vis,
                mask_vis,
                meta_s,
            )

        def do_goto(keys: List[str], index: float):
            rgb, depth_vis, mask_vis, meta_s, st, idx = load_frame(keys, int(index))
            return idx, st, rgb, depth_vis, mask_vis, meta_s, gr.update(value=idx)

        def do_step(keys: List[str], index: int, delta: int):
            new_i = step_index(keys, index, delta)
            rgb, depth_vis, mask_vis, meta_s, st, idx = load_frame(keys, new_i)
            return idx, st, rgb, depth_vis, mask_vis, meta_s, gr.update(value=idx)

        outs_scan = [
            keys_state,
            index_state,
            index_slider,
            split_in,
            info,
            status,
            rgb_img,
            depth_img,
            mask_img,
            meta_box,
        ]
        outs_nav = [
            index_state,
            status,
            rgb_img,
            depth_img,
            mask_img,
            meta_box,
            index_slider,
        ]

        scan_btn.click(do_scan, [root_in, split_in], outs_scan)
        split_in.change(do_scan, [root_in, split_in], outs_scan)
        demo.load(do_scan, [root_in, split_in], outs_scan)
        demo.load(None, None, None, js=ARROW_JS)

        prev_btn.click(lambda k, i: do_step(k, i, -1), [keys_state, index_state], outs_nav)
        next_btn.click(lambda k, i: do_step(k, i, +1), [keys_state, index_state], outs_nav)
        index_slider.release(do_goto, [keys_state, index_slider], outs_nav)

    return demo


def main():
    parser = argparse.ArgumentParser(description="SOPE 样本 Gradio 浏览器")
    parser.add_argument(
        "--root",
        type=str,
        default=str(DEFAULT_ROOT),
        help="数据集根目录（默认 datasets/train_set_1st_0810）",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_ui(Path(args.root))
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
