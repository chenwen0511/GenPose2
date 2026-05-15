import argparse
import asyncio
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict


class InferTiming(TypedDict, total=False):
    upload_s: float
    templates_s: float
    ism_s: Optional[float]
    yolo_s: Optional[float]
    pose_s: float
    pipeline_s: float
    total_s: float

from fastapi import FastAPI, File, Form, HTTPException, UploadFile


ROOT_DIR = Path(__file__).resolve().parent
ENV_DIR = Path(os.environ.get("SAM6D_ENV_DIR", "/home/mui/.micromamba/envs/sam6d"))
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "service_outputs"

# Eager-import PEM so missing CUDA/torch/deps fails at process start, not on first /infer.
_pem_dir = str(ROOT_DIR / "Pose_Estimation_Model")
if _pem_dir not in sys.path:
    sys.path.insert(0, _pem_dir)
from run_warmup_inference_custom import run_pose_inference  # noqa: E402

app = FastAPI(title="SAM-6D HTTP Service")
infer_lock = asyncio.Lock()


@app.on_event("startup")
async def _startup_preload_yolo() -> None:
    """Optionally preload YOLO weights and run one warmup predict."""
    yolo_weights = os.environ.get("SAM6D_YOLO_WEIGHTS")
    if not yolo_weights:
        print("[warmup_http_service] startup preload YOLO skipped: SAM6D_YOLO_WEIGHTS is not set")
        return

    weights_path = Path(yolo_weights).expanduser().resolve()
    if not weights_path.is_file():
        print(f"[warmup_http_service] startup preload YOLO skipped: weights not found: {weights_path}")
        return

    try:
        from yolo_seg_backend import preload_yolo_model

        yolo_imgsz = int(os.environ.get("SAM6D_YOLO_IMGSZ", "640"))
        yolo_conf = float(os.environ.get("SAM6D_YOLO_CONF", "0.25"))
        yolo_class_id = int(os.environ.get("SAM6D_YOLO_CLASS_ID", "0"))
        print(
            "[warmup_http_service] startup preload YOLO begin: "
            f"weights={weights_path} imgsz={yolo_imgsz} conf={yolo_conf} class_id={yolo_class_id}"
        )
        preload_yolo_model(
            weights_path=weights_path,
            imgsz=yolo_imgsz,
            conf=yolo_conf,
            class_id=yolo_class_id,
        )
        print("[warmup_http_service] startup preload YOLO done")
    except Exception as exc:
        print(f"[warmup_http_service] startup preload YOLO failed: {type(exc).__name__}: {exc}")


@app.on_event("startup")
async def _startup_preload_pem_templates_gpu() -> None:
    """Load CAD templates from disk into GPU tensors once; run_pose_inference reuses the cache."""
    flag = os.environ.get("SAM6D_PEM_PRELOAD_TEMPLATES", "1").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        print("[warmup_http_service] PEM template GPU preload skipped: SAM6D_PEM_PRELOAD_TEMPLATES=0")
        return

    cad_env = os.environ.get("SAM6D_CAD_PATH")
    if not cad_env:
        print("[warmup_http_service] PEM template GPU preload skipped: SAM6D_CAD_PATH is not set")
        return

    cad_path = Path(cad_env).expanduser().resolve()
    if not cad_path.is_file():
        print(f"[warmup_http_service] PEM template GPU preload skipped: CAD not found: {cad_path}")
        return

    try:
        templates_dir = _ensure_templates(cad_path)
        from run_warmup_inference_custom import preload_pem_templates_gpu_cache

        gpu_ids = os.environ.get("SAM6D_CUDA_VISIBLE_DEVICES", "0")
        pem_cfg = os.environ.get("SAM6D_PEM_CONFIG_PATH")
        rd_env = os.environ.get("SAM6D_RD_SEED")
        kwargs: Dict[str, Any] = {"tem_path": str(templates_dir), "gpus": gpu_ids, "verbose": True}
        if pem_cfg:
            kwargs["config_path"] = pem_cfg
        if rd_env is not None and rd_env.strip() != "":
            kwargs["rd_seed"] = int(rd_env.strip())

        print(
            "[warmup_http_service] PEM template GPU preload begin: "
            f"templates_dir={templates_dir} gpus={gpu_ids}"
        )
        preload_pem_templates_gpu_cache(**kwargs)
        print("[warmup_http_service] PEM template GPU preload done")
    except Exception as exc:
        print(f"[warmup_http_service] PEM template GPU preload failed: {type(exc).__name__}: {exc}")


def _env() -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MAMBA_ROOT_PREFIX", "/home/mui/.micromamba")

    cuda_runtime_lib = ENV_DIR / "lib/python3.9/site-packages/nvidia/cuda_runtime/lib"
    ld_parts = [str(ENV_DIR / "lib"), str(cuda_runtime_lib)]
    if env.get("LD_LIBRARY_PATH"):
        ld_parts.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts)

    if os.environ.get("SAM6D_CUDA_VISIBLE_DEVICES"):
        env["CUDA_VISIBLE_DEVICES"] = os.environ["SAM6D_CUDA_VISIBLE_DEVICES"]
    return env


def _cad_path() -> Path:
    value = os.environ.get("SAM6D_CAD_PATH")
    if not value:
        raise HTTPException(
            status_code=500,
            detail="SAM6D_CAD_PATH is not set. Export it before starting the service.",
        )
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise HTTPException(status_code=500, detail=f"SAM6D_CAD_PATH does not exist: {path}")
    return path


def _output_root() -> Path:
    return Path(os.environ.get("SAM6D_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT))).expanduser().resolve()


def _run_command(command: List[str], cwd: Path, stage: str) -> None:
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        env=_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-80:])
        raise RuntimeError(f"{stage} failed with exit code {proc.returncode}\n{tail}")


def _template_cache_dir(cad_path: Path) -> Path:
    stat = cad_path.stat()
    fingerprint = f"{cad_path}:{stat.st_mtime_ns}:{stat.st_size}".encode("utf-8")
    cache_key = hashlib.sha256(fingerprint).hexdigest()[:16]
    return _output_root() / "_template_cache" / cache_key


def _ensure_templates(cad_path: Path) -> Path:
    cache_dir = _template_cache_dir(cad_path)
    templates_dir = cache_dir / "templates"
    if (templates_dir / "rgb_0.png").is_file():
        return templates_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    if templates_dir.exists():
        shutil.rmtree(templates_dir)

    blenderproc = ENV_DIR / "bin/blenderproc"
    _run_command(
        [
            str(blenderproc),
            "run",
            str(ROOT_DIR / "Render/render_custom_templates.py"),
            "--output_dir",
            str(cache_dir),
            "--cad_path",
            str(cad_path),
        ],
        cwd=ROOT_DIR / "Render",
        stage="template rendering",
    )

    if not (templates_dir / "rgb_0.png").is_file():
        raise RuntimeError(f"template rendering did not create {templates_dir / 'rgb_0.png'}")
    return templates_dir


async def _save_upload(upload: UploadFile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def _link_templates(source: Path, output_dir: Path) -> None:
    target = output_dir / "templates"
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.is_dir():
        shutil.rmtree(target)
    target.symlink_to(source, target_is_directory=True)


def _best_detection(result_path: Path) -> Dict[str, Any]:
    if not result_path.is_file():
        raise RuntimeError(f"missing result file: {result_path}")

    with result_path.open("r", encoding="utf-8") as f:
        detections = json.load(f)
    if not detections:
        raise RuntimeError("SAM-6D returned no detections")

    best = max(detections, key=lambda item: float(item.get("score", 0.0)))
    if "t" not in best:
        raise RuntimeError("best detection does not contain translation field 't'")
    return best


def _rotation_matrix_to_euler_zyx(rotation: List[List[float]]) -> List[float]:
    """Return rx, ry, rz in radians, using ZYX order: R = Rz(rz) * Ry(ry) * Rx(rx)."""
    r00, r10 = float(rotation[0][0]), float(rotation[1][0])
    r20, r21, r22 = float(rotation[2][0]), float(rotation[2][1]), float(rotation[2][2])
    r01, r11 = float(rotation[0][1]), float(rotation[1][1])

    sy = math.sqrt(r00 * r00 + r10 * r10)
    if sy > 1e-6:
        rx = math.atan2(r21, r22)
        ry = math.atan2(-r20, sy)
        rz = math.atan2(r10, r00)
    else:
        rx = math.atan2(-r01, r11)
        ry = math.atan2(-r20, sy)
        rz = 0.0
    return [rx, ry, rz]


def _run_sam6d_pipeline(
    cad_path: Path,
    rgb_path: Path,
    depth_path: Path,
    camera_path: Path,
    output_dir: Path,
    segmentor_model: str,
    seg_backend: str,
    yolo_weights: Optional[Path],
    yolo_conf: float,
    yolo_imgsz: int,
    yolo_class_id: int,
    det_score_thresh: float,
) -> Dict[str, Any]:
    timing: InferTiming = {
        "ism_s": None,
        "yolo_s": None,
    }

    t_templates = time.perf_counter()
    templates_dir = _ensure_templates(cad_path)
    _link_templates(templates_dir, output_dir)
    timing["templates_s"] = time.perf_counter() - t_templates

    python = sys.executable
    seg_path = output_dir / "sam6d_results/detection_ism.json"
    if seg_backend == "sam6d_ism":
        t_seg = time.perf_counter()
        _run_command(
            [
                python,
                "run_inference_custom.py",
                "--segmentor_model",
                segmentor_model,
                "--output_dir",
                str(output_dir),
                "--cad_path",
                str(cad_path),
                "--rgb_path",
                str(rgb_path),
                "--depth_path",
                str(depth_path),
                "--cam_path",
                str(camera_path),
            ],
            cwd=ROOT_DIR / "Instance_Segmentation_Model",
            stage="instance segmentation",
        )
        timing["ism_s"] = time.perf_counter() - t_seg
        pem_det_score_thresh = det_score_thresh
    elif seg_backend == "yolo_seg":
        if yolo_weights is None:
            raise RuntimeError("SAM6D_YOLO_WEIGHTS is not set and no yolo_weights query parameter was provided")
        from yolo_seg_backend import run_yolo_segmentation

        t_seg = time.perf_counter()
        seg_path = run_yolo_segmentation(
            weights_path=yolo_weights,
            rgb_path=rgb_path,
            output_dir=output_dir,
            conf=yolo_conf,
            imgsz=yolo_imgsz,
            class_id=yolo_class_id,
        )
        timing["yolo_s"] = time.perf_counter() - t_seg
        # PEM 显存随 batch（实例数）暴涨；勿写死 0.0，否则所有 YOLO 框进 PEM 易在 coarse 阶段 OOM。
        pem_det_score_thresh = float(det_score_thresh)
    else:
        raise RuntimeError("seg_backend must be 'sam6d_ism' or 'yolo_seg'")

    t_pose = time.perf_counter()
    gpu_ids = os.environ.get("SAM6D_CUDA_VISIBLE_DEVICES", "0")
    pem_verbose = os.environ.get("SAM6D_PEM_VERBOSE", "true").lower() in {"1", "true", "yes"}
    run_pose_inference(
        output_dir,
        cad_path,
        rgb_path,
        depth_path,
        camera_path,
        seg_path,
        det_score_thresh=float(pem_det_score_thresh),
        gpus=gpu_ids,
        verbose=pem_verbose,
        save_visualization=False,
    )
    timing["pose_s"] = time.perf_counter() - t_pose

    seg_used = timing.get("ism_s") if seg_backend == "sam6d_ism" else timing.get("yolo_s")
    timing["pipeline_s"] = timing["templates_s"] + float(seg_used or 0.0) + timing["pose_s"]

    result_path = output_dir / "sam6d_results/detection_pem.json"
    best = _best_detection(result_path)
    xyz_mm = best["t"]
    rotation_matrix = best.get("R")
    euler_zyx_rad = _rotation_matrix_to_euler_zyx(rotation_matrix) if rotation_matrix else None
    return {
        "score": float(best.get("score", 0.0)),
        "xyz_mm": xyz_mm,
        "rotation_euler_zyx_rad": euler_zyx_rad,
        "rotation_order": "zyx",
        "pose_convention": "xyz is camera-frame translation in mm; rx, ry, rz are ZYX Euler angles in radians.",
        "xyzrxryrz": list(xyz_mm) + list(euler_zyx_rad) if euler_zyx_rad else None,
        "xyzrxryrz_unit": "mm_rad",
        "result_dir": str(output_dir),
        "detection_ism_path": str(seg_path),
        "detection_pem_path": str(result_path),
        "vis_ism_path": str(output_dir / "sam6d_results/vis_ism.png"),
        "vis_pem_path": str(output_dir / "sam6d_results/vis_pem.png"),
        "all_detections": str(result_path),
        "timing": timing,
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    cad_path = os.environ.get("SAM6D_CAD_PATH")
    yolo_weights = os.environ.get("SAM6D_YOLO_WEIGHTS")
    return {
        "status": "ok",
        "root_dir": str(ROOT_DIR),
        "env_dir": str(ENV_DIR),
        "cad_path": cad_path,
        "cad_exists": bool(cad_path and Path(cad_path).expanduser().is_file()),
        "output_root": str(_output_root()),
        "default_seg_backend": os.environ.get("SAM6D_SEG_BACKEND", "sam6d_ism"),
        "yolo_weights": yolo_weights,
        "yolo_weights_exists": bool(yolo_weights and Path(yolo_weights).expanduser().is_file()),
    }


@app.post("/infer")
async def infer(
    rgb: UploadFile = File(...),
    depth: UploadFile = File(...),
    camera: UploadFile = File(...),
    segmentor_model: str = Form(default="sam"),
    seg_backend: Optional[str] = Form(default=None),
    yolo_weights: Optional[str] = Form(default=None),
    yolo_conf: float = Form(default=0.25),
    yolo_imgsz: int = Form(default=640),
    yolo_class_id: int = Form(default=0),
    det_score_thresh: float = Form(default=0.3),
) -> Dict[str, Any]:
    if segmentor_model not in {"sam", "fastsam"}:
        raise HTTPException(status_code=400, detail="segmentor_model must be 'sam' or 'fastsam'")

    cad_path = _cad_path()
    seg_backend_clean = seg_backend.strip() if seg_backend else None
    selected_backend = seg_backend_clean or os.environ.get("SAM6D_SEG_BACKEND", "sam6d_ism")
    yolo_weights_clean = yolo_weights.strip() if yolo_weights else None
    selected_yolo_weights_value = yolo_weights_clean or os.environ.get("SAM6D_YOLO_WEIGHTS")
    selected_yolo_weights = (
        Path(selected_yolo_weights_value).expanduser().resolve() if selected_yolo_weights_value else None
    )
    request_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    output_dir = _output_root() / request_id
    input_dir = output_dir / "inputs"
    output_dir.mkdir(parents=True, exist_ok=False)

    rgb_path = input_dir / "rgb.png"
    depth_path = input_dir / "depth.png"
    camera_path = input_dir / "camera.json"

    t_upload = time.perf_counter()
    await _save_upload(rgb, rgb_path)
    await _save_upload(depth, depth_path)
    await _save_upload(camera, camera_path)
    upload_s = time.perf_counter() - t_upload

    try:
        with camera_path.open("r", encoding="utf-8") as f:
            camera_json = json.load(f)
        if "cam_K" not in camera_json or "depth_scale" not in camera_json:
            raise ValueError("camera.json must contain cam_K and depth_scale")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid camera.json: {exc}") from exc

    async with infer_lock:
        try:
            payload = _run_sam6d_pipeline(
                cad_path=cad_path,
                rgb_path=rgb_path,
                depth_path=depth_path,
                camera_path=camera_path,
                output_dir=output_dir,
                segmentor_model=segmentor_model,
                seg_backend=selected_backend,
                yolo_weights=selected_yolo_weights,
                yolo_conf=yolo_conf,
                yolo_imgsz=yolo_imgsz,
                yolo_class_id=yolo_class_id,
                det_score_thresh=det_score_thresh,
            )
            timing = payload["timing"]
            timing["upload_s"] = upload_s
            timing["total_s"] = float(upload_s) + float(timing["pipeline_s"])
            return payload
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8001, type=int)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
