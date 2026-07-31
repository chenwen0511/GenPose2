"""多模态大模型生成 SAM3 实例分割提示词（OpenAI 兼容 /v1/chat/completions）。"""

from __future__ import annotations

import base64
import io
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import requests
from PIL import Image

_ROOT_DIR = Path(__file__).resolve().parents[1]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from config import get_vlm_conf  # noqa: E402

logger = logging.getLogger("vlm_prompt")

VLM_SYSTEM_TEMPLATE = (
    "You are a visual prompting assistant for SAM3 instance segmentation.\n"
    "Rules:\n"
    "1. Translate the Chinese object name in double quotes into standard retail "
    "English commodity name, add shape/package feature description.\n"
    "2. Generate SAM3 prompt following exact format: instance segmentation, translated_name\n"
    "3. Do not add any comments, line breaks, symbols, descriptions. Only one single line result.\n"
    'Task: Look at the image, find all "{chinese_name}".'
)


def _load_vlm_defaults() -> Dict[str, Any]:
    cfg = get_vlm_conf()
    return {
        "api_url": str(
            cfg.get("api_url") or "http://192.168.100.63:8000/v1/chat/completions"
        ),
        "model": str(cfg.get("model") or "qwen3-vl-4b"),
        "timeout_s": float(cfg.get("timeout_s") or 60.0),
        "temperature": float(cfg.get("temperature") or 0.1),
        "max_tokens": int(cfg.get("max_tokens") or 128),
    }


_VLM_DEFAULTS = _load_vlm_defaults()
DEFAULT_VLM_API_URL = _VLM_DEFAULTS["api_url"]
DEFAULT_VLM_MODEL = _VLM_DEFAULTS["model"]
DEFAULT_VLM_TIMEOUT_S = _VLM_DEFAULTS["timeout_s"]
DEFAULT_VLM_TEMPERATURE = _VLM_DEFAULTS["temperature"]
DEFAULT_VLM_MAX_TOKENS = _VLM_DEFAULTS["max_tokens"]


def build_vlm_user_text(chinese_name: str) -> str:
    name = (chinese_name or "").strip()
    if not name:
        raise ValueError("商品中文名不能为空")
    return VLM_SYSTEM_TEMPLATE.format(chinese_name=name)


def _image_to_data_url(image: Union[Image.Image, Path, str]) -> str:
    if isinstance(image, (str, Path)):
        path = Path(image).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"图像不存在: {path}")
        raw = path.read_bytes()
        suffix = path.suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        if suffix in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        elif suffix == ".webp":
            mime = "image/webp"
        else:
            mime = "image/png"
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{b64}"

    rgb = image.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _extract_message_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"VLM 响应无 choices: {payload}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in (None, "text"):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    raise RuntimeError(f"无法解析 VLM message.content: {message!r}")


def sanitize_sam3_prompt(raw: str) -> str:
    """清洗模型输出，只保留单行 SAM3 提示词。"""
    text = (raw or "").strip()
    if not text:
        raise RuntimeError("VLM 返回空提示词")

    # 去掉常见思考/代码围栏残留
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```(?:\w+)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()

    for line in text.splitlines():
        line = line.strip().strip("`\"'")
        if not line:
            continue
        # 去掉前缀编号 / 标签
        line = re.sub(r"^(?:SAM3\s*)?(?:prompt\s*[:=]\s*)", "", line, flags=re.IGNORECASE)
        if "instance segmentation" in line.lower() or "," in line:
            return line
        return line

    raise RuntimeError(f"无法从 VLM 输出解析提示词: {raw!r}")


def generate_sam3_prompt_from_image(
    image: Union[Image.Image, Path, str],
    chinese_name: str,
    *,
    api_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout_s: Optional[float] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """调用多模态大模型，根据 RGB 与中文商品名生成 SAM3 提示词。"""
    url = (api_url or DEFAULT_VLM_API_URL).strip()
    model_name = (model or DEFAULT_VLM_MODEL).strip()
    timeout = float(timeout_s if timeout_s is not None else DEFAULT_VLM_TIMEOUT_S)
    temp = float(temperature if temperature is not None else DEFAULT_VLM_TEMPERATURE)
    tokens = int(max_tokens if max_tokens is not None else DEFAULT_VLM_MAX_TOKENS)

    user_text = build_vlm_user_text(chinese_name)
    data_url = _image_to_data_url(image)

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": user_text},
                ],
            }
        ],
        "temperature": temp,
        "max_tokens": tokens,
    }

    logger.info(
        "VLM generate prompt: url=%s model=%s chinese_name=%r",
        url,
        model_name,
        chinese_name,
    )
    resp = requests.post(url, json=payload, timeout=timeout)
    if resp.status_code >= 400:
        raise RuntimeError(
            f"VLM 请求失败 HTTP {resp.status_code}: {resp.text[:500]}"
        )
    try:
        body = resp.json()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"VLM 响应非 JSON: {resp.text[:500]}") from exc

    raw = _extract_message_text(body)
    prompt = sanitize_sam3_prompt(raw)
    logger.info("VLM generated SAM3 prompt: %r (raw=%r)", prompt, raw)
    return prompt
