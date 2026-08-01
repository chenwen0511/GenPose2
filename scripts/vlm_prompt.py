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

# 缺货商品识别默认提示词（可在 UI 与 conf.json 覆盖）
DEFAULT_MISSING_PRODUCT_PROMPT = (
    "You are a professional supermarket shelf inspector. Your job is to find out-of-stock goods.\n"
    "Goods placement rule: The same goods are placed vertically.\n"
    "Check if there are empty positions with missing goods on the outermost row "
    "(the row closest to the shelf edge).\n"
    "The missing goods name is the Chinese name of the goods sitting directly behind "
    "that empty outermost position.\n"
    "Do not name goods that are already present on the outermost row.\n"
    "Only output the Chinese name of the missing goods."
)

_MISSING_INSPECT_SUFFIX = (
    "\n\nBefore answering, inspect carefully in Chinese (short):\n"
    "- Which outermost-row positions are truly EMPTY "
    "(no product standing at the shelf edge there)?\n"
    "- For each EMPTY, what Chinese-brand goods sit directly BEHIND it "
    "(same column, deeper on the shelf)?\n"
    "If a brand already stands at the front edge, that column is NOT empty.\n"
    "Write at most 2 short sentences."
)
# 兼容旧常量名
DEFAULT_MISPLACED_PRODUCT_PROMPT = DEFAULT_MISSING_PRODUCT_PROMPT


def _load_vlm_defaults() -> Dict[str, Any]:
    cfg = get_vlm_conf()
    return {
        "api_url": str(
            cfg.get("api_url") or "http://192.168.130.88:8000/v1/chat/completions"
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


def _strip_model_noise(raw: str) -> str:
    text = (raw or "").strip()
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```(?:\w+)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    return text


def sanitize_sam3_prompt(raw: str) -> str:
    """清洗模型输出，只保留单行 SAM3 提示词。"""
    text = _strip_model_noise(raw)
    if not text:
        raise RuntimeError("VLM 返回空提示词")

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


def sanitize_product_name(raw: str) -> str:
    """从 VLM 回复中提取商品中文名（取首行有效内容）。"""
    text = _strip_model_noise(raw)
    if not text:
        raise RuntimeError("VLM 未返回商品名")

    for line in text.splitlines():
        line = line.strip().strip("`\"'「」『』")
        if not line:
            continue
        # 兼容两阶段定位格式：左起第2列|怡宝
        if "|" in line:
            line = line.split("|")[-1].strip()
        line = re.sub(
            r"^(?:商品(?:名称|名)?|名字|名称|答[：:]?)\s*[：:\-]?\s*",
            "",
            line,
        )
        line = line.strip().strip("`\"'「」『』。；;")
        if line:
            return line
    raise RuntimeError(f"无法从 VLM 输出解析商品名: {raw!r}")


def chat_completions_vision(
    image: Union[Image.Image, Path, str],
    user_text: str,
    *,
    api_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout_s: Optional[float] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """OpenAI 兼容视觉对话：传图 + 文本，返回 message 文本。"""
    url = (api_url or DEFAULT_VLM_API_URL).strip()
    model_name = (model or DEFAULT_VLM_MODEL).strip()
    timeout = float(timeout_s if timeout_s is not None else DEFAULT_VLM_TIMEOUT_S)
    temp = float(temperature if temperature is not None else DEFAULT_VLM_TEMPERATURE)
    tokens = int(max_tokens if max_tokens is not None else DEFAULT_VLM_MAX_TOKENS)
    prompt = (user_text or "").strip()
    if not prompt:
        raise ValueError("提示词不能为空")

    data_url = _image_to_data_url(image)
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": temp,
        "max_tokens": tokens,
    }

    logger.info("VLM vision chat: url=%s model=%s prompt[:80]=%r", url, model_name, prompt[:80])
    # 内网 VLM 不走系统 HTTP_PROXY，避免局域网地址被代理拦截
    resp = requests.post(
        url,
        json=payload,
        timeout=timeout,
        proxies={"http": None, "https": None},
    )
    if resp.status_code >= 400:
        raise RuntimeError(
            f"VLM 请求失败 HTTP {resp.status_code}: {resp.text[:500]}"
        )
    try:
        body = resp.json()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"VLM 响应非 JSON: {resp.text[:500]}") from exc

    return _extract_message_text(body)


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
    raw = chat_completions_vision(
        image,
        build_vlm_user_text(chinese_name),
        api_url=api_url,
        model=model,
        timeout_s=timeout_s,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    prompt = sanitize_sam3_prompt(raw)
    logger.info("VLM generated SAM3 prompt: %r (raw=%r)", prompt, raw)
    return prompt


def identify_missing_product(
    image: Union[Image.Image, Path, str],
    prompt: Optional[str] = None,
    *,
    api_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout_s: Optional[float] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, str]:
    """识别货架边缘排中缺货的商品名。

    两阶段：先描述「外侧空位 + 空位正后方商品」，再据此输出中文商品名，
    避免把前排已有商品（如薯愿）误报为缺货。

    Returns:
      ``{"product_name": ..., "raw_reply": ...}``
    """
    user_text = (prompt or "").strip() or DEFAULT_MISSING_PRODUCT_PROMPT
    tokens = int(max_tokens if max_tokens is not None else max(DEFAULT_VLM_MAX_TOKENS, 160))
    common = dict(
        api_url=api_url,
        model=model,
        timeout_s=timeout_s,
        temperature=temperature,
    )

    note = chat_completions_vision(
        image,
        user_text + _MISSING_INSPECT_SUFFIX,
        max_tokens=tokens,
        **common,
    ).strip()

    extract_prompt = (
        f"Shelf inspection note:\n{note}\n\n"
        f"Now follow the original task:\n{user_text}\n\n"
        "Critical:\n"
        "- Missing goods name = Chinese name of the goods BEHIND the empty outermost slot.\n"
        "- Never output goods that already stand on the outermost row.\n"
        "- If multiple empties are mentioned, prefer the empty whose behind-goods match "
        "neighboring front-row goods of the same brand "
        "(e.g. empty between/beside 可比克 with 可比克 behind → 可比克).\n"
        "Only output one Chinese product name."
    )
    raw = chat_completions_vision(
        image,
        extract_prompt,
        max_tokens=min(64, tokens),
        **common,
    ).strip()
    name = sanitize_product_name(raw)
    # 无效占位回退到 note 里最后一个疑似中文商品片段
    if name.upper() in {"EMPTY", "NONE", "N/A"} or name in {"无", "无缺货", "没有"}:
        name = sanitize_product_name(note)
    logger.info("VLM missing product: name=%r note=%r raw=%r", name, note, raw)
    return {
        "product_name": name,
        "raw_reply": f"{note}\n{raw}".strip(),
    }


# 兼容旧函数名
identify_misplaced_product = identify_missing_product
