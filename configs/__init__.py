"""Load project runtime config from ``configs/conf.json``.

Training argparse lives in ``configs/config.py`` (``from configs.config import get_config``).
UI / SAM3 / VLM / GenPose2 权重默认值走本模块。
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = Path(__file__).resolve().parent
CONF_PATH = CONFIGS_DIR / "conf.json"
CONF_EXAMPLE_PATH = CONFIGS_DIR / "conf.json.example"
SECRETS_PATH = CONFIGS_DIR / "secrets.local.json"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@lru_cache(maxsize=1)
def load_conf(path: str | None = None) -> Dict[str, Any]:
    if path:
        conf_path = Path(path).expanduser().resolve()
    elif CONF_PATH.is_file():
        conf_path = CONF_PATH
    elif CONF_EXAMPLE_PATH.is_file():
        conf_path = CONF_EXAMPLE_PATH
    else:
        raise FileNotFoundError(
            f"缺少 {CONF_PATH}（可从 {CONF_EXAMPLE_PATH.name} 复制一份）"
        )
    with conf_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config must be a JSON object: {conf_path}")
    if SECRETS_PATH.is_file() and path is None:
        try:
            secrets = json.loads(SECRETS_PATH.read_text(encoding="utf-8"))
            if isinstance(secrets, dict):
                data = _deep_merge(data, secrets)
        except Exception:  # noqa: BLE001
            pass
    return data


def get_sam3_conf() -> Dict[str, Any]:
    return dict(load_conf().get("sam3") or {})


def get_vlm_conf() -> Dict[str, Any]:
    """Raw ``vlm`` section (may contain nested ``sam3_prompt`` / ``reason``)."""
    cfg = dict(load_conf().get("vlm") or {})
    for env_key in ("ANTHROPIC_API_KEY", "MINIMAX_API_KEY", "VLM_API_KEY"):
        val = (os.environ.get(env_key) or "").strip()
        if val:
            cfg["api_key"] = val
            break
    return cfg


def get_vlm_profile(name: str = "reason") -> Dict[str, Any]:
    """Return one VLM endpoint profile.

    - ``sam3_prompt``: local qwen3-vl for SAM3 text prompts
    - ``reason``: MiniMax-M3 for missing-product / place-offset reasoning
    """
    cfg = get_vlm_conf()
    key = (name or "reason").strip().lower()
    aliases = {
        "sam3": "sam3_prompt",
        "sam3_prompt": "sam3_prompt",
        "prompt": "sam3_prompt",
        "qwen": "sam3_prompt",
        "reason": "reason",
        "missing": "reason",
        "place": "reason",
        "m3": "reason",
        "minimax": "reason",
        "advanced": "reason",
    }
    profile_key = aliases.get(key, key)
    nested = cfg.get(profile_key)
    if isinstance(nested, dict) and nested:
        out = dict(nested)
    elif profile_key == "reason" and cfg.get("api_url"):
        out = {
            k: cfg[k]
            for k in (
                "provider",
                "api_url",
                "model",
                "timeout_s",
                "temperature",
                "max_tokens",
                "api_key",
            )
            if k in cfg
        }
    else:
        out = {}

    if not out.get("api_key") and cfg.get("api_key"):
        out["api_key"] = cfg["api_key"]

    if profile_key == "reason":
        base = (os.environ.get("ANTHROPIC_BASE_URL") or "").strip()
        if base:
            out["api_url"] = base
        model = (os.environ.get("VLM_REASON_MODEL") or os.environ.get("VLM_MODEL") or "").strip()
        if model:
            out["model"] = model
        out.setdefault("provider", "anthropic")
        out.setdefault("api_url", "https://api.minimaxi.com/anthropic")
        out.setdefault("model", "MiniMax-M3")
        out.setdefault("timeout_s", 120.0)
        out.setdefault("temperature", 0.2)
        out.setdefault("max_tokens", 1024)
    else:
        url = (os.environ.get("VLM_SAM3_API_URL") or "").strip()
        if url:
            out["api_url"] = url
        model = (os.environ.get("VLM_SAM3_MODEL") or "").strip()
        if model:
            out["model"] = model
        out.setdefault("provider", "openai")
        out.setdefault("api_url", "http://127.0.0.1:8000/v1/chat/completions")
        out.setdefault("model", "qwen3-vl-4b")
        out.setdefault("timeout_s", 60.0)
        out.setdefault("temperature", 0.1)
        out.setdefault("max_tokens", 128)

    return out


def get_genpose2_conf() -> Dict[str, Any]:
    return dict(load_conf().get("genpose2") or {})


def resolve_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (ROOT_DIR / path).resolve()
    else:
        path = path.resolve()
    return path


def _ckpt_epoch_num(path: Path) -> int:
    name = path.stem  # ckpt_epoch50
    if name.startswith("ckpt_epoch"):
        suffix = name[len("ckpt_epoch") :]
        if suffix.isdigit():
            return int(suffix)
    return -1


def resolve_ckpt_path(value: str) -> Path:
    """Resolve a checkpoint file, or the latest ``ckpt_epoch*.pth`` inside a directory."""
    path = resolve_repo_path(value)
    if path.is_file():
        return path
    if path.is_dir():
        epochs = sorted(path.glob("ckpt_epoch*.pth"), key=_ckpt_epoch_num)
        if epochs:
            return epochs[-1]
        pths = sorted(path.glob("*.pth"), key=lambda p: p.stat().st_mtime)
        if pths:
            return pths[-1]
        raise FileNotFoundError(f"checkpoint directory has no .pth: {path}")
    raise FileNotFoundError(f"checkpoint not found: {path}")
