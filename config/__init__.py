"""Load project external dependency config from ``config/conf.json``."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

CONFIG_DIR = Path(__file__).resolve().parent
CONF_PATH = CONFIG_DIR / "conf.json"
ROOT_DIR = CONFIG_DIR.parent


@lru_cache(maxsize=1)
def load_conf(path: str | None = None) -> Dict[str, Any]:
    conf_path = Path(path).expanduser().resolve() if path else CONF_PATH
    with conf_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config must be a JSON object: {conf_path}")
    return data


def get_sam3_conf() -> Dict[str, Any]:
    return dict(load_conf().get("sam3") or {})


def get_genpose2_conf() -> Dict[str, Any]:
    return dict(load_conf().get("genpose2") or {})


def resolve_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (ROOT_DIR / path).resolve()
    else:
        path = path.resolve()
    return path
