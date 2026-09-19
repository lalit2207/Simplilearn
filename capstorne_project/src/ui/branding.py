"""Site logo helpers for the header and welcome screens."""

from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

from src.config import settings

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"


@lru_cache(maxsize=8)
def asset_data_uri(filename: str) -> str:
    raw = (ASSETS_DIR / filename).read_bytes()
    mime = "image/png" if raw[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


def logo_data_uri() -> str:
    return asset_data_uri("logo.png")


def brand_markup(size: str = "nav") -> str:
    """Logo plus app title. `size` is nav (header) or hero (welcome screens)."""
    title = settings.app_title
    inner = (
        f'<img class="app-logo" src="{logo_data_uri()}" alt="{title}" />'
        f"<span>{title}</span>"
    )
    if size == "hero":
        return f'<div class="brand-mark brand-mark--hero">{inner}</div>'
    return f'<div class="brand">{inner}</div>'
