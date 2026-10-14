"""Language of the text baked into rendered images and videos.

The browser UI switches language on its own (static/i18n.js), but pictures are
rendered once on the server, so their labels follow one deployment-wide
setting: ``SIMREADY_RENDER_LANG=en`` renders English labels, anything else
(the default) keeps the Chinese labels of the private deployment.
"""

from __future__ import annotations

import os


ENV_VAR = "SIMREADY_RENDER_LANG"


def render_lang() -> str:
    """Return ``"en"`` or ``"zh"`` for the current process."""
    value = os.environ.get(ENV_VAR, "").strip().lower()
    return "en" if value.startswith("en") else "zh"


def label(zh: str, en: str) -> str:
    """Pick the Chinese or English text of a rendered label."""
    return en if render_lang() == "en" else zh


def sans_serif_fonts() -> list[str]:
    """Matplotlib sans-serif fonts: CJK fonts first only when labels are Chinese."""
    fonts = ["DejaVu Sans"]
    if render_lang() == "zh":
        fonts = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", *fonts]
    return fonts
