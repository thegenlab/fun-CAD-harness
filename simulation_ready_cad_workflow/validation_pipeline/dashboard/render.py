from __future__ import annotations

import json
from pathlib import Path
from typing import Any


STATIC_ROOT = Path(__file__).with_name("static")


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )


def render_live_shell(
    base_path: str = "", features: dict[str, bool] | None = None, page: str = "index.html"
) -> str:
    template = (STATIC_ROOT / page).read_text(encoding="utf-8")
    template = template.replace('="/static/', f'="{base_path}/static/')
    boot = {
        "mode": "live",
        "apiBase": f"{base_path}/api",
        "basePath": base_path,
        "features": features or {"maintenance": False, "team_library": False, "launch": False},
    }
    return template.replace(
        "<!-- QUEST_BOOT -->",
        f"<script>window.QUEST_BOOT={_safe_json(boot)};</script>",
    )


def _inline_script(source: str) -> str:
    # Only "</script" can end an inline script early; "<\/" means the same in JS strings.
    return "<script>\n" + source.replace("</script", "<\\/script") + "\n</script>"


def render_snapshot(data: dict[str, Any]) -> str:
    template = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    css = (STATIC_ROOT / "app.css").read_text(encoding="utf-8")
    # The language switch runs in the head; the English catalog goes first so
    # an offline snapshot translates without fetching anything.
    i18n_runtime = _inline_script(
        (STATIC_ROOT / "i18n" / "en.js").read_text(encoding="utf-8")
    ) + "\n" + _inline_script((STATIC_ROOT / "i18n.js").read_text(encoding="utf-8"))
    viewer_js = (STATIC_ROOT / "viewer3d.js").read_text(encoding="utf-8")
    js = (STATIC_ROOT / "workbench.js").read_text(encoding="utf-8") + "\n" + (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    template = template.replace(
        '<link rel="stylesheet" href="/static/app.css">',
        f"<style>\n{css}\n</style>",
    )
    # Classic inline scripts ignore the `defer` attribute. Keep the boot payload
    # in the head, remove the two external placeholders, and append their inline
    # bodies after #app so a standalone snapshot cannot initialize against a null
    # root element.
    template = template.replace(
        '<script src="/static/viewer3d.js" defer></script>', ""
    )
    template = template.replace('<script src="/static/app.js" defer></script>', "")
    template = template.replace('<script src="/static/i18n.js"></script>', i18n_runtime)
    template = template.replace('<script src="/static/workbench.js" defer></script>', "")
    template = template.replace(
        "<!-- QUEST_BOOT -->",
        f"<script>window.QUEST_BOOT={{mode:'snapshot',data:{_safe_json(data)}}};</script>",
    )
    inline_runtime = (
        f"<script>\n{viewer_js}\n</script>\n"
        f"<script>\n{js}\n</script>\n"
    )
    return template.replace("</body>", f"{inline_runtime}</body>")
