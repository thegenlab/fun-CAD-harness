"""Public dashboard URLs when the server is mounted below a URL prefix."""

from typing import Any


def prefix_url_fields(value: Any, base_path: str, field: str = "") -> Any:
    """Prefix generated API links without changing saved paths or source text."""
    if not base_path:
        return value
    if isinstance(value, dict):
        return {key: prefix_url_fields(item, base_path, key) for key, item in value.items()}
    if isinstance(value, list):
        return [prefix_url_fields(item, base_path, field) for item in value]
    is_url = field in {"url", "urls", "href", "src", "image", "poster"} or field.endswith(("_url", "_urls"))
    if is_url and isinstance(value, str) and value.startswith("/api/"):
        return base_path + value
    return value
