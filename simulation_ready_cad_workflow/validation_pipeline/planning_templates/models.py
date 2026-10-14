"""Immutable public records for the built-in planning-template library."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


def _detached(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


@dataclass(frozen=True)
class TemplateRecord:
    kind: str
    template_id: str
    version: str
    path: str
    sha256: str
    _payload: dict[str, Any]

    def payload(self) -> dict[str, Any]:
        """Return a detached copy; callers cannot mutate the loaded record."""

        return _detached(self._payload)

    def reference(self) -> dict[str, str]:
        return {
            "template_id": self.template_id,
            "version": self.version,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class TemplateCatalog:
    catalog_id: str
    catalog_version: str
    catalog_sha256: str
    records: tuple[TemplateRecord, ...]

    def records_for(self, kind: str) -> tuple[TemplateRecord, ...]:
        return tuple(item for item in self.records if item.kind == kind)

    def resolve(self, kind: str, template_id: str, version: str) -> TemplateRecord:
        matches = tuple(
            item
            for item in self.records
            if item.kind == kind
            and item.template_id == template_id
            and item.version == version
        )
        if len(matches) != 1:
            raise KeyError(f"unknown {kind} template {template_id}@{version}")
        return matches[0]
