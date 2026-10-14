"""Formal input JSON Schemas for intake-facing documents.

Two documents cross the trust boundary from the authoring AI into the
Harness: the immutable external requirements file and the design package
manifest (``package.json``).  Their structural contracts are frozen as
JSON Schema files next to this module:

* ``requirements_input.v1.schema.json``
* ``design_package_input.v1.schema.json``

The schemas own the structural rules (required fields, types, constants,
enumerations).  Deep semantics — path safety, artifact hash verification,
prompt-clause binding, metric registry compatibility — stay in the intake
rules and the acceptance compiler, which remain the authority.  Both
layers run: a document must satisfy the schema *and* the code checks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema_check import SchemaCheckError, check_against_schema

REQUIREMENTS_INPUT_SCHEMA_ID = "simulation_ready_cad_requirements_input.v1"
DESIGN_PACKAGE_INPUT_SCHEMA_ID = "simulation_ready_cad_design_package_input.v1"

_REQUIREMENTS_SCHEMA_FILE = (
    Path(__file__).resolve().parent / "requirements_input.v1.schema.json"
)
_DESIGN_PACKAGE_SCHEMA_FILE = (
    Path(__file__).resolve().parent / "design_package_input.v1.schema.json"
)

_cache: dict[str, dict[str, Any]] = {}


def _schema(path: Path) -> dict[str, Any]:
    key = str(path)
    if key not in _cache:
        _cache[key] = json.loads(path.read_text(encoding="utf-8"))
    return _cache[key]


def requirements_input_schema() -> dict[str, Any]:
    return _schema(_REQUIREMENTS_SCHEMA_FILE)


def design_package_input_schema() -> dict[str, Any]:
    return _schema(_DESIGN_PACKAGE_SCHEMA_FILE)


def check_requirements_input(document: Any) -> None:
    """Raise :class:`SchemaCheckError` unless the requirements document is
    structurally valid."""

    check_against_schema(document, requirements_input_schema())


def check_design_package_manifest(document: Any) -> None:
    """Raise :class:`SchemaCheckError` unless the package manifest is
    structurally valid."""

    check_against_schema(document, design_package_input_schema())


__all__ = [
    "DESIGN_PACKAGE_INPUT_SCHEMA_ID",
    "REQUIREMENTS_INPUT_SCHEMA_ID",
    "SchemaCheckError",
    "check_design_package_manifest",
    "check_requirements_input",
    "design_package_input_schema",
    "requirements_input_schema",
]
