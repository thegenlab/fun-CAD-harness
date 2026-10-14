"""Minimal JSON Schema interpreter for the pipeline's frozen contracts.

The pinned validation environments deliberately carry no third-party
``jsonschema`` dependency, so contract files are validated by this
interpreter.  It supports exactly the keyword subset the contract schemas
use — ``type``, ``const``, ``enum``, ``required``, ``properties``,
``additionalProperties`` (boolean form), ``items``, ``allOf``, and
``if``/``then``/``else`` — and fails closed on any unknown constraint
keyword so a schema author cannot silently rely on an unenforced rule.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


_TYPE_CHECKS = {
    "object": lambda value: isinstance(value, dict),
    "array": lambda value: isinstance(value, list),
    "string": lambda value: isinstance(value, str),
    "boolean": lambda value: isinstance(value, bool),
    "null": lambda value: value is None,
    "integer": lambda value: isinstance(value, int)
    and not isinstance(value, bool),
    "number": lambda value: isinstance(value, (int, float))
    and not isinstance(value, bool),
}

_SUPPORTED_KEYWORDS = frozenset(
    {
        "$schema",
        "$id",
        "title",
        "description",
        "type",
        "const",
        "enum",
        "required",
        "properties",
        "additionalProperties",
        "items",
        "allOf",
        "if",
        "then",
        "else",
    }
)


class SchemaCheckError(ValueError):
    """The document does not satisfy the contract schema."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = list(errors)
        super().__init__("; ".join(self.errors) or "schema check failed")


def _type_matches(value: Any, declared: Any) -> bool:
    names = declared if isinstance(declared, list) else [declared]
    return any(
        name in _TYPE_CHECKS and _TYPE_CHECKS[name](value) for name in names
    )


def _check(
    value: Any,
    schema: Mapping[str, Any],
    *,
    path: str,
    errors: list[str],
) -> None:
    unknown = set(schema) - _SUPPORTED_KEYWORDS
    if unknown:
        errors.append(
            f"{path}: schema uses unsupported keywords: {sorted(unknown)}"
        )
        return
    if "type" in schema and not _type_matches(value, schema["type"]):
        errors.append(
            f"{path}: expected type {schema['type']}, "
            f"got {type(value).__name__}"
        )
        return
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected constant {schema['const']!r}")
        return
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value {value!r} is not one of the allowed set")
        return
    if isinstance(value, dict):
        for field in schema.get("required", ()):
            if field not in value:
                errors.append(f"{path}: missing required field {field!r}")
        properties = schema.get("properties", {})
        for field, subschema in properties.items():
            if field in value:
                _check(
                    value[field],
                    subschema,
                    path=f"{path}.{field}",
                    errors=errors,
                )
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(properties)
            if extra:
                errors.append(
                    f"{path}: unexpected fields {sorted(extra)}"
                )
    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            _check(
                item,
                schema["items"],
                path=f"{path}[{index}]",
                errors=errors,
            )
    for subschema in schema.get("allOf", ()):
        _check(value, subschema, path=path, errors=errors)
    if "if" in schema:
        condition: list[str] = []
        _check(value, schema["if"], path=path, errors=condition)
        branch = schema.get("then") if not condition else schema.get("else")
        if branch is not None:
            _check(value, branch, path=path, errors=errors)


def check_against_schema(value: Any, schema: Mapping[str, Any]) -> None:
    """Raise :class:`SchemaCheckError` unless ``value`` satisfies ``schema``."""

    errors: list[str] = []
    _check(value, schema, path="$", errors=errors)
    if errors:
        raise SchemaCheckError(errors[:16])
