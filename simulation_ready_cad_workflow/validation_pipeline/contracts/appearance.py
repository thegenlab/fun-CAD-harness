"""Closed appearance-authoring plan for fast CAD drafting and review."""

from __future__ import annotations

import math
from pathlib import PurePosixPath
import re
from typing import Any, Mapping


APPEARANCE_PLAN_SCHEMA = "simulation_ready_cad_appearance_plan_v1"
STATIC_VISUAL_GATE_VERSION = 2
STATIC_VISUAL_BASE_ROLES_V1 = (
    "front",
    "rear",
    "left",
    "right",
    "top",
    "bottom",
)
STATIC_VISUAL_BASE_ROLES_V2 = (*STATIC_VISUAL_BASE_ROLES_V1, "iso")

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_VIEW_KINDS = frozenset({"view", "detail", "section", "pose"})


class AppearancePlanError(ValueError):
    pass


def _text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise AppearancePlanError(f"{label} must be non-empty")
    return text


def _text_list(value: Any, label: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list):
        raise AppearancePlanError(f"{label} must be a list")
    result = [_text(item, f"{label}[]") for item in value]
    if not allow_empty and not result:
        raise AppearancePlanError(f"{label} must not be empty")
    return result


def _identifier(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if _ID.fullmatch(text) is None:
        raise AppearancePlanError(f"{label} must be a bounded identifier")
    return text


def _angle(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AppearancePlanError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise AppearancePlanError(f"{label} must be finite")
    return number


def _source_step(value: Any, label: str) -> str:
    text = _text(value, label)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or path.suffix.lower() not in {".step", ".stp"}
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise AppearancePlanError(
            f"{label} must be a package-relative STEP path"
        )
    return text


def validate_appearance_plan(value: Any) -> dict[str, Any]:
    """Validate and normalize the InnoCAD-derived appearance plan."""

    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "product_identity",
        "components",
        "view_requirements",
    }:
        raise AppearancePlanError(
            "appearance_plan must contain schema_version, product_identity, "
            "components, and view_requirements"
        )
    if value.get("schema_version") != APPEARANCE_PLAN_SCHEMA:
        raise AppearancePlanError("appearance_plan schema_version is unsupported")
    identity = value.get("product_identity")
    if not isinstance(identity, Mapping) or set(identity) != {
        "recognizable_as",
        "must_show",
        "must_not_look_like",
    }:
        raise AppearancePlanError("product_identity has an unsupported shape")
    normalized_identity = {
        "recognizable_as": _text(
            identity.get("recognizable_as"),
            "product_identity.recognizable_as",
        ),
        "must_show": _text_list(
            identity.get("must_show"), "product_identity.must_show"
        ),
        "must_not_look_like": _text_list(
            identity.get("must_not_look_like"),
            "product_identity.must_not_look_like",
        ),
    }

    raw_components = value.get("components")
    if not isinstance(raw_components, list) or not raw_components:
        raise AppearancePlanError("components must be a non-empty list")
    if len(raw_components) > 64:
        raise AppearancePlanError("components exceeds the 64-item limit")
    components: list[dict[str, Any]] = []
    component_ids: set[str] = set()
    component_fields = {
        "component_id",
        "name",
        "owner_rigid_group",
        "visible_features",
        "interfaces",
        "positive_checks",
        "negative_checks",
    }
    for index, raw in enumerate(raw_components):
        if not isinstance(raw, Mapping) or set(raw) != component_fields:
            raise AppearancePlanError(
                f"components[{index}] has an unsupported shape"
            )
        component_id = _identifier(
            raw.get("component_id"), f"components[{index}].component_id"
        )
        if component_id in component_ids:
            raise AppearancePlanError(f"duplicate component_id {component_id!r}")
        component_ids.add(component_id)
        components.append(
            {
                "component_id": component_id,
                "name": _text(raw.get("name"), f"components[{index}].name"),
                "owner_rigid_group": _identifier(
                    raw.get("owner_rigid_group"),
                    f"components[{index}].owner_rigid_group",
                ),
                "visible_features": _text_list(
                    raw.get("visible_features"),
                    f"components[{index}].visible_features",
                ),
                "interfaces": _text_list(
                    raw.get("interfaces"),
                    f"components[{index}].interfaces",
                    allow_empty=True,
                ),
                "positive_checks": _text_list(
                    raw.get("positive_checks"),
                    f"components[{index}].positive_checks",
                ),
                "negative_checks": _text_list(
                    raw.get("negative_checks"),
                    f"components[{index}].negative_checks",
                ),
            }
        )

    raw_views = value.get("view_requirements")
    if not isinstance(raw_views, list) or len(raw_views) > 16:
        raise AppearancePlanError("view_requirements must contain at most 16 views")
    views: list[dict[str, Any]] = []
    roles: set[str] = set(STATIC_VISUAL_BASE_ROLES_V2)
    for index, raw in enumerate(raw_views):
        if not isinstance(raw, Mapping):
            raise AppearancePlanError(
                f"view_requirements[{index}] must be an object"
            )
        role = _identifier(raw.get("role"), f"view_requirements[{index}].role")
        if role in roles:
            raise AppearancePlanError(f"duplicate or reserved view role {role!r}")
        roles.add(role)
        kind = str(raw.get("kind") or "").strip().lower()
        if kind not in _VIEW_KINDS:
            raise AppearancePlanError(
                f"view_requirements[{index}].kind is unsupported"
            )
        expected = {"role", "kind", "purpose", "elevation_deg", "azimuth_deg"}
        if kind == "pose":
            expected.add("source_step")
        elif "source_step" in raw:
            expected.add("source_step")
        if kind == "section":
            expected.update(
                {"section_axis", "section_position_mm", "section_keep"}
            )
        if set(raw) != expected:
            raise AppearancePlanError(
                f"view_requirements[{index}] has an unsupported shape"
            )
        normalized: dict[str, Any] = {
            "role": role,
            "kind": kind,
            "purpose": _text(
                raw.get("purpose"), f"view_requirements[{index}].purpose"
            ),
            "elevation_deg": _angle(
                raw.get("elevation_deg"),
                f"view_requirements[{index}].elevation_deg",
            ),
            "azimuth_deg": _angle(
                raw.get("azimuth_deg"),
                f"view_requirements[{index}].azimuth_deg",
            ),
        }
        if "source_step" in raw:
            normalized["source_step"] = _source_step(
                raw.get("source_step"),
                f"view_requirements[{index}].source_step",
            )
        if kind == "section":
            axis = str(raw.get("section_axis") or "").lower()
            keep = str(raw.get("section_keep") or "").lower()
            if axis not in {"x", "y", "z"} or keep not in {"below", "above"}:
                raise AppearancePlanError(
                    f"view_requirements[{index}] section axis/keep is invalid"
                )
            normalized.update(
                {
                    "section_axis": axis,
                    "section_position_mm": _angle(
                        raw.get("section_position_mm"),
                        f"view_requirements[{index}].section_position_mm",
                    ),
                    "section_keep": keep,
                }
            )
        views.append(normalized)
    return {
        "schema_version": APPEARANCE_PLAN_SCHEMA,
        "product_identity": normalized_identity,
        "components": components,
        "view_requirements": views,
    }


def required_visual_roles(
    design_input: Mapping[str, Any], *, gate_version: int
) -> tuple[str, ...]:
    """Return the ordered evidence roles required by one gate version."""

    base = (
        STATIC_VISUAL_BASE_ROLES_V2
        if gate_version == STATIC_VISUAL_GATE_VERSION
        else STATIC_VISUAL_BASE_ROLES_V1
    )
    extra: list[str] = []
    if gate_version == STATIC_VISUAL_GATE_VERSION:
        plan = validate_appearance_plan(design_input.get("appearance_plan"))
        extra.extend(item["role"] for item in plan["view_requirements"])
    for raw in design_input.get("motion_review_roles") or ():
        role = str(raw or "").strip()
        if role and role not in extra and role not in base:
            extra.append(role)
    return (*base, *extra)


__all__ = [
    "APPEARANCE_PLAN_SCHEMA",
    "AppearancePlanError",
    "STATIC_VISUAL_BASE_ROLES_V1",
    "STATIC_VISUAL_BASE_ROLES_V2",
    "STATIC_VISUAL_GATE_VERSION",
    "required_visual_roles",
    "validate_appearance_plan",
]
