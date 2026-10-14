"""Geometry-bound finite engagement for telescoping and sliding interfaces.

An ideal or compliant kinematic joint can otherwise transmit force at any
separation.  A finite capture declaration binds the useful guide lengths to
specific authoritative STEP solids and requires a positive overlap throughout
the certified travel.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .models import MechanismValidationError, finite, identifier, vector3


FINITE_LINEAR_CAPTURE_SCHEMA = "finite_linear_capture_v1"


@dataclass(frozen=True)
class FiniteLinearCaptureSpec:
    capture_id: str
    parent_body_id: str
    child_body_id: str
    parent_anchor_abs_m: tuple[float, float, float]
    child_anchor_abs_m: tuple[float, float, float]
    axis_abs: tuple[float, float, float]
    parent_capture_solid_ids: tuple[str, ...]
    child_capture_solid_ids: tuple[str, ...]
    minimum_overlap_m: float
    maximum_separation_m: float
    schema_version: str = FINITE_LINEAR_CAPTURE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _solid_ids(value: Any, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
    ):
        raise MechanismValidationError(
            "FINITE_CAPTURE_SOLID_SET_INVALID",
            f"{label} must name at least one STEP solid",
        )
    result = tuple(str(item) for item in value)
    for index, solid_id in enumerate(result):
        identifier(solid_id, f"{label}[{index}]")
        if not solid_id.startswith("solid_"):
            raise MechanismValidationError(
                "FINITE_CAPTURE_SOLID_SET_INVALID",
                f"{label}[{index}] must be a solid_N identifier",
            )
    if len(set(result)) != len(result):
        raise MechanismValidationError(
            "FINITE_CAPTURE_SOLID_SET_INVALID",
            f"{label} contains duplicate STEP solids",
        )
    return result


def finite_linear_capture_from_dict(
    value: Mapping[str, Any],
    *,
    label: str = "finite_linear_capture",
) -> FiniteLinearCaptureSpec:
    if value.get("schema_version") != FINITE_LINEAR_CAPTURE_SCHEMA:
        raise MechanismValidationError(
            "FINITE_CAPTURE_SCHEMA_INVALID",
            f"{label} must use {FINITE_LINEAR_CAPTURE_SCHEMA}",
        )
    capture_id = str(value.get("capture_id") or "")
    parent = str(value.get("parent_body_id") or "")
    child = str(value.get("child_body_id") or "")
    identifier(capture_id, f"{label}.capture_id")
    identifier(parent, f"{label}.parent_body_id")
    identifier(child, f"{label}.child_body_id")
    if parent == child:
        raise MechanismValidationError(
            "FINITE_CAPTURE_BODY_INVALID",
            f"{label} cannot capture a body to itself",
        )
    parent_solids = _solid_ids(
        value.get("parent_capture_solid_ids"),
        f"{label}.parent_capture_solid_ids",
    )
    child_solids = _solid_ids(
        value.get("child_capture_solid_ids"),
        f"{label}.child_capture_solid_ids",
    )
    if set(parent_solids) & set(child_solids):
        raise MechanismValidationError(
            "FINITE_CAPTURE_SOLID_SET_INVALID",
            f"{label} assigns one STEP solid to both capture sides",
        )
    minimum_overlap = finite(
        value.get("minimum_overlap_m"),
        f"{label}.minimum_overlap_m",
    )
    maximum_separation = finite(
        value.get("maximum_separation_m"),
        f"{label}.maximum_separation_m",
    )
    if minimum_overlap <= 0.0 or maximum_separation <= minimum_overlap:
        raise MechanismValidationError(
            "FINITE_CAPTURE_RANGE_INVALID",
            f"{label} overlap and separation limits must be positive and ordered",
        )
    return FiniteLinearCaptureSpec(
        capture_id=capture_id,
        parent_body_id=parent,
        child_body_id=child,
        parent_anchor_abs_m=vector3(
            tuple(value.get("parent_anchor_abs_m") or ()),
            f"{label}.parent_anchor_abs_m",
            allow_zero=True,
        ),
        child_anchor_abs_m=vector3(
            tuple(value.get("child_anchor_abs_m") or ()),
            f"{label}.child_anchor_abs_m",
            allow_zero=True,
        ),
        axis_abs=vector3(
            tuple(value.get("axis_abs") or ()),
            f"{label}.axis_abs",
            unit=True,
            allow_zero=False,
        ),
        parent_capture_solid_ids=parent_solids,
        child_capture_solid_ids=child_solids,
        minimum_overlap_m=minimum_overlap,
        maximum_separation_m=maximum_separation,
    )


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _sub(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, float, float]:
    return tuple(float(a) - float(b) for a, b in zip(left, right))


def build_finite_linear_capture_report(
    declarations: Sequence[FiniteLinearCaptureSpec],
    *,
    raw_meshes: Mapping[int, Any],
    body_solid_indices: Mapping[str, Sequence[int]],
    package_binding_sha256: str,
    mesh_tolerance_m: float,
) -> dict[str, Any]:
    """Derive guide lengths from selected authoritative STEP solids."""

    captures: list[dict[str, Any]] = []
    for declaration in declarations:
        body_ids = set(body_solid_indices)
        if (
            declaration.parent_body_id not in body_ids
            or declaration.child_body_id not in body_ids
        ):
            raise MechanismValidationError(
                "FINITE_CAPTURE_BODY_INVALID",
                f"capture {declaration.capture_id!r} references an unknown body",
            )
        parent_allowed = {
            f"solid_{index}"
            for index in body_solid_indices[declaration.parent_body_id]
        }
        child_allowed = {
            f"solid_{index}"
            for index in body_solid_indices[declaration.child_body_id]
        }
        if not set(declaration.parent_capture_solid_ids).issubset(parent_allowed):
            raise MechanismValidationError(
                "FINITE_CAPTURE_SOLID_BODY_MISMATCH",
                f"capture {declaration.capture_id!r} parent solids do not belong to "
                f"{declaration.parent_body_id!r}",
            )
        if not set(declaration.child_capture_solid_ids).issubset(child_allowed):
            raise MechanismValidationError(
                "FINITE_CAPTURE_SOLID_BODY_MISMATCH",
                f"capture {declaration.capture_id!r} child solids do not belong to "
                f"{declaration.child_body_id!r}",
            )

        def vertices(solid_ids: Sequence[str]) -> list[Sequence[float]]:
            return [
                vertex
                for solid_id in solid_ids
                for vertex in raw_meshes[int(solid_id.removeprefix("solid_"))].vertices
            ]

        parent_length = max(
            _dot(
                _sub(vertex, declaration.parent_anchor_abs_m),
                declaration.axis_abs,
            )
            for vertex in vertices(declaration.parent_capture_solid_ids)
        )
        child_length = max(
            _dot(
                _sub(declaration.child_anchor_abs_m, vertex),
                declaration.axis_abs,
            )
            for vertex in vertices(declaration.child_capture_solid_ids)
        )
        anchor_delta = _sub(
            declaration.child_anchor_abs_m,
            declaration.parent_anchor_abs_m,
        )
        initial_axial_separation = _dot(anchor_delta, declaration.axis_abs)
        lateral = _sub(
            anchor_delta,
            tuple(item * initial_axial_separation for item in declaration.axis_abs),
        )
        initial_lateral_offset = math.sqrt(sum(item * item for item in lateral))
        initial_overlap = parent_length + child_length - initial_axial_separation
        supported_maximum = (
            parent_length
            + child_length
            - declaration.minimum_overlap_m
        )
        failures: list[str] = []
        if parent_length <= mesh_tolerance_m or child_length <= mesh_tolerance_m:
            failures.append("capture_geometry_has_no_positive_guide_length")
        if initial_axial_separation <= 0.0:
            failures.append("capture_initial_separation_not_positive")
        if initial_lateral_offset > mesh_tolerance_m:
            failures.append("capture_anchors_not_coaxial")
        if initial_overlap + mesh_tolerance_m < declaration.minimum_overlap_m:
            failures.append("capture_initial_overlap_below_minimum")
        if (
            declaration.maximum_separation_m
            > supported_maximum + mesh_tolerance_m
        ):
            failures.append("capture_declared_travel_exceeds_step_geometry")
        captures.append(
            {
                **declaration.to_dict(),
                "parent_capture_length_m": parent_length,
                "child_capture_length_m": child_length,
                "initial_axial_separation_m": initial_axial_separation,
                "initial_lateral_offset_m": initial_lateral_offset,
                "initial_overlap_m": initial_overlap,
                "maximum_supported_separation_m": supported_maximum,
                "passed": not failures,
                "failure_reasons": failures,
                "length_source": "authoritative_step_selected_solids_projection",
            }
        )
    passed = all(item["passed"] is True for item in captures)
    return {
        "schema_version": "finite_linear_capture_report_v1",
        "status": (
            "NOT_REQUIRED"
            if not captures
            else "PASS"
            if passed
            else "FAIL"
        ),
        "passed": passed,
        "required": bool(captures),
        "package_binding_sha256": package_binding_sha256,
        "capture_count": len(captures),
        "captures": captures,
        "failure_reasons": [
            f"{item['capture_id']}:{reason}"
            for item in captures
            for reason in item["failure_reasons"]
        ],
    }
