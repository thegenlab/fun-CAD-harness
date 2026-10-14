from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .io import relative_to_root, sha256_file, write_json_atomic
from .models import MechanismValidationError


RIGID_ATTACHMENT_PROOF_SCHEMA = "rigid_attachment_proof_v1"
_ABSOLUTE_MIN_OVERLAP_VOLUME_MM3 = 1e-6
_RELATIVE_MIN_OVERLAP_VOLUME = 1e-6
_MAX_BOOLEAN_CANDIDATE_PAIRS = 16_384


def _fail(code: str, message: str) -> MechanismValidationError:
    return MechanismValidationError(code, message)


def _shape_volume_mm3(shape: Any) -> float:
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, properties)
    return abs(float(properties.Mass()))


def _shape_bounds_mm(shape: Any, solid_id: str) -> tuple[float, ...]:
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    if box.IsVoid():
        raise _fail(
            "RIGID_ATTACHMENT_BOUNDS_FAILED",
            f"cannot bound {solid_id} for rigid-attachment proof",
        )
    return tuple(float(value) for value in box.Get())


def _boxes_overlap(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(
        float(left[axis + 3]) >= float(right[axis])
        and float(right[axis + 3]) >= float(left[axis])
        for axis in range(3)
    )


def _connected_components(
    indices: Sequence[int],
    adjacency: Mapping[int, set[int]],
) -> list[list[int]]:
    components: list[list[int]] = []
    remaining = set(indices)
    while remaining:
        component: list[int] = []
        pending = [min(remaining)]
        while pending:
            index = pending.pop()
            if index not in remaining:
                continue
            remaining.remove(index)
            component.append(index)
            pending.extend(adjacency[index] & remaining)
        components.append(sorted(component))
    return components


def build_rigid_attachment_proof(
    solids: Sequence[Any],
    body_indices: Mapping[str, Sequence[int]],
    *,
    package_binding_sha256: str,
    body_group_ids: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Prove conservative STEP-only rigid-body membership.

    A proof edge requires validator-measured positive common material volume.
    Surface contact, distance, friction, gravity, and authored prose never add
    an edge.
    """

    normalized: dict[str, tuple[int, ...]] = {}
    claimed_by: dict[int, str] = {}
    for body_id, members in body_indices.items():
        if not isinstance(body_id, str) or not body_id:
            raise _fail(
                "RIGID_ATTACHMENT_BODY_ID_INVALID",
                "rigid-attachment body IDs must be non-empty strings",
            )
        indices = tuple(sorted({int(index) for index in members}))
        if not indices:
            raise _fail(
                "RIGID_ATTACHMENT_BODY_EMPTY",
                f"{body_id!r} has no canonical STEP solids",
            )
        for index in indices:
            if index < 1 or index > len(solids):
                raise _fail(
                    "RIGID_ATTACHMENT_SOLID_OUT_OF_RANGE",
                    f"solid_{index} is outside the imported STEP inventory",
                )
            previous = claimed_by.get(index)
            if previous is not None and previous != body_id:
                raise _fail(
                    "RIGID_ATTACHMENT_SOLID_BODY_OVERLAP",
                    f"solid_{index} is assigned to both {previous!r} and {body_id!r}",
                )
            claimed_by[index] = body_id
        normalized[body_id] = indices

    selected = sorted(claimed_by)
    # A one-solid rigid body is already one connected graph node.  Volume
    # and bounds are used only to screen and measure pairwise proof edges, so
    # evaluating them for singleton bodies adds no evidence and can be very
    # expensive for a complex fused STEP solid.
    pairwise_indices = sorted(
        {
            index
            for indices in normalized.values()
            if len(indices) > 1
            for index in indices
        }
    )
    volumes = {
        index: _shape_volume_mm3(solids[index - 1])
        for index in pairwise_indices
    }
    bounds = {
        index: _shape_bounds_mm(solids[index - 1], f"solid_{index}")
        for index in pairwise_indices
    }

    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common

    candidate_count = 0
    body_reports: list[dict[str, Any]] = []
    failure_reasons: list[str] = []
    for body_id, indices in normalized.items():
        adjacency = {index: set() for index in indices}
        edges: list[dict[str, Any]] = []
        for left_position, left_index in enumerate(indices):
            for right_index in indices[left_position + 1 :]:
                if not _boxes_overlap(bounds[left_index], bounds[right_index]):
                    continue
                candidate_count += 1
                if candidate_count > _MAX_BOOLEAN_CANDIDATE_PAIRS:
                    raise _fail(
                        "RIGID_ATTACHMENT_PROOF_TOO_LARGE",
                        "rigid-attachment Boolean candidate budget exceeded",
                    )
                common = BRepAlgoAPI_Common(
                    solids[left_index - 1],
                    solids[right_index - 1],
                )
                common.Build()
                if not common.IsDone():
                    raise _fail(
                        "RIGID_ATTACHMENT_BOOLEAN_FAILED",
                        f"OCCT Boolean common failed for solid_{left_index} and "
                        f"solid_{right_index}",
                    )
                common_volume = _shape_volume_mm3(common.Shape())
                minimum_volume = max(
                    _ABSOLUTE_MIN_OVERLAP_VOLUME_MM3,
                    min(volumes[left_index], volumes[right_index])
                    * _RELATIVE_MIN_OVERLAP_VOLUME,
                )
                if common_volume <= minimum_volume:
                    continue
                adjacency[left_index].add(right_index)
                adjacency[right_index].add(left_index)
                edges.append(
                    {
                        "solid_pair": [
                            f"solid_{left_index}",
                            f"solid_{right_index}",
                        ],
                        "proof_type": "POSITIVE_VOLUME_MATERIAL_CONNECTION",
                        "common_volume_mm3": common_volume,
                        "minimum_required_common_volume_mm3": minimum_volume,
                        "constraint_rank": 6,
                        "remaining_relative_dof": 0,
                    }
                )

        components = _connected_components(indices, adjacency)
        passed = len(components) == 1
        component_ids = [
            [f"solid_{index}" for index in component]
            for component in components
        ]
        if not passed:
            failure_reasons.append(
                f"{body_id} has {len(components)} material-disconnected "
                f"components: "
                + " | ".join(",".join(component) for component in component_ids)
            )
        body_reports.append(
            {
                "body_id": body_id,
                "declared_group_ids": list(
                    (body_group_ids or {}).get(body_id, ())
                ),
                "source_solid_ids": [
                    f"solid_{index}" for index in indices
                ],
                "solid_count": len(indices),
                "proven_edge_count": len(edges),
                "proven_fixed_edges": edges,
                "connected_components": component_ids,
                "passed": passed,
            }
        )

    passed = not failure_reasons
    return {
        "schema_version": RIGID_ATTACHMENT_PROOF_SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "package_binding_sha256": package_binding_sha256,
        "policy": {
            "proof_class": "STEP_POSITIVE_VOLUME_MATERIAL_PATH",
            "absolute_minimum_overlap_volume_mm3": (
                _ABSOLUTE_MIN_OVERLAP_VOLUME_MM3
            ),
            "relative_minimum_overlap_volume": (
                _RELATIVE_MIN_OVERLAP_VOLUME
            ),
            "maximum_boolean_candidate_pairs": (
                _MAX_BOOLEAN_CANDIDATE_PAIRS
            ),
            "surface_contact_creates_fixed_edge": False,
            "distance_proximity_creates_fixed_edge": False,
            "friction_or_gravity_creates_fixed_edge": False,
            "text_connection_evidence_used_for_verdict": False,
            "ai_authored_pass_record_used": False,
            "unknown_attachment_fails_closed": True,
        },
        "body_count": len(body_reports),
        "solid_count": len(selected),
        "boolean_candidate_pair_count": candidate_count,
        "bodies": body_reports,
        "failure_reasons": failure_reasons,
    }


def materialize_rigid_attachment_proof(
    solids: Sequence[Any],
    body_indices: Mapping[str, Sequence[int]],
    *,
    package_binding_sha256: str,
    workspace_root: str | Path,
    destination: str | Path,
    body_group_ids: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    workspace = Path(workspace_root).resolve()
    path = Path(destination).resolve()
    try:
        path.relative_to(workspace)
    except ValueError as exc:
        raise _fail(
            "RIGID_ATTACHMENT_REPORT_PATH_UNSAFE",
            "rigid-attachment proof must stay inside the workspace",
        ) from exc

    report = build_rigid_attachment_proof(
        solids,
        body_indices,
        package_binding_sha256=package_binding_sha256,
        body_group_ids=body_group_ids,
    )
    write_json_atomic(path, report)
    binding = {
        "schema_version": report["schema_version"],
        "status": report["status"],
        "passed": report["passed"],
        "report": relative_to_root(workspace, path),
        "report_sha256": sha256_file(path),
        "package_binding_sha256": package_binding_sha256,
    }
    if not report["passed"]:
        detail = "; ".join(str(item) for item in report["failure_reasons"])
        raise _fail(
            "RIGID_ATTACHMENT_PROOF_FAILED",
            "declared rigid bodies lack a validator-proven six-DOF fixed "
            f"material path: {detail}",
        )
    return binding


def verify_rigid_attachment_binding(
    binding: Mapping[str, Any] | None,
    *,
    workspace_root: str | Path,
    package_binding_sha256: str,
) -> dict[str, Any]:
    """Verify immutable rigid-membership evidence before physical execution."""

    from .io import ensure_inside, read_json

    if not isinstance(binding, Mapping):
        raise _fail(
            "RIGID_ATTACHMENT_EVIDENCE_REQUIRED",
            "physical execution requires a rigid-attachment proof binding",
        )
    relative = binding.get("report")
    expected_hash = binding.get("report_sha256")
    if not isinstance(relative, str) or not isinstance(expected_hash, str):
        raise _fail(
            "RIGID_ATTACHMENT_BINDING_INVALID",
            "rigid-attachment report path or hash is missing",
        )
    workspace = Path(workspace_root).resolve()
    path = ensure_inside(workspace, relative)
    if (
        not path.is_file()
        or path.is_symlink()
        or sha256_file(path) != expected_hash
    ):
        raise _fail(
            "RIGID_ATTACHMENT_REPORT_CHANGED",
            "rigid-attachment evidence is missing or no longer hash-bound",
        )
    report = read_json(path)
    if (
        report.get("schema_version") != RIGID_ATTACHMENT_PROOF_SCHEMA
        or report.get("passed") is not True
        or report.get("status") != "PASS"
        or report.get("package_binding_sha256") != package_binding_sha256
        or binding.get("package_binding_sha256") != package_binding_sha256
        or not isinstance(report.get("bodies"), list)
        or not report["bodies"]
        or any(body.get("passed") is not True for body in report["bodies"])
    ):
        raise _fail(
            "RIGID_ATTACHMENT_REPORT_INVALID",
            "rigid-attachment evidence cannot authorize physical execution",
        )
    return report
