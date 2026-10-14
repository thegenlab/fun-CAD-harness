"""Deterministic STEP-to-joint realization proofs before physical execution."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io import ensure_inside, read_json, sha256_file
from .models import MechanismValidationError


REPORT_SCHEMA = "simulation_ready_cad_joint_realization_report_v1"
SUPPORTED_PROOF_KINDS = frozenset({"revolute.pin_bore.v1"})
MAXIMUM_AXIS_OFFSET_M = 0.001
MAXIMUM_AXIS_ANGLE_ERROR_RAD = 0.01
MAXIMUM_RADIAL_CLEARANCE_M = 0.0005
MINIMUM_AXIAL_OVERLAP_M = 0.001


def _proof_policy() -> dict[str, float]:
    return {
        "maximum_axis_offset_m": MAXIMUM_AXIS_OFFSET_M,
        "maximum_axis_angle_error_rad": MAXIMUM_AXIS_ANGLE_ERROR_RAD,
        "maximum_radial_clearance_m": MAXIMUM_RADIAL_CLEARANCE_M,
        "minimum_axial_overlap_m": MINIMUM_AXIAL_OVERLAP_M,
    }


@dataclass(frozen=True)
class RevoluteProofRequirement:
    joint_id: str
    proof_kind: str
    parent_body_id: str
    child_body_id: str
    origin_abs_m: tuple[float, float, float]
    axis_abs: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint_id": self.joint_id,
            "proof_kind": self.proof_kind,
            "parent_body_id": self.parent_body_id,
            "child_body_id": self.child_body_id,
            "origin_abs_m": list(self.origin_abs_m),
            "axis_abs": list(self.axis_abs),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RevoluteProofRequirement":
        allowed = {
            "joint_id",
            "proof_kind",
            "parent_body_id",
            "child_body_id",
            "origin_abs_m",
            "axis_abs",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise MechanismValidationError(
                "JOINT_REALIZATION_REQUIREMENT_INVALID",
                "geometry proof requirements cannot define Harness policy "
                "fields: " + ", ".join(unknown),
            )
        return cls(
            joint_id=str(value["joint_id"]),
            proof_kind=str(value["proof_kind"]),
            parent_body_id=str(value["parent_body_id"]),
            child_body_id=str(value["child_body_id"]),
            origin_abs_m=tuple(float(item) for item in value["origin_abs_m"]),
            axis_abs=tuple(float(item) for item in value["axis_abs"]),
        )


@dataclass(frozen=True)
class _PackageFamilySpecRecord:
    capsule_id: str
    capsule_family: str | None
    family_spec: dict[str, Any]

    @property
    def has_revolute_claim(self) -> bool:
        joints = self.family_spec.get("revolute_joints")
        return self.capsule_family == "revolute" or bool(joints)


def _unit(value: Sequence[float], *, label: str) -> tuple[float, float, float]:
    length = math.sqrt(sum(float(item) ** 2 for item in value))
    if length <= 1.0e-12:
        raise MechanismValidationError(
            "JOINT_REALIZATION_REQUIREMENT_INVALID",
            f"{label} must be nonzero",
        )
    return tuple(float(item) / length for item in value)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _sub(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, float, float]:
    return tuple(float(a) - float(b) for a, b in zip(left, right))


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(_dot(value, value))


def _load_step_solids(step_path: Path) -> list[Any]:
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader

    from .occt_collision_fidelity import shape_solids

    reader = STEPControl_Reader()
    if (
        reader.ReadFile(str(step_path)) != IFSelect_RetDone
        or reader.TransferRoots() <= 0
    ):
        raise MechanismValidationError(
            "JOINT_REALIZATION_STEP_IMPORT_FAILED",
            "OCCT could not import the frozen assembly STEP",
        )
    solids = shape_solids(reader.OneShape())
    if not solids:
        raise MechanismValidationError(
            "JOINT_REALIZATION_STEP_EMPTY",
            "the frozen assembly STEP contains no solids",
        )
    return solids


def step_solid_geometry_sha256s(step_path: str | Path) -> tuple[str, ...]:
    """Return ordered OCCT BRep identities for one assembly STEP."""

    from .reuse import solid_geometry_sha256

    return tuple(
        solid_geometry_sha256(solid)
        for solid in _load_step_solids(Path(step_path).resolve())
    )


def _cylindrical_features(shape: Any) -> list[dict[str, Any]]:
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    features: list[dict[str, Any]] = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_index = 0
    while explorer.More():
        face_index += 1
        face = TopoDS.Face_s(explorer.Current())
        surface = BRepAdaptor_Surface(face)
        if surface.GetType() == GeomAbs_Cylinder:
            cylinder = surface.Cylinder()
            axis = cylinder.Axis()
            location = axis.Location()
            direction = axis.Direction()
            feature_axis = _unit(
                (
                    float(direction.X()),
                    float(direction.Y()),
                    float(direction.Z()),
                ),
                label="cylindrical feature axis",
            )
            first_u = float(surface.FirstUParameter())
            last_u = float(surface.LastUParameter())
            first_v = float(surface.FirstVParameter())
            last_v = float(surface.LastVParameter())
            middle_u = 0.5 * (first_u + last_u)
            axial_endpoints_abs_m: list[tuple[float, float, float]] = []
            for parameter in (first_v, last_v):
                point = surface.Value(middle_u, parameter)
                point_m = (
                    float(point.X()) * 1.0e-3,
                    float(point.Y()) * 1.0e-3,
                    float(point.Z()) * 1.0e-3,
                )
                axial_endpoints_abs_m.append(point_m)
            features.append(
                {
                    "face_index": face_index,
                    "origin_abs_m": (
                        float(location.X()) * 1.0e-3,
                        float(location.Y()) * 1.0e-3,
                        float(location.Z()) * 1.0e-3,
                    ),
                    "axis_abs": feature_axis,
                    "radius_m": float(cylinder.Radius()) * 1.0e-3,
                    "surface_role": (
                        "bore"
                        if face.Orientation() == TopAbs_REVERSED
                        else "pin"
                    ),
                    "axial_endpoints_abs_m": tuple(axial_endpoints_abs_m),
                }
            )
        explorer.Next()
    return features


def _axis_match(
    feature: Mapping[str, Any],
    requirement: RevoluteProofRequirement,
) -> tuple[float, float, tuple[float, float, float]]:
    declared_axis = _unit(requirement.axis_abs, label="declared joint axis")
    feature_axis = _unit(feature["axis_abs"], label="feature axis")
    alignment = _dot(declared_axis, feature_axis)
    angle_error = math.acos(max(-1.0, min(1.0, abs(alignment))))
    origin_to_joint = _sub(
        requirement.origin_abs_m,
        feature["origin_abs_m"],
    )
    axial = _dot(origin_to_joint, feature_axis)
    axis_anchor = tuple(
        float(feature["origin_abs_m"][index])
        + axial * feature_axis[index]
        for index in range(3)
    )
    offset_m = _norm(_sub(requirement.origin_abs_m, axis_anchor))
    return angle_error, offset_m, axis_anchor


def _matching_cylinders(
    features: Sequence[Mapping[str, Any]],
    requirement: RevoluteProofRequirement,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    declared_axis = _unit(requirement.axis_abs, label="declared joint axis")
    for feature in features:
        angle_error, offset_m, axis_anchor = _axis_match(feature, requirement)
        if (
            angle_error <= MAXIMUM_AXIS_ANGLE_ERROR_RAD
            and offset_m <= MAXIMUM_AXIS_OFFSET_M
        ):
            axial_coordinates = [
                _dot(endpoint, declared_axis)
                for endpoint in feature["axial_endpoints_abs_m"]
            ]
            matches.append(
                {
                    **dict(feature),
                    "axis_angle_error_rad": angle_error,
                    "axis_offset_m": offset_m,
                    "axis_anchor_abs_m": axis_anchor,
                    "axial_interval_abs_m": (
                        min(axial_coordinates),
                        max(axial_coordinates),
                    ),
                }
            )
    return matches


def _missing_support_failure(
    requirement: RevoluteProofRequirement,
    *,
    body_id: str,
    side: str,
    bound_solid_ids: Sequence[str],
    bound_solid_cylindrical_feature_counts: Mapping[str, int],
    step_solid_count: int,
) -> dict[str, Any]:
    cylindrical_feature_count = sum(
        bound_solid_cylindrical_feature_counts.values()
    )
    required_change = (
        f"Add a finite cylindrical pin or bearing support for "
        f"{requirement.joint_id} to {body_id}, coaxial with the declared "
        "joint axis, then regenerate assembly.step."
    )
    if not bound_solid_ids:
        required_change = (
            f"Bind {body_id} to the STEP solids that realize "
            f"{requirement.joint_id}: assembly_contract.json binds no solid "
            "to this body. Add the carrying solids to its rigid group's "
            "source_solid_ids, add a finite cylindrical pin or bearing "
            "support coaxial with the declared joint axis, then regenerate "
            "assembly.step."
        )
    elif cylindrical_feature_count == 0:
        required_change += (
            " The solids bound to this body contain no cylindrical feature "
            "at all; if the pin or bore geometry exists in other STEP "
            "solids, update the body-to-solid binding (source_solid_ids) in "
            "assembly_contract.json instead of reshaping geometry."
        )
    else:
        required_change += (
            " The bound solids do contain cylindrical features, but none "
            "is coaxial with the declared joint axis; realign the support "
            "or add a coaxial one."
        )
    return {
        "code": f"JOINT_{side.upper()}_SUPPORT_MISSING",
        "joint_id": requirement.joint_id,
        "proof_kind": requirement.proof_kind,
        "body_id": body_id,
        "target_file": "assembly.step",
        "measured": {
            "cylindrical_feature_count": cylindrical_feature_count,
            "matching_axis_feature_count": 0,
            "bound_solid_ids": list(bound_solid_ids),
            "bound_solid_cylindrical_feature_counts": dict(
                bound_solid_cylindrical_feature_counts
            ),
            "step_solid_count": step_solid_count,
        },
        "required_change": required_change,
    }


def _axis_point_at_declared_coordinate(
    feature: Mapping[str, Any],
    declared_axis: Sequence[float],
    coordinate_m: float,
) -> tuple[float, float, float]:
    axis = _unit(feature["axis_abs"], label="feature axis")
    declared = _unit(declared_axis, label="declared joint axis")
    denominator = _dot(axis, declared)
    if abs(denominator) <= 1.0e-12:
        raise MechanismValidationError(
            "JOINT_REALIZATION_REQUIREMENT_INVALID",
            "a matched cylindrical feature is perpendicular to the joint axis",
        )
    origin = tuple(float(item) for item in feature["origin_abs_m"])
    distance = (coordinate_m - _dot(origin, declared)) / denominator
    return tuple(
        origin[index] + distance * axis[index] for index in range(3)
    )


def _paired_measurements(
    parent_matches: Sequence[Mapping[str, Any]],
    child_matches: Sequence[Mapping[str, Any]],
    *,
    declared_axis: Sequence[float],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    axis = _unit(declared_axis, label="declared joint axis")
    for parent in parent_matches:
        for child in child_matches:
            if parent.get("surface_role") == child.get("surface_role"):
                continue
            pin = parent if parent.get("surface_role") == "pin" else child
            bore = parent if parent.get("surface_role") == "bore" else child
            radial_clearance_m = (
                float(bore["radius_m"]) - float(pin["radius_m"])
            )
            parent_interval = parent["axial_interval_abs_m"]
            child_interval = child["axial_interval_abs_m"]
            overlap_start_m = max(
                float(parent_interval[0]), float(child_interval[0])
            )
            overlap_end_m = min(
                float(parent_interval[1]), float(child_interval[1])
            )
            axial_overlap_m = max(0.0, overlap_end_m - overlap_start_m)
            sample_coordinates = (
                (overlap_start_m, overlap_end_m)
                if axial_overlap_m > 0.0
                else (0.5 * (overlap_start_m + overlap_end_m),)
            )
            separation_samples: list[dict[str, float]] = []
            for coordinate_m in sample_coordinates:
                parent_point = _axis_point_at_declared_coordinate(
                    parent, axis, coordinate_m
                )
                child_point = _axis_point_at_declared_coordinate(
                    child, axis, coordinate_m
                )
                separation_samples.append(
                    {
                        "declared_axis_coordinate_m": coordinate_m,
                        "axis_separation_m": _norm(
                            _sub(parent_point, child_point)
                        ),
                    }
                )
            axis_separation_m = max(
                item["axis_separation_m"] for item in separation_samples
            )
            pair_alignment = abs(
                _dot(parent["axis_abs"], child["axis_abs"])
            )
            pair_axis_angle_error_rad = math.acos(
                max(-1.0, min(1.0, pair_alignment))
            )
            candidates.append(
                {
                    "parent_feature": dict(parent),
                    "child_feature": dict(child),
                    "radial_clearance_m": radial_clearance_m,
                    "axis_separation_m": axis_separation_m,
                    "axis_separation_samples": separation_samples,
                    "pair_axis_angle_error_rad": pair_axis_angle_error_rad,
                    "axial_overlap_m": axial_overlap_m,
                }
            )
    return sorted(
        candidates,
        key=lambda item: (
            str(item["parent_feature"].get("solid_id") or ""),
            int(item["parent_feature"].get("face_index") or 0),
            str(item["child_feature"].get("solid_id") or ""),
            int(item["child_feature"].get("face_index") or 0),
        ),
    )


def _pair_failure(
    requirement: RevoluteProofRequirement,
    *,
    code: str,
    measured: Mapping[str, Any],
    required_change: str,
) -> dict[str, Any]:
    return {
        "code": code,
        "joint_id": requirement.joint_id,
        "proof_kind": requirement.proof_kind,
        "target_file": "assembly.step",
        "measured": dict(measured),
        "required_change": required_change,
    }


def _measurement_failures(
    requirement: RevoluteProofRequirement,
    measured: Mapping[str, Any],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    radial_clearance_m = float(measured["radial_clearance_m"])
    axis_separation_m = float(measured["axis_separation_m"])
    pair_axis_angle_error_rad = float(
        measured["pair_axis_angle_error_rad"]
    )
    if radial_clearance_m >= 0.0 and (
        axis_separation_m > radial_clearance_m + 1.0e-12
        or pair_axis_angle_error_rad > MAXIMUM_AXIS_ANGLE_ERROR_RAD
    ):
        failures.append(
            _pair_failure(
                requirement,
                code="JOINT_AXIS_PAIR_MISALIGNED",
                measured={
                    "axis_separation_m": axis_separation_m,
                    "available_radial_clearance_m": radial_clearance_m,
                    "pair_axis_angle_error_rad": pair_axis_angle_error_rad,
                    "maximum_axis_angle_error_rad": (
                        MAXIMUM_AXIS_ANGLE_ERROR_RAD
                    ),
                },
                required_change=(
                    f"Make the pin and bore centerlines for "
                    f"{requirement.joint_id} mutually coaxial; their relative "
                    "offset must fit inside the available radial clearance and "
                    "their angular mismatch must stay within the Harness limit."
                ),
            )
        )
    if radial_clearance_m < -1.0e-12:
        failures.append(
            _pair_failure(
                requirement,
                code="JOINT_RADIAL_INTERFERENCE",
                measured={
                    "radial_clearance_m": radial_clearance_m,
                    "minimum_radial_clearance_m": 0.0,
                },
                required_change=(
                    f"Increase the bore radius or reduce the pin radius for "
                    f"{requirement.joint_id}; the pin is currently larger than "
                    "the bore."
                ),
            )
        )
    elif radial_clearance_m > MAXIMUM_RADIAL_CLEARANCE_M:
        failures.append(
            _pair_failure(
                requirement,
                code="JOINT_RADIAL_CLEARANCE_OUT_OF_RANGE",
                measured={
                    "radial_clearance_m": radial_clearance_m,
                    "maximum_radial_clearance_m": (
                        MAXIMUM_RADIAL_CLEARANCE_M
                    ),
                },
                required_change=(
                    f"Resize the pin/bore features for "
                    f"{requirement.joint_id} so their radial clearance is at "
                    "most the Harness limit."
                ),
            )
        )
    axial_overlap_m = float(measured["axial_overlap_m"])
    if axial_overlap_m < MINIMUM_AXIAL_OVERLAP_M:
        failures.append(
            _pair_failure(
                requirement,
                code="JOINT_AXIAL_OVERLAP_INSUFFICIENT",
                measured={
                    "axial_overlap_m": axial_overlap_m,
                    "minimum_axial_overlap_m": MINIMUM_AXIAL_OVERLAP_M,
                },
                required_change=(
                    f"Extend the coaxial support features for "
                    f"{requirement.joint_id} to provide the Harness-required "
                    "finite axial overlap."
                ),
            )
        )
    return failures


def validate_joint_realizations(
    *,
    step_path: str | Path,
    body_solid_ids: Mapping[str, Sequence[str]],
    requirements: Sequence[RevoluteProofRequirement],
) -> dict[str, Any]:
    """Validate declared revolutes against authoritative STEP BRep features."""

    raw_path = Path(step_path)
    if raw_path.is_symlink() or not raw_path.is_file():
        raise MechanismValidationError(
            "JOINT_REALIZATION_STEP_MISSING",
            "joint realization requires a regular frozen assembly.step",
        )
    path = raw_path.resolve()
    solids = _load_step_solids(path)
    by_id = {f"solid_{index}": solid for index, solid in enumerate(solids, 1)}
    feature_cache: dict[str, list[dict[str, Any]]] = {}

    def body_features(body_id: str) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for solid_id in body_solid_ids.get(body_id, ()):
            if solid_id not in by_id:
                raise MechanismValidationError(
                    "JOINT_REALIZATION_BODY_BINDING_INVALID",
                    f"{body_id!r} references unknown STEP solid {solid_id!r}; "
                    f"available solids: {', '.join(sorted(by_id))}",
                )
            if solid_id not in feature_cache:
                feature_cache[solid_id] = _cylindrical_features(by_id[solid_id])
            output.extend(
                {**item, "solid_id": solid_id}
                for item in feature_cache[solid_id]
            )
        return output

    def bound_context(body_id: str) -> tuple[tuple[str, ...], dict[str, int]]:
        solid_ids = tuple(body_solid_ids.get(body_id, ()))
        return solid_ids, {
            solid_id: len(feature_cache[solid_id])
            for solid_id in solid_ids
            if solid_id in feature_cache
        }

    proofs: list[dict[str, Any]] = []
    for requirement in requirements:
        if requirement.proof_kind != "revolute.pin_bore.v1":
            raise MechanismValidationError(
                "JOINT_PROOF_VALIDATOR_UNAVAILABLE",
                f"unsupported proof kind {requirement.proof_kind!r}",
            )
        parent_features = body_features(requirement.parent_body_id)
        child_features = body_features(requirement.child_body_id)
        parent_matches = _matching_cylinders(parent_features, requirement)
        child_matches = _matching_cylinders(child_features, requirement)
        failures: list[dict[str, Any]] = []
        if not parent_matches:
            bound_ids, bound_counts = bound_context(requirement.parent_body_id)
            failures.append(
                _missing_support_failure(
                    requirement,
                    body_id=requirement.parent_body_id,
                    side="parent",
                    bound_solid_ids=bound_ids,
                    bound_solid_cylindrical_feature_counts=bound_counts,
                    step_solid_count=len(by_id),
                )
            )
        if not child_matches:
            bound_ids, bound_counts = bound_context(requirement.child_body_id)
            failures.append(
                _missing_support_failure(
                    requirement,
                    body_id=requirement.child_body_id,
                    side="child",
                    bound_solid_ids=bound_ids,
                    bound_solid_cylindrical_feature_counts=bound_counts,
                    step_solid_count=len(by_id),
                )
            )
        candidates = _paired_measurements(
            parent_matches,
            child_matches,
            declared_axis=requirement.axis_abs,
        )
        measured: dict[str, Any] | None = None
        if parent_matches and child_matches and not candidates:
            failures.append(
                _pair_failure(
                    requirement,
                    code="JOINT_PIN_BORE_PAIR_MISSING",
                    measured={
                        "parent_surface_roles": sorted(
                            {
                                str(item.get("surface_role") or "")
                                for item in parent_matches
                            }
                        ),
                        "child_surface_roles": sorted(
                            {
                                str(item.get("surface_role") or "")
                                for item in child_matches
                            }
                        ),
                    },
                    required_change=(
                        f"Create one internal cylindrical bore and one external "
                        f"cylindrical pin for {requirement.joint_id}; two "
                        "external bosses or two bores do not realize a retained "
                        "revolute joint."
                    ),
                )
            )
        if candidates:
            evaluated = [
                (candidate, _measurement_failures(requirement, candidate))
                for candidate in candidates
            ]
            passing = [item for item in evaluated if not item[1]]
            measured, pair_failures = passing[0] if passing else evaluated[0]
            failures.extend(pair_failures)
        proofs.append(
            {
                "joint_id": requirement.joint_id,
                "proof_kind": requirement.proof_kind,
                "status": "FAIL" if failures else "PASS",
                "failures": failures,
                "measured": measured,
                "candidate_pair_count": len(candidates),
                "parent_candidates": parent_matches,
                "child_candidates": child_matches,
            }
        )

    passed = all(item["status"] == "PASS" for item in proofs)
    status = (
        "NOT_APPLICABLE"
        if not proofs
        else "PASS"
        if passed
        else "FAIL"
    )
    return {
        "schema_version": REPORT_SCHEMA,
        "assembly_step_sha256": sha256_file(path),
        "policy": _proof_policy(),
        "status": status,
        "passed": passed,
        "certifying_physics_authorized": bool(proofs) and passed,
        "physical_simulation_started": False,
        "step_solid_ids": sorted(by_id),
        "proofs": proofs,
    }


def _body_solid_ids(
    assembly_contract: Mapping[str, Any],
    family_spec: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    group_solid_ids: dict[str, list[str]] = {}
    for part in assembly_contract.get("parts") or ():
        if not isinstance(part, Mapping):
            continue
        group_id = str(part.get("rigid_group_id") or "")
        if not group_id:
            continue
        group_solid_ids.setdefault(group_id, []).extend(
            str(item)
            for item in part.get("source_solid_ids") or ()
            if str(item)
        )
    result: dict[str, tuple[str, ...]] = {}
    for body in family_spec.get("bodies") or ():
        if not isinstance(body, Mapping):
            continue
        body_id = str(body.get("body_id") or "")
        if not body_id:
            continue
        result[body_id] = tuple(
            solid_id
            for group_id in body.get("group_ids") or ()
            for solid_id in group_solid_ids.get(str(group_id), ())
        )
    return result


def _validate_revolute_requirement_coverage(
    family_spec: Mapping[str, Any],
    requirements: Sequence[Any],
) -> None:
    raw_claims = family_spec.get("revolute_joints")
    if not isinstance(raw_claims, list) or not raw_claims:
        return
    claims: dict[str, Mapping[str, Any]] = {}
    for raw in raw_claims:
        if not isinstance(raw, Mapping):
            raise MechanismValidationError(
                "JOINT_REALIZATION_REQUIREMENT_COVERAGE_INVALID",
                "revolute_joints must contain objects before proof binding",
            )
        joint_id = str(raw.get("joint_id") or "")
        if not joint_id or joint_id in claims:
            raise MechanismValidationError(
                "JOINT_REALIZATION_REQUIREMENT_COVERAGE_INVALID",
                "revolute_joints must have unique non-empty joint_id values",
            )
        claims[joint_id] = raw
    bound: dict[str, Any] = {}
    for requirement in requirements:
        joint_id = str(getattr(requirement, "joint_id", ""))
        if not joint_id or joint_id in bound:
            raise MechanismValidationError(
                "JOINT_REALIZATION_REQUIREMENT_COVERAGE_INVALID",
                "geometry proof requirements must have unique joint_id values",
            )
        bound[joint_id] = requirement
    if set(bound) != set(claims):
        missing = sorted(set(claims) - set(bound))
        extra = sorted(set(bound) - set(claims))
        raise MechanismValidationError(
            "JOINT_REALIZATION_REQUIREMENT_COVERAGE_INVALID",
            "geometry proof requirements must cover every revolute_joints "
            f"entry exactly once; missing={missing}, extra={extra}",
        )
    for joint_id, claim in claims.items():
        requirement = bound[joint_id]
        expected_origin = tuple(
            float(item) for item in claim.get("origin_abs_m") or ()
        )
        expected_axis = tuple(
            float(item) for item in claim.get("axis_abs") or ()
        )
        actual_origin = tuple(
            float(item) for item in requirement.origin_abs_m
        )
        actual_axis = tuple(float(item) for item in requirement.axis_abs)
        if (
            str(requirement.proof_kind) != "revolute.pin_bore.v1"
            or str(requirement.parent_body_id)
            != str(claim.get("parent_body_id") or "")
            or str(requirement.child_body_id)
            != str(claim.get("child_body_id") or "")
            or actual_origin != expected_origin
            or actual_axis != expected_axis
        ):
            raise MechanismValidationError(
                "JOINT_REALIZATION_REQUIREMENT_COVERAGE_INVALID",
                f"geometry proof requirement {joint_id!r} must exactly bind "
                "the approved revolute_joints declaration",
            )


def validate_adapter_joint_realizations(
    *,
    step_path: str | Path,
    assembly_contract: Mapping[str, Any],
    family_spec: Mapping[str, Any],
    adapter: Any,
    plugin: Any,
    capsule_id: str,
    has_revolute_claim: bool = False,
) -> dict[str, Any] | None:
    from .family_plugins import (
        MechanismAdapter,
        geometry_proof_contract_error,
    )

    family_spec_snapshot = copy.deepcopy(dict(family_spec))
    requirements = MechanismAdapter.validated_geometry_proof_requirements(
        plugin,
        adapter,
        capsule_id=capsule_id,
    )
    if not requirements:
        if has_revolute_claim and str(plugin.plugin_id).startswith("reviewer."):
            raise geometry_proof_contract_error(
                plugin,
                "JOINT_REALIZATION_PROOF_REQUIREMENTS_MISSING",
                f"reviewer adapter {plugin.plugin_id!r} returned no STEP proof "
                f"requirements for revolute capsule {capsule_id!r}",
            )
        return None
    try:
        _validate_revolute_requirement_coverage(
            family_spec_snapshot,
            requirements,
        )
    except MechanismValidationError as exc:
        raise geometry_proof_contract_error(
            plugin,
            exc.code,
            exc.message,
        ) from exc
    return validate_joint_realizations(
        step_path=step_path,
        body_solid_ids=_body_solid_ids(
            assembly_contract,
            family_spec_snapshot,
        ),
        requirements=requirements,
    )


def _package_family_specs(
    package: Path,
) -> list[_PackageFamilySpecRecord]:
    records: list[_PackageFamilySpecRecord] = []
    referenced_paths: set[Path] = set()
    capsules_root = package / "mechanism_capsules"
    for capsule_path in sorted(capsules_root.glob("*.json")):
        capsule = read_json(capsule_path)
        claim = capsule.get("reduced_model_claim")
        family_spec_file = (
            claim.get("family_spec_file")
            if isinstance(claim, Mapping)
            else None
        )
        if not isinstance(family_spec_file, str) or not family_spec_file:
            continue
        try:
            family_path = ensure_inside(package, family_spec_file)
        except ValueError as exc:
            raise MechanismValidationError(
                "JOINT_REALIZATION_FAMILY_SPEC_INVALID",
                f"{capsule_path.name} references an invalid family spec: {exc}",
            ) from exc
        if not family_path.is_file():
            raise MechanismValidationError(
                "JOINT_REALIZATION_FAMILY_SPEC_MISSING",
                f"{capsule_path.name} references missing {family_spec_file}",
            )
        referenced_paths.add(family_path.resolve())
        capsule_family = (
            str(claim.get("family"))
            if isinstance(claim.get("family"), str)
            else None
        )
        records.append(
            _PackageFamilySpecRecord(
                capsule_id=str(
                    capsule.get("capsule_id") or capsule_path.stem
                ),
                capsule_family=capsule_family,
                family_spec=read_json(family_path),
            )
        )
    for family_path in sorted(capsules_root.glob("*.family.json")):
        if family_path.resolve() in referenced_paths:
            continue
        records.append(
            _PackageFamilySpecRecord(
                capsule_id=family_path.name.removesuffix(".family.json"),
                capsule_family=None,
                family_spec=read_json(family_path),
            )
        )
    return records


def validate_package_joint_realizations(
    package_dir: str | Path,
) -> dict[str, Any]:
    """Run every installed Adapter's STEP proof gate on one frozen package."""

    from .adapters import AdapterSpec
    from .family_plugins import family_spec_plugin_for_kind

    package = Path(package_dir).resolve()
    assembly_contract = read_json(package / "assembly_contract.json")

    aggregate_proofs: list[dict[str, Any]] = []
    capsule_reports: list[dict[str, Any]] = []
    step_solid_ids: list[str] = []
    for record in _package_family_specs(package):
        capsule_id = record.capsule_id
        family_spec = record.family_spec
        mechanism_kind = family_spec.get("mechanism_kind")
        plugin = family_spec_plugin_for_kind(mechanism_kind)
        if plugin is None:
            if record.has_revolute_claim:
                raise MechanismValidationError(
                    "JOINT_PROOF_ADAPTER_UNAVAILABLE",
                    f"revolute capsule {capsule_id!r} has no registered "
                    f"MechanismAdapter for {mechanism_kind!r}",
                )
            continue
        adapter = AdapterSpec(
            adapter_id="joint_realization_preflight",
            family=sorted(plugin.families)[0],
            required_source_hashes=frozenset(),
            supported_programs=frozenset(),
            family_spec=dict(family_spec),
        )
        report = validate_adapter_joint_realizations(
            step_path=package / "assembly.step",
            assembly_contract=assembly_contract,
            family_spec=family_spec,
            adapter=adapter,
            plugin=plugin,
            capsule_id=capsule_id,
            has_revolute_claim=record.has_revolute_claim,
        )
        if report is None:
            continue
        if not step_solid_ids:
            step_solid_ids = list(report["step_solid_ids"])
        aggregate_proofs.extend(report["proofs"])
        capsule_reports.append(
            {
                "capsule_id": capsule_id,
                "mechanism_kind": mechanism_kind,
                "status": report["status"],
                "passed": report["passed"],
                "assembly_step_sha256": report["assembly_step_sha256"],
                "policy": report["policy"],
                "proofs": report["proofs"],
            }
        )

    passed = all(item["passed"] for item in capsule_reports)
    status = (
        "NOT_APPLICABLE"
        if not capsule_reports
        else "PASS"
        if passed
        else "FAIL"
    )
    result = {
        "schema_version": REPORT_SCHEMA,
        "assembly_step_sha256": (
            capsule_reports[0]["assembly_step_sha256"]
            if capsule_reports
            else None
        ),
        "policy": _proof_policy(),
        "status": status,
        "passed": passed,
        "certifying_physics_authorized": bool(capsule_reports) and passed,
        "physical_simulation_started": False,
        "proof_count": len(aggregate_proofs),
        "proofs": aggregate_proofs,
        "capsules": capsule_reports,
    }
    if capsule_reports:
        assigned_solid_ids = {
            str(solid_id)
            for part in assembly_contract.get("parts") or ()
            if isinstance(part, Mapping)
            for solid_id in part.get("source_solid_ids") or ()
        }
        result["unassigned_solid_ids"] = [
            solid_id
            for solid_id in step_solid_ids
            if solid_id not in assigned_solid_ids
        ]
    return result


__all__ = [
    "MAXIMUM_AXIS_ANGLE_ERROR_RAD",
    "MAXIMUM_AXIS_OFFSET_M",
    "MAXIMUM_RADIAL_CLEARANCE_M",
    "MINIMUM_AXIAL_OVERLAP_M",
    "REPORT_SCHEMA",
    "SUPPORTED_PROOF_KINDS",
    "RevoluteProofRequirement",
    "step_solid_geometry_sha256s",
    "validate_adapter_joint_realizations",
    "validate_package_joint_realizations",
    "validate_joint_realizations",
]
