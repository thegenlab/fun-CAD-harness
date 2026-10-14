from __future__ import annotations

import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from ..contracts.failure_envelope import (
    EvidenceValidity,
    FailureOwner,
    make_failure_envelope,
)
from .bounded_support import (
    BoundedSupportRuntime,
    RELATIVE_ANCHOR_DRIFT,
    body_component_refs,
    bounded_support_gate,
    conditional_runtime_summary,
)
from .collision_policy import (
    COMPLIANT_INTERFACE_PHYSICAL,
    CONTACT_PHYSICAL,
    collision_policy_disclosure,
    interface_declaration,
)
from .contact_intent import (
    EXPECTED_REAL_CONTACT,
    build_runtime_component_contact_intent_policy,
    contact_intent_declaration,
)
from .compound_spec import CompoundLinkageSpec, compound_linkage_from_dict
from .conditional_interface import (
    build_functional_interface_geometry_binding,
)
from .io import canonical_sha256
from .models import MechanismValidationError
from .task_motion_evidence import (
    registered_adaptive_termination_evidence,
    registered_stop_comparator_reached,
)
from .wrench_ledger import (
    add_ledgered_load,
    create_body_force,
    create_body_torque,
    current_wrench_ledger,
    set_current_wrench_phase,
    standard_wrench_declaration,
    transition_current_wrench_phase,
)

if TYPE_CHECKING:
    from .worker import ExecutionRequest


PHYSICAL_COMPOUND_JOINT_DRIFT_EXCEEDED = (
    "PHYSICAL_COMPOUND_JOINT_DRIFT_EXCEEDED"
)


def _compound_joint_drift_owner(
    registration_source: str,
) -> FailureOwner:
    """Keep mixed CAD/model drift unclassified for reviewer adapters."""

    return (
        FailureOwner.DESIGN_PACKAGE
        if str(registration_source or "")
        in {"installed_harness_code", "project_design_package"}
        else FailureOwner.UNCLASSIFIED
    )


def _compound_joint_drift_failure(
    *,
    branch: str,
    joint_drifts_m: Mapping[str, float],
    declared_max_joint_drift_m: float,
    mechanism_plugin_id: str,
    mechanism_registration_source: str,
    load_case_id: str,
    phase_index: int,
    phase_id: str,
    time_s: float,
    step_s: float,
    sample_interval_s: float,
) -> MechanismValidationError | None:
    """Return the terminal measured failure once saved physical drift is fatal.

    Compound certification retains the maximum observed joint drift.  Once a
    physical state is above the declared limit, later recovery cannot make the
    completed load case pass.  Equality remains admissible, matching the
    comparison gate's strict ``>`` check.
    """

    if branch != "physical" or not joint_drifts_m:
        return None
    joint_id, observed = max(
        (
            (str(candidate_id), float(candidate_drift))
            for candidate_id, candidate_drift in joint_drifts_m.items()
        ),
        key=lambda item: item[1],
    )
    declared = float(declared_max_joint_drift_m)
    if observed <= declared:
        return None
    details = {
        "mechanism_registration_source": str(
            mechanism_registration_source
        ),
        "load_case_id": str(load_case_id),
        "phase_index": int(phase_index),
        "phase_id": str(phase_id),
        "time_s": float(time_s),
        "step_s": float(step_s),
        "sample_interval_s": float(sample_interval_s),
        "joint_id": joint_id,
        "observed_joint_drift_m": observed,
        "declared_max_joint_drift_m": declared,
    }
    message = (
        f"physical joint {joint_id!r} drifted {observed:.9g} m, above "
        f"the declared {declared:.9g} m certification limit"
    )
    return MechanismValidationError(
        PHYSICAL_COMPOUND_JOINT_DRIFT_EXCEEDED,
        message,
        details=details,
        failure=make_failure_envelope(
            PHYSICAL_COMPOUND_JOINT_DRIFT_EXCEEDED,
            _compound_joint_drift_owner(mechanism_registration_source),
            source_component="mechanism.compound_runtime",
            message=message,
            plugin_id=str(mechanism_plugin_id),
            plugin_kind="mechanism",
            evidence_validity=EvidenceValidity.VALID_MEASUREMENT,
            details=details,
        ),
    )


def compound_bushing_parameters() -> dict[str, float]:
    """Return finite, engineering-scale bearing-seat compliance values.

    These values intentionally remain finite so that the physical branch can
    drift, react, and fail independently of the reduced ideal-joint branch.
    They are stiff enough to represent steel pin joints seated in supported
    bearings instead of elastomeric mounts.
    """
    return {
        "translational_stiffness_n_m": 5_000_000.0,
        "translational_damping_n_s_m": 1_500.0,
        "axis_stiffness_n_m_rad": 5_000.0,
        "axis_damping_n_m_s_rad": 20.0,
        "free_axis_damping_n_m_s_rad": 0.05,
        "limit_stiffness_n_m_rad": 650.0,
        "limit_damping_n_m_s_rad": 5.0,
    }


def _registered_passive_fixture(
    execution: ExecutionRequest,
    linkage: CompoundLinkageSpec,
) -> dict[str, Any] | None:
    """Return one immutable Harness-supplied fixture for a registered task."""

    family_spec = execution.adapter.family_spec
    if not isinstance(family_spec, Mapping):
        return None
    if (
        family_spec.get("mechanism_kind") == "clothespin_dual_jaw_torsion"
        and execution.load_case.program == "task_clothespin_hold"
    ):
        moving = {
            body.body_id for body in linkage.bodies if not body.fixed
        }
        required = {linkage.input_body_id, linkage.output_body_id}
        if not required <= moving:
            raise MechanismValidationError(
                "PASSIVE_WOOD_ROD_FIXTURE_TARGET_INVALID",
                "the wood rod fixture requires both moving jaw bodies",
            )
        radius_m = 0.004
        length_m = 0.080
        density_kg_m3 = 700.0
        return {
            "schema_version": "validator_passive_fixture_v1",
            "fixture_id": "harness_wood_rod_fixture",
            "shape": "passive_wood_rod_cylinder",
            "target_body_id": linkage.output_body_id,
            "required_contact_body_ids": tuple(sorted(required)),
            "radius_m": radius_m,
            "length_m": length_m,
            "density_kg_m3": density_kg_m3,
            "mass_kg": math.pi * radius_m * radius_m * length_m * density_kg_m3,
            "friction": 1.00,
            "collision_enabled": False,
            "unloaded_pad_gap_m": 0.006,
            "rod_diameter_m": 0.008,
            "pad_lever_arm_m": 0.048,
            "contact_envelope_m": 0.0002,
            "contact_normal_stiffness_n_m": 10_000.0,
            "contact_normal_damping_n_s_m": 8.0,
            "contact_normal_max_force_per_jaw_n": 8.0,
            "grip_velocity_damping_n_s_m": 0.25,
            "world_center_m": (0.048, 0.0, 0.0),
            "contact_setup_phase_ids": ("jaw_release_spring_close",),
            "release_phase_ids": ("jaw_release_spring_close",),
            "active_phase_ids": (
                "jaw_release_spring_close",
                "bilateral_grip_settle",
                "tong_hold_verify",
            ),
            "evidence_phase_ids": ("tong_hold_verify",),
            "disturbance_phase_ids": ("tong_hold_verify",),
            "centering_phase_ids": (
                "jaw_release_spring_close",
                "bilateral_grip_settle",
            ),
            "centering_stiffness_n_m": 200.0,
            "centering_damping_n_s_m": 1.5,
            "centering_max_force_n": 5.0,
            "minimum_contact_sample_fraction": 0.80,
            "source": "installed_validator_wood_rod_fixture_v1",
        }
    if (
        family_spec.get("mechanism_kind") != "rotary_detent_phone_stand"
        or execution.load_case.program != "task_phone_stand_loaded_hold"
    ):
        return None
    if linkage.output_body_id not in {
        body.body_id for body in linkage.bodies if not body.fixed
    }:
        raise MechanismValidationError(
            "PASSIVE_PHONE_FIXTURE_TARGET_INVALID",
            "the registered phone fixture requires the moving task output body",
        )
    return {
        "schema_version": "validator_passive_fixture_v1",
        "fixture_id": "harness_phone_fixture",
        "shape": "passive_phone_box",
        "target_body_id": linkage.output_body_id,
        "size_m": (0.060, 0.014, 0.100),
        "mass_kg": 0.200,
        "friction": 0.80,
        "local_center_m": (
            -0.0035,
            -0.004207050807584095,
            0.009415254037842232,
        ),
        "active_phase_ids": ("working_detent_settle", "loaded_phone_hold"),
        "contact_setup_phase_ids": (),
        "release_phase_ids": ("working_detent_settle",),
        "disturbance_phase_ids": (),
        "evidence_phase_ids": ("loaded_phone_hold",),
        "minimum_contact_sample_fraction": 0.80,
        "source": "installed_validator_phone_fixture_v1",
    }


def _registered_compound_torsion_support(
    execution: ExecutionRequest,
    linkage: CompoundLinkageSpec,
) -> dict[str, Any] | None:
    """Resolve the approved grounded/coupled clothespin spring law."""

    family_spec = execution.adapter.family_spec
    if not isinstance(family_spec, Mapping) or family_spec.get(
        "mechanism_kind"
    ) != "clothespin_dual_jaw_torsion":
        return None
    raw = family_spec.get("torsion_spring_parameters")
    stops = family_spec.get("jaw_stop_geometry")
    if not isinstance(raw, Mapping) or not isinstance(stops, Mapping):
        raise MechanismValidationError(
            "COMPOUND_TORSION_SUPPORT_INVALID",
            "clothespin torsion support requires spring and jaw-stop objects",
        )
    body_a_id = str(raw.get("body_a_id") or "")
    body_b_id = str(raw.get("body_b_id") or "")
    anchor_body_id = str(raw.get("anchor_body_id") or "")
    bodies = {body.body_id: body for body in linkage.bodies}
    if (
        body_a_id not in bodies
        or body_b_id not in bodies
        or anchor_body_id not in bodies
        or bodies[body_a_id].fixed
        or bodies[body_b_id].fixed
        or not bodies[anchor_body_id].fixed
        or body_a_id == body_b_id
    ):
        raise MechanismValidationError(
            "COMPOUND_TORSION_SUPPORT_BODY_INVALID",
            "clothespin spring must couple two moving jaws to one fixed anchor",
        )
    axis_raw = raw.get("axis_abs")
    if not isinstance(axis_raw, Sequence) or isinstance(
        axis_raw, (str, bytes)
    ) or len(axis_raw) != 3:
        raise MechanismValidationError(
            "COMPOUND_TORSION_SUPPORT_AXIS_INVALID",
            "clothespin spring axis_abs must contain three finite numbers",
        )
    axis = tuple(float(value) for value in axis_raw)
    axis_norm = math.sqrt(sum(value * value for value in axis))
    if not all(math.isfinite(value) for value in axis) or axis_norm <= 1e-12:
        raise MechanismValidationError(
            "COMPOUND_TORSION_SUPPORT_AXIS_INVALID",
            "clothespin spring axis_abs must be finite and non-zero",
        )
    axis = tuple(value / axis_norm for value in axis)
    joint_by_child = {
        joint.child_body_id: joint for joint in linkage.revolute_joints
    }
    if body_a_id not in joint_by_child or body_b_id not in joint_by_child:
        raise MechanismValidationError(
            "COMPOUND_TORSION_SUPPORT_BODY_INVALID",
            "each spring-driven jaw must be the child of one revolute joint",
        )
    body_a_axis = tuple(joint_by_child[body_a_id].axis_abs)
    body_b_axis = tuple(joint_by_child[body_b_id].axis_abs)
    if (
        abs(sum(a * b for a, b in zip(axis, body_a_axis))) < 1.0 - 1e-6
        or abs(sum(a * b for a, b in zip(axis, body_b_axis))) < 1.0 - 1e-6
    ):
        raise MechanismValidationError(
            "COMPOUND_TORSION_SUPPORT_AXIS_INVALID",
            "spring and both jaw revolute axes must be collinear",
        )
    values = {
        "return_stiffness_n_m_rad": raw.get("stiffness_n_m_rad"),
        "return_damping_n_m_s_rad": raw.get("damping_n_m_s_rad"),
        "coupling_stiffness_n_m_rad": raw.get(
            "coupling_stiffness_n_m_rad"
        ),
        "coupling_damping_n_m_s_rad": raw.get(
            "coupling_damping_n_m_s_rad"
        ),
        "preload_torque_nm": raw.get("preload_torque_nm"),
        "rest_relative_angle_rad": raw.get("rest_relative_angle_rad"),
        "home_angle_rad": stops.get("closed_angle_rad"),
    }
    parsed: dict[str, float] = {}
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MechanismValidationError(
                "COMPOUND_TORSION_SUPPORT_PARAMETER_INVALID",
                f"clothespin spring {name} must be finite",
            )
        parsed[name] = float(value)
        if not math.isfinite(parsed[name]):
            raise MechanismValidationError(
                "COMPOUND_TORSION_SUPPORT_PARAMETER_INVALID",
                f"clothespin spring {name} must be finite",
            )
    if (
        parsed["return_stiffness_n_m_rad"] <= 0.0
        or parsed["coupling_stiffness_n_m_rad"] <= 0.0
        or parsed["return_damping_n_m_s_rad"] < 0.0
        or parsed["coupling_damping_n_m_s_rad"] < 0.0
        or parsed["preload_torque_nm"] < 0.0
    ):
        raise MechanismValidationError(
            "COMPOUND_TORSION_SUPPORT_PARAMETER_INVALID",
            "clothespin spring stiffness must be positive and damping non-negative",
        )
    return {
        "schema_version": "validator_compound_torsion_support_v1",
        "support_id": "clothespin_grounded_dual_jaw_torsion_v1",
        "body_a_id": body_a_id,
        "body_b_id": body_b_id,
        "anchor_body_id": anchor_body_id,
        "axis_abs": axis,
        "body_a_axis_abs": body_a_axis,
        "body_b_axis_abs": body_b_axis,
        **parsed,
        "source": "reviewer_declared_finite_torsion_spring_v1",
    }


def _compound_torsion_support_torques(
    support: Mapping[str, Any],
    *,
    angle_a_rad: float,
    speed_a_rad_s: float,
    angle_b_rad: float,
    speed_b_rad_s: float,
) -> tuple[float, float]:
    """Return grounded-return plus equal/opposite coupling torques."""

    home = float(support["home_angle_rad"])
    relative_error = (
        angle_b_rad
        - angle_a_rad
        - float(support["rest_relative_angle_rad"])
    )
    relative_speed = speed_b_rad_s - speed_a_rad_s
    preload = float(support["preload_torque_nm"])
    coupling = (
        float(support["coupling_stiffness_n_m_rad"]) * relative_error
        + float(support["coupling_damping_n_m_s_rad"]) * relative_speed
    )
    torque_a = (
        -float(support["return_stiffness_n_m_rad"])
        * (angle_a_rad - home)
        - float(support["return_damping_n_m_s_rad"]) * speed_a_rad_s
        + coupling
        - preload
    )
    torque_b = (
        -float(support["return_stiffness_n_m_rad"])
        * (angle_b_rad - home)
        - float(support["return_damping_n_m_s_rad"]) * speed_b_rad_s
        - coupling
        - preload
    )
    return torque_a, torque_b


def _compound_physical_solver(chrono: Any) -> Any:
    """Build the contact-compatible solver for finite compound supports."""

    # Compound physical lanes combine unilateral STEP contact with finite
    # bushing stiffness/damping matrices. ADMM supports both terms while a
    # linear MINRES solve can inject catastrophic velocities into dense
    # frictional contact graphs.
    solver = chrono.ChSolverADMM()
    if hasattr(solver, "SetMaxIterations"):
        solver.SetMaxIterations(500)
    if hasattr(solver, "SetTolerance"):
        solver.SetTolerance(1.0e-10)
    return solver


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _cross(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(sum(item * item for item in value))


def _sub(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float, float]:
    return tuple(a - b for a, b in zip(left, right))


def _add(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float, float]:
    return tuple(a + b for a, b in zip(left, right))


def _scale(value: Sequence[float], factor: float) -> tuple[float, float, float]:
    return tuple(item * factor for item in value)


def _soft_limit_torque(
    *,
    joint_angle_rad: float,
    free_speed_rad_s: float,
    minimum_rad: float | None,
    maximum_rad: float | None,
    stiffness_n_m_rad: float,
    damping_n_m_s_rad: float,
    approach_zone_rad: float,
    approach_damping_n_m_s_rad: float,
) -> float:
    """Return a finite end-stop torque with optional one-way approach damping."""

    torque = 0.0
    if approach_zone_rad > 0.0:
        if (
            minimum_rad is not None
            and minimum_rad <= joint_angle_rad < minimum_rad + approach_zone_rad
            and free_speed_rad_s < 0.0
        ):
            blend = 1.0 - (joint_angle_rad - minimum_rad) / approach_zone_rad
            torque = -approach_damping_n_m_s_rad * free_speed_rad_s * blend
        elif (
            maximum_rad is not None
            and maximum_rad - approach_zone_rad < joint_angle_rad <= maximum_rad
            and free_speed_rad_s > 0.0
        ):
            blend = 1.0 - (maximum_rad - joint_angle_rad) / approach_zone_rad
            torque = -approach_damping_n_m_s_rad * free_speed_rad_s * blend

    if minimum_rad is not None and joint_angle_rad < minimum_rad:
        return (
            -stiffness_n_m_rad * (joint_angle_rad - minimum_rad)
            - damping_n_m_s_rad * min(free_speed_rad_s, 0.0)
        )
    if maximum_rad is not None and joint_angle_rad > maximum_rad:
        return (
            -stiffness_n_m_rad * (joint_angle_rad - maximum_rad)
            - damping_n_m_s_rad * max(free_speed_rad_s, 0.0)
        )
    return torque


def _point_velocity(body: Any, point_abs: Sequence[float], tuple3: Any) -> tuple[float, float, float]:
    center = tuple3(body.GetPos())
    linear = tuple3(body.GetPosDt())
    angular = tuple3(body.GetAngVelParent())
    return _add(linear, _cross(angular, _sub(point_abs, center)))


def _body_axis_abs(body: Any, axis_local: Any, tuple3: Any) -> tuple[float, float, float]:
    return tuple3(body.TransformDirectionLocalToParent(axis_local))


def _validated_compliant_surface_contacts(
    family_spec: Mapping[str, Any],
    collision_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Validate hash-bound local compliant contacts owned by the validator.

    This is intentionally narrower than a general collision mask: it may
    suppress named moving components only, never a complete body pair, and
    every named component remains present in the independent STEP legality
    audit inventory.
    """

    raw_contacts = family_spec.get("compliant_surface_contacts", ())
    if not isinstance(raw_contacts, (list, tuple)):
        raise MechanismValidationError(
            "COMPLIANT_SURFACE_CONTACTS_INVALID",
            "compliant_surface_contacts must be a list",
        )
    body_components: dict[str, set[str]] = {}
    component_owner: dict[str, str] = {}
    for raw_body in collision_manifest.get("bodies") or ():
        if not isinstance(raw_body, Mapping):
            continue
        body_id = str(raw_body.get("body_id") or "")
        solids = {
            str(item.get("solid_id"))
            for item in raw_body.get("components") or ()
            if isinstance(item, Mapping) and item.get("solid_id")
        }
        body_components[body_id] = solids
        for solid_id in solids:
            component_owner[f"{body_id}/{solid_id}"] = body_id

    retained: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_contacts:
        if not isinstance(raw, Mapping):
            raise MechanismValidationError(
                "COMPLIANT_SURFACE_CONTACT_INVALID",
                "a compliant surface contact entry is not an object",
            )
        item = dict(raw)
        contact_id = str(item.get("contact_id") or "")
        if not contact_id or any(character.isspace() for character in contact_id):
            raise MechanismValidationError(
                "COMPLIANT_SURFACE_CONTACT_INVALID",
                "contact_id is invalid",
            )
        if contact_id in seen:
            raise MechanismValidationError(
                "COMPLIANT_SURFACE_CONTACT_DUPLICATE",
                f"duplicate compliant surface contact: {contact_id}",
            )
        seen.add(contact_id)
        if item.get("model") != "validator_compliant_surface_drag_v1":
            raise MechanismValidationError(
                "COMPLIANT_SURFACE_MODEL_INVALID",
                f"unsupported compliant surface model for {contact_id}",
            )
        fixed_body_id = str(item.get("fixed_body_id") or "")
        moving_body_id = str(item.get("moving_body_id") or "")
        if (
            fixed_body_id == moving_body_id
            or fixed_body_id not in body_components
            or moving_body_id not in body_components
        ):
            raise MechanismValidationError(
                "COMPLIANT_SURFACE_BODY_BINDING_INVALID",
                f"invalid body binding for {contact_id}",
            )

        def component_refs(field: str, owner: str) -> tuple[str, ...]:
            values = item.get(field)
            if not isinstance(values, (list, tuple)) or not values:
                raise MechanismValidationError(
                    "COMPLIANT_SURFACE_COMPONENT_BINDING_INVALID",
                    f"{contact_id} requires {field}",
                )
            refs = tuple(sorted({str(value) for value in values}))
            if any(component_owner.get(ref) != owner for ref in refs):
                raise MechanismValidationError(
                    "COMPLIANT_SURFACE_COMPONENT_BINDING_INVALID",
                    f"{contact_id} has an unknown or wrong-body component ref",
                )
            return refs

        fixed_refs = component_refs("fixed_component_refs", fixed_body_id)
        moving_refs = component_refs("moving_component_refs", moving_body_id)
        point = tuple(float(value) for value in item.get("contact_point_abs_m") or ())
        normal = tuple(float(value) for value in item.get("normal_abs") or ())
        if len(point) != 3 or len(normal) != 3 or not all(
            math.isfinite(value) for value in point + normal
        ):
            raise MechanismValidationError(
                "COMPLIANT_SURFACE_FRAME_INVALID",
                f"{contact_id} requires finite 3D contact point and normal",
            )
        normal_norm = _norm(normal)
        if not 0.999 <= normal_norm <= 1.001:
            raise MechanismValidationError(
                "COMPLIANT_SURFACE_NORMAL_INVALID",
                f"{contact_id} normal must be a unit vector",
            )
        normal_load = float(item.get("normal_load_n", 0.0))
        friction = float(item.get("friction_coefficient", -1.0))
        damping = float(item.get("tangential_damping_n_s_m", -1.0))
        maximum_penetration = float(
            item.get(
                "max_penetration_m",
                collision_manifest.get("interpenetration_tolerance_m", 0.00035),
            )
        )
        if not 0.0 < normal_load <= 1_000_000.0:
            raise MechanismValidationError(
                "COMPLIANT_SURFACE_LOAD_INVALID",
                f"{contact_id} normal load is outside validator bounds",
            )
        if not 0.0 <= friction <= 2.0 or not 0.0 <= damping <= 100_000.0:
            raise MechanismValidationError(
                "COMPLIANT_SURFACE_DRAG_INVALID",
                f"{contact_id} drag parameters are outside validator bounds",
            )
        if not 0.0 <= maximum_penetration <= 0.01:
            raise MechanismValidationError(
                "COMPLIANT_SURFACE_PENETRATION_LIMIT_INVALID",
                f"{contact_id} penetration limit is outside validator bounds",
            )
        retained.append(
            {
                "contact_id": contact_id,
                "model": item["model"],
                "fixed_body_id": fixed_body_id,
                "moving_body_id": moving_body_id,
                "fixed_component_refs": fixed_refs,
                "moving_component_refs": moving_refs,
                "contact_point_abs_m": point,
                "normal_abs": _scale(normal, 1.0 / normal_norm),
                "normal_load_n": normal_load,
                "friction_coefficient": friction,
                "tangential_damping_n_s_m": damping,
                "max_penetration_m": maximum_penetration,
            }
        )
    return tuple(retained)


def _output_state(
    linkage: CompoundLinkageSpec,
    bodies: dict[str, Any],
    initial_quaternions: dict[str, tuple[float, float, float, float]],
    signed_axis_angle: Any,
    angular_velocity: Any,
) -> tuple[float, float, float, float]:
    joints = {item.joint_id: item for item in linkage.revolute_joints}
    input_joint = joints[linkage.input_joint_id]
    output_joint = joints[linkage.output_joint_id]
    from .worker import _pose

    def relative_state(joint: Any) -> tuple[float, float]:
        parent = bodies[joint.parent_body_id]
        child = bodies[joint.child_body_id]
        parent_quaternion = tuple(_pose(parent)["rotation_abs_quat_wxyz"])
        child_quaternion = tuple(_pose(child)["rotation_abs_quat_wxyz"])
        parent_delta = signed_axis_angle(parent_quaternion, joint.axis_abs) - signed_axis_angle(
            initial_quaternions[joint.parent_body_id], joint.axis_abs
        )
        child_delta = signed_axis_angle(child_quaternion, joint.axis_abs) - signed_axis_angle(
            initial_quaternions[joint.child_body_id], joint.axis_abs
        )
        speed = _dot(
            _sub(angular_velocity(child), angular_velocity(parent)), joint.axis_abs
        )
        return child_delta - parent_delta, speed

    input_angle, input_speed = relative_state(input_joint)
    output_delta, output_speed = relative_state(output_joint)
    return (
        input_angle,
        input_speed,
        linkage.home_output_angle_rad + output_delta,
        output_speed,
    )


def _compound_worker_backend_matches(mechanism_plugin: Any) -> bool:
    plugin_id = str(getattr(mechanism_plugin, "plugin_id", "")).strip()
    worker_backend = str(
        getattr(mechanism_plugin, "worker_backend", "")
    ).strip()
    return bool(
        worker_backend == "revolute_linkage_v1"
        or (plugin_id and worker_backend == plugin_id)
    )


def _require_compound_task_actuation_backend(
    execution: Any,
) -> None:
    from .task_plugins import claimed_reviewer_task_programs

    program = str(execution.load_case.program)
    if program not in claimed_reviewer_task_programs(execution.plan):
        return
    registry = execution.plan.task_channel_registry
    input_body_ids = sorted(
        {
            str(item)
            for item in (
                registry.get("input_body_ids")
                if isinstance(registry, Mapping)
                else ()
            )
            or ()
            if str(item)
        }
    )
    if len(input_body_ids) <= 1:
        return
    plugin_id = str(execution.mechanism_plugin.plugin_id)
    code = (
        "REVIEWER_MECHANISM_ADAPTER_MULTI_INPUT_TASK_RUNTIME_REQUIRED"
    )
    message = (
        "the shared compound runtime has one task-wrench target, but the "
        "bound reviewer task program selects multiple input bodies"
    )
    details = {
        "plugin_id": plugin_id,
        "program": program,
        "input_body_ids": input_body_ids,
        "required_repair": (
            "provide a reviewer-owned simulation callback with explicit "
            "per-phase actuation targets"
        ),
    }
    raise MechanismValidationError(
        code,
        message,
        details=details,
        failure=make_failure_envelope(
            code,
            FailureOwner.REVIEWER_MECHANISM_PLUGIN,
            source_component="mechanism.compound_runtime.task_actuation",
            message=message,
            plugin_id=plugin_id,
            plugin_kind="mechanism",
            registration_source="reviewer_mechanism_adapter",
            evidence_validity=EvidenceValidity.NOT_APPLICABLE,
            details=details,
        ),
    )


def simulate_compound_linkage(
    execution: ExecutionRequest,
    workspace: Path,
) -> dict[str, Any]:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(os.path.join(conda_prefix, "Library", "bin"))
    import pychrono as chrono

    from .worker import (
        _angular_velocity,
        _configure_revolute_angle_limit,
        _contact_count,
        _contact_pair_counts,
        _inertia_box,
        _make_material,
        _make_nsc_system,
        _pose,
        _quat_from_z_to_axis,
        _reference_schedule,
        _signed_axis_angle,
        _tuple3,
        _write_trace_checkpoint,
        lane_execution_source_sha256,
        _collision_progress,
    )

    if not _compound_worker_backend_matches(execution.mechanism_plugin):
        raise MechanismValidationError(
            "MECHANISM_WORKER_BACKEND_UNAVAILABLE",
            "compound runtime received a different mechanism backend",
        )
    plugin_id = str(execution.mechanism_plugin.plugin_id)
    mechanism_registration_source = str(
        current_wrench_ledger().capability_registration.get(
            "registration_source"
        )
        or ""
    )
    linkage = compound_linkage_from_dict(execution.adapter.family_spec)
    _require_compound_task_actuation_backend(execution)
    registered_torsion_support = _registered_compound_torsion_support(
        execution, linkage
    )
    from .family_collision import (
        family_collision_geometry,
        make_family_body,
        trace_collision_thresholds,
    )

    collision_manifest = family_collision_geometry(
        execution.adapter,
        expected_bodies=linkage.bodies,
    )
    if collision_manifest["package_binding_sha256"] != execution.plan.package_binding_sha256:
        raise MechanismValidationError(
            "FAMILY_COLLIDER_PACKAGE_BINDING_CHANGED",
            "compound collision geometry does not match the executed package",
        )
    compliant_surface_contacts = _validated_compliant_surface_contacts(
        execution.adapter.family_spec,
        collision_manifest,
    )
    runtime_suppressed_by_body: dict[str, set[str]] = {}
    for contact in compliant_surface_contacts:
        for component_ref in contact["moving_component_refs"]:
            body_id, solid_id = component_ref.split("/", 1)
            runtime_suppressed_by_body.setdefault(body_id, set()).add(solid_id)
    declared_joint_limits = {
        joint.joint_id: (
            (
                linkage.home_output_angle_rad
                if joint.min_angle_rad is None and joint.joint_id == linkage.output_joint_id
                else joint.min_angle_rad
            ),
            (
                linkage.outbound_output_angle_rad
                if joint.max_angle_rad is None and joint.joint_id == linkage.output_joint_id
                else joint.max_angle_rad
            ),
        )
        for joint in linkage.revolute_joints
    }
    reference_schedule = _reference_schedule(execution)
    system = _make_nsc_system(
        chrono,
        collision_envelope_m=12e-6,
        collision_margin_m=2e-6,
    )
    if execution.branch == "physical":
        system.SetSolver(_compound_physical_solver(chrono))
    system.SetGravitationalAcceleration(
        chrono.ChVector3d(*execution.plan.gravity_abs_m_s2)
    )
    material = _make_material(chrono, linkage.friction)
    bodies: dict[str, Any] = {}
    inertias: dict[str, list[float]] = {}
    initial_poses: dict[str, dict[str, Any]] = {}
    initial_quaternions: dict[str, tuple[float, float, float, float]] = {}
    collision_audit: list[dict[str, Any]] = []
    collision_families: dict[str, int] = {}
    for body_index, spec in enumerate(linkage.bodies):
        inertia = _inertia_box(spec.mass_kg, spec.size_m)
        body, audit = make_family_body(
            chrono,
            body_spec=spec,
            geometry=collision_manifest,
            workspace=workspace,
            material=material,
            inertia_xx=inertia,
            branch=execution.branch,
            runtime_suppressed_solid_ids=(
                sorted(runtime_suppressed_by_body.get(spec.body_id, ()))
                if execution.branch == "physical"
                else ()
            ),
        )
        if execution.branch == "physical":
            if body_index >= 15:
                raise MechanismValidationError(
                    "COMPOUND_COLLISION_FAMILY_CAPACITY_EXCEEDED",
                    "exact joint-pair collision filtering currently supports at most 15 bodies",
                )
            family = body_index + 1
            body.GetCollisionModel().SetFamily(family)
            collision_families[spec.body_id] = family
            audit["collision_family"] = family
        _collision_progress("system_add_body_begin", body_id=spec.body_id)
        system.Add(body)
        _collision_progress("system_add_body_complete", body_id=spec.body_id)
        bodies[spec.body_id] = body
        collision_audit.append(audit)
        inertias[spec.body_id] = list(inertia)
        initial_poses[spec.body_id] = _pose(body)
        initial_quaternions[spec.body_id] = tuple(
            initial_poses[spec.body_id]["rotation_abs_quat_wxyz"]
        )

    passive_fixture_spec = _registered_passive_fixture(execution, linkage)
    passive_fixture_body = None
    passive_fixture_released = False
    passive_fixture_collision_enabled = False
    if execution.branch == "physical" and passive_fixture_spec is not None:
        fixture_material = _make_material(
            chrono, passive_fixture_spec["friction"]
        )
        if passive_fixture_spec["shape"] == "passive_phone_box":
            fixture_density = passive_fixture_spec["mass_kg"] / math.prod(
                passive_fixture_spec["size_m"]
            )
            passive_fixture_body = chrono.ChBodyEasyBox(
                *passive_fixture_spec["size_m"],
                fixture_density,
                False,
                True,
                fixture_material,
            )
        elif passive_fixture_spec["shape"] == "passive_wood_rod_cylinder":
            passive_fixture_body = chrono.ChBodyEasyCylinder(
                chrono.ChAxis_Y,
                passive_fixture_spec["radius_m"],
                passive_fixture_spec["length_m"],
                passive_fixture_spec["density_kg_m3"],
                False,
                True,
                fixture_material,
            )
        else:
            raise MechanismValidationError(
                "PASSIVE_FIXTURE_SHAPE_UNSUPPORTED",
                "registered passive fixture has an unsupported shape",
            )
        passive_fixture_body.SetName(passive_fixture_spec["fixture_id"])
        passive_fixture_body.SetFixed(True)
        passive_fixture_body.EnableCollision(False)
        if "world_center_m" in passive_fixture_spec:
            passive_fixture_body.SetPos(
                chrono.ChVector3d(*passive_fixture_spec["world_center_m"])
            )
            passive_fixture_body.SetRot(chrono.ChQuaterniond(1.0, 0.0, 0.0, 0.0))
        else:
            fixture_target = bodies[passive_fixture_spec["target_body_id"]]
            passive_fixture_body.SetPos(
                fixture_target.TransformPointLocalToParent(
                    chrono.ChVector3d(
                        *passive_fixture_spec["local_center_m"]
                    )
                )
            )
            passive_fixture_body.SetRot(fixture_target.GetRot())
        system.Add(passive_fixture_body)

    support_gate_by_joint = {
        joint.joint_id: bounded_support_gate(
            body_a_id=joint.parent_body_id,
            body_b_id=joint.child_body_id,
            endpoint_a_component_refs=body_component_refs(
                collision_manifest,
                joint.parent_body_id,
            ),
            endpoint_b_component_refs=body_component_refs(
                collision_manifest,
                joint.child_body_id,
            ),
            maximum_drift_m=linkage.max_joint_drift_m,
            coordinate_mode=RELATIVE_ANCHOR_DRIFT,
            maximum_axis_misalignment_rad=(
                linkage.max_off_plane_rotation_rad
            ),
            template_id="finite_revolute_pin_capture_v1",
            proof_source="validator_compound_support_envelope_v1",
        )
        for joint in linkage.revolute_joints
    }
    physical_interfaces = [
        interface_declaration(
            interface_id=f"support-{joint.joint_id}",
            classification=COMPLIANT_INTERFACE_PHYSICAL,
            body_a_id=joint.parent_body_id,
            body_b_id=joint.child_body_id,
            model="finite_revolute_bushing",
            source="trusted_revolute_linkage_runtime_v1",
            free_coordinates=("rotation_about_axis",),
            parameters={
                "free_axis_damping_n_m_s_rad": joint.free_axis_damping_n_m_s_rad,
                "limit_stiffness_n_m_rad": joint.limit_stiffness_n_m_rad,
                "limit_damping_n_m_s_rad": joint.limit_damping_n_m_s_rad,
                "limit_approach_zone_rad": joint.limit_approach_zone_rad,
                "limit_approach_damping_n_m_s_rad": (
                    joint.limit_approach_damping_n_m_s_rad
                ),
            },
            transmission_gate=support_gate_by_joint[joint.joint_id],
        )
        for joint in linkage.revolute_joints
    ]
    physical_interfaces.extend(
        interface_declaration(
            interface_id=f"contact-{item['contact_id']}",
            classification=CONTACT_PHYSICAL,
            body_a_id=item["fixed_body_id"],
            body_b_id=item["moving_body_id"],
            model=item["model"],
            source="trusted_compound_compliant_surface_runtime_v1",
            parameters={
                "fixed_component_refs": list(item["fixed_component_refs"]),
                "moving_component_refs": list(item["moving_component_refs"]),
                "validator_owned_suppressed_component_refs": list(
                    item["moving_component_refs"]
                ),
                "normal_load_n": item["normal_load_n"],
                "friction_coefficient": item["friction_coefficient"],
                "tangential_damping_n_s_m": item[
                    "tangential_damping_n_s_m"
                ],
            },
        )
        for item in compliant_surface_contacts
    )
    functional_interface_geometry_binding = (
        build_functional_interface_geometry_binding(
            physical_interfaces,
            collision_manifest,
        )
    )
    # The finite bushing declares how force crosses the bearing. It never
    # authorizes disabling collision for the complete connected body pair.

    ideal_links: list[str] = []
    if execution.branch == "reduced":
        for joint in linkage.revolute_joints:
            child = bodies[joint.child_body_id]
            parent = bodies[joint.parent_body_id]
            revolute = chrono.ChLinkLockRevolute()
            revolute.Initialize(
                child,
                parent,
                chrono.ChFramed(
                    chrono.ChVector3d(*joint.origin_abs_m),
                    _quat_from_z_to_axis(chrono, joint.axis_abs),
                ),
            )
            minimum, maximum = declared_joint_limits[joint.joint_id]
            if minimum is not None and maximum is not None:
                _configure_revolute_angle_limit(
                    revolute, maximum, min_angle_rad=minimum
                )
            system.AddLink(revolute)
            ideal_links.append("ChLinkLockRevolute")
    link_count = int(system.GetNumLinks()) if hasattr(system, "GetNumLinks") else None
    if execution.branch == "physical" and link_count not in {0, None}:
        raise RuntimeError("compound physical branch unexpectedly contains an ideal link")
    if execution.branch == "reduced" and link_count != len(linkage.revolute_joints):
        raise RuntimeError("compound reduced branch did not create every declared pivot")

    load_container = chrono.ChLoadContainer()
    system.Add(load_container)
    input_body = bodies[linkage.input_body_id]
    input_point_local = input_body.TransformPointParentToLocal(
        chrono.ChVector3d(*linkage.input_point_abs_m)
    )
    input_force = create_body_force(
        chrono,
        input_body,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        False,
        chrono.ChVector3d(*linkage.input_point_abs_m),
        False,
        declaration=standard_wrench_declaration(
            plugin_id=plugin_id,
            profile="task_input",
            wrench_kind="force",
            load_id="declared-input-force",
            owner=plugin_id,
            target_body=linkage.input_body_id,
            source_role="declared_input_port",
            target_role="declared_input",
            geometry_gate="load_case_phase_and_input_frame",
            active_interval="declared_load_case_phases",
            release_required=True,
        ),
    )
    add_ledgered_load(load_container, input_force)
    input_torque = create_body_torque(
        chrono,
        input_body,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        False,
        declaration=standard_wrench_declaration(
            plugin_id=plugin_id,
            profile="task_input",
            wrench_kind="torque",
            load_id="declared-input-torque",
            owner=plugin_id,
            target_body=linkage.input_body_id,
            source_role="declared_input_port",
            target_role="declared_input",
            geometry_gate="load_case_phase_and_input_frame",
            active_interval="declared_load_case_phases",
            release_required=True,
        ),
    )
    add_ledgered_load(load_container, input_torque)
    torsion_torque_a = None
    torsion_torque_b = None
    current_torsion_torques_nm = {
        "body_a": 0.0,
        "body_b": 0.0,
    }
    if execution.branch == "physical" and registered_torsion_support is not None:
        torsion_torque_a = create_body_torque(
            chrono,
            bodies[registered_torsion_support["body_a_id"]],
            chrono.ChVector3d(0.0, 0.0, 0.0),
            False,
            declaration=standard_wrench_declaration(
                plugin_id=plugin_id,
                profile="constitutive",
                wrench_kind="torque",
                load_id="clothespin-torsion-body-a",
                owner=plugin_id,
                target_body=registered_torsion_support["body_a_id"],
                source_role="finite_torsion_spring",
                target_role="declared_input",
                geometry_gate=registered_torsion_support["support_id"],
                active_interval="all_physical_phases",
            ),
        )
        add_ledgered_load(load_container, torsion_torque_a)
        torsion_torque_b = create_body_torque(
            chrono,
            bodies[registered_torsion_support["body_b_id"]],
            chrono.ChVector3d(0.0, 0.0, 0.0),
            False,
            declaration=standard_wrench_declaration(
                plugin_id=plugin_id,
                profile="constitutive",
                wrench_kind="torque",
                load_id="clothespin-torsion-body-b",
                owner=plugin_id,
                target_body=registered_torsion_support["body_b_id"],
                source_role="finite_torsion_spring",
                target_role="task_output",
                geometry_gate=registered_torsion_support["support_id"],
                active_interval="all_physical_phases",
            ),
        )
        add_ledgered_load(load_container, torsion_torque_b)
    passive_fixture_disturbance = None
    current_passive_fixture_force_abs_n = (0.0, 0.0, 0.0)
    if (
        execution.branch == "physical"
        and passive_fixture_body is not None
        and passive_fixture_spec is not None
        and passive_fixture_spec.get("disturbance_phase_ids")
    ):
        passive_fixture_disturbance = create_body_force(
            chrono,
            passive_fixture_body,
            chrono.ChVector3d(0.0, 0.0, 0.0),
            False,
            passive_fixture_body.GetPos(),
            False,
            declaration=standard_wrench_declaration(
                plugin_id=plugin_id,
                profile="passive_test_load",
                wrench_kind="force",
                load_id="passive-fixture-disturbance",
                owner=plugin_id,
                target_body=passive_fixture_spec["fixture_id"],
                source_role="harness_passive_fixture_disturbance",
                target_role="external_passive_fixture",
                geometry_gate=passive_fixture_spec["fixture_id"],
                active_interval="registered_fixture_disturbance_phases",
                release_required=True,
            ),
        )
        add_ledgered_load(load_container, passive_fixture_disturbance)
    passive_fixture_grip_support = None
    current_passive_fixture_support_force_abs_n = (0.0, 0.0, 0.0)
    passive_fixture_bilateral_contact_active = False
    passive_fixture_pad_gap_m = None
    current_passive_fixture_contact_force_per_jaw_n = 0.0
    current_passive_fixture_contact_torques_nm = {
        "body_a": 0.0,
        "body_b": 0.0,
    }
    passive_fixture_contact_torque_a = None
    passive_fixture_contact_torque_b = None
    if (
        execution.branch == "physical"
        and passive_fixture_body is not None
        and passive_fixture_spec is not None
        and passive_fixture_spec["shape"]
        == "passive_wood_rod_cylinder"
    ):
        passive_fixture_grip_support = create_body_force(
            chrono,
            passive_fixture_body,
            chrono.ChVector3d(0.0, 0.0, 0.0),
            False,
            passive_fixture_body.GetPos(),
            False,
            declaration=standard_wrench_declaration(
                plugin_id=plugin_id,
                profile="passive_test_load",
                wrench_kind="force",
                load_id="passive-fixture-grip-support",
                owner=plugin_id,
                target_body=passive_fixture_spec["fixture_id"],
                source_role="geometry_gated_bilateral_grip_friction",
                target_role="external_passive_fixture",
                geometry_gate="bilateral_jaw_pad_gap",
                active_interval="registered_grip_and_centering_phases",
                release_required=True,
            ),
        )
        add_ledgered_load(load_container, passive_fixture_grip_support)
        if registered_torsion_support is not None:
            passive_fixture_contact_torque_a = create_body_torque(
                chrono,
                bodies[registered_torsion_support["body_a_id"]],
                chrono.ChVector3d(0.0, 0.0, 0.0),
                False,
                declaration=standard_wrench_declaration(
                    plugin_id=plugin_id,
                    profile="constitutive",
                    wrench_kind="torque",
                    load_id="passive-fixture-contact-body-a",
                    owner=plugin_id,
                    target_body=registered_torsion_support["body_a_id"],
                    source_role="finite_pad_rod_normal_contact",
                    target_role="declared_input",
                    geometry_gate="bilateral_jaw_pad_gap",
                    active_interval="geometry_gated_fixture_contact_interval",
                ),
            )
            add_ledgered_load(
                load_container, passive_fixture_contact_torque_a
            )
            passive_fixture_contact_torque_b = create_body_torque(
                chrono,
                bodies[registered_torsion_support["body_b_id"]],
                chrono.ChVector3d(0.0, 0.0, 0.0),
                False,
                declaration=standard_wrench_declaration(
                    plugin_id=plugin_id,
                    profile="constitutive",
                    wrench_kind="torque",
                    load_id="passive-fixture-contact-body-b",
                    owner=plugin_id,
                    target_body=registered_torsion_support["body_b_id"],
                    source_role="finite_pad_rod_normal_contact",
                    target_role="task_output",
                    geometry_gate="bilateral_jaw_pad_gap",
                    active_interval="geometry_gated_fixture_contact_interval",
                ),
            )
            add_ledgered_load(
                load_container, passive_fixture_contact_torque_b
            )
    compliant_surface_loads: list[dict[str, Any]] = []
    current_compliant_surface_forces_n: dict[str, float] = {}
    if execution.branch == "physical":
        for item in compliant_surface_contacts:
            moving_body = bodies[item["moving_body_id"]]
            contact_point = chrono.ChVector3d(*item["contact_point_abs_m"])
            target_body_id = str(item["moving_body_id"])
            drag_load = create_body_force(
                chrono,
                moving_body,
                chrono.ChVector3d(0.0, 0.0, 0.0),
                False,
                contact_point,
                False,
                declaration=standard_wrench_declaration(
                    plugin_id=plugin_id,
                    profile="transmission",
                    wrench_kind="force",
                    load_id=f"compliant-surface-{item['contact_id']}",
                    owner=plugin_id,
                    target_body=target_body_id,
                    source_role="finite_contact_interface",
                    target_role=(
                        "task_output"
                        if target_body_id == linkage.output_body_id
                        else "internal_moving"
                    ),
                    geometry_gate=(
                        f"compliant_surface_contact:{item['contact_id']}"
                    ),
                    active_interval="geometry_gated_physical_contact_interval",
                ),
            )
            add_ledgered_load(load_container, drag_load)
            compliant_surface_loads.append(
                {
                    "spec": item,
                    "body": moving_body,
                    "point_local": moving_body.TransformPointParentToLocal(
                        contact_point
                    ),
                    "load": drag_load,
                }
            )
            current_compliant_surface_forces_n[item["contact_id"]] = 0.0

    def update_compliant_surfaces() -> None:
        for runtime in compliant_surface_loads:
            item = runtime["spec"]
            body = runtime["body"]
            point_abs = _tuple3(
                body.TransformPointLocalToParent(runtime["point_local"])
            )
            velocity = _point_velocity(body, point_abs, _tuple3)
            normal = item["normal_abs"]
            tangent_velocity = _sub(
                velocity,
                _scale(normal, _dot(velocity, normal)),
            )
            speed = _norm(tangent_velocity)
            maximum_drag = (
                item["friction_coefficient"] * item["normal_load_n"]
            )
            magnitude = min(
                item["tangential_damping_n_s_m"] * speed,
                maximum_drag,
            )
            force = (
                _scale(tangent_velocity, -magnitude / speed)
                if speed > 1e-12
                else (0.0, 0.0, 0.0)
            )
            runtime["load"].SetApplicationPoint(
                chrono.ChVector3d(*point_abs), False
            )
            runtime["load"].SetForce(chrono.ChVector3d(*force), False)
            current_compliant_surface_forces_n[item["contact_id"]] = magnitude

    # Physical pivots remain finite bushings, not ideal mathematical joints.
    bushing_parameters = compound_bushing_parameters()
    bushing_stiffness_n_m = bushing_parameters["translational_stiffness_n_m"]
    bushing_damping_n_s_m = bushing_parameters["translational_damping_n_s_m"]
    axis_stiffness_n_m_rad = bushing_parameters["axis_stiffness_n_m_rad"]
    axis_damping_n_m_s_rad = bushing_parameters["axis_damping_n_m_s_rad"]
    runtime_joints: list[dict[str, Any]] = []
    if execution.branch == "physical":
        for joint in linkage.revolute_joints:
            parent = bodies[joint.parent_body_id]
            child = bodies[joint.child_body_id]
            origin = chrono.ChVector3d(*joint.origin_abs_m)
            stiffness = chrono.ChMatrix66d()
            damping = chrono.ChMatrix66d()
            stiffness.SetZero()
            damping.SetZero()
            for index, value in enumerate((
                bushing_stiffness_n_m,
                bushing_stiffness_n_m,
                bushing_stiffness_n_m,
                axis_stiffness_n_m_rad,
                axis_stiffness_n_m_rad,
                0.0,
            )):
                stiffness.SetItem(index, index, value)
            for index, value in enumerate((
                bushing_damping_n_s_m,
                bushing_damping_n_s_m,
                bushing_damping_n_s_m,
                axis_damping_n_m_s_rad,
                axis_damping_n_m_s_rad,
                joint.free_axis_damping_n_m_s_rad,
            )):
                damping.SetItem(index, index, value)
            bushing = chrono.ChLoadBodyBodyBushingGeneric(
                child,
                parent,
                chrono.ChFramed(
                    origin, _quat_from_z_to_axis(chrono, joint.axis_abs)
                ),
                stiffness,
                damping,
            )
            load_container.Add(bushing)
            child_role = (
                "declared_input"
                if joint.child_body_id == linkage.input_body_id
                else "task_output"
                if joint.child_body_id == linkage.output_body_id
                else "internal_moving"
            )
            child_limit_torque = create_body_torque(
                chrono,
                child,
                chrono.ChVector3d(0.0, 0.0, 0.0),
                False,
                declaration=standard_wrench_declaration(
                    plugin_id=plugin_id,
                    profile="limit",
                    wrench_kind="torque",
                    load_id=f"{joint.joint_id}-child-limit",
                    owner=plugin_id,
                    target_body=joint.child_body_id,
                    source_role="finite_joint_stop",
                    target_role=child_role,
                    geometry_gate=f"joint_angle_limit:{joint.joint_id}",
                    active_interval="geometry_gated_physical_interval",
                ),
            )
            add_ledgered_load(load_container, child_limit_torque)
            parent_limit_torque = None
            if not parent.IsFixed():
                parent_role = (
                    "declared_input"
                    if joint.parent_body_id == linkage.input_body_id
                    else "task_output"
                    if joint.parent_body_id == linkage.output_body_id
                    else "internal_moving"
                )
                parent_limit_torque = create_body_torque(
                    chrono,
                    parent,
                    chrono.ChVector3d(0.0, 0.0, 0.0),
                    False,
                    declaration=standard_wrench_declaration(
                        plugin_id=plugin_id,
                        profile="limit",
                        wrench_kind="torque",
                        load_id=f"{joint.joint_id}-parent-limit",
                        owner=plugin_id,
                        target_body=joint.parent_body_id,
                        source_role="finite_joint_stop_reaction",
                        target_role=parent_role,
                        geometry_gate=f"joint_angle_limit:{joint.joint_id}",
                        active_interval="geometry_gated_physical_interval",
                    ),
                )
                add_ledgered_load(load_container, parent_limit_torque)
            runtime_joints.append(
                {
                    "spec": joint,
                    "parent": parent,
                    "child": child,
                    "bushing": bushing,
                    "stiffness_matrix": stiffness,
                    "damping_matrix": damping,
                    "zero_stiffness_matrix": chrono.ChMatrix66d(),
                    "zero_damping_matrix": chrono.ChMatrix66d(),
                    "support_gate": BoundedSupportRuntime(
                        interface_id=f"support-{joint.joint_id}",
                        gate=support_gate_by_joint[joint.joint_id],
                    ),
                    "parent_anchor_local": parent.TransformPointParentToLocal(origin),
                    "child_anchor_local": child.TransformPointParentToLocal(origin),
                    "parent_axis_local": parent.TransformDirectionParentToLocal(
                        chrono.ChVector3d(*joint.axis_abs)
                    ),
                    "child_axis_local": child.TransformDirectionParentToLocal(
                        chrono.ChVector3d(*joint.axis_abs)
                    ),
                    "parent_limit_torque": parent_limit_torque,
                    "child_limit_torque": child_limit_torque,
                    "min_angle_rad": declared_joint_limits[joint.joint_id][0],
                    "max_angle_rad": declared_joint_limits[joint.joint_id][1],
                    "limit_stiffness_n_m_rad": joint.limit_stiffness_n_m_rad,
                    "limit_damping_n_m_s_rad": joint.limit_damping_n_m_s_rad,
                    "limit_approach_zone_rad": joint.limit_approach_zone_rad,
                    "limit_approach_damping_n_m_s_rad": (
                        joint.limit_approach_damping_n_m_s_rad
                    ),
                }
            )
            runtime_joints[-1]["zero_stiffness_matrix"].SetZero()
            runtime_joints[-1]["zero_damping_matrix"].SetZero()

    current_joint_drifts: dict[str, float] = {
        joint.joint_id: 0.0 for joint in linkage.revolute_joints
    }
    current_reaction_forces: dict[str, float] = {
        joint.joint_id: 0.0 for joint in linkage.revolute_joints
    }
    current_conditional_interface_states: dict[str, dict[str, Any]] = {}

    def update_bushings() -> None:
        for runtime in runtime_joints:
            joint = runtime["spec"]
            parent = runtime["parent"]
            child = runtime["child"]
            parent_point = _tuple3(parent.TransformPointLocalToParent(runtime["parent_anchor_local"]))
            child_point = _tuple3(child.TransformPointLocalToParent(runtime["child_anchor_local"]))
            displacement = _sub(child_point, parent_point)
            parent_axis = _body_axis_abs(
                parent, runtime["parent_axis_local"], _tuple3
            )
            child_axis = _body_axis_abs(
                child, runtime["child_axis_local"], _tuple3
            )
            mean_axis_raw = _add(parent_axis, child_axis)
            mean_axis_norm = max(_norm(mean_axis_raw), 1e-12)
            mean_axis = _scale(mean_axis_raw, 1.0 / mean_axis_norm)
            axis_misalignment = math.acos(
                max(-1.0, min(1.0, abs(_dot(parent_axis, child_axis))))
            )
            support_state = runtime["support_gate"].evaluate(
                time_s=time_s,
                anchor_a_abs_m=parent_point,
                anchor_b_abs_m=child_point,
                velocity_a_abs_m_s=_point_velocity(
                    parent,
                    parent_point,
                    _tuple3,
                ),
                velocity_b_abs_m_s=_point_velocity(
                    child,
                    child_point,
                    _tuple3,
                ),
                requested_effort_si=_norm(
                    _tuple3(runtime["bushing"].GetForce())
                ),
                axis_misalignment_rad=axis_misalignment,
            )
            current_conditional_interface_states[
                support_state["interface_id"]
            ] = support_state
            support_active = bool(support_state["transmission_active"])
            runtime["bushing"].SetStiffnessMatrix(
                runtime[
                    "stiffness_matrix"
                    if support_active
                    else "zero_stiffness_matrix"
                ]
            )
            runtime["bushing"].SetDampingMatrix(
                runtime[
                    "damping_matrix"
                    if support_active
                    else "zero_damping_matrix"
                ]
            )
            relative_omega = _sub(_angular_velocity(child), _angular_velocity(parent))
            free_speed = _dot(relative_omega, mean_axis)
            parent_quaternion = tuple(_pose(parent)["rotation_abs_quat_wxyz"])
            child_quaternion = tuple(_pose(child)["rotation_abs_quat_wxyz"])
            parent_delta = _signed_axis_angle(
                parent_quaternion, joint.axis_abs
            ) - _signed_axis_angle(
                initial_quaternions[joint.parent_body_id], joint.axis_abs
            )
            child_delta = _signed_axis_angle(
                child_quaternion, joint.axis_abs
            ) - _signed_axis_angle(
                initial_quaternions[joint.child_body_id], joint.axis_abs
            )
            joint_angle = child_delta - parent_delta
            minimum = runtime["min_angle_rad"]
            maximum = runtime["max_angle_rad"]
            limit_torque = _soft_limit_torque(
                joint_angle_rad=joint_angle,
                free_speed_rad_s=free_speed,
                minimum_rad=minimum,
                maximum_rad=maximum,
                stiffness_n_m_rad=runtime["limit_stiffness_n_m_rad"],
                damping_n_m_s_rad=runtime["limit_damping_n_m_s_rad"],
                approach_zone_rad=runtime["limit_approach_zone_rad"],
                approach_damping_n_m_s_rad=runtime[
                    "limit_approach_damping_n_m_s_rad"
                ],
            )
            requested_limit_torque = limit_torque
            if not support_active:
                limit_torque = 0.0
            support_state["requested_support_torque_nm"] = abs(
                requested_limit_torque
            )
            support_state["transmitted_support_torque_nm"] = abs(
                limit_torque
            )
            limit_torque_abs = _scale(mean_axis, limit_torque)
            runtime["child_limit_torque"].SetTorque(
                chrono.ChVector3d(*limit_torque_abs), False
            )
            if runtime["parent_limit_torque"] is not None:
                runtime["parent_limit_torque"].SetTorque(
                    chrono.ChVector3d(*_scale(limit_torque_abs, -1.0)), False
                )
            current_joint_drifts[joint.joint_id] = _norm(displacement)
            current_reaction_forces[joint.joint_id] = (
                _norm(_tuple3(runtime["bushing"].GetForce()))
                if support_active
                else 0.0
            )

    def update_registered_torsion_support() -> None:
        if (
            registered_torsion_support is None
            or torsion_torque_a is None
            or torsion_torque_b is None
        ):
            return
        input_angle, input_speed, output_angle, output_speed = _output_state(
            linkage,
            bodies,
            initial_quaternions,
            _signed_axis_angle,
            _angular_velocity,
        )
        torque_a, torque_b = _compound_torsion_support_torques(
            registered_torsion_support,
            angle_a_rad=input_angle,
            speed_a_rad_s=input_speed,
            angle_b_rad=output_angle,
            speed_b_rad_s=output_speed,
        )
        torsion_torque_a.SetTorque(
            chrono.ChVector3d(
                *_scale(
                    registered_torsion_support["body_a_axis_abs"],
                    torque_a,
                )
            ),
            False,
        )
        torsion_torque_b.SetTorque(
            chrono.ChVector3d(
                *_scale(
                    registered_torsion_support["body_b_axis_abs"],
                    torque_b,
                )
            ),
            False,
        )
        current_torsion_torques_nm["body_a"] = torque_a
        current_torsion_torques_nm["body_b"] = torque_b

    def update_registered_rod_contact(phase_id: str) -> None:
        nonlocal passive_fixture_bilateral_contact_active
        nonlocal passive_fixture_pad_gap_m
        nonlocal current_passive_fixture_contact_force_per_jaw_n
        if (
            passive_fixture_spec is None
            or passive_fixture_body is None
            or passive_fixture_spec["shape"]
            != "passive_wood_rod_cylinder"
            or phase_id not in passive_fixture_spec["active_phase_ids"]
        ):
            passive_fixture_bilateral_contact_active = False
            passive_fixture_pad_gap_m = None
            current_passive_fixture_contact_force_per_jaw_n = 0.0
            current_passive_fixture_contact_torques_nm["body_a"] = 0.0
            current_passive_fixture_contact_torques_nm["body_b"] = 0.0
            if passive_fixture_contact_torque_a is not None:
                passive_fixture_contact_torque_a.SetTorque(
                    chrono.ChVector3d(0.0, 0.0, 0.0), False
                )
            if passive_fixture_contact_torque_b is not None:
                passive_fixture_contact_torque_b.SetTorque(
                    chrono.ChVector3d(0.0, 0.0, 0.0), False
                )
            return
        input_angle, input_speed, output_angle, output_speed = (
            _output_state(
                linkage,
                bodies,
                initial_quaternions,
                _signed_axis_angle,
                _angular_velocity,
            )
        )
        gap_m = passive_fixture_spec["unloaded_pad_gap_m"] + (
            passive_fixture_spec["pad_lever_arm_m"]
            * (math.sin(input_angle) + math.sin(output_angle))
        )
        passive_fixture_pad_gap_m = gap_m
        fixture_position = _tuple3(passive_fixture_body.GetPos())
        center = passive_fixture_spec["world_center_m"]
        transverse_offset_m = math.hypot(
            fixture_position[0] - center[0],
            fixture_position[2] - center[2],
        )
        passive_fixture_bilateral_contact_active = bool(
            gap_m
            <= passive_fixture_spec["rod_diameter_m"]
            + passive_fixture_spec["contact_envelope_m"]
            and transverse_offset_m <= 0.001
        )
        gap_rate_m_s = passive_fixture_spec["pad_lever_arm_m"] * (
            math.cos(input_angle) * input_speed
            + math.cos(output_angle) * output_speed
        )
        penetration_m = max(
            0.0,
            passive_fixture_spec["rod_diameter_m"] - gap_m,
        )
        normal_force_per_jaw_n = (
            passive_fixture_spec["contact_normal_stiffness_n_m"]
            * penetration_m
            - passive_fixture_spec["contact_normal_damping_n_s_m"]
            * gap_rate_m_s
        )
        normal_force_per_jaw_n = min(
            passive_fixture_spec[
                "contact_normal_max_force_per_jaw_n"
            ],
            max(0.0, normal_force_per_jaw_n),
        )
        if not passive_fixture_bilateral_contact_active:
            normal_force_per_jaw_n = 0.0
        current_passive_fixture_contact_force_per_jaw_n = (
            normal_force_per_jaw_n
        )
        contact_torque_nm = (
            normal_force_per_jaw_n
            * passive_fixture_spec["pad_lever_arm_m"]
        )
        current_passive_fixture_contact_torques_nm["body_a"] = (
            contact_torque_nm
        )
        current_passive_fixture_contact_torques_nm["body_b"] = (
            contact_torque_nm
        )
        if (
            passive_fixture_contact_torque_a is not None
            and registered_torsion_support is not None
        ):
            passive_fixture_contact_torque_a.SetTorque(
                chrono.ChVector3d(
                    *_scale(
                        registered_torsion_support["body_a_axis_abs"],
                        contact_torque_nm,
                    )
                ),
                False,
            )
        if (
            passive_fixture_contact_torque_b is not None
            and registered_torsion_support is not None
        ):
            passive_fixture_contact_torque_b.SetTorque(
                chrono.ChVector3d(
                    *_scale(
                        registered_torsion_support["body_b_axis_abs"],
                        contact_torque_nm,
                    )
                ),
                False,
            )

    output_body = bodies[linkage.output_body_id]
    output_local = output_body.TransformPointParentToLocal(
        chrono.ChVector3d(*linkage.output_observable_point_abs_m)
    )
    output_axis_joint = next(
        item for item in linkage.revolute_joints
        if item.joint_id == linkage.output_joint_id
    )
    output_axis_local = output_body.TransformDirectionParentToLocal(
        chrono.ChVector3d(*output_axis_joint.axis_abs)
    )
    samples: list[dict[str, Any]] = []
    executed_phases: list[dict[str, Any]] = []
    time_s = 0.0
    next_sample_s = 0.0

    def ensure_passive_fixture(phase_id: str) -> None:
        nonlocal passive_fixture_released, passive_fixture_collision_enabled
        if (
            execution.branch != "physical"
            or passive_fixture_spec is None
            or passive_fixture_body is None
            or passive_fixture_released
            or phase_id not in passive_fixture_spec["active_phase_ids"]
        ):
            return
        if not passive_fixture_collision_enabled:
            if "world_center_m" in passive_fixture_spec:
                passive_fixture_body.SetPos(
                    chrono.ChVector3d(*passive_fixture_spec["world_center_m"])
                )
                passive_fixture_body.SetRot(
                    chrono.ChQuaterniond(1.0, 0.0, 0.0, 0.0)
                )
            else:
                passive_fixture_body.SetPos(
                    output_body.TransformPointLocalToParent(
                        chrono.ChVector3d(
                            *passive_fixture_spec["local_center_m"]
                        )
                    )
                )
                passive_fixture_body.SetRot(output_body.GetRot())
            passive_fixture_body.SetPosDt(
                chrono.ChVector3d(0.0, 0.0, 0.0)
            )
            passive_fixture_body.SetAngVelParent(
                chrono.ChVector3d(0.0, 0.0, 0.0)
            )
            passive_fixture_body.SetFixed(True)
            passive_fixture_body.EnableCollision(
                bool(passive_fixture_spec.get("collision_enabled", True))
            )
            passive_fixture_collision_enabled = True
        if phase_id in passive_fixture_spec["release_phase_ids"]:
            passive_fixture_body.SetFixed(False)
            passive_fixture_released = True

    def record(
        phase_id: str,
        applied_force: tuple[float, float, float],
        applied_torque: tuple[float, float, float],
    ) -> None:
        input_angle, input_speed, output_angle, output_speed = _output_state(
            linkage,
            bodies,
            initial_quaternions,
            _signed_axis_angle,
            _angular_velocity,
        )
        output_point = _tuple3(output_body.TransformPointLocalToParent(output_local))
        output_axis = _body_axis_abs(output_body, output_axis_local, _tuple3)
        off_plane = math.acos(max(-1.0, min(1.0, _dot(output_axis, output_axis_joint.axis_abs))))
        contact_pairs = (
            _contact_pair_counts(chrono, system)
            if execution.branch == "physical"
            else {}
        )
        if (
            passive_fixture_bilateral_contact_active
            and passive_fixture_spec is not None
        ):
            contact_pairs = dict(contact_pairs)
            for body_id in passive_fixture_spec.get(
                "required_contact_body_ids", ()
            ):
                contact_pairs[
                    f"{passive_fixture_spec['fixture_id']}|{body_id}"
                ] = max(
                    1,
                    int(
                        contact_pairs.get(
                            f"{passive_fixture_spec['fixture_id']}|{body_id}",
                            0,
                        )
                    ),
                )
        fixture_contact_count = 0
        fixture_contact_counts_by_body: dict[str, int] = {}
        fixture_pose = None
        fixture_relative_position = None
        if passive_fixture_body is not None and passive_fixture_spec is not None:
            fixture_id = passive_fixture_spec["fixture_id"]
            target_id = passive_fixture_spec["target_body_id"]
            fixture_contact_count = sum(
                int(count)
                for pair, count in contact_pairs.items()
                if fixture_id in str(pair) and target_id in str(pair)
            )
            fixture_contact_counts_by_body = {
                body_id: sum(
                    int(count)
                    for pair, count in contact_pairs.items()
                    if fixture_id in str(pair) and body_id in str(pair)
                )
                for body_id in passive_fixture_spec.get(
                    "required_contact_body_ids", (target_id,)
                )
            }
            fixture_contact_count = sum(
                fixture_contact_counts_by_body.values()
            )
            fixture_pose = _pose(passive_fixture_body)
            fixture_relative_position = list(
                _tuple3(
                    output_body.TransformPointParentToLocal(
                        passive_fixture_body.GetPos()
                    )
                )
            )
        body_poses = {
            body_id: _pose(body) for body_id, body in bodies.items()
        }
        if fixture_pose is not None and passive_fixture_spec is not None:
            body_poses[passive_fixture_spec["fixture_id"]] = fixture_pose
        samples.append(
            {
                "time_s": time_s,
                "phase_id": phase_id,
                "applied_force_abs_n": list(applied_force),
                "applied_torque_abs_nm": list(applied_torque),
                "moving_pose": _pose(output_body),
                "body_poses": body_poses,
                "observable_point_abs_m": list(output_point),
                "generalized_angle_rad": output_angle,
                "generalized_speed_rad_s": output_speed,
                "compound_input_angle_rad": input_angle,
                "compound_input_speed_rad_s": input_speed,
                "compound_output_angle_rad": output_angle,
                "compound_output_speed_rad_s": output_speed,
                "joint_drifts_m": dict(current_joint_drifts),
                "max_joint_drift_m": max(current_joint_drifts.values(), default=0.0),
                "bushing_reaction_forces_n": dict(current_reaction_forces),
                "max_bushing_reaction_force_n": max(current_reaction_forces.values(), default=0.0),
                "conditional_interface_states": [
                    dict(current_conditional_interface_states[interface_id])
                    for interface_id in sorted(
                        current_conditional_interface_states
                    )
                ],
                "compliant_surface_drag_forces_n": dict(
                    current_compliant_surface_forces_n
                ),
                "max_compliant_surface_drag_force_n": max(
                    current_compliant_surface_forces_n.values(),
                    default=0.0,
                ),
                "registered_torsion_torques_nm": dict(
                    current_torsion_torques_nm
                ),
                "off_plane_rotation_rad": off_plane,
                "contact_count": _contact_count(system)
                + (2 if passive_fixture_bilateral_contact_active else 0),
                "contact_pair_counts": contact_pairs,
                "passive_fixture_active": passive_fixture_released,
                "passive_fixture_setup_placement": (
                    passive_fixture_body is not None
                    and not passive_fixture_released
                ),
                "passive_fixture_mass_kg": (
                    passive_fixture_spec["mass_kg"]
                    if passive_fixture_spec is not None
                    else None
                ),
                "passive_fixture_contact_count": fixture_contact_count,
                "passive_fixture_contact_counts_by_body": dict(
                    fixture_contact_counts_by_body
                ),
                "passive_fixture_applied_force_abs_n": list(
                    current_passive_fixture_force_abs_n
                ),
                "passive_fixture_support_force_abs_n": list(
                    current_passive_fixture_support_force_abs_n
                ),
                "passive_fixture_bilateral_contact_active": (
                    passive_fixture_bilateral_contact_active
                ),
                "passive_fixture_pad_gap_m": passive_fixture_pad_gap_m,
                "passive_fixture_contact_normal_force_per_jaw_n": (
                    current_passive_fixture_contact_force_per_jaw_n
                ),
                "passive_fixture_contact_torques_nm": dict(
                    current_passive_fixture_contact_torques_nm
                ),
                "passive_fixture_target_relative_position_m": (
                    fixture_relative_position
                ),
                "direct_passive_fixture_actuation": False,
            }
        )
        _write_trace_checkpoint(
            execution,
            samples,
            executed_phases=executed_phases,
        )

    zero_wrench = (0.0, 0.0, 0.0)
    update_bushings()
    update_registered_torsion_support()
    record("initial", zero_wrench, zero_wrench)
    next_sample_s = execution.load_case.sample_interval_s
    first_dynamics_step = True
    dynamics_step_count = 0
    for phase_index, phase in enumerate(execution.load_case.phases):
        ensure_passive_fixture(phase.phase_id)
        transition_current_wrench_phase(phase.phase_id)
        direction = phase.direction_abs or zero_wrench
        declared_force_abs = (
            tuple(value * phase.magnitude_si for value in direction)
            if phase.wrench_kind == "force"
            else zero_wrench
        )
        fixture_disturbance_active = bool(
            passive_fixture_disturbance is not None
            and passive_fixture_spec is not None
            and phase.phase_id
            in passive_fixture_spec.get("disturbance_phase_ids", ())
        )
        fixture_centering_active = bool(
            passive_fixture_grip_support is not None
            and passive_fixture_spec is not None
            and phase.phase_id
            in passive_fixture_spec.get("centering_phase_ids", ())
        )
        force_abs = (
            zero_wrench
            if fixture_disturbance_active
            else declared_force_abs
        )
        current_passive_fixture_force_abs_n = (
            declared_force_abs
            if fixture_disturbance_active
            else zero_wrench
        )
        torque_abs = (
            tuple(value * phase.magnitude_si for value in direction)
            if phase.wrench_kind == "torque"
            else zero_wrench
        )
        phase_limit_s = (
            reference_schedule[phase_index]["duration_s"]
            if reference_schedule is not None
            else phase.duration_s
        )
        elapsed = 0.0
        stop_reached = False
        while elapsed < phase_limit_s - 1e-12:
            input_point = _tuple3(input_body.TransformPointLocalToParent(input_point_local))
            input_force.SetApplicationPoint(chrono.ChVector3d(*input_point), False)
            input_force.SetForce(chrono.ChVector3d(*force_abs), False)
            input_torque.SetTorque(chrono.ChVector3d(*torque_abs), False)
            update_registered_rod_contact(phase.phase_id)
            if passive_fixture_disturbance is not None:
                current_passive_fixture_force_abs_n = (
                    declared_force_abs
                    if fixture_disturbance_active
                    else zero_wrench
                )
                passive_fixture_disturbance.SetApplicationPoint(
                    passive_fixture_body.GetPos(), False
                )
                passive_fixture_disturbance.SetForce(
                    chrono.ChVector3d(
                        *current_passive_fixture_force_abs_n
                    ),
                    False,
                )
            if passive_fixture_grip_support is not None:
                if fixture_centering_active:
                    displacement = _sub(
                        _tuple3(passive_fixture_body.GetPos()),
                        passive_fixture_spec["world_center_m"],
                    )
                    velocity = _tuple3(passive_fixture_body.GetPosDt())
                    centering_force = _add(
                        _scale(
                            displacement,
                            -passive_fixture_spec[
                                "centering_stiffness_n_m"
                            ],
                        ),
                        _scale(
                            velocity,
                            -passive_fixture_spec[
                                "centering_damping_n_s_m"
                            ],
                        ),
                    )
                    centering_magnitude = _norm(centering_force)
                    maximum = passive_fixture_spec[
                        "centering_max_force_n"
                    ]
                    current_passive_fixture_support_force_abs_n = (
                        _scale(
                            centering_force,
                            maximum / centering_magnitude,
                        )
                        if centering_magnitude > maximum
                        else centering_force
                    )
                elif (
                    fixture_disturbance_active
                    and passive_fixture_bilateral_contact_active
                    and registered_torsion_support is not None
                ):
                    velocity = _tuple3(passive_fixture_body.GetPosDt())
                    desired_friction = _add(
                        _add(
                            _scale(declared_force_abs, -1.0),
                            _scale(
                                execution.plan.gravity_abs_m_s2,
                                -passive_fixture_spec["mass_kg"],
                            ),
                        ),
                        _scale(
                            velocity,
                            -passive_fixture_spec[
                                "grip_velocity_damping_n_s_m"
                            ],
                        ),
                    )
                    normal_force_per_jaw = (
                        registered_torsion_support["preload_torque_nm"]
                        / passive_fixture_spec["pad_lever_arm_m"]
                    )
                    maximum_friction = (
                        2.0
                        * passive_fixture_spec["friction"]
                        * normal_force_per_jaw
                    )
                    friction_magnitude = _norm(desired_friction)
                    current_passive_fixture_support_force_abs_n = (
                        _scale(
                            desired_friction,
                            maximum_friction / friction_magnitude,
                        )
                        if friction_magnitude > maximum_friction
                        else desired_friction
                    )
                else:
                    current_passive_fixture_support_force_abs_n = zero_wrench
                passive_fixture_grip_support.SetApplicationPoint(
                    passive_fixture_body.GetPos(), False
                )
                passive_fixture_grip_support.SetForce(
                    chrono.ChVector3d(
                        *current_passive_fixture_support_force_abs_n
                    ),
                    False,
                )
            update_compliant_surfaces()
            update_registered_torsion_support()
            step = min(execution.load_case.step_s, phase_limit_s - elapsed)
            if first_dynamics_step:
                _collision_progress("first_dynamics_step_begin", capsule_id=execution.plan.capsule_id)
            system.DoStepDynamics(step)
            dynamics_step_count += 1
            if first_dynamics_step:
                _collision_progress("first_dynamics_step_complete", capsule_id=execution.plan.capsule_id)
                first_dynamics_step = False
            if dynamics_step_count % 1000 == 0:
                _collision_progress(
                    "dynamics_progress",
                    capsule_id=execution.plan.capsule_id,
                    step_count=dynamics_step_count,
                    phase_id=phase.phase_id,
                    simulated_time_s=time_s + step,
                )
            elapsed += step
            time_s += step
            # Refresh the live support observation at the just-integrated
            # state.  This both prepares the bushing for the next step and
            # makes an over-limit sample available before any early stop.
            update_bushings()
            update_registered_torsion_support()
            update_registered_rod_contact(phase.phase_id)
            drift_failure = _compound_joint_drift_failure(
                branch=execution.branch,
                joint_drifts_m=current_joint_drifts,
                declared_max_joint_drift_m=linkage.max_joint_drift_m,
                mechanism_plugin_id=plugin_id,
                mechanism_registration_source=(
                    mechanism_registration_source
                ),
                load_case_id=execution.load_case.load_case_id,
                phase_index=phase_index,
                phase_id=phase.phase_id,
                time_s=time_s,
                step_s=execution.load_case.step_s,
                sample_interval_s=execution.load_case.sample_interval_s,
            )
            if drift_failure is not None:
                # The saved state is the valid failure evidence.  Release the
                # task-input channel and close its phase only after the atomic
                # checkpoint exists, then let the worker preserve both facts.
                record(phase.phase_id, force_abs, torque_abs)
                input_force.SetForce(
                    chrono.ChVector3d(0.0, 0.0, 0.0), False
                )
                input_torque.SetTorque(
                    chrono.ChVector3d(0.0, 0.0, 0.0), False
                )
                set_current_wrench_phase(None)
                raise drift_failure
            if time_s + 1e-12 >= next_sample_s:
                record(phase.phase_id, force_abs, torque_abs)
                while next_sample_s <= time_s + 1e-12:
                    next_sample_s += execution.load_case.sample_interval_s
            if execution.branch == "physical" and phase.stop_condition is not None:
                output_angle = _output_state(
                    linkage, bodies, initial_quaternions,
                    _signed_axis_angle, _angular_velocity,
                )[2]
                target = float(phase.target_coordinate)
                reached = registered_stop_comparator_reached(
                    phase.stop_condition,
                    target_coordinate_si=target,
                    actual_coordinate_si=output_angle,
                )
                if reached:
                    stop_reached = True
                    break
        if not samples or samples[-1]["time_s"] < time_s - 1e-12:
            record(phase.phase_id, force_abs, torque_abs)
        input_force.SetForce(chrono.ChVector3d(0.0, 0.0, 0.0), False)
        input_torque.SetTorque(chrono.ChVector3d(0.0, 0.0, 0.0), False)
        if passive_fixture_disturbance is not None:
            passive_fixture_disturbance.SetForce(
                chrono.ChVector3d(0.0, 0.0, 0.0), False
            )
            current_passive_fixture_force_abs_n = zero_wrench
        if passive_fixture_grip_support is not None:
            passive_fixture_grip_support.SetForce(
                chrono.ChVector3d(0.0, 0.0, 0.0), False
            )
            current_passive_fixture_support_force_abs_n = zero_wrench
        executed_phase = {
            "phase_id": phase.phase_id,
            "wrench_kind": phase.wrench_kind,
            "direction_abs": (
                None
                if phase.direction_abs is None
                else list(phase.direction_abs)
            ),
            "magnitude_si": phase.magnitude_si,
            "duration_s": (
                elapsed if reference_schedule is None else phase_limit_s
            ),
        }
        if phase.stop_condition is not None:
            if (
                reference_schedule is not None
                and isinstance(
                    reference_schedule[phase_index].get("termination"),
                    dict,
                )
            ):
                executed_phase["termination"] = dict(
                    reference_schedule[phase_index]["termination"]
                )
            else:
                output_angle = _output_state(
                    linkage,
                    bodies,
                    initial_quaternions,
                    _signed_axis_angle,
                    _angular_velocity,
                )[2]
                executed_phase["termination"] = (
                    registered_adaptive_termination_evidence(
                        phase,
                        actual_coordinate_si=output_angle,
                        reached=stop_reached,
                    )
                )
        executed_phases.append(executed_phase)

    input_force.SetForce(chrono.ChVector3d(0.0, 0.0, 0.0), False)
    input_torque.SetTorque(chrono.ChVector3d(0.0, 0.0, 0.0), False)
    if passive_fixture_disturbance is not None:
        passive_fixture_disturbance.SetForce(
            chrono.ChVector3d(0.0, 0.0, 0.0), False
        )
    if passive_fixture_grip_support is not None:
        passive_fixture_grip_support.SetForce(
            chrono.ChVector3d(0.0, 0.0, 0.0), False
        )
    set_current_wrench_phase(None)
    passive_fixture_summary = None
    if execution.branch == "physical" and passive_fixture_spec is not None:
        evidence_phases = set(passive_fixture_spec["evidence_phase_ids"])
        fixture_samples = [
            sample
            for sample in samples
            if sample.get("phase_id") in evidence_phases
        ]
        required_contact_body_ids = tuple(
            passive_fixture_spec.get(
                "required_contact_body_ids",
                (passive_fixture_spec["target_body_id"],),
            )
        )
        contact_samples_by_body = {
            body_id: sum(
                int(
                    (
                        sample.get(
                            "passive_fixture_contact_counts_by_body"
                        )
                        or {}
                    ).get(body_id)
                    or 0
                )
                > 0
                for sample in fixture_samples
            )
            for body_id in required_contact_body_ids
        }
        contact_fractions_by_body = {
            body_id: (
                count / len(fixture_samples)
                if fixture_samples
                else 0.0
            )
            for body_id, count in contact_samples_by_body.items()
        }
        contact_samples = sum(
            int(sample.get("passive_fixture_contact_count") or 0) > 0
            for sample in fixture_samples
        )
        contact_fraction = min(
            contact_fractions_by_body.values(), default=0.0
        )
        passive_fixture_summary = dict(passive_fixture_spec)
        for key in (
            "size_m",
            "local_center_m",
            "world_center_m",
            "active_phase_ids",
            "contact_setup_phase_ids",
            "release_phase_ids",
            "evidence_phase_ids",
            "disturbance_phase_ids",
            "centering_phase_ids",
            "required_contact_body_ids",
        ):
            if key in passive_fixture_summary:
                passive_fixture_summary[key] = list(
                    passive_fixture_summary[key]
                )
        passive_fixture_summary.update({
            "runtime_enabled": passive_fixture_body is not None,
            "evidence_sample_count": len(fixture_samples),
            "contact_sample_count": contact_samples,
            "contact_sample_fraction": contact_fraction,
            "contact_sample_counts_by_body": contact_samples_by_body,
            "contact_sample_fractions_by_body": contact_fractions_by_body,
            "direct_actuation": False,
            "supported": (
                bool(fixture_samples)
                and all(
                    fraction
                    >= passive_fixture_spec[
                        "minimum_contact_sample_fraction"
                    ]
                    for fraction in contact_fractions_by_body.values()
                )
            ),
        })
        if passive_fixture_summary["supported"] is not True:
            is_phone = (
                passive_fixture_spec["shape"] == "passive_phone_box"
            )
            code = (
                "PASSIVE_PHONE_FIXTURE_SUPPORT_MISSING"
                if is_phone
                else "PASSIVE_WOOD_ROD_FIXTURE_GRIP_MISSING"
            )
            message = (
                "the Harness-supplied passive phone fixture did not remain "
                "in contact with the phone-support body through the hold window"
                if is_phone
                else "the Harness-supplied passive wood rod did not retain "
                "bilateral jaw contact through the 3 N hold window"
            )
            raise MechanismValidationError(
                code,
                message,
                details=passive_fixture_summary,
                failure=make_failure_envelope(
                    code,
                    FailureOwner.DESIGN_PACKAGE,
                    source_component="mechanism.compound_runtime.passive_fixture",
                    message=message,
                    evidence_validity=EvidenceValidity.VALID_MEASUREMENT,
                    details=passive_fixture_summary,
                ),
            )
    topology = linkage.to_dict()
    total_mass = sum(item.mass_kg for item in linkage.bodies if not item.fixed)
    bushing_disclosure = [
        {
            "joint_id": item.joint_id,
            "chrono_type": "ChLoadBodyBodyBushingGeneric",
            "translational_stiffness_n_m": bushing_stiffness_n_m,
            "translational_damping_n_s_m": bushing_damping_n_s_m,
            "axis_stiffness_n_m_rad": axis_stiffness_n_m_rad,
            "axis_damping_n_m_s_rad": axis_damping_n_m_s_rad,
            "free_axis_damping_n_m_s_rad": item.free_axis_damping_n_m_s_rad,
            "limit_stiffness_n_m_rad": item.limit_stiffness_n_m_rad,
            "limit_damping_n_m_s_rad": item.limit_damping_n_m_s_rad,
            "limit_approach_zone_rad": item.limit_approach_zone_rad,
            "limit_approach_damping_n_m_s_rad": (
                item.limit_approach_damping_n_m_s_rad
            ),
            "ideal_constraint": False,
        }
        for item in linkage.revolute_joints
    ] if execution.branch == "physical" else []
    compliant_surface_disclosure = [
        {
            "contact_id": item["contact_id"],
            "model": item["model"],
            "fixed_body_id": item["fixed_body_id"],
            "moving_body_id": item["moving_body_id"],
            "fixed_component_refs": list(item["fixed_component_refs"]),
            "moving_component_refs": list(item["moving_component_refs"]),
            "runtime_suppressed_component_refs": (
                list(item["moving_component_refs"])
                if execution.branch == "physical"
                else []
            ),
            "normal_load_n": item["normal_load_n"],
            "friction_coefficient": item["friction_coefficient"],
            "tangential_damping_n_s_m": item[
                "tangential_damping_n_s_m"
            ],
            "max_penetration_m": item["max_penetration_m"],
            "whole_body_pair_collision_enabled": True,
            "independent_step_geometry_legality_required": True,
        }
        for item in compliant_surface_contacts
    ]
    active_contact_phases = [
        "initial",
        *[
            str(item["phase_id"])
            for item in executed_phases
            if item.get("phase_id")
        ],
    ]
    explicit_contact_declarations = [
        contact_intent_declaration(
            intent_id=f"surface-{item['contact_id']}",
            kind=EXPECTED_REAL_CONTACT,
            component_a_refs=item["fixed_component_refs"],
            component_b_refs=item["moving_component_refs"],
            source="trusted_compound_compliant_surface_runtime_v1",
            interface_id=f"contact-{item['contact_id']}",
            active_phase_ids=active_contact_phases,
            max_penetration_m=item["max_penetration_m"],
            binding_scope="validator_owned_exact_component_subset",
        )
        for item in compliant_surface_contacts
    ]
    component_contact_policy = build_runtime_component_contact_intent_policy(
        physical_interfaces,
        collision_manifest,
        collision_audit,
        executed_phases,
        explicit_declarations=explicit_contact_declarations,
    )
    conditional_interface_runtime = (
        conditional_runtime_summary(
            [
                runtime["support_gate"]
                for runtime in runtime_joints
            ],
            physics_step_count=dynamics_step_count,
        )
        if execution.branch == "physical"
        else None
    )
    chrono_version = str(chrono.GetChronoVersion()) if hasattr(chrono, "GetChronoVersion") else None
    return {
        "schema_version": "0.1",
        "status": "COMPLETED",
        "validator_source_sha256": lane_execution_source_sha256(),
        "branch": execution.branch,
        "adapter_id": execution.adapter.adapter_id,
        "mechanism_plugin_id": execution.mechanism_plugin.plugin_id,
        "package_id": execution.plan.package_id,
        "design_version": execution.plan.design_version,
        "package_binding_sha256": execution.plan.package_binding_sha256,
        "capsule_id": execution.plan.capsule_id,
        "load_case_id": execution.load_case.load_case_id,
        "plan_sha256": canonical_sha256(execution.plan.to_dict()),
        "load_case_sha256": canonical_sha256(execution.load_case.to_dict()),
        "engine": {
            "name": "PyChrono",
            "version": chrono_version,
            "system": "ChSystemNSC",
            "solver": (
                "ChSolverADMM" if execution.branch == "physical" else "ChSolverPSOR"
            ),
            "mechanism_plugin_id": execution.mechanism_plugin.plugin_id,
            "worker_backend": execution.mechanism_plugin.worker_backend,
            "collision_backend": "BULLET",
            "step_s": execution.load_case.step_s,
            "gravity_abs_m_s2": list(execution.plan.gravity_abs_m_s2),
            "chrono_link_count": link_count,
            "ideal_links": ideal_links,
            "internal_ideal_joint_count": len(ideal_links),
            "compliant_supports": [],
            "registered_torsion_support": (
                None
                if registered_torsion_support is None
                else {
                    **registered_torsion_support,
                    "axis_abs": list(registered_torsion_support["axis_abs"]),
                    "body_a_axis_abs": list(
                        registered_torsion_support["body_a_axis_abs"]
                    ),
                    "body_b_axis_abs": list(
                        registered_torsion_support["body_b_axis_abs"]
                    ),
                }
            ),
            "compound_bushings": bushing_disclosure,
            "collision_shape_counts": {
                body_id: (
                    int(body.GetCollisionModel().GetNumShapes())
                    if body.GetCollisionModel() is not None
                    else 0
                )
                for body_id, body in bodies.items()
            } | {
                "external_passive_fixture": (
                    int(
                        passive_fixture_body.GetCollisionModel().GetNumShapes()
                    )
                    if passive_fixture_body is not None
                    else 0
                )
            },
        },
        "branch_semantics": {
            "same_mass_kg": total_mass,
            "same_inertia_xx_kg_m2": inertias,
            "same_initial_pose": initial_poses,
            "same_external_load_sha256": canonical_sha256(executed_phases),
            "task_input_body_id": linkage.input_body_id,
            "unique_task_output_body_id": linkage.output_body_id,
            "direct_output_actuation_used": False,
            "runtime_bindings": {
                "fixed_body_id": next(
                    body.body_id for body in linkage.bodies if body.fixed
                ),
                "input_body_id": linkage.input_body_id,
                "output_body_id": linkage.output_body_id,
            },
            "physical_auxiliary_contact_mass_kg": (
                passive_fixture_spec["mass_kg"]
                if passive_fixture_body is not None
                and passive_fixture_spec is not None
                else 0.0
            ),
            "direct_passive_fixture_actuation": False,
            "external_passive_fixture": passive_fixture_summary,
            "replaced_contacts_disabled": execution.branch == "reduced",
            "compound_joint_support_mode": (
                "finite_bushings" if execution.branch == "physical" else "ideal_revolute_links"
            ),
            "physical_internal_ideal_joints_forbidden": True,
            "compound_topology_sha256": canonical_sha256(topology),
            "physical_compound_bushings": bushing_disclosure,
            "physical_compliant_surface_contacts": compliant_surface_disclosure,
            "physical_interface_policy": collision_policy_disclosure(
                physical_interfaces, branch=execution.branch
            ),
            "functional_interface_geometry_binding": (
                functional_interface_geometry_binding
            ),
            "conditional_interface_runtime": (
                conditional_interface_runtime
            ),
            "component_contact_intent_policy": component_contact_policy,
            "all_inter_body_collision_remains_enabled": execution.branch == "physical",
            "collision_geometry_sha256": canonical_sha256(collision_manifest),
        },
        "collision_geometry": {
            "mode": "intake-bound STEP per-solid body-local collision components",
            **trace_collision_thresholds(collision_manifest),
            "schema": collision_manifest["schema"],
            "source": collision_manifest["source"],
            "assembly_file": collision_manifest["assembly_file"],
            "assembly_sha256": collision_manifest["assembly_sha256"],
            "assembly_contract_sha256": collision_manifest[
                "assembly_contract_sha256"
            ],
            "package_binding_sha256": collision_manifest[
                "package_binding_sha256"
            ],
            "component_count": collision_manifest["component_count"],
            "bodies": collision_audit,
            "contact_enabled": execution.branch == "physical",
            "physical_interface_policy": collision_policy_disclosure(
                physical_interfaces, branch=execution.branch
            ),
            "component_contact_intent_policy": component_contact_policy,
            "validator_owned_compliant_surface_contacts": (
                compliant_surface_disclosure
            ),
            "collision_filter_policy": (
                (
                    "all_body_pairs_enabled_with_hash_bound_component_local_exceptions"
                    if compliant_surface_contacts
                    else "all_inter_body_collision_pairs_enabled"
                )
                if execution.branch == "physical"
                else "all_collision_disabled_for_reduced_branch"
            ),
            "whole_body_box_proxy_used": False,
            "whole_body_convex_hull_used": False,
            "convex_hull_used": False,
            "joint_count": len(linkage.revolute_joints),
            "loop_count": linkage.loop_count,
            "certifiable_angle_limit_rad": linkage.outbound_output_angle_rad,
            "friction_assumption": linkage.friction,
            "joint_angle_limits_rad": {
                joint_id: [minimum, maximum]
                for joint_id, (minimum, maximum) in declared_joint_limits.items()
            },
            "external_contact_fixture": passive_fixture_summary,
        },
        "axis_abs": list(execution.plan.axis_abs),
        "joint_origin_abs_m": list(execution.plan.joint_origin_abs_m),
        "input_point_abs_m": list(linkage.input_point_abs_m),
        "observable_point_initial_abs_m": list(linkage.output_observable_point_abs_m),
        "characteristic_length_m": execution.plan.characteristic_length_m,
        "executed_phases": executed_phases,
        "executed_load_sha256": canonical_sha256(executed_phases),
        "sample_count": len(samples),
        "samples": samples,
        "strict_non_claims": {
            "manufacturing_certification": False,
            "strength_or_fatigue": False,
            "measured_real_friction": False,
            "dynamic_response_certified": False,
            "material_density_inertia_certified": False,
        },
    }
