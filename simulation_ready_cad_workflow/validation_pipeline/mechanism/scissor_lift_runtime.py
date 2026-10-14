"""Reduced and physical PyChrono runtimes for the ``hydraulic_scissor_lift`` family.

Mirrors ``compound_runtime.py``: the reduced branch builds the declared ideal
topology (revolute pins, prismatic roller tracks, one cylindrical barrel-rod
link, in spec declaration order) while the physical branch uses ZERO ideal
links — every joint is a ``ChLoadBodyBodyBushingGeneric`` compliance mount and
the declared travel limits (cylinder min/max length, revolute angle limits)
are enforced as limit forces/torques, never as hard constraints.  The
hydraulic cylinder is a force pair between the barrel and rod anchors whose
axis is recomputed from the current body poses at every step.

``pychrono`` is imported lazily inside :func:`simulate_scissor_lift`; this
module is safe to import in the validator host environment.  Physics only ever
runs inside a harness queue worker (``worker.py`` dispatch).
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from .bounded_support import (
    BoundedSupportRuntime,
    RELATIVE_ANCHOR_DRIFT,
    TRANSVERSE_ANCHOR_DRIFT,
    body_component_refs,
    bounded_support_gate,
    conditional_runtime_summary,
)
from .conditional_interface import (
    build_functional_interface_geometry_binding,
)
from .io import canonical_sha256, write_json_atomic
from .models import MechanismValidationError
from .scissor_lift_spec import ScissorLiftSpec, scissor_lift_from_dict
from .task_motion_evidence import (
    registered_adaptive_termination_evidence,
    registered_stop_comparator_reached,
)
from .wrench_ledger import (
    add_ledgered_load,
    create_body_force,
    create_body_torque,
    set_current_wrench_phase,
    standard_wrench_declaration,
    transition_current_wrench_phase,
)

if TYPE_CHECKING:
    from .worker import ExecutionRequest


# Steel pin/roller-seat compliance.  These are finite (therefore physically
# distinguishable from the reduced ideal links) while keeping sub-millimetre
# drift bounded under kilonewton-scale joint loads, matching the magnitudes
# validated for the rack-pinion support bushings.
BUSHING_STIFFNESS_N_M = 25_000_000.0
BUSHING_DAMPING_N_S_M = 32_000.0
AXIS_STIFFNESS_N_M_RAD = 50_000.0
AXIS_DAMPING_N_M_S_RAD = 500.0
FREE_TRANSLATION_DAMPING_N_S_M = 3.0
FREE_AXIS_DAMPING_N_M_S_RAD = 0.08
# Backup revolute angle stops (the cylinder travel stop binds first in normal
# operation) and the cylinder axial travel stop.
LIMIT_STIFFNESS_N_M_RAD = 25_000.0
LIMIT_DAMPING_N_M_S_RAD = 50.0
CYLINDER_STOP_STIFFNESS_N_M = 750_000.0
CYLINDER_STOP_DAMPING_N_S_M = 9_000.0
# Hydraulic lock: whenever no cylinder-drive phase is active the closed valve
# holds the trapped oil volume, modelled as a stiff finite axial spring-damper
# referenced to the cylinder length at the moment the valve closed.  This is a
# finite stiffness (not an ideal constraint) so the branches stay physically
# distinguishable, and it applies to both branches because the reduced ideal
# cylindrical link is also axially free.
HYDRAULIC_LOCK_STIFFNESS_N_M = 10_000_000.0
# Lock damping sized for zeta ~= 0.5 of the platform/lock oscillator
# (omega ~= sqrt(k_lock / (mech_advantage^2 * m_moving)) ~= 135 rad/s,
# c_crit ~= 148 kN*s/m at the cylinder): the platform rings less than one
# visible cycle after a drive phase cuts, instead of the ~14 mm sustained
# oscillation measured with the original 8 kN*s/m (zeta ~= 0.054).
HYDRAULIC_LOCK_DAMPING_N_S_M = 74_000.0
# Relief-valve cap on the lock force: a real hydraulic lock circuit relieves
# above a set pressure, so the damping force never exceeds a few times the
# drive force.  Without the cap, closing the valve on a moving platform hits
# the 0.74 kg rod body with a ~30 kN step force (a ~40 km/s^2 jerk) which is
# both physically wrong and numerically toxic (it drove the physics worker
# past its 768 MiB containment limit right at the drive->settle transition).
HYDRAULIC_LOCK_RELIEF_N = 6_000.0
# Meter-out flow-control throttling on the drive phases: a real hydraulic
# cylinder circuit restricts the return-line flow, so the drive force does
# not slam the platform into the phase-cut target at full acceleration.
# The retract coefficient is larger because gravity assists the closing
# motion (the valve only meters it), while the extend coefficient must stay
# small enough that the light load case still finishes the stroke inside
# the phase time budget.  Applied as -c * cylinder_rate during drive
# phases, i.e. pure back-pressure opposing the motion in either direction.
CYLINDER_THROTTLE_EXTEND_DAMPING_N_S_M = 3_400.0
CYLINDER_THROTTLE_RETRACT_DAMPING_N_S_M = 12_000.0


def _support_alignment_axis_abs(
    kind: str,
    joint_axis_abs: Sequence[float],
    scissor_hinge_axis_abs: Sequence[float],
) -> tuple[float, float, float]:
    """Return the axis whose loss of alignment means physical capture loss.

    A roller-track joint deliberately frees both translation along the rail
    and rotation about the scissor hinge.  Its two rail directions therefore
    rotate relative to one another during ordinary deployment and cannot be
    used as a capture-alignment gate.  The common hinge/roller axis remains
    aligned in valid motion and only diverges when the roller tips out of its
    retained track.  Revolute pins and cylindrical sleeves continue to use
    their declared joint axis.
    """

    selected = (
        scissor_hinge_axis_abs if kind == "prismatic" else joint_axis_abs
    )
    return tuple(float(value) for value in selected)


def _unsigned_axis_misalignment_rad(
    axis_a: Sequence[float],
    axis_b: Sequence[float],
) -> float:
    """Measure undirected axis skew while treating opposite signs as equal."""

    norm_a = math.sqrt(sum(float(value) ** 2 for value in axis_a))
    norm_b = math.sqrt(sum(float(value) ** 2 for value in axis_b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        raise MechanismValidationError(
            "SUPPORT_ALIGNMENT_AXIS_INVALID",
            "support alignment axes must be non-zero",
        )
    cosine = sum(
        float(left) * float(right)
        for left, right in zip(axis_a, axis_b, strict=True)
    ) / (norm_a * norm_b)
    return math.acos(max(-1.0, min(1.0, abs(cosine))))


def _branch_owns_stop_time(
    reference_schedule: list[dict[str, Any]] | None,
    phase: Any,
) -> bool:
    return phase.stop_condition is not None and (
        reference_schedule is None
        or phase.branch_termination_policy == "each_branch_target"
    )


def _recorded_phase_duration(
    reference_schedule: list[dict[str, Any]] | None,
    *,
    phase: Any,
    phase_limit_s: float,
    elapsed_s: float,
) -> float:
    if (
        reference_schedule is not None
        and phase.branch_termination_policy != "each_branch_target"
    ):
        return phase_limit_s
    return elapsed_s


def _external_load_schedule_sha256(
    executed_phases: Sequence[dict[str, Any]],
) -> str:
    return canonical_sha256(list(executed_phases))


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


def _point_velocity(body: Any, point_abs: Sequence[float], tuple3: Any) -> tuple[float, float, float]:
    center = tuple3(body.GetPos())
    linear = tuple3(body.GetPosDt())
    angular = tuple3(body.GetAngVelParent())
    return _add(linear, _cross(angular, _sub(point_abs, center)))


def _body_axis_abs(body: Any, axis_local: Any, tuple3: Any) -> tuple[float, float, float]:
    return tuple3(body.TransformDirectionLocalToParent(axis_local))


def _quat_slide_x_hinge_z(
    chrono: Any,
    slide_axis: Sequence[float],
    hinge_axis: Sequence[float],
) -> Any:
    """Quaternion for the roller-track joint frame: local +X runs along the
    track slide axis and local +Z runs along the scissor hinge axis.

    ``ChLinkLockRevolutePrismatic`` frees translation along its local X and
    rotation about its local Z (verified numerically against the installed
    pychrono), which is exactly the roller-in-track interface: the roller
    slides along the track and the arm rotates about the hinge axis.  The
    physical-branch bushing uses the same frame so its released compliance
    components (tx, rz) coincide with the ideal link's free DOF.
    """
    x = _scale(slide_axis, 1.0 / max(_norm(slide_axis), 1e-12))
    z = _scale(hinge_axis, 1.0 / max(_norm(hinge_axis), 1e-12))
    y = _unit3(_cross(z, x))
    # Rotation matrix with columns (x, y, z); convert to a quaternion (wxyz)
    # with the standard Shepperd method.
    m00, m01, m02 = x[0], y[0], z[0]
    m10, m11, m12 = x[1], y[1], z[1]
    m20, m21, m22 = x[2], y[2], z[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = (0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s)
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        quat = ((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s)
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        quat = ((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s)
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        quat = ((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s)
    norm = math.sqrt(sum(item * item for item in quat))
    return chrono.ChQuaterniond(*(item / norm for item in quat))


def _unit3(value: Sequence[float]) -> tuple[float, float, float]:
    norm = _norm(value)
    if norm <= 1e-12:
        raise MechanismValidationError(
            "SCISSOR_LIFT_AXIS_INVALID", "joint frame axes cannot be parallel"
        )
    return _scale(value, 1.0 / norm)


def simulate_scissor_lift(
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

    if execution.mechanism_plugin.worker_backend != "hydraulic_scissor_lift_v1":
        raise MechanismValidationError(
            "MECHANISM_WORKER_BACKEND_UNAVAILABLE",
            "scissor-lift runtime received a different mechanism backend",
        )
    spec = scissor_lift_from_dict(execution.adapter.family_spec)
    from .family_collision import (
        family_collision_geometry,
        make_family_body,
        trace_collision_thresholds,
    )
    from .collision_policy import (
        COMPLIANT_INTERFACE_PHYSICAL,
        collision_policy_disclosure,
        interface_declaration,
    )

    collision_manifest = family_collision_geometry(
        execution.adapter,
        expected_bodies=spec.bodies,
    )
    if collision_manifest["package_binding_sha256"] != execution.plan.package_binding_sha256:
        raise MechanismValidationError(
            "FAMILY_COLLIDER_PACKAGE_BINDING_CHANGED",
            "scissor-lift collision geometry does not match the executed package",
        )
    finite_capture_report = collision_manifest.get("finite_linear_capture")
    finite_capture_spec = spec.finite_linear_captures[0]
    finite_capture_records = (
        finite_capture_report.get("captures")
        if isinstance(finite_capture_report, dict)
        else None
    )
    finite_capture_geometry = next(
        (
            item
            for item in finite_capture_records or ()
            if isinstance(item, dict)
            and item.get("capture_id") == finite_capture_spec.capture_id
        ),
        None,
    )
    if (
        not isinstance(finite_capture_report, dict)
        or finite_capture_report.get("passed") is not True
        or not isinstance(finite_capture_geometry, dict)
        or finite_capture_geometry.get("passed") is not True
    ):
        raise MechanismValidationError(
            "FINITE_CAPTURE_GEOMETRY_REQUIRED",
            "scissor-lift execution requires passing STEP-bound finite "
            "capture evidence",
        )
    parent_capture_length_m = float(
        finite_capture_geometry["parent_capture_length_m"]
    )
    child_capture_length_m = float(
        finite_capture_geometry["child_capture_length_m"]
    )
    all_joint_specs: list[tuple[str, Any]] = [
        *(("revolute", item) for item in spec.revolute_joints),
        *(("prismatic", item) for item in spec.prismatic_joints),
        ("cylindrical", spec.cylindrical_joint),
    ]
    support_gate_by_joint: dict[str, dict[str, Any]] = {}
    for kind, joint in all_joint_specs:
        support_gate_by_joint[joint.joint_id] = bounded_support_gate(
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
            maximum_drift_m=spec.max_joint_drift_m,
            coordinate_mode=(
                RELATIVE_ANCHOR_DRIFT
                if kind == "revolute"
                else TRANSVERSE_ANCHOR_DRIFT
            ),
            maximum_axis_misalignment_rad=spec.max_off_plane_rotation_rad,
            template_id=(
                "finite_revolute_pin_capture_v1"
                if kind == "revolute"
                else (
                    "finite_roller_track_retention_v1"
                    if kind == "prismatic"
                    else "finite_cylindrical_lateral_capture_v1"
                )
            ),
            proof_source="validator_scissor_support_envelope_v1",
        )
    # The scissor hinge axis (shared by every revolute joint) orients the
    # roller-track joint frames: their local +Z is the hinge axis so the freed
    # rotation is the arm-swing rotation.
    scissor_hinge_axis_abs = spec.revolute_joints[0].axis_abs
    reference_schedule = _reference_schedule(execution)
    system = _make_nsc_system(
        chrono,
        collision_envelope_m=12e-6,
        collision_margin_m=2e-6,
    )
    if execution.branch == "physical":
        # Generic body-body bushings contribute stiffness and damping matrices;
        # MINRES supports those matrices while the NSC default PSOR solver does not.
        solver = chrono.ChSolverMINRES()
        if hasattr(solver, "SetMaxIterations"):
            solver.SetMaxIterations(250)
        if hasattr(solver, "SetTolerance"):
            solver.SetTolerance(1e-10)
        system.SetSolver(solver)
    system.SetGravitationalAcceleration(
        chrono.ChVector3d(*execution.plan.gravity_abs_m_s2)
    )
    material = _make_material(chrono, spec.friction)
    bodies: dict[str, Any] = {}
    inertias: dict[str, list[float]] = {}
    initial_poses: dict[str, dict[str, Any]] = {}
    initial_quaternions: dict[str, tuple[float, float, float, float]] = {}
    collision_audit: list[dict[str, Any]] = []
    collision_families: dict[str, int] = {}
    for body_index, body_spec in enumerate(spec.bodies):
        inertia = _inertia_box(body_spec.mass_kg, body_spec.size_m)
        body, audit = make_family_body(
            chrono,
            body_spec=body_spec,
            geometry=collision_manifest,
            workspace=workspace,
            material=material,
            inertia_xx=inertia,
            branch=execution.branch,
        )
        if execution.branch == "physical":
            if body_index >= 15:
                raise MechanismValidationError(
                    "SCISSOR_LIFT_COLLISION_FAMILY_CAPACITY_EXCEEDED",
                    "exact joint-pair collision filtering currently supports at most 15 bodies",
                )
            family = body_index + 1
            body.GetCollisionModel().SetFamily(family)
            collision_families[body_spec.body_id] = family
            audit["collision_family"] = family
        _collision_progress("system_add_body_begin", body_id=body_spec.body_id)
        system.Add(body)
        _collision_progress("system_add_body_complete", body_id=body_spec.body_id)
        bodies[body_spec.body_id] = body
        collision_audit.append(audit)
        inertias[body_spec.body_id] = list(inertia)
        initial_poses[body_spec.body_id] = _pose(body)
        initial_quaternions[body_spec.body_id] = tuple(
            initial_poses[body_spec.body_id]["rotation_abs_quat_wxyz"]
        )

    excluded_joint_collision_pairs: list[dict[str, Any]] = []
    # Keep all physical body pairs collision-enabled.  Joint bushings constrain
    # motion but do not own or erase the surrounding solid geometry.

    ideal_links: list[str] = []
    ideal_limit_disclosure: dict[str, Any] = {}
    if execution.branch == "reduced":
        for joint in spec.revolute_joints:
            revolute = chrono.ChLinkLockRevolute()
            revolute.Initialize(
                bodies[joint.child_body_id],
                bodies[joint.parent_body_id],
                chrono.ChFramed(
                    chrono.ChVector3d(*joint.origin_abs_m),
                    _quat_from_z_to_axis(chrono, joint.axis_abs),
                ),
            )
            if joint.min_angle_rad is not None and joint.max_angle_rad is not None:
                ideal_limit_disclosure[joint.joint_id] = _configure_revolute_angle_limit(
                    revolute, joint.max_angle_rad, min_angle_rad=joint.min_angle_rad
                )
            system.AddLink(revolute)
            ideal_links.append("ChLinkLockRevolute")
        for joint in spec.prismatic_joints:
            # Roller-track interface: slide along the track (local X) plus
            # rotation about the scissor hinge axis (local Z).  A plain
            # ChLinkLockPrismatic would lock the relative rotation and
            # over-constrain the 1-DOF scissor loop.
            prismatic = chrono.ChLinkLockRevolutePrismatic()
            prismatic.Initialize(
                bodies[joint.child_body_id],
                bodies[joint.parent_body_id],
                chrono.ChFramed(
                    chrono.ChVector3d(*joint.origin_abs_m),
                    _quat_slide_x_hinge_z(
                        chrono, joint.axis_abs, scissor_hinge_axis_abs
                    ),
                ),
            )
            system.AddLink(prismatic)
            ideal_links.append("ChLinkLockRevolutePrismatic")
        cylindrical = chrono.ChLinkLockCylindrical()
        cylindrical.Initialize(
            bodies[spec.cylindrical_joint.child_body_id],
            bodies[spec.cylindrical_joint.parent_body_id],
            chrono.ChFramed(
                chrono.ChVector3d(*spec.cylindrical_joint.origin_abs_m),
                _quat_from_z_to_axis(chrono, spec.cylindrical_joint.axis_abs),
            ),
        )
        system.AddLink(cylindrical)
        ideal_links.append("ChLinkLockCylindrical")
    link_count = int(system.GetNumLinks()) if hasattr(system, "GetNumLinks") else None
    if execution.branch == "physical" and link_count not in {0, None}:
        raise RuntimeError("scissor-lift physical branch unexpectedly contains an ideal link")
    if execution.branch == "reduced" and link_count != spec.joint_count:
        raise RuntimeError("scissor-lift reduced branch did not create every declared joint")

    load_container = chrono.ChLoadContainer()
    system.Add(load_container)
    platform = bodies[spec.output_body_id]
    output_local = platform.TransformPointParentToLocal(
        chrono.ChVector3d(*spec.output_reference_point_abs_m)
    )
    platform_up_local = platform.TransformDirectionParentToLocal(
        chrono.ChVector3d(0.0, 0.0, 1.0)
    )
    platform_hinge_local = platform.TransformDirectionParentToLocal(
        chrono.ChVector3d(*scissor_hinge_axis_abs)
    )
    barrel_body = bodies[spec.actuator.barrel_body_id]
    rod_body = bodies[spec.actuator.rod_body_id]
    barrel_anchor_local = barrel_body.TransformPointParentToLocal(
        chrono.ChVector3d(*spec.actuator.barrel_anchor_abs_m)
    )
    rod_anchor_local = rod_body.TransformPointParentToLocal(
        chrono.ChVector3d(*spec.actuator.rod_anchor_abs_m)
    )
    plugin_id = execution.mechanism_plugin.plugin_id
    barrel_force = create_body_force(
        chrono,
        barrel_body,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        False,
        barrel_body.GetPos(),
        False,
        declaration=standard_wrench_declaration(
            plugin_id=plugin_id,
            profile="task_input_dynamic_axis_reaction",
            wrench_kind="force",
            load_id="hydraulic-actuator-barrel-reaction",
            owner=plugin_id,
            target_body=spec.actuator.barrel_body_id,
            source_role="declared_input_port",
            target_role="declared_input",
            geometry_gate="finite_cylinder_capture_and_relief_gate",
            active_interval="declared_actuator_phases",
            release_required=True,
        ),
    )
    rod_force = create_body_force(
        chrono,
        rod_body,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        False,
        rod_body.GetPos(),
        False,
        declaration=standard_wrench_declaration(
            plugin_id=plugin_id,
            profile="task_input_dynamic_axis",
            wrench_kind="force",
            load_id="hydraulic-actuator-rod-force",
            owner=plugin_id,
            target_body=spec.actuator.rod_body_id,
            source_role="declared_input_port",
            target_role="declared_input",
            geometry_gate="finite_cylinder_capture_and_relief_gate",
            active_interval="declared_actuator_phases",
            release_required=True,
        ),
    )
    probe_force = create_body_force(
        chrono,
        platform,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        False,
        platform.GetPos(),
        False,
        declaration=standard_wrench_declaration(
            plugin_id=plugin_id,
            profile="probe",
            wrench_kind="force",
            load_id="platform-output-resistance-probe",
            owner=plugin_id,
            target_body=spec.output_body_id,
            source_role="validator_output_probe",
            target_role="task_output",
            geometry_gate="non_actuator_probe_phase_only",
            active_interval="declared_probe_phases",
            release_required=True,
        ),
    )
    add_ledgered_load(load_container, barrel_force)
    add_ledgered_load(load_container, rod_force)
    add_ledgered_load(load_container, probe_force)
    barrel_lock_force = create_body_force(
        chrono,
        barrel_body,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        False,
        barrel_body.GetPos(),
        False,
        declaration=standard_wrench_declaration(
            plugin_id=plugin_id,
            profile="constitutive",
            wrench_kind="force",
            load_id="hydraulic-lock-barrel-reaction",
            owner=plugin_id,
            target_body=spec.actuator.barrel_body_id,
            source_role="finite_hydraulic_lock",
            target_role="declared_input",
            geometry_gate="finite_cylinder_capture_and_relief_gate",
            active_interval="geometry_gated_cylinder_interval",
            release_required=True,
        ),
    )
    rod_lock_force = create_body_force(
        chrono,
        rod_body,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        False,
        rod_body.GetPos(),
        False,
        declaration=standard_wrench_declaration(
            plugin_id=plugin_id,
            profile="constitutive",
            wrench_kind="force",
            load_id="hydraulic-lock-rod-force",
            owner=plugin_id,
            target_body=spec.actuator.rod_body_id,
            source_role="finite_hydraulic_lock",
            target_role="declared_input",
            geometry_gate="finite_cylinder_capture_and_relief_gate",
            active_interval="geometry_gated_cylinder_interval",
            release_required=True,
        ),
    )
    barrel_stop_force = create_body_force(
        chrono,
        barrel_body,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        False,
        barrel_body.GetPos(),
        False,
        declaration=standard_wrench_declaration(
            plugin_id=plugin_id,
            profile="limit",
            wrench_kind="force",
            load_id="cylinder-stop-barrel-reaction",
            owner=plugin_id,
            target_body=spec.actuator.barrel_body_id,
            source_role="finite_cylinder_travel_stop",
            target_role="declared_input",
            geometry_gate="declared_cylinder_travel_interval",
            active_interval="geometry_gated_physical_interval",
            release_required=True,
        ),
    )
    rod_stop_force = create_body_force(
        chrono,
        rod_body,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        False,
        rod_body.GetPos(),
        False,
        declaration=standard_wrench_declaration(
            plugin_id=plugin_id,
            profile="limit",
            wrench_kind="force",
            load_id="cylinder-stop-rod-force",
            owner=plugin_id,
            target_body=spec.actuator.rod_body_id,
            source_role="finite_cylinder_travel_stop",
            target_role="declared_input",
            geometry_gate="declared_cylinder_travel_interval",
            active_interval="geometry_gated_physical_interval",
            release_required=True,
        ),
    )
    for load in (
        barrel_lock_force,
        rod_lock_force,
        barrel_stop_force,
        rod_stop_force,
    ):
        add_ledgered_load(load_container, load)

    runtime_joints: list[dict[str, Any]] = []
    if execution.branch == "physical":
        for kind, joint in all_joint_specs:
            parent = bodies[joint.parent_body_id]
            child = bodies[joint.child_body_id]
            origin = chrono.ChVector3d(*joint.origin_abs_m)
            stiffness = chrono.ChMatrix66d()
            damping = chrono.ChMatrix66d()
            stiffness.SetZero()
            damping.SetZero()
            if kind == "revolute":
                # Free only rotation about the joint axis (frame-local rz).
                stiffness_diag = (
                    BUSHING_STIFFNESS_N_M,
                    BUSHING_STIFFNESS_N_M,
                    BUSHING_STIFFNESS_N_M,
                    AXIS_STIFFNESS_N_M_RAD,
                    AXIS_STIFFNESS_N_M_RAD,
                    0.0,
                )
                damping_diag = (
                    BUSHING_DAMPING_N_S_M,
                    BUSHING_DAMPING_N_S_M,
                    BUSHING_DAMPING_N_S_M,
                    AXIS_DAMPING_N_M_S_RAD,
                    AXIS_DAMPING_N_M_S_RAD,
                    FREE_AXIS_DAMPING_N_M_S_RAD,
                )
            elif kind == "prismatic":
                # Roller-track interface: free translation along the track
                # (frame-local tx) and free rotation about the scissor hinge
                # (frame-local rz); the frame comes from
                # _quat_slide_x_hinge_z so these match the reduced ideal link.
                stiffness_diag = (
                    0.0,
                    BUSHING_STIFFNESS_N_M,
                    BUSHING_STIFFNESS_N_M,
                    AXIS_STIFFNESS_N_M_RAD,
                    AXIS_STIFFNESS_N_M_RAD,
                    0.0,
                )
                damping_diag = (
                    FREE_TRANSLATION_DAMPING_N_S_M,
                    BUSHING_DAMPING_N_S_M,
                    BUSHING_DAMPING_N_S_M,
                    AXIS_DAMPING_N_M_S_RAD,
                    AXIS_DAMPING_N_M_S_RAD,
                    FREE_AXIS_DAMPING_N_M_S_RAD,
                )
            else:
                # Cylindrical barrel-rod: free translation along the cylinder
                # axis (tz) AND free rotation about it (rz).
                stiffness_diag = (
                    BUSHING_STIFFNESS_N_M,
                    BUSHING_STIFFNESS_N_M,
                    0.0,
                    AXIS_STIFFNESS_N_M_RAD,
                    AXIS_STIFFNESS_N_M_RAD,
                    0.0,
                )
                damping_diag = (
                    BUSHING_DAMPING_N_S_M,
                    BUSHING_DAMPING_N_S_M,
                    FREE_TRANSLATION_DAMPING_N_S_M,
                    AXIS_DAMPING_N_M_S_RAD,
                    AXIS_DAMPING_N_M_S_RAD,
                    FREE_AXIS_DAMPING_N_M_S_RAD,
                )
            for index, value in enumerate(stiffness_diag):
                stiffness.SetItem(index, index, value)
            for index, value in enumerate(damping_diag):
                damping.SetItem(index, index, value)
            frame_quat = (
                _quat_slide_x_hinge_z(
                    chrono, joint.axis_abs, scissor_hinge_axis_abs
                )
                if kind == "prismatic"
                else _quat_from_z_to_axis(chrono, joint.axis_abs)
            )
            bushing = chrono.ChLoadBodyBodyBushingGeneric(
                child,
                parent,
                chrono.ChFramed(origin, frame_quat),
                stiffness,
                damping,
            )
            load_container.Add(bushing)
            child_role = (
                "task_output"
                if joint.child_body_id == spec.output_body_id
                else "declared_input"
                if joint.child_body_id
                in {
                    spec.actuator.barrel_body_id,
                    spec.actuator.rod_body_id,
                }
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
                    "task_output"
                    if joint.parent_body_id == spec.output_body_id
                    else "declared_input"
                    if joint.parent_body_id
                    in {
                        spec.actuator.barrel_body_id,
                        spec.actuator.rod_body_id,
                    }
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
                    "kind": kind,
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
                    "parent_alignment_axis_local": (
                        parent.TransformDirectionParentToLocal(
                            chrono.ChVector3d(
                                *_support_alignment_axis_abs(
                                    kind,
                                    joint.axis_abs,
                                    scissor_hinge_axis_abs,
                                )
                            )
                        )
                    ),
                    "child_alignment_axis_local": (
                        child.TransformDirectionParentToLocal(
                            chrono.ChVector3d(
                                *_support_alignment_axis_abs(
                                    kind,
                                    joint.axis_abs,
                                    scissor_hinge_axis_abs,
                                )
                            )
                        )
                    ),
                    "parent_limit_torque": parent_limit_torque,
                    "child_limit_torque": child_limit_torque,
                    "min_angle_rad": (
                        joint.min_angle_rad if kind == "revolute" else None
                    ),
                    "max_angle_rad": (
                        joint.max_angle_rad if kind == "revolute" else None
                    ),
                }
            )
            runtime_joints[-1]["zero_stiffness_matrix"].SetZero()
            runtime_joints[-1]["zero_damping_matrix"].SetZero()

    current_joint_drifts: dict[str, float] = {
        joint.joint_id: 0.0 for _kind, joint in all_joint_specs
    }
    current_reaction_forces: dict[str, float] = {
        joint.joint_id: 0.0 for _kind, joint in all_joint_specs
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
            parent_alignment_axis = _body_axis_abs(
                parent,
                runtime["parent_alignment_axis_local"],
                _tuple3,
            )
            child_alignment_axis = _body_axis_abs(
                child,
                runtime["child_alignment_axis_local"],
                _tuple3,
            )
            mean_axis_raw = _add(parent_axis, child_axis)
            mean_axis_norm = max(_norm(mean_axis_raw), 1e-12)
            mean_axis = _scale(mean_axis_raw, 1.0 / mean_axis_norm)
            axis_misalignment = _unsigned_axis_misalignment_rad(
                parent_alignment_axis,
                child_alignment_axis,
            )
            parent_velocity = _point_velocity(
                parent,
                parent_point,
                _tuple3,
            )
            child_velocity = _point_velocity(
                child,
                child_point,
                _tuple3,
            )
            free_axis = (
                None
                if runtime["kind"] == "revolute"
                else (
                    parent_axis
                    if runtime["kind"] == "prismatic"
                    else mean_axis
                )
            )
            capture_overlap = (
                finite_capture_overlap_m(cylinder_state()[3])
                if runtime["kind"] == "cylindrical"
                else None
            )
            support_state = runtime["support_gate"].evaluate(
                time_s=time_s,
                anchor_a_abs_m=parent_point,
                anchor_b_abs_m=child_point,
                velocity_a_abs_m_s=parent_velocity,
                velocity_b_abs_m_s=child_velocity,
                requested_effort_si=_norm(
                    _tuple3(runtime["bushing"].GetForce())
                ),
                free_axis_abs=free_axis,
                axis_misalignment_rad=axis_misalignment,
                additional_constraints_satisfied=(
                    capture_overlap is None
                    or capture_overlap
                    >= finite_capture_spec.minimum_overlap_m
                ),
                additional_metrics=(
                    {}
                    if capture_overlap is None
                    else {"finite_capture_overlap_m": capture_overlap}
                ),
                additional_blockers=(
                    ()
                    if capture_overlap is None
                    else ("finite_capture_overlap_lost",)
                ),
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
            limit_torque = 0.0
            minimum = runtime["min_angle_rad"]
            maximum = runtime["max_angle_rad"]
            if minimum is not None and joint_angle < minimum:
                limit_torque = (
                    -LIMIT_STIFFNESS_N_M_RAD * (joint_angle - minimum)
                    - LIMIT_DAMPING_N_M_S_RAD * min(free_speed, 0.0)
                )
            elif maximum is not None and joint_angle > maximum:
                limit_torque = (
                    -LIMIT_STIFFNESS_N_M_RAD * (joint_angle - maximum)
                    - LIMIT_DAMPING_N_M_S_RAD * max(free_speed, 0.0)
                )
            if not support_active:
                limit_torque = 0.0
            limit_torque_abs = _scale(mean_axis, limit_torque)
            runtime["child_limit_torque"].SetTorque(
                chrono.ChVector3d(*limit_torque_abs), False
            )
            if runtime["parent_limit_torque"] is not None:
                runtime["parent_limit_torque"].SetTorque(
                    chrono.ChVector3d(*_scale(limit_torque_abs, -1.0)), False
                )
            # Joint drift is measured only along the constrained directions:
            # roller tracks and the cylinder legitimately slide along their
            # free axis, so that component is not drift.  Revolute pins
            # constrain every translational direction.  NOTE: the joint kind
            # must come from the runtime record (not the setup-loop variable,
            # which leaks its last value "cylindrical" into every joint).
            # For roller tracks the slide axis is carried by the track
            # (parent) body: the scissor arm (child) rotates as it opens, so
            # subtracting the mean axis would leak the legitimate slide
            # travel into the drift metric.
            if runtime["kind"] == "revolute":
                current_joint_drifts[joint.joint_id] = _norm(displacement)
            elif runtime["kind"] == "prismatic":
                along_free = _dot(displacement, parent_axis)
                current_joint_drifts[joint.joint_id] = _norm(
                    _sub(displacement, _scale(parent_axis, along_free))
                )
            else:
                along_free = _dot(displacement, mean_axis)
                current_joint_drifts[joint.joint_id] = _norm(
                    _sub(displacement, _scale(mean_axis, along_free))
                )
            current_reaction_forces[joint.joint_id] = (
                _norm(_tuple3(runtime["bushing"].GetForce()))
                if support_active
                else 0.0
            )

    def cylinder_state() -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        float,
    ]:
        barrel_point = _tuple3(barrel_body.TransformPointLocalToParent(barrel_anchor_local))
        rod_point = _tuple3(rod_body.TransformPointLocalToParent(rod_anchor_local))
        axis_raw = _sub(rod_point, barrel_point)
        length = max(_norm(axis_raw), 1e-12)
        return _scale(axis_raw, 1.0 / length), barrel_point, rod_point, length

    def finite_capture_overlap_m(cylinder_length_m: float) -> float:
        return (
            parent_capture_length_m
            + child_capture_length_m
            - cylinder_length_m
        )

    initial_reference_height_m = spec.output_reference_point_abs_m[2]
    samples: list[dict[str, Any]] = []
    executed_phases: list[dict[str, Any]] = []
    time_s = 0.0
    next_sample_s = 0.0
    last_cylinder_length_m = cylinder_state()[3]
    current_cylinder_force_n = 0.0

    def record(phase_id: str, applied_force: tuple[float, float, float]) -> None:
        output_point = _tuple3(platform.TransformPointLocalToParent(output_local))
        output_velocity = _point_velocity(platform, output_point, _tuple3)
        platform_up = _body_axis_abs(platform, platform_up_local, _tuple3)
        level_error = math.acos(
            max(-1.0, min(1.0, _dot(platform_up, (0.0, 0.0, 1.0))))
        )
        hinge_axis = _body_axis_abs(platform, platform_hinge_local, _tuple3)
        off_plane = math.acos(
            max(-1.0, min(1.0, _dot(hinge_axis, scissor_hinge_axis_abs)))
        )
        height = output_point[2]
        progress = height - initial_reference_height_m
        _axis, _barrel_point, _rod_point, cylinder_length = cylinder_state()
        capture_overlap = finite_capture_overlap_m(cylinder_length)
        samples.append(
            {
                "time_s": time_s,
                "phase_id": phase_id,
                "applied_force_abs_n": list(applied_force),
                "actuator_axial_force_n": current_cylinder_force_n,
                "moving_pose": _pose(platform),
                "body_poses": {body_id: _pose(body) for body_id, body in bodies.items()},
                "observable_point_abs_m": list(output_point),
                "platform_height_m": height,
                "platform_speed_m_s": output_velocity[2],
                "platform_progress_m": progress,
                "generalized_progress_m": progress,
                "generalized_speed_m_s": output_velocity[2],
                "cylinder_length_m": cylinder_length,
                "finite_capture_overlaps_m": {
                    finite_capture_spec.capture_id: capture_overlap,
                },
                "minimum_finite_capture_overlap_m": capture_overlap,
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
                "off_plane_rotation_rad": off_plane,
                "platform_level_error_rad": level_error,
                "contact_count": _contact_count(system),
                "contact_pair_counts": (
                    _contact_pair_counts(chrono, system)
                    if execution.branch == "physical"
                    else {}
                ),
            }
        )
        _write_trace_checkpoint(
            execution,
            samples,
            executed_phases=executed_phases,
        )

    def platform_progress() -> float:
        return _tuple3(platform.TransformPointLocalToParent(output_local))[2] - (
            initial_reference_height_m
        )

    update_bushings()
    record("initial", (0.0, 0.0, 0.0))
    next_sample_s = execution.load_case.sample_interval_s
    first_dynamics_step = True
    dynamics_step_count = 0
    for phase_index, phase in enumerate(execution.load_case.phases):
        transition_current_wrench_phase(phase.phase_id)
        if phase.wrench_kind == "torque":
            raise MechanismValidationError(
                "SCISSOR_LIFT_TORQUE_LOAD_UNSUPPORTED",
                "scissor-lift backend accepts a cylinder force pair, not torque",
            )
        direction = phase.direction_abs or (0.0, 0.0, 0.0)
        force_abs = tuple(value * phase.magnitude_si for value in direction)
        # Cylinder phases drive the actuator force pair; any other force phase
        # (the transverse probe) acts directly on the platform reference point.
        actuator_phase = phase.phase_id.startswith("cylinder_")
        phase_limit_s = (
            phase.duration_s
            if (
                reference_schedule is None
                or phase.branch_termination_policy == "each_branch_target"
            )
            else reference_schedule[phase_index]["duration_s"]
        )
        # Close the hydraulic valve as each phase begins: non-drive phases
        # (settles, zero-input, transverse probe) hold the oil volume trapped
        # at this instant, so the platform stays where the preceding drive
        # phase left it (a finite lock stiffness, not an ideal constraint).
        hold_reference_m = cylinder_state()[3]
        elapsed = 0.0
        stop_reached = False
        while elapsed < phase_limit_s - 1e-12:
            step = min(execution.load_case.step_s, phase_limit_s - elapsed)
            axis, barrel_point, rod_point, cylinder_length = cylinder_state()
            capture_overlap = finite_capture_overlap_m(cylinder_length)
            capture_engaged = (
                capture_overlap
                >= finite_capture_spec.minimum_overlap_m - 1e-9
            )
            cylinder_rate = (
                (cylinder_length - last_cylinder_length_m) / step if step > 0 else 0.0
            )
            last_cylinder_length_m = cylinder_length
            if actuator_phase:
                drive_scalar = _dot(force_abs, axis)
                throttle = (
                    CYLINDER_THROTTLE_EXTEND_DAMPING_N_S_M
                    if drive_scalar >= 0.0
                    else CYLINDER_THROTTLE_RETRACT_DAMPING_N_S_M
                )
                constitutive_scalar = -throttle * cylinder_rate
            else:
                lock_scalar = (
                    -HYDRAULIC_LOCK_STIFFNESS_N_M
                    * (cylinder_length - hold_reference_m)
                    - HYDRAULIC_LOCK_DAMPING_N_S_M * cylinder_rate
                )
                # Relief valve: cap the trapped-oil force magnitude.
                constitutive_scalar = max(
                    -HYDRAULIC_LOCK_RELIEF_N,
                    min(HYDRAULIC_LOCK_RELIEF_N, lock_scalar),
                )
                drive_scalar = 0.0
            limit_scalar = 0.0
            if execution.branch == "physical":
                if cylinder_length < spec.actuator.min_length_m:
                    limit_scalar = (
                        CYLINDER_STOP_STIFFNESS_N_M
                        * (spec.actuator.min_length_m - cylinder_length)
                        - CYLINDER_STOP_DAMPING_N_S_M * min(cylinder_rate, 0.0)
                    )
                elif cylinder_length > spec.actuator.max_length_m:
                    limit_scalar = (
                        -CYLINDER_STOP_STIFFNESS_N_M
                        * (cylinder_length - spec.actuator.max_length_m)
                        - CYLINDER_STOP_DAMPING_N_S_M * max(cylinder_rate, 0.0)
                    )
            if not capture_engaged:
                drive_scalar = 0.0
                constitutive_scalar = 0.0
                limit_scalar = 0.0
            total_scalar = drive_scalar + constitutive_scalar + limit_scalar
            current_cylinder_force_n = total_scalar
            rod_force.SetApplicationPoint(chrono.ChVector3d(*rod_point), False)
            rod_force.SetForce(
                chrono.ChVector3d(*_scale(axis, drive_scalar)),
                False,
            )
            barrel_force.SetApplicationPoint(chrono.ChVector3d(*barrel_point), False)
            barrel_force.SetForce(
                chrono.ChVector3d(*_scale(axis, -drive_scalar)), False
            )
            rod_lock_force.SetApplicationPoint(
                chrono.ChVector3d(*rod_point), False
            )
            rod_lock_force.SetForce(
                chrono.ChVector3d(*_scale(axis, constitutive_scalar)),
                False,
            )
            barrel_lock_force.SetApplicationPoint(
                chrono.ChVector3d(*barrel_point), False
            )
            barrel_lock_force.SetForce(
                chrono.ChVector3d(*_scale(axis, -constitutive_scalar)),
                False,
            )
            rod_stop_force.SetApplicationPoint(
                chrono.ChVector3d(*rod_point), False
            )
            rod_stop_force.SetForce(
                chrono.ChVector3d(*_scale(axis, limit_scalar)),
                False,
            )
            barrel_stop_force.SetApplicationPoint(
                chrono.ChVector3d(*barrel_point), False
            )
            barrel_stop_force.SetForce(
                chrono.ChVector3d(*_scale(axis, -limit_scalar)),
                False,
            )
            probe_point = _tuple3(platform.TransformPointLocalToParent(output_local))
            probe_force.SetApplicationPoint(chrono.ChVector3d(*probe_point), False)
            probe_force.SetForce(
                chrono.ChVector3d(*(0.0, 0.0, 0.0) if actuator_phase else force_abs),
                False,
            )
            update_bushings()
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
            if time_s + 1e-12 >= next_sample_s:
                record(phase.phase_id, force_abs)
                while next_sample_s <= time_s + 1e-12:
                    next_sample_s += execution.load_case.sample_interval_s
            # Fixed-duration phases still replay the certified physical
            # schedule.  A phase explicitly declared as an operator-style
            # "apply effort until this position" policy instead evaluates the
            # same target independently in each branch, under the same effort
            # and declared timeout.
            if _branch_owns_stop_time(
                reference_schedule,
                phase,
            ):
                progress = platform_progress()
                target = float(phase.target_position_si)
                reached = registered_stop_comparator_reached(
                    phase.stop_condition,
                    target_coordinate_si=target,
                    actual_coordinate_si=progress,
                )
                if reached:
                    stop_reached = True
                    break
        if not samples or samples[-1]["time_s"] < time_s - 1e-12:
            record(phase.phase_id, force_abs)
        rod_force.SetForce(chrono.ChVector3d(0.0, 0.0, 0.0), False)
        barrel_force.SetForce(chrono.ChVector3d(0.0, 0.0, 0.0), False)
        executed_phase = {
            "phase_id": phase.phase_id,
            "wrench_kind": phase.wrench_kind,
            "direction_abs": (
                None if phase.direction_abs is None else list(phase.direction_abs)
            ),
            "magnitude_si": phase.magnitude_si,
            "duration_s": (
                _recorded_phase_duration(
                    reference_schedule,
                    phase=phase,
                    phase_limit_s=phase_limit_s,
                    elapsed_s=elapsed,
                )
            ),
        }
        if phase.stop_condition is not None:
            if (
                reference_schedule is not None
                and phase.branch_termination_policy != "each_branch_target"
                and isinstance(
                    reference_schedule[phase_index].get("termination"),
                    dict,
                )
            ):
                executed_phase["termination"] = dict(
                    reference_schedule[phase_index]["termination"]
                )
            else:
                executed_phase["termination"] = (
                    registered_adaptive_termination_evidence(
                        phase,
                        actual_coordinate_si=platform_progress(),
                        reached=stop_reached,
                    )
                )
        executed_phases.append(executed_phase)

    for load in (
        rod_force,
        barrel_force,
        rod_lock_force,
        barrel_lock_force,
        rod_stop_force,
        barrel_stop_force,
        probe_force,
    ):
        load.SetForce(chrono.ChVector3d(0.0, 0.0, 0.0), False)
    set_current_wrench_phase(None)
    topology = spec.to_dict()
    total_mass = sum(item.mass_kg for item in spec.bodies if not item.fixed)
    bushing_disclosure = [
        {
            "joint_id": joint.joint_id,
            "joint_kind": kind,
            "chrono_type": "ChLoadBodyBodyBushingGeneric",
            "constrained_translation_stiffness_n_m": BUSHING_STIFFNESS_N_M,
            "constrained_translation_damping_n_s_m": BUSHING_DAMPING_N_S_M,
            "constrained_axis_stiffness_n_m_rad": AXIS_STIFFNESS_N_M_RAD,
            "constrained_axis_damping_n_m_s_rad": AXIS_DAMPING_N_M_S_RAD,
            "freed_components": (
                ["tx_slide", "rz_hinge"]
                if kind == "prismatic"
                else (["rz_hinge"] if kind == "revolute" else ["tz_axis", "rz_axis"])
            ),
            "free_translation_damping_n_s_m": FREE_TRANSLATION_DAMPING_N_S_M,
            "free_axis_damping_n_m_s_rad": FREE_AXIS_DAMPING_N_M_S_RAD,
            "ideal_constraint": False,
        }
        for kind, joint in all_joint_specs
    ] if execution.branch == "physical" else []
    physical_interfaces = [
        interface_declaration(
            interface_id=f"support-{joint.joint_id}",
            classification=COMPLIANT_INTERFACE_PHYSICAL,
            body_a_id=joint.parent_body_id,
            body_b_id=joint.child_body_id,
            model=(
                "finite_revolute_bushing"
                if kind == "revolute"
                else (
                    "finite_roller_track_bushing"
                    if kind == "prismatic"
                    else "finite_cylindrical_bushing"
                )
            ),
            source="trusted_hydraulic_scissor_lift_runtime_v1",
            free_coordinates=(
                ("rotation_about_axis",)
                if kind == "revolute"
                else (
                    ("translation_along_track", "rotation_about_hinge_axis")
                    if kind == "prismatic"
                    else ("translation_along_axis", "rotation_about_axis")
                )
            ),
            parameters={
                "constrained_translation_stiffness_n_m": BUSHING_STIFFNESS_N_M,
                "constrained_translation_damping_n_s_m": BUSHING_DAMPING_N_S_M,
                "constrained_axis_stiffness_n_m_rad": AXIS_STIFFNESS_N_M_RAD,
                "constrained_axis_damping_n_m_s_rad": AXIS_DAMPING_N_M_S_RAD,
                "free_translation_damping_n_s_m": FREE_TRANSLATION_DAMPING_N_S_M,
                "free_axis_damping_n_m_s_rad": FREE_AXIS_DAMPING_N_M_S_RAD,
                **(
                    {
                        "finite_capture_id": finite_capture_spec.capture_id,
                        "minimum_overlap_m": (
                            finite_capture_spec.minimum_overlap_m
                        ),
                        "parent_capture_length_m": parent_capture_length_m,
                        "child_capture_length_m": child_capture_length_m,
                        "out_of_range_policy": (
                            "continue_dynamics_with_exact_zero_interface_force"
                        ),
                    }
                    if kind == "cylindrical"
                    else {}
                ),
            },
            transmission_gate=support_gate_by_joint[joint.joint_id],
        )
        for kind, joint in all_joint_specs
    ]
    functional_interface_geometry_binding = (
        build_functional_interface_geometry_binding(
            physical_interfaces,
            collision_manifest,
        )
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
    payload = {
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
                "ChSolverMINRES" if execution.branch == "physical" else "ChSolverPSOR"
            ),
            "mechanism_plugin_id": execution.mechanism_plugin.plugin_id,
            "worker_backend": execution.mechanism_plugin.worker_backend,
            "collision_backend": "BULLET",
            "step_s": execution.load_case.step_s,
            "gravity_abs_m_s2": list(execution.plan.gravity_abs_m_s2),
            "chrono_link_count": link_count,
            "ideal_links": ideal_links,
            "internal_ideal_joint_count": len(ideal_links),
            "ideal_limits": ideal_limit_disclosure,
            "compliant_supports": [],
            "scissor_lift_bushings": bushing_disclosure,
            "actuator": "hydraulic_cylinder_force_pair_axis_recomputed_each_step",
            "hydraulic_lock": {
                "model": "closed_valve_trapped_oil_volume_spring_damper_relief_capped",
                "stiffness_n_m": HYDRAULIC_LOCK_STIFFNESS_N_M,
                "damping_n_s_m": HYDRAULIC_LOCK_DAMPING_N_S_M,
                "relief_n": HYDRAULIC_LOCK_RELIEF_N,
                "reference": "cylinder length at the start of each non-drive phase",
                "applies_to_branches": ["physical", "reduced"],
            },
            "drive_throttle": {
                "model": "meter_out_back_pressure_linear_in_cylinder_rate",
                "extend_damping_n_s_m": CYLINDER_THROTTLE_EXTEND_DAMPING_N_S_M,
                "retract_damping_n_s_m": CYLINDER_THROTTLE_RETRACT_DAMPING_N_S_M,
                "applies_to_branches": ["physical", "reduced"],
            },
            "collision_shape_counts": {
                body_id: (
                    int(body.GetCollisionModel().GetNumShapes())
                    if body.GetCollisionModel() is not None
                    else 0
                )
                for body_id, body in bodies.items()
            },
        },
        "branch_semantics": {
            "same_mass_kg": total_mass,
            "same_inertia_xx_kg_m2": inertias,
            "same_initial_pose": initial_poses,
            "same_external_load_sha256": _external_load_schedule_sha256(
                executed_phases
            ),
            "replaced_contacts_disabled": execution.branch == "reduced",
            "support_mode": (
                "finite_bushings_with_force_limited_stops"
                if execution.branch == "physical"
                else "ideal_revolute_prismatic_cylindrical_links"
            ),
            "physical_internal_ideal_joints_forbidden": True,
            "family_spec_sha256": canonical_sha256(topology),
            "physical_interface_policy": collision_policy_disclosure(
                physical_interfaces,
                branch=execution.branch,
            ),
            "functional_interface_geometry_binding": (
                functional_interface_geometry_binding
            ),
            "conditional_interface_runtime": (
                conditional_interface_runtime
            ),
            "finite_linear_capture": finite_capture_report,
            "collision_geometry_sha256": canonical_sha256(collision_manifest),
            "excluded_joint_collision_pairs": excluded_joint_collision_pairs,
            "all_inter_body_collision_remains_enabled": execution.branch == "physical",
            "actuator_model": {
                "kind": spec.actuator.kind,
                "nominal_input_force_n": spec.nominal_input_force_n,
                "return_input_force_n": spec.return_input_force_n,
                "free_axial_rotation": spec.actuator.free_axial_rotation,
                "length_limits_m": [
                    spec.actuator.min_length_m,
                    spec.actuator.max_length_m,
                ],
                "stop_stiffness_n_m": CYLINDER_STOP_STIFFNESS_N_M,
                "stop_damping_n_s_m": CYLINDER_STOP_DAMPING_N_S_M,
                "source": "validator_hydraulic_cylinder_force_pair_v1",
            },
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
                physical_interfaces,
                branch=execution.branch,
            ),
            "excluded_joint_collision_pairs": excluded_joint_collision_pairs,
            "collision_filter_policy": (
                "all_inter_body_collision_pairs_enabled"
                if execution.branch == "physical"
                else "all_collision_disabled_for_reduced_branch"
            ),
            "whole_body_box_proxy_used": False,
            "whole_body_convex_hull_used": False,
            "convex_hull_used": False,
            "joint_count": spec.joint_count,
            "loop_count": spec.loop_count,
            "progress_limits_m": [spec.home_progress_m, spec.outbound_progress_m],
            "cylinder_length_limits_m": [
                spec.cylindrical_joint.min_length_m,
                spec.cylindrical_joint.max_length_m,
            ],
            "revolute_angle_limits_rad": {
                joint.joint_id: [joint.min_angle_rad, joint.max_angle_rad]
                for joint in spec.revolute_joints
                if joint.min_angle_rad is not None
            },
            "friction_assumption": spec.friction,
        },
        "axis_abs": list(spec.output_axis_abs),
        "joint_origin_abs_m": list(execution.plan.joint_origin_abs_m),
        "actuator_barrel_anchor_abs_m": list(spec.actuator.barrel_anchor_abs_m),
        "actuator_rod_anchor_abs_m": list(spec.actuator.rod_anchor_abs_m),
        "observable_point_initial_abs_m": list(spec.output_reference_point_abs_m),
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
            "hydraulic_pressure_or_flow_certified": False,
        },
    }
    # ``run_request`` normally persists the returned terminal trace.  The
    # whole-machine worker calls this family runtime directly, so close the
    # last partial checkpoint here as well.  This keeps executed phase
    # metadata and saved samples internally consistent in both entry paths.
    write_json_atomic(execution.output_file, payload)
    return payload
