from __future__ import annotations

"""Finite-contact physical runtime for a guided upright bicycle roll test."""

import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .bicycle_ground_roll_spec import (
    BICYCLE_GROUND_ROLL_PLUGIN_ID,
    BICYCLE_GROUND_ROLL_PROGRAM,
    BicycleGroundRollSpec,
    bicycle_ground_roll_from_dict,
)
from .collision_policy import (
    CONTACT_PHYSICAL,
    collision_policy_disclosure,
    interface_declaration,
)
from .io import canonical_sha256
from .models import MechanismValidationError
from .wrench_ledger import (
    add_ledgered_load,
    create_body_torque,
    set_current_wrench_phase,
    standard_wrench_declaration,
    transition_current_wrench_phase,
)

if TYPE_CHECKING:
    from .worker import ExecutionRequest


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _sub(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(a) - float(b) for a, b in zip(left, right))


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(_dot(value, value))


def _family_spec_without_collision(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MechanismValidationError(
            "BICYCLE_GROUND_ROLL_SPEC_REQUIRED",
            "runtime has no bicycle ground-roll family_spec",
        )
    return {key: item for key, item in value.items() if key != "collision_geometry"}


def _inertia_box(mass_kg: float, size_m: Sequence[float]) -> tuple[float, float, float]:
    x, y, z = (float(item) for item in size_m)
    return (
        mass_kg * (y * y + z * z) / 12.0,
        mass_kg * (x * x + z * z) / 12.0,
        mass_kg * (x * x + y * y) / 12.0,
    )


def _world_point(body: Any, local_point: Any) -> tuple[float, float, float]:
    value = body.TransformPointLocalToParent(local_point)
    return (float(value.x), float(value.y), float(value.z))


def _world_axis(body: Any, local_axis: Any) -> tuple[float, float, float]:
    value = body.TransformDirectionLocalToParent(local_axis)
    return (float(value.x), float(value.y), float(value.z))


def _axis_error(actual: Sequence[float], expected: Sequence[float]) -> float:
    denominator = max(_norm(actual) * _norm(expected), 1.0e-30)
    cosine = abs(_dot(actual, expected) / denominator)
    return math.acos(max(-1.0, min(1.0, cosine)))


def _joint_reaction(link: Any) -> dict[str, list[float]]:
    reaction = link.GetReaction2()
    return {
        "force_joint_frame_n": [
            float(reaction.force.x),
            float(reaction.force.y),
            float(reaction.force.z),
        ],
        "torque_joint_frame_nm": [
            float(reaction.torque.x),
            float(reaction.torque.y),
            float(reaction.torque.z),
        ],
    }


def _validate_program(spec: BicycleGroundRollSpec, execution: ExecutionRequest) -> None:
    load = execution.load_case
    if load.program != BICYCLE_GROUND_ROLL_PROGRAM:
        raise MechanismValidationError(
            "BICYCLE_GROUND_ROLL_PROGRAM_INVALID",
            "ground-roll runtime accepts only task_bicycle_ground_roll",
        )
    expected = (
        ("ground_contact_settle", "none", 0.50),
        ("rear_wheel_powered_drive", "torque", 0.70),
        ("passive_roll_observe", "none", 0.10),
    )
    actual = tuple((p.phase_id, p.wrench_kind, p.duration_s) for p in load.phases)
    if actual != expected:
        raise MechanismValidationError(
            "BICYCLE_GROUND_ROLL_PROGRAM_CHANGED",
            "ground-roll phases differ from the public v1 program",
        )
    drive = load.phases[1]
    if (
        drive.direction_abs != spec.wheel_axis_abs
        or drive.magnitude_si != spec.nominal_rear_drive_torque_nm
    ):
        raise MechanismValidationError(
            "BICYCLE_GROUND_ROLL_INPUT_CHANGED",
            "the powered phase must apply the declared rear-wheel torque",
        )


def simulate_bicycle_ground_roll(
    execution: ExecutionRequest,
    workspace: Path,
) -> dict[str, Any]:
    if execution.mechanism_plugin.worker_backend != BICYCLE_GROUND_ROLL_PLUGIN_ID:
        raise MechanismValidationError(
            "MECHANISM_WORKER_BACKEND_UNAVAILABLE",
            "bicycle ground-roll runtime received another backend",
        )
    if execution.branch != "physical":
        raise MechanismValidationError(
            "BICYCLE_GROUND_ROLL_IDEAL_DYNAMICS_NOT_REQUIRED",
            "the deterministic ideal reference is generated from physical saved times",
        )
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(os.path.join(conda_prefix, "Library", "bin"))
    import pychrono as chrono

    from .family_collision import (
        family_collision_geometry,
        make_family_body,
        trace_collision_thresholds,
    )
    from .worker import (
        _angular_velocity,
        _collision_progress,
        _contact_count,
        _contact_pair_counts,
        _make_material,
        _make_nsc_system,
        _pose,
        _quat_from_z_to_axis,
        _reset_body_velocity,
        _tuple3,
        _write_trace_checkpoint,
        lane_execution_source_sha256,
    )

    spec = bicycle_ground_roll_from_dict(
        _family_spec_without_collision(execution.adapter.family_spec)
    )
    _validate_program(spec, execution)
    collision = family_collision_geometry(execution.adapter, expected_bodies=spec.bodies)
    if collision["package_binding_sha256"] != execution.plan.package_binding_sha256:
        raise MechanismValidationError(
            "FAMILY_COLLIDER_PACKAGE_BINDING_CHANGED",
            "ground-roll collision geometry differs from the executed package",
        )
    geometry = execution.functional_geometry_evidence
    if not isinstance(geometry, Mapping) or geometry.get("passed") is not True:
        raise MechanismValidationError(
            "BICYCLE_GROUND_ROLL_FUNCTIONAL_GEOMETRY_INVALID",
            "ground-roll execution requires passing hash-bound CAD preflight evidence",
        )

    system = _make_nsc_system(
        chrono,
        collision_envelope_m=float(collision["collision_envelope_m"]),
        collision_margin_m=float(collision["collision_margin_m"]),
    )
    solver = chrono.ChSolverAPGD()
    solver.SetMaxIterations(200)
    solver.EnableWarmStart(True)
    system.SetSolver(solver)
    solver_name = chrono.ChSolver.GetTypeAsString(solver.GetType())
    system.SetGravitationalAcceleration(chrono.ChVector3d(*execution.plan.gravity_abs_m_s2))
    material = _make_material(chrono, spec.ground_friction)

    bodies: dict[str, Any] = {}
    inertias: dict[str, list[float]] = {}
    initial_poses: dict[str, dict[str, Any]] = {}
    collision_audit: list[dict[str, Any]] = []
    for index, body_spec in enumerate(spec.bodies):
        inertia = _inertia_box(body_spec.mass_kg, body_spec.size_m)
        body, audit = make_family_body(
            chrono,
            body_spec=body_spec,
            geometry=collision,
            workspace=workspace,
            material=material,
            inertia_xx=inertia,
            branch="physical",
        )
        body.GetCollisionModel().SetFamily(index + 1)
        audit["collision_family"] = index + 1
        _collision_progress("system_add_body_begin", body_id=body_spec.body_id)
        system.AddBody(body)
        _collision_progress("system_add_body_complete", body_id=body_spec.body_id)
        bodies[body_spec.body_id] = body
        inertias[body_spec.body_id] = list(inertia)
        initial_poses[body_spec.body_id] = _pose(body)
        collision_audit.append(audit)

    ground = bodies[spec.ground_body_id]
    chassis = bodies[spec.chassis_body_id]
    steering = bodies[spec.steering_body_id]
    rear = bodies[spec.rear_wheel_body_id]
    front = bodies[spec.front_wheel_body_id]
    if hasattr(system, "GetNumLinks") and int(system.GetNumLinks()) != 0:
        raise RuntimeError("ground-roll physical branch contains an ideal link")

    load_container = chrono.ChLoadContainer()
    system.Add(load_container)
    # The authored and visible test-stand guide keeps the unmanned bicycle
    # upright while leaving translation along the declared travel axis free.
    # It is a declared support boundary, never a wheel rolling prescription.
    chassis_guide = chrono.ChLinkLockPrismatic()
    chassis_guide.Initialize(
        chassis,
        ground,
        chrono.ChFramed(
            chassis.GetPos(),
            _quat_from_z_to_axis(chrono, spec.travel_axis_abs),
        ),
    )
    system.AddLink(chassis_guide)

    rear_frame = chrono.ChFramed(
        chrono.ChVector3d(*spec.rear_axle_origin_abs_m),
        _quat_from_z_to_axis(chrono, spec.wheel_axis_abs),
    )
    front_frame = chrono.ChFramed(
        chrono.ChVector3d(*spec.front_axle_origin_abs_m),
        _quat_from_z_to_axis(chrono, spec.wheel_axis_abs),
    )
    # The wheel hubs are declared revolute bearings. They constrain the axle
    # centers and axes but leave spin free for arbitrarily many turns. This is
    # not a rolling/no-slip shortcut: chassis travel and passive front-wheel
    # motion still arise only from the native wheel-ground contact solver.
    rear_bearing = chrono.ChLinkLockRevolute()
    rear_bearing.Initialize(chassis, rear, rear_frame)
    system.AddLink(rear_bearing)
    front_bearing = chrono.ChLinkLockRevolute()
    front_bearing.Initialize(steering, front, front_frame)
    system.AddLink(front_bearing)
    steering_bearing = chrono.ChLinkLockLock()
    steering_bearing.Initialize(
        steering,
        chassis,
        chrono.ChFramed(
            chrono.ChVector3d(*spec.steering_origin_abs_m),
            _quat_from_z_to_axis(chrono, spec.steering_axis_abs),
        ),
    )
    system.AddLink(steering_bearing)
    support_link_count = int(system.GetNumLinks()) if hasattr(system, "GetNumLinks") else None
    if support_link_count != 4:
        raise RuntimeError("ground-roll runtime did not create its four declared support joints")

    rear_task_torque = create_body_torque(
        chrono,
        rear,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        False,
        declaration=standard_wrench_declaration(
            plugin_id=BICYCLE_GROUND_ROLL_PLUGIN_ID,
            profile="task_input",
            wrench_kind="torque",
            load_id="validator-rear-wheel-ground-roll-torque",
            owner=BICYCLE_GROUND_ROLL_PLUGIN_ID,
            target_body=spec.rear_wheel_body_id,
            source_role="declared_input_port",
            target_role="declared_input",
            geometry_gate="load_case_phase_and_rear_axle_frame",
            active_interval="declared_load_case_phases",
            release_required=True,
        ),
    )
    add_ledgered_load(load_container, rear_task_torque)

    chassis_start = tuple(initial_poses[spec.chassis_body_id]["position_abs_m"])
    rear_axle_chassis_local = chassis.TransformPointParentToLocal(
        chrono.ChVector3d(*spec.rear_axle_origin_abs_m)
    )
    rear_axle_wheel_local = rear.TransformPointParentToLocal(
        chrono.ChVector3d(*spec.rear_axle_origin_abs_m)
    )
    front_axle_steering_local = steering.TransformPointParentToLocal(
        chrono.ChVector3d(*spec.front_axle_origin_abs_m)
    )
    front_axle_wheel_local = front.TransformPointParentToLocal(
        chrono.ChVector3d(*spec.front_axle_origin_abs_m)
    )
    rear_axis_local = rear.TransformDirectionParentToLocal(
        chrono.ChVector3d(*spec.wheel_axis_abs)
    )
    front_axis_local = front.TransformDirectionParentToLocal(
        chrono.ChVector3d(*spec.wheel_axis_abs)
    )

    rear_angle = 0.0
    front_angle = 0.0
    time_s = 0.0
    dynamics_step_count = 0
    samples: list[dict[str, Any]] = []
    executed_phases: list[dict[str, Any]] = []

    rear_ground_key = "|".join(sorted((spec.ground_body_id, spec.rear_wheel_body_id)))
    front_ground_key = "|".join(sorted((spec.ground_body_id, spec.front_wheel_body_id)))
    ground_spec = next(
        body for body in spec.bodies if body.body_id == spec.ground_body_id
    )
    ground_min_x = ground_spec.center_abs_m[0] - 0.5 * ground_spec.size_m[0]
    ground_max_x = ground_spec.center_abs_m[0] + 0.5 * ground_spec.size_m[0]
    ground_min_y = ground_spec.center_abs_m[1] - 0.5 * ground_spec.size_m[1]
    ground_max_y = ground_spec.center_abs_m[1] + 0.5 * ground_spec.size_m[1]

    def record(phase_id: str) -> None:
        chassis_position = tuple(_pose(chassis)["position_abs_m"])
        displacement = _sub(chassis_position, chassis_start)
        chassis_progress = _dot(displacement, spec.travel_axis_abs)
        rear_anchor_drift = _norm(
            _sub(
                _world_point(rear, rear_axle_wheel_local),
                _world_point(chassis, rear_axle_chassis_local),
            )
        )
        front_anchor_drift = _norm(
            _sub(
                _world_point(front, front_axle_wheel_local),
                _world_point(steering, front_axle_steering_local),
            )
        )
        rear_axis_error = _axis_error(
            _world_axis(rear, rear_axis_local), spec.wheel_axis_abs
        )
        front_axis_error = _axis_error(
            _world_axis(front, front_axis_local), spec.wheel_axis_abs
        )
        pair_counts = _contact_pair_counts(chrono, system)
        rear_contacts = int(pair_counts.get(rear_ground_key, 0))
        front_contacts = int(pair_counts.get(front_ground_key, 0))
        rear_center_z = float(rear.GetPos().z)
        front_center_z = float(front.GetPos().z)
        maximum_ground_penetration = max(
            0.0,
            spec.ground_plane_height_m + spec.rear_wheel_radius_m - rear_center_z,
            spec.ground_plane_height_m + spec.front_wheel_radius_m - front_center_z,
        )
        rear_center_x = float(rear.GetPos().x)
        rear_center_y = float(rear.GetPos().y)
        front_center_x = float(front.GetPos().x)
        front_center_y = float(front.GetPos().y)
        finite_surface_horizontal_gap = max(
            0.0,
            ground_min_x - rear_center_x,
            rear_center_x - ground_max_x,
            ground_min_y - rear_center_y,
            rear_center_y - ground_max_y,
            ground_min_x - front_center_x,
            front_center_x - ground_max_x,
            ground_min_y - front_center_y,
            front_center_y - ground_max_y,
        )
        maximum_ground_separation = max(
            0.0,
            rear_center_z - spec.ground_plane_height_m - spec.rear_wheel_radius_m,
            front_center_z - spec.ground_plane_height_m - spec.front_wheel_radius_m,
            finite_surface_horizontal_gap,
        )
        rolling_distance = abs(front_angle) * spec.front_wheel_radius_m
        rolling_slip_ratio = abs(rolling_distance - abs(chassis_progress)) / max(
            rolling_distance,
            abs(chassis_progress),
            1.0e-6,
        )
        samples.append(
            {
                "time_s": time_s,
                "phase_id": phase_id,
                "rear_wheel_angle_rad": rear_angle,
                "front_wheel_angle_rad": front_angle,
                "chassis_position_m": chassis_progress,
                "generalized_progress_m": chassis_progress,
                "rear_wheel_speed_rad_s": _dot(_angular_velocity(rear), spec.wheel_axis_abs),
                "front_wheel_speed_rad_s": _dot(_angular_velocity(front), spec.wheel_axis_abs),
                "chassis_speed_m_s": _dot(_tuple3(chassis.GetPosDt()), spec.travel_axis_abs),
                "rolling_slip_ratio": rolling_slip_ratio,
                "rear_axle_drift_m": rear_anchor_drift,
                "front_axle_drift_m": front_anchor_drift,
                "chassis_guide_transverse_drift_m": math.hypot(displacement[1], displacement[2]),
                "max_off_axis_rotation_rad": max(rear_axis_error, front_axis_error),
                "rear_ground_contact_count": rear_contacts,
                "front_ground_contact_count": front_contacts,
                "expected_contact_count": min(rear_contacts, front_contacts),
                "contact_count": _contact_count(system),
                "contact_pair_counts": pair_counts,
                "support_reactions": {
                    "chassis_guide": _joint_reaction(chassis_guide),
                    "rear_bearing": _joint_reaction(rear_bearing),
                    "front_bearing": _joint_reaction(front_bearing),
                    "steering_lock": _joint_reaction(steering_bearing),
                },
                "max_penetration_m": maximum_ground_penetration,
                "max_ground_separation_m": maximum_ground_separation,
                "body_poses": {body_id: _pose(body) for body_id, body in bodies.items()},
                "moving_pose": _pose(chassis),
            }
        )

    record("initial")
    next_sample_s = execution.load_case.sample_interval_s
    for phase_index, phase in enumerate(execution.load_case.phases):
        if phase.phase_id == "ground_contact_settle":
            chassis.SetFixed(True)
        transition_current_wrench_phase(phase.phase_id)
        torque = (
            phase.magnitude_si * _dot(phase.direction_abs or (0.0, 0.0, 0.0), spec.wheel_axis_abs)
            if phase.wrench_kind == "torque"
            else 0.0
        )
        rear_task_torque.SetTorque(
            chrono.ChVector3d(*(torque * item for item in spec.wheel_axis_abs)),
            False,
        )
        elapsed = 0.0
        while elapsed < phase.duration_s - 1.0e-12:
            step = min(execution.load_case.step_s, phase.duration_s - elapsed)
            system.DoStepDynamics(step)
            dynamics_step_count += 1
            elapsed += step
            time_s += step
            rear_angle += _dot(_angular_velocity(rear), spec.wheel_axis_abs) * step
            front_angle += _dot(_angular_velocity(front), spec.wheel_axis_abs) * step
            if time_s + 1.0e-12 >= next_sample_s:
                record(phase.phase_id)
                while next_sample_s <= time_s + 1.0e-12:
                    next_sample_s += execution.load_case.sample_interval_s
        if samples[-1]["time_s"] < time_s - 1.0e-12:
            record(phase.phase_id)
        rear_task_torque.SetTorque(chrono.ChVector3d(0.0, 0.0, 0.0), False)
        if phase.phase_id == "ground_contact_settle":
            chassis.SetFixed(False)
            for moving_body in (chassis, steering, rear, front):
                _reset_body_velocity(chrono, moving_body)
        executed_phases.append(
            {
                "phase_id": phase.phase_id,
                "wrench_kind": phase.wrench_kind,
                "direction_abs": None if phase.direction_abs is None else list(phase.direction_abs),
                "magnitude_si": phase.magnitude_si,
                "duration_s": elapsed,
            }
        )
        if phase_index < len(execution.load_case.phases) - 1:
            _write_trace_checkpoint(execution, samples, executed_phases=executed_phases)

    rear_task_torque.SetTorque(chrono.ChVector3d(0.0, 0.0, 0.0), False)
    set_current_wrench_phase(None)
    executed_load_sha256 = canonical_sha256(executed_phases)
    chrono_version = str(chrono.GetChronoVersion()) if hasattr(chrono, "GetChronoVersion") else None
    # ATTENTION AI EDITORS (2026-08-08): this disclosure is load-bearing, not
    # decorative.  The saved-pose interpenetration audit exempts ONLY solid
    # pairs whose expected contact is declared here as CONTACT_PHYSICAL (see
    # physical_legality.audit_family_collision_geometry).  If you add, rename,
    # or remove a contacting body pair in this runtime, update these
    # declarations IN THE SAME COMMIT, or the lane fails with
    # physical_illegal_interpenetration / policy binding errors. (DEFECT #4)
    physical_interfaces = (
        interface_declaration(
            interface_id="contact-ground-rear-wheel-roll",
            classification=CONTACT_PHYSICAL,
            body_a_id=spec.ground_body_id,
            body_b_id=spec.rear_wheel_body_id,
            model="finite_coulomb_wheel_ground_contact",
            source="built_in_bicycle_ground_roll_runtime_v1",
        ),
        interface_declaration(
            interface_id="contact-ground-front-wheel-roll",
            classification=CONTACT_PHYSICAL,
            body_a_id=spec.ground_body_id,
            body_b_id=spec.front_wheel_body_id,
            model="finite_coulomb_wheel_ground_contact",
            source="built_in_bicycle_ground_roll_runtime_v1",
        ),
    )
    physical_interface_policy = collision_policy_disclosure(
        physical_interfaces,
        branch="physical",
    )
    return {
        "schema_version": "0.1",
        "status": "COMPLETED",
        "validator_source_sha256": lane_execution_source_sha256(),
        "branch": "physical",
        "adapter_id": execution.adapter.adapter_id,
        "mechanism_plugin_id": BICYCLE_GROUND_ROLL_PLUGIN_ID,
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
            "solver": solver_name,
            "mechanism_plugin_id": BICYCLE_GROUND_ROLL_PLUGIN_ID,
            "worker_backend": BICYCLE_GROUND_ROLL_PLUGIN_ID,
            "step_s": execution.load_case.step_s,
            "gravity_abs_m_s2": list(execution.plan.gravity_abs_m_s2),
            "chrono_link_count": 4,
            # Declared support links are Harness-owned upright/axle/steering
            # supports, NOT internal ideal joints: internal_ideal_joint_count
            # must stay 0 so the universal physical-branch gate
            # (physical_ideal_comparison.py) passes. (DEFECT #2, 2026-08-07)
            "declared_support_links": [
                "ChLinkLockPrismatic",
                "ChLinkLockRevolute",
                "ChLinkLockRevolute",
                "ChLinkLockLock",
            ],
            "internal_ideal_joint_count": 0,
            "declared_axle_revolute_joint_count": 2,
            "declared_upright_guide_count": 1,
            "declared_straight_steering_lock_count": 1,
            "contact_bypass_link_count": 0,
            "contact_model": "native_finite_coulomb_wheel_ground_contact",
        },
        "branch_semantics": {
            "same_mass_kg": sum(item.mass_kg for item in spec.bodies if not item.fixed),
            "body_inertia_diagonal_kg_m2": inertias,
            "same_initial_pose": initial_poses,
            "replaced_contacts_disabled": False,
            "pose_setters_used": False,
            "direct_front_wheel_actuation_used": False,
            "task_input_body_id": spec.rear_wheel_body_id,
            "unique_task_output_body_id": spec.front_wheel_body_id,
            "chassis_fixed": False,
            "chassis_fixed_during_contact_settle": True,
            "chassis_released_and_velocities_zeroed_before_power": True,
            "front_wheel_passive": True,
            "declared_axle_revolute_joints_used": True,
            "contact_bypass_ideal_links_forbidden": True,
            "upright_fixture_mode": "visible_declared_prismatic_test_stand_guide_free_in_travel_axis",
            "steering_fixture_mode": "declared_ground_roll_task_lock_straight_ahead",
            "ground_contact_force_model": "native_finite_coulomb_contact",
            "ground_plane_height_m": spec.ground_plane_height_m,
            "rolling_constraint_used": False,
            "family_spec_sha256": canonical_sha256(spec.to_dict()),
            "physical_interface_policy": physical_interface_policy,
        },
        "functional_geometry": dict(geometry),
        "collision_geometry": {
            "mode": "intake-bound STEP per-solid body-local collision components",
            **trace_collision_thresholds(collision),
            "schema": collision["schema"],
            "source": collision["source"],
            "assembly_file": collision["assembly_file"],
            "assembly_sha256": collision["assembly_sha256"],
            "assembly_contract_sha256": collision["assembly_contract_sha256"],
            "package_binding_sha256": collision["package_binding_sha256"],
            "component_count": collision["component_count"],
            "bodies": collision_audit,
            "contact_enabled": True,
            "collision_filter_policy": "all_inter_body_collision_pairs_enabled",
            "whole_body_box_proxy_used": False,
            "whole_body_convex_hull_used": False,
            "convex_hull_used": False,
            "joint_count": 0,
            "loop_count": 0,
            "friction_assumption": spec.ground_friction,
        },
        "axis_abs": list(spec.travel_axis_abs),
        "joint_origin_abs_m": list(spec.rear_axle_origin_abs_m),
        "input_point_abs_m": list(spec.rear_axle_origin_abs_m),
        "observable_point_initial_abs_m": list(spec.front_axle_origin_abs_m),
        "characteristic_length_m": execution.plan.characteristic_length_m,
        "executed_phases": executed_phases,
        "executed_load_sha256": executed_load_sha256,
        "sample_count": len(samples),
        "samples": samples,
        "strict_non_claims": {
            "manufacturing_certification": False,
            "strength_fatigue_or_tire_wear": False,
            "unassisted_bicycle_balance_certified": False,
            "steering_dynamics_certified": False,
            "dynamic_response_certified": False,
            "ideal_no_slip_constraint_used": False,
        },
    }
