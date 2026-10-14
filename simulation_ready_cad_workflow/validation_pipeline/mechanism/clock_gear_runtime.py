from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from .bounded_support import (
    BoundedSupportRuntime,
    RELATIVE_ANCHOR_DRIFT,
    body_component_refs,
    bounded_support_gate,
    conditional_runtime_summary,
)
from .collision_policy import (
    ANALYTICAL_TRANSMISSION,
    COMPLIANT_INTERFACE_PHYSICAL,
    collision_policy_disclosure,
    interface_declaration,
)
from .conditional_interface import (
    AxialClutchGeometry,
    AxialClutchPose,
    AxialGearPose,
    AxialSlidingSpurGearGeometry,
    ConditionalCompliantInterface,
    CoplanarPulleyTransferGeometry,
    RadialReachGeometry,
    RadialReachPose,
    RightAngleBevelGearGeometry,
    build_functional_interface_geometry_binding,
)
from .io import canonical_sha256
from .initial_functional_geometry import (
    build_clock_functional_geometry_contracts,
    build_multi_output_shaft_functional_geometry_contracts,
)
from .models import MechanismValidationError
from .task_motion_evidence import registered_adaptive_termination_evidence
from .wrench_ledger import (
    add_ledgered_load,
    create_body_torque,
    current_wrench_phase,
    current_wrench_ledger,
    restore_current_wrench_checkpoint,
    set_current_wrench_phase,
    standard_wrench_declaration,
    transition_current_wrench_phase,
)

if TYPE_CHECKING:
    from .worker import ExecutionRequest


_CLOCK_CHECKPOINT_RESUME_POLICY = "resume_complete_clock_state_v1"


def _clock_checkpoint_interface_ids(
    functional_interface_ids: Sequence[str],
    support_interface_ids: Sequence[str],
) -> set[str]:
    """Return every interface whose current state is checkpointed."""

    return {*functional_interface_ids, *support_interface_ids}


def _clock_checkpoint_resume(
    prior_checkpoint: Any,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Select a bound checkpoint only when it contains the full clock state."""

    checkpoint_detected = prior_checkpoint is not None
    if prior_checkpoint is None:
        return None, {
            "schema_version": "clock_checkpoint_resume_policy_v1",
            "policy": _CLOCK_CHECKPOINT_RESUME_POLICY,
            "checkpoint_detected": False,
            "checkpoint_discarded": False,
            "dynamics_state_restored": False,
            "wrench_ledger_state_restored": False,
            "execution_origin": "time_zero",
        }
    if not isinstance(prior_checkpoint, dict):
        raise MechanismValidationError(
            "CLOCK_CHECKPOINT_STATE_INVALID",
            "clock checkpoint must be a JSON object",
        )
    checkpoint = prior_checkpoint.get("checkpoint")
    resume_state = (
        checkpoint.get("resume_state")
        if isinstance(checkpoint, dict)
        else None
    )
    if (
        not isinstance(resume_state, dict)
        or resume_state.get("schema_version") != "clock_gear_resume_v1"
        or not isinstance(resume_state.get("wrench_ledger_state"), dict)
    ):
        raise MechanismValidationError(
            "CLOCK_CHECKPOINT_STATE_INVALID",
            "clock checkpoint is missing dynamics or wrench-ledger state",
        )
    return prior_checkpoint, {
        "schema_version": "clock_checkpoint_restart_policy_v1",
        "policy": _CLOCK_CHECKPOINT_RESUME_POLICY,
        "checkpoint_detected": checkpoint_detected,
        "checkpoint_discarded": False,
        "dynamics_state_restored": True,
        "wrench_ledger_state_restored": True,
        "execution_origin": "checkpoint",
    }


# Keep the finite bushing coefficients explicit in both the runtime and the
# interface disclosure.  They are intentionally not ideal revolute joints;
# saved-pose support drift and off-axis rotation remain measured outcomes.
CLOCK_SUPPORT_TRANSLATIONAL_STIFFNESS_N_M = 32_000_000.0
CLOCK_SUPPORT_TRANSLATIONAL_DAMPING_N_S_M = 14_000.0
CLOCK_SUPPORT_ROTATIONAL_STIFFNESS_NM_RAD = 65_000.0
CLOCK_SUPPORT_ROTATIONAL_DAMPING_NM_S_RAD = 90.0
CLOCK_SUPPORT_FREE_AXIS_DAMPING_NM_S_RAD = 0.00001


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _scale(value: Sequence[float], factor: float) -> tuple[float, float, float]:
    return tuple(item * factor for item in value)


def _add(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, float, float]:
    return tuple(a + b for a, b in zip(left, right))


def _sub(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return tuple(a - b for a, b in zip(left, right))


def _cross(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(_dot(value, value))


def _support_drift_from_gate_state(state: dict[str, Any]) -> float:
    """Return anchor-to-anchor drift, never body-centre-to-axis offset."""

    value = state.get("support_drift_m")
    if not isinstance(value, (int, float)):
        raise MechanismValidationError(
            "SUPPORT_DRIFT_EVIDENCE_INVALID",
            "bounded support state is missing numeric anchor drift",
        )
    drift = float(value)
    if not math.isfinite(drift) or drift < 0.0:
        raise MechanismValidationError(
            "SUPPORT_DRIFT_EVIDENCE_INVALID",
            "bounded support anchor drift must be finite and non-negative",
        )
    return drift


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _is_external_target_phase(program: str, phase_id: str) -> bool:
    return (program, phase_id) in {
        ('manual_zero_and_resume_clock', 'calibration_return'),
        ('scheduled_alarm_clock', 'advance_to_alarm_time'),
    }


def _phase_limit_seconds(
    *,
    branch: str,
    program: str,
    phase_id: str,
    declared_duration_s: float,
    reference_duration_s: float | None,
) -> float:
    # A reduced replacement has intentionally different dynamics.  Replaying
    # the physical time-to-target as an open-loop duration can overshoot a
    # zeroing task or undershoot a scheduled trigger.  These two phases retain
    # their declared timeout and re-evaluate the same external target below.
    if branch == 'reduced' and _is_external_target_phase(program, phase_id):
        return declared_duration_s
    return (
        reference_duration_s
        if reference_duration_s is not None
        else declared_duration_s
    )


def _clock_physical_solver(chrono: Any) -> Any:
    """Build the nonsymmetric solver required by the clock contact graph."""

    # The clock starts with hundreds of frictional tooth/support contacts.
    # GMRES handles the resulting non-symmetric KKT system together with the
    # finite bushing stiffness matrices. MINRES assumes symmetry and can
    # inject catastrophic velocities into this contact graph.
    solver = chrono.ChSolverGMRES()
    if hasattr(solver, 'SetMaxIterations'):
        solver.SetMaxIterations(350)
    if hasattr(solver, 'SetTolerance'):
        solver.SetTolerance(1.0e-10)
    if hasattr(solver, 'EnableDiagonalPreconditioner'):
        solver.EnableDiagonalPreconditioner(True)
    return solver


def _local_frame(body: Any, absolute_frame: Any, chrono: Any) -> Any:
    return chrono.ChFramed(
        body.TransformPointParentToLocal(absolute_frame.GetPos()),
        body.GetRot().GetConjugate() * absolute_frame.GetRot(),
    )


def _body_orientation(spec: Any, body_id: str) -> float:
    ratios = {spec.input_body_id: 1.0}
    pending = [spec.input_body_id]
    while pending:
        current = pending.pop(0)
        for mesh in spec.gear_meshes:
            if mesh.driver_body_id == current and mesh.driven_body_id not in ratios:
                ratios[mesh.driven_body_id] = -mesh.ratio * ratios[current]
                pending.append(mesh.driven_body_id)
            elif mesh.driven_body_id == current and mesh.driver_body_id not in ratios:
                ratios[mesh.driver_body_id] = -ratios[current] / mesh.ratio
                pending.append(mesh.driver_body_id)
        for transfer in spec.shaft_transfers:
            if transfer.driver_body_id == current and transfer.driven_body_id not in ratios:
                ratios[transfer.driven_body_id] = transfer.ratio * ratios[current]
                pending.append(transfer.driven_body_id)
            elif transfer.driven_body_id == current and transfer.driver_body_id not in ratios:
                ratios[transfer.driver_body_id] = ratios[current] / transfer.ratio
                pending.append(transfer.driver_body_id)
    output_ratio = ratios.get(body_id)
    if output_ratio is None or abs(output_ratio) <= 1.0e-12:
        raise MechanismValidationError(
            'CLOCK_OUTPUT_PATH_MISSING',
            f'clock input has no gear path to {body_id!r}',
        )
    return math.copysign(1.0, output_ratio)


def simulate_clock_gear_train(
    execution: ExecutionRequest,
    workspace: Path,
) -> dict[str, Any]:
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix and hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(os.path.join(conda_prefix, 'Library', 'bin'))
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
        _inertia_box,
        _load_trace_checkpoint_for_resume,
        _make_material,
        _make_nsc_system,
        _off_axis_angle,
        _pose,
        _quat_from_z_to_axis,
        _reference_schedule,
        _tuple3,
        _write_trace_checkpoint,
        lane_execution_source_sha256,
    )

    backend = execution.mechanism_plugin.worker_backend
    if backend not in {
        'spur_gear_clock_v1',
        'multi_output_shaft_coupling_v1',
    }:
        raise MechanismValidationError(
            'MECHANISM_WORKER_BACKEND_UNAVAILABLE',
            'multi-shaft runtime received a different mechanism backend',
        )
    is_multi_output = backend == 'multi_output_shaft_coupling_v1'
    if is_multi_output:
        from .multi_output_shaft_spec import multi_output_shaft_from_dict

        spec = multi_output_shaft_from_dict(execution.adapter.family_spec)
        runtime_source = 'trusted_multi_output_shaft_runtime_v1'
    else:
        from .clock_gear_train_spec import clock_gear_train_from_dict

        spec = clock_gear_train_from_dict(execution.adapter.family_spec)
        runtime_source = 'trusted_clock_gear_runtime_v1'
    if spec.collision_exclusions:
        raise MechanismValidationError(
            'CLOCK_BODY_PAIR_COLLISION_EXCLUSION_FORBIDDEN',
            'clock packages may not disable collision for an entire pair of rigid bodies; '
            'model real clearances or use validator-owned per-component collision geometry',
        )
    collision_manifest = family_collision_geometry(
        execution.adapter,
        expected_bodies=spec.bodies,
    )
    if collision_manifest['package_binding_sha256'] != execution.plan.package_binding_sha256:
        raise MechanismValidationError(
            'FAMILY_COLLIDER_PACKAGE_BINDING_CHANGED',
            'clock collision geometry does not match the executed package',
        )
    functional_geometry_contracts = (
        build_multi_output_shaft_functional_geometry_contracts(
            spec,
            collision_manifest,
            workspace_root=workspace,
        )
        if is_multi_output
        else build_clock_functional_geometry_contracts(
            spec,
            collision_manifest,
            workspace_root=workspace,
        )
    )
    functional_contract_by_id = {
        str(item["interface_id"]): item
        for item in functional_geometry_contracts
    }
    expected_functional_interface_ids = {
        *(f"gear-{item.mesh_id}" for item in spec.gear_meshes),
        *(
            f"transfer-{item.transfer_id}"
            for item in spec.shaft_transfers
        ),
        *(f"cam-{item.follower_id}" for item in spec.cam_followers),
        *(
            [f"clutch-{spec.setting_clutch.clutch_id}"]
            if spec.setting_clutch is not None
            else []
        ),
        *(
            [f"clutch-{spec.alarm_setting_clutch.clutch_id}"]
            if spec.alarm_setting_clutch is not None
            else []
        ),
    }
    if set(functional_contract_by_id) != expected_functional_interface_ids:
        raise MechanismValidationError(
            "CLOCK_FUNCTIONAL_GEOMETRY_GATE_INVENTORY_MISMATCH",
            "clock analytical transmissions do not have one shared "
            "STEP-bound geometry gate each",
        )
    runtime_suppressed_by_body: dict[str, set[str]] = {}
    for contract in functional_geometry_contracts:
        gate = contract["gate"]
        for endpoint_name in ("endpoint_a", "endpoint_b"):
            endpoint = gate[endpoint_name]
            endpoint_body_id = str(endpoint["body_id"])
            for component_ref in endpoint[
                "local_collision_component_refs"
            ]:
                body_id, separator, solid_id = str(component_ref).partition("/")
                if (
                    separator != "/"
                    or body_id != endpoint_body_id
                    or not solid_id
                ):
                    raise MechanismValidationError(
                        "CLOCK_LOCAL_COLLISION_EXCEPTION_BINDING_INVALID",
                        "clock analytical-interface collision suppression "
                        "must use its exact STEP-bound endpoint components",
                    )
                runtime_suppressed_by_body.setdefault(
                    body_id,
                    set(),
                ).add(solid_id)

    system = _make_nsc_system(
        chrono,
        collision_envelope_m=12.0e-6,
        collision_margin_m=2.0e-6,
    )
    if execution.branch == 'physical':
        system.SetSolver(_clock_physical_solver(chrono))
    system.SetGravitationalAcceleration(chrono.ChVector3d(*execution.plan.gravity_abs_m_s2))

    material = _make_material(chrono, spec.friction)
    bodies: dict[str, Any] = {}
    inertias: dict[str, list[float]] = {}
    initial_poses: dict[str, dict[str, Any]] = {}
    collision_audit: list[dict[str, Any]] = []
    collision_families: dict[str, int] = {}
    for index, body_spec in enumerate(spec.bodies):
        inertia = _inertia_box(body_spec.mass_kg, body_spec.size_m)
        body, audit = make_family_body(
            chrono,
            body_spec=body_spec,
            geometry=collision_manifest,
            workspace=workspace,
            material=material,
            inertia_xx=inertia,
            branch=execution.branch,
            runtime_suppressed_solid_ids=(
                sorted(
                    runtime_suppressed_by_body.get(
                        body_spec.body_id,
                        (),
                    )
                )
                if execution.branch == "physical"
                else ()
            ),
        )
        if execution.branch == 'physical':
            family = index + 1
            body.GetCollisionModel().SetFamily(family)
            collision_families[body_spec.body_id] = family
            audit['collision_family'] = family
        _collision_progress('system_add_body_begin', body_id=body_spec.body_id)
        system.AddBody(body)
        _collision_progress('system_add_body_complete', body_id=body_spec.body_id)
        bodies[body_spec.body_id] = body
        inertias[body_spec.body_id] = list(inertia)
        initial_poses[body_spec.body_id] = _pose(body)
        collision_audit.append(audit)

    ground = bodies[spec.fixed_body_id]
    shaft_by_body = {item.body_id: item for item in spec.shafts}
    shaft_frames = {
        item.body_id: chrono.ChFramed(
            chrono.ChVector3d(*item.origin_abs_m),
            _quat_from_z_to_axis(chrono, item.axis_abs),
        )
        for item in spec.shafts
    }
    support_gates = {
        shaft.shaft_id: bounded_support_gate(
            body_a_id=spec.fixed_body_id,
            body_b_id=shaft.body_id,
            endpoint_a_component_refs=body_component_refs(
                collision_manifest,
                spec.fixed_body_id,
            ),
            endpoint_b_component_refs=body_component_refs(
                collision_manifest,
                shaft.body_id,
            ),
            maximum_drift_m=spec.max_support_drift_m,
            coordinate_mode=RELATIVE_ANCHOR_DRIFT,
            maximum_axis_misalignment_rad=spec.max_off_axis_rotation_rad,
            template_id="finite_clock_shaft_capture_v1",
            proof_source="validator_clock_shaft_support_envelope_v1",
        )
        for shaft in spec.shafts
    }
    expected_conditional_interface_ids = _clock_checkpoint_interface_ids(
        tuple(expected_functional_interface_ids),
        tuple(f"support-{shaft.shaft_id}" for shaft in spec.shafts),
    )

    physical_interfaces = [
        interface_declaration(
            interface_id=f"support-{shaft.shaft_id}",
            classification=COMPLIANT_INTERFACE_PHYSICAL,
            body_a_id=spec.fixed_body_id,
            body_b_id=shaft.body_id,
            model="finite_shaft_bushing",
            source=runtime_source,
            free_coordinates=("rotation_about_axis",),
            parameters={
                "translational_stiffness_n_m": CLOCK_SUPPORT_TRANSLATIONAL_STIFFNESS_N_M,
                "translational_damping_n_s_m": CLOCK_SUPPORT_TRANSLATIONAL_DAMPING_N_S_M,
                "rotational_stiffness_nm_rad": CLOCK_SUPPORT_ROTATIONAL_STIFFNESS_NM_RAD,
                "rotational_damping_nm_s_rad": CLOCK_SUPPORT_ROTATIONAL_DAMPING_NM_S_RAD,
                "free_axis_damping_nm_s_rad": CLOCK_SUPPORT_FREE_AXIS_DAMPING_NM_S_RAD,
            },
            transmission_gate=support_gates[shaft.shaft_id],
        )
        for shaft in spec.shafts
    ]
    physical_interfaces.extend(
        interface_declaration(
            interface_id=str(contract["interface_id"]),
            classification=ANALYTICAL_TRANSMISSION,
            body_a_id=str(contract["body_a_id"]),
            body_b_id=str(contract["body_b_id"]),
            model=str(contract["model"]),
            source=(
                "trusted_multi_output_shaft_geometry_contract_v1"
                if is_multi_output
                else "trusted_clock_functional_geometry_contract_v1"
            ),
            transmission_gate=contract["gate"],
        )
        for contract in functional_geometry_contracts
    )
    functional_interface_geometry_binding = (
        build_functional_interface_geometry_binding(
            physical_interfaces,
            collision_manifest,
        )
    )
    interface_geometry_plugins: dict[str, Any] = {}
    interface_feature_frames: dict[
        tuple[str, str],
        dict[str, Any],
    ] = {}
    for contract in functional_geometry_contracts:
        interface_id = str(contract["interface_id"])
        gate = contract["gate"]
        plugin_id = str(gate["geometry_plugin_id"])
        parameters = gate["geometry_parameters"]
        if plugin_id == "axial_sliding_spur_gear_mesh_v1":
            interface_geometry_plugins[interface_id] = (
                AxialSlidingSpurGearGeometry(
                    driver_pitch_radius_m=float(
                        parameters["driver_pitch_radius_m"]
                    ),
                    driven_pitch_radius_m=float(
                        parameters["driven_pitch_radius_m"]
                    ),
                    radial_tolerance_m=float(
                        parameters["radial_tolerance_m"]
                    ),
                    axis_parallel_tolerance_rad=float(
                        parameters["axis_parallel_tolerance_rad"]
                    ),
                    driver_module_m=float(
                        parameters["driver_module_m"]
                    ),
                    driven_module_m=float(
                        parameters["driven_module_m"]
                    ),
                    module_relative_tolerance=float(
                        parameters["module_relative_tolerance"]
                    ),
                )
            )
            endpoint_definitions = (
                (
                    "a",
                    str(contract["body_a_id"]),
                    float(parameters["driver_axial_center_offset_m"]),
                    float(parameters["driver_face_width_m"]),
                    None,
                ),
                (
                    "b",
                    str(contract["body_b_id"]),
                    float(parameters["driven_axial_center_offset_m"]),
                    float(parameters["driven_face_width_m"]),
                    None,
                ),
            )
        elif plugin_id == "right_angle_bevel_gear_mesh_v1":
            interface_geometry_plugins[interface_id] = (
                RightAngleBevelGearGeometry(
                    driver_pitch_radius_m=float(
                        parameters["driver_pitch_radius_m"]
                    ),
                    driven_pitch_radius_m=float(
                        parameters["driven_pitch_radius_m"]
                    ),
                    pitch_surface_tolerance_m=float(
                        parameters["pitch_surface_tolerance_m"]
                    ),
                    required_shaft_angle_rad=float(
                        parameters["required_shaft_angle_rad"]
                    ),
                    shaft_angle_tolerance_rad=float(
                        parameters["shaft_angle_tolerance_rad"]
                    ),
                    driver_face_width_m=float(
                        parameters["driver_face_width_m"]
                    ),
                    driven_face_width_m=float(
                        parameters["driven_face_width_m"]
                    ),
                )
            )
            endpoint_definitions = (
                (
                    "a",
                    str(contract["body_a_id"]),
                    0.0,
                    float(parameters["driver_face_width_m"]),
                    None,
                ),
                (
                    "b",
                    str(contract["body_b_id"]),
                    0.0,
                    float(parameters["driven_face_width_m"]),
                    None,
                ),
            )
        elif plugin_id == "coplanar_pulley_transfer_v1":
            interface_geometry_plugins[interface_id] = (
                CoplanarPulleyTransferGeometry(
                    required_center_distance_m=float(
                        parameters["required_center_distance_m"]
                    ),
                    center_distance_tolerance_m=float(
                        parameters["center_distance_tolerance_m"]
                    ),
                    axis_parallel_tolerance_rad=float(
                        parameters["axis_parallel_tolerance_rad"]
                    ),
                )
            )
            endpoint_definitions = (
                (
                    "a",
                    str(contract["body_a_id"]),
                    float(parameters["driver_axial_center_offset_m"]),
                    float(parameters["driver_face_width_m"]),
                    None,
                ),
                (
                    "b",
                    str(contract["body_b_id"]),
                    float(parameters["driven_axial_center_offset_m"]),
                    float(parameters["driven_face_width_m"]),
                    None,
                ),
            )
        elif plugin_id == "radial_cam_follower_reach_v1":
            interface_geometry_plugins[interface_id] = (
                RadialReachGeometry(
                    radial_gap_tolerance_m=float(
                        parameters["radial_gap_tolerance_m"]
                    ),
                    axis_parallel_tolerance_rad=float(
                        parameters["axis_parallel_tolerance_rad"]
                    ),
                )
            )
            endpoint_definitions = (
                (
                    "a",
                    str(contract["body_a_id"]),
                    float(parameters["driver_axial_center_offset_m"]),
                    float(parameters["driver_face_width_m"]),
                    float(parameters["driver_radial_reach_m"]),
                ),
                (
                    "b",
                    str(contract["body_b_id"]),
                    float(parameters["follower_axial_center_offset_m"]),
                    float(parameters["follower_face_width_m"]),
                    float(parameters["follower_radial_reach_m"]),
                ),
            )
        elif plugin_id == "coaxial_clutch_face_gap_v1":
            interface_geometry_plugins[interface_id] = (
                AxialClutchGeometry(
                    radial_alignment_tolerance_m=float(
                        parameters["radial_alignment_tolerance_m"]
                    ),
                    axis_parallel_tolerance_rad=float(
                        parameters["axis_parallel_tolerance_rad"]
                    ),
                )
            )
            endpoint_definitions = (
                (
                    "a",
                    str(contract["body_a_id"]),
                    float(parameters["input_face_center_offset_m"]),
                    None,
                    float(parameters["input_radial_extent_m"]),
                ),
                (
                    "b",
                    str(contract["body_b_id"]),
                    float(parameters["driven_face_center_offset_m"]),
                    None,
                    float(parameters["driven_radial_extent_m"]),
                ),
            )
        else:
            raise MechanismValidationError(
                "CLOCK_FUNCTIONAL_GEOMETRY_PLUGIN_UNSUPPORTED",
                f"clock runtime cannot evaluate {plugin_id!r}",
            )
        for (
            endpoint_name,
            body_id,
            axial_offset,
            face_width,
            radial_reach,
        ) in endpoint_definitions:
            shaft = shaft_by_body[body_id]
            feature_point_abs = _add(
                shaft.origin_abs_m,
                _scale(shaft.axis_abs, axial_offset),
            )
            body = bodies[body_id]
            local_point = _tuple3(
                body.TransformPointParentToLocal(
                    chrono.ChVector3d(*feature_point_abs)
                )
            )
            local_axis = _tuple3(
                body.GetRot().GetConjugate().Rotate(
                    chrono.ChVector3d(*shaft.axis_abs)
                )
            )
            interface_feature_frames[
                (interface_id, endpoint_name)
            ] = {
                "body_id": body_id,
                "local_point_m": local_point,
                "local_axis": local_axis,
                "face_width_m": face_width,
                "radial_reach_m": radial_reach,
            }
    # Finite analytical loads guide motion but never erase complete body pairs.

    ideal_links: list[str] = []
    if execution.branch == 'reduced':
        for shaft in spec.shafts:
            revolute = chrono.ChLinkLockRevolute()
            revolute.Initialize(bodies[shaft.body_id], ground, shaft_frames[shaft.body_id])
            system.AddLink(revolute)
            ideal_links.append('ChLinkLockRevolute')
        for mesh in spec.gear_meshes:
            driver = bodies[mesh.driver_body_id]
            driven = bodies[mesh.driven_body_id]
            gear = chrono.ChLinkLockGear()
            gear.Initialize(driver, driven, chrono.ChFramed())
            gear.SetFrameShaft1(_local_frame(driver, shaft_frames[mesh.driver_body_id], chrono))
            gear.SetFrameShaft2(_local_frame(driven, shaft_frames[mesh.driven_body_id], chrono))
            gear.SetTransmissionRatio(mesh.ratio)
            gear.SetPhaseAutomatically()
            gear.SetEnforcePhase(True)
            system.AddLink(gear)
            ideal_links.append('ChLinkLockGear')
        ideal_links.extend('AnalyticalCamFollower' for _ in spec.cam_followers)
        ideal_links.extend('AnalyticalShaftTransfer' for _ in spec.shaft_transfers)
        if spec.setting_clutch is not None:
            ideal_links.append('AnalyticalSettingClutch')
        if spec.alarm_setting_clutch is not None:
            ideal_links.append('AnalyticalAlarmSettingClutch')

    actual_link_count = int(system.GetNumLinks()) if hasattr(system, 'GetNumLinks') else None
    if execution.branch == 'physical' and actual_link_count not in {0, None}:
        raise RuntimeError('clock physical branch unexpectedly contains an ideal link')
    expected_chrono_links = len(spec.shafts) + len(spec.gear_meshes)
    if execution.branch == 'reduced' and actual_link_count != expected_chrono_links:
        raise RuntimeError('clock reduced branch did not create the declared shaft and gear links')

    load_container = chrono.ChLoadContainer()
    system.Add(load_container)
    plugin_id = execution.mechanism_plugin.plugin_id
    declared_input_bodies = {
        spec.input_body_id,
        spec.setting_input_body_id,
        spec.effective_alarm_input_body_id,
    }
    torque_loads: dict[str, Any] = {}
    for shaft in spec.shafts:
        target_role = (
            "declared_input"
            if shaft.body_id in declared_input_bodies
            else "task_output"
            if shaft.body_id == spec.output_body_id
            else "internal_moving"
        )
        load = create_body_torque(
            chrono,
            bodies[shaft.body_id],
            chrono.ChVector3d(0.0, 0.0, 0.0),
            False,
            declaration=standard_wrench_declaration(
                plugin_id=plugin_id,
                profile="accumulator",
                wrench_kind="torque",
                load_id=f"shaft-wrench-accumulator-{shaft.body_id}",
                owner=plugin_id,
                target_body=shaft.body_id,
                source_role="audited_clock_wrench_components",
                target_role=target_role,
                geometry_gate="phase_schedule_and_finite_interface_gates",
                active_interval="simulation_component_sum_interval",
            ),
        )
        add_ledgered_load(load_container, load)
        torque_loads[shaft.body_id] = load

    bushings: dict[str, Any] = {}
    support_runtimes: dict[str, dict[str, Any]] = {}
    if execution.branch == 'physical':
        for shaft in spec.shafts:
            stiffness = chrono.ChMatrix66d()
            damping = chrono.ChMatrix66d()
            stiffness.SetZero()
            damping.SetZero()
            for axis_index, value in enumerate(
                (
                    CLOCK_SUPPORT_TRANSLATIONAL_STIFFNESS_N_M,
                    CLOCK_SUPPORT_TRANSLATIONAL_STIFFNESS_N_M,
                    CLOCK_SUPPORT_TRANSLATIONAL_STIFFNESS_N_M,
                    CLOCK_SUPPORT_ROTATIONAL_STIFFNESS_NM_RAD,
                    CLOCK_SUPPORT_ROTATIONAL_STIFFNESS_NM_RAD,
                    0.0,
                )
            ):
                stiffness.SetItem(axis_index, axis_index, value)
            for axis_index, value in enumerate(
                (
                    CLOCK_SUPPORT_TRANSLATIONAL_DAMPING_N_S_M,
                    CLOCK_SUPPORT_TRANSLATIONAL_DAMPING_N_S_M,
                    CLOCK_SUPPORT_TRANSLATIONAL_DAMPING_N_S_M,
                    CLOCK_SUPPORT_ROTATIONAL_DAMPING_NM_S_RAD,
                    CLOCK_SUPPORT_ROTATIONAL_DAMPING_NM_S_RAD,
                    CLOCK_SUPPORT_FREE_AXIS_DAMPING_NM_S_RAD,
                )
            ):
                damping.SetItem(axis_index, axis_index, value)
            bushing = chrono.ChLoadBodyBodyBushingGeneric(
                bodies[shaft.body_id],
                ground,
                shaft_frames[shaft.body_id],
                stiffness,
                damping,
            )
            load_container.Add(bushing)
            bushings[shaft.shaft_id] = bushing
            zero_stiffness = chrono.ChMatrix66d()
            zero_damping = chrono.ChMatrix66d()
            zero_stiffness.SetZero()
            zero_damping.SetZero()
            origin = chrono.ChVector3d(*shaft.origin_abs_m)
            support_runtimes[shaft.shaft_id] = {
                "interface_id": f"support-{shaft.shaft_id}",
                "parent": ground,
                "child": bodies[shaft.body_id],
                "bushing": bushing,
                "stiffness_matrix": stiffness,
                "damping_matrix": damping,
                "zero_stiffness_matrix": zero_stiffness,
                "zero_damping_matrix": zero_damping,
                "parent_anchor_local": (
                    ground.TransformPointParentToLocal(origin)
                ),
                "child_anchor_local": (
                    bodies[shaft.body_id].TransformPointParentToLocal(origin)
                ),
                "parent_axis_local": (
                    ground.TransformDirectionParentToLocal(
                        chrono.ChVector3d(*shaft.axis_abs)
                    )
                ),
                "child_axis_local": (
                    bodies[shaft.body_id].TransformDirectionParentToLocal(
                        chrono.ChVector3d(*shaft.axis_abs)
                    )
                ),
                "gate": BoundedSupportRuntime(
                    interface_id=f"support-{shaft.shaft_id}",
                    gate=support_gates[shaft.shaft_id],
                ),
            }

    reference_schedule = _reference_schedule(execution)
    output_orientation = _body_orientation(spec, spec.output_body_id)
    shaft_angles = {shaft.body_id: 0.0 for shaft in spec.shafts}
    current_mesh_errors = {
        **{mesh.mesh_id: 0.0 for mesh in spec.gear_meshes},
        **{item.transfer_id: 0.0 for item in spec.shaft_transfers},
    }
    current_follower_errors = {item.follower_id: 0.0 for item in spec.cam_followers}
    current_setting_clutch_error = 0.0
    current_alarm_setting_clutch_error = 0.0
    current_support_drifts = {shaft.shaft_id: 0.0 for shaft in spec.shafts}
    current_bushing_reactions = {shaft.shaft_id: 0.0 for shaft in spec.shafts}
    current_applied_torque = (0.0, 0.0, 0.0)
    conditional_interfaces = {
        interface_id: ConditionalCompliantInterface(
            contract["gate"]
        )
        for interface_id, contract in functional_contract_by_id.items()
    }
    conditional_interface_evaluation_counts = {
        interface_id: 0 for interface_id in conditional_interfaces
    }
    conditional_interface_state_paths = {
        interface_id: [] for interface_id in conditional_interfaces
    }
    conditional_interface_max_inactive_efforts = {
        interface_id: 0.0 for interface_id in conditional_interfaces
    }
    current_conditional_interface_states: dict[
        str,
        dict[str, Any],
    ] = {}
    conditional_interface_reference_errors = {
        interface_id: 0.0 for interface_id in conditional_interfaces
    }
    torque_limit_nm = max(
        1.5,
        8.0 * spec.nominal_input_torque_nm,
        8.0 * spec.effective_nominal_alarm_input_torque_nm,
    )
    # Clock wheel trains use low-friction pivots; a generic machine-joint drag
    # would be reflected roughly 60^2 through the seconds train and stall the
    # barrel before it reaches the display.
    shaft_drag_nm_s_rad = 0.00001
    manual_setting_program = execution.load_case.program == 'manual_time_setting_clock'
    zero_resume_program = execution.load_case.program == 'manual_zero_and_resume_clock'
    hour_alarm_setting_program = (
        execution.load_case.program == 'hour_hand_alarm_setting_clock'
    )
    scheduled_alarm_program = execution.load_case.program == 'scheduled_alarm_clock'
    hour_set_alarm_program = (
        (hour_alarm_setting_program or scheduled_alarm_program)
        and spec.alarm_setting_clutch is not None
    )
    setting_program = manual_setting_program or zero_resume_program
    isolated_alarm_program = (
        execution.load_case.program == 'alarm_clock_cycle' and spec.alarm_isolated
    )
    if scheduled_alarm_program and spec.scheduled_alarm is None:
        raise MechanismValidationError(
            'CLOCK_SCHEDULED_ALARM_REQUIRED',
            'scheduled alarm program requires a declared scheduled_alarm spec',
        )
    if hour_alarm_setting_program and (
        spec.scheduled_alarm is None or spec.alarm_setting_clutch is None
    ):
        raise MechanismValidationError(
            'CLOCK_HOUR_ALARM_SETTING_REQUIRED',
            'hour-hand alarm setting requires scheduled_alarm and '
            'alarm_setting_clutch declarations',
        )
    holding_program = execution.load_case.program in {
        'zero_input_clock_train',
        'transverse_clock_probe',
    }
    current_phase_id = 'initial'
    current_setting_clutch_engaged = manual_setting_program or zero_resume_program
    current_alarm_setting_clutch_engaged = hour_set_alarm_program
    current_additional_alarm_torque_nm = 0.0
    current_alarm_hold_angle_rad = 0.0
    current_alarm_target_hold_angle_rad = (
        shaft_angles[spec.scheduled_alarm.setting_body_id]
        if spec.scheduled_alarm is not None
        else 0.0
    )

    def phase_input_body_id(phase_id: str) -> str:
        if zero_resume_program:
            return (
                spec.setting_input_body_id
                if phase_id in {'initial', 'calibration_offset', 'calibration_return'}
                else spec.input_body_id
            )
        if scheduled_alarm_program:
            if hour_set_alarm_program and phase_id in {
                'initial',
                'turn_hour_hand_to_alarm_target',
                'restore_hour_hand_to_current_time',
            }:
                return spec.hour_body_id
            return (
                spec.setting_input_body_id
                if phase_id in {'initial', 'set_alarm_for_one_minute'}
                else spec.input_body_id
            )
        if hour_alarm_setting_program:
            return spec.hour_body_id
        if manual_setting_program:
            return spec.setting_input_body_id
        if isolated_alarm_program:
            return spec.effective_alarm_input_body_id
        return spec.input_body_id

    def input_frame_for_body(
        body_id: str,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        if body_id == spec.setting_input_body_id:
            return spec.setting_input_point_abs_m, spec.setting_input_axis_abs
        if body_id == spec.effective_alarm_input_body_id and spec.alarm_isolated:
            return spec.effective_alarm_input_point_abs_m, spec.effective_alarm_input_axis_abs
        if body_id in shaft_by_body:
            shaft = shaft_by_body[body_id]
            return shaft.origin_abs_m, shaft.axis_abs
        return spec.input_point_abs_m, spec.input_axis_abs

    current_active_input_body_id = phase_input_body_id(current_phase_id)
    current_active_input_point_abs_m, current_active_input_axis_abs = input_frame_for_body(
        current_active_input_body_id
    )

    def shaft_speeds() -> dict[str, float]:
        return {
            shaft.body_id: _dot(_angular_velocity(bodies[shaft.body_id]), shaft.axis_abs)
            for shaft in spec.shafts
        }

    def live_feature(interface_id: str, endpoint_name: str) -> dict[str, Any]:
        feature = interface_feature_frames[
            (interface_id, endpoint_name)
        ]
        body = bodies[str(feature["body_id"])]
        return {
            **feature,
            "point_abs_m": _tuple3(
                body.TransformPointLocalToParent(
                    chrono.ChVector3d(*feature["local_point_m"])
                )
            ),
            "axis_abs": _tuple3(
                body.GetRot().Rotate(
                    chrono.ChVector3d(*feature["local_axis"])
                )
            ),
            "linear_velocity_abs_m_s": _tuple3(body.GetPosDt()),
        }

    def interface_observation(interface_id: str) -> Any:
        plugin = interface_geometry_plugins[interface_id]
        endpoint_a = live_feature(interface_id, "a")
        endpoint_b = live_feature(interface_id, "b")
        if isinstance(
            plugin,
            (
                AxialSlidingSpurGearGeometry,
                CoplanarPulleyTransferGeometry,
                RightAngleBevelGearGeometry,
            ),
        ):
            driver = AxialGearPose(
                center_abs_m=endpoint_a["point_abs_m"],
                axis_abs=endpoint_a["axis_abs"],
                face_width_m=float(endpoint_a["face_width_m"]),
                linear_velocity_abs_m_s=endpoint_a[
                    "linear_velocity_abs_m_s"
                ],
            )
            driven = AxialGearPose(
                center_abs_m=endpoint_b["point_abs_m"],
                axis_abs=endpoint_b["axis_abs"],
                face_width_m=float(endpoint_b["face_width_m"]),
                linear_velocity_abs_m_s=endpoint_b[
                    "linear_velocity_abs_m_s"
                ],
            )
            return plugin.evaluate(
                time_s=time_s,
                driver=driver,
                driven=driven,
            )
        if isinstance(plugin, RadialReachGeometry):
            driver = RadialReachPose(
                center_abs_m=endpoint_a["point_abs_m"],
                axis_abs=endpoint_a["axis_abs"],
                radial_reach_m=float(endpoint_a["radial_reach_m"]),
                face_width_m=float(endpoint_a["face_width_m"]),
                linear_velocity_abs_m_s=endpoint_a[
                    "linear_velocity_abs_m_s"
                ],
            )
            follower = RadialReachPose(
                center_abs_m=endpoint_b["point_abs_m"],
                axis_abs=endpoint_b["axis_abs"],
                radial_reach_m=float(endpoint_b["radial_reach_m"]),
                face_width_m=float(endpoint_b["face_width_m"]),
                linear_velocity_abs_m_s=endpoint_b[
                    "linear_velocity_abs_m_s"
                ],
            )
            return plugin.evaluate(
                time_s=time_s,
                driver=driver,
                follower=follower,
            )
        if isinstance(plugin, AxialClutchGeometry):
            input_face = AxialClutchPose(
                face_center_abs_m=endpoint_a["point_abs_m"],
                axis_abs=endpoint_a["axis_abs"],
                radial_extent_m=float(endpoint_a["radial_reach_m"]),
                linear_velocity_abs_m_s=endpoint_a[
                    "linear_velocity_abs_m_s"
                ],
            )
            driven_face = AxialClutchPose(
                face_center_abs_m=endpoint_b["point_abs_m"],
                axis_abs=endpoint_b["axis_abs"],
                radial_extent_m=float(endpoint_b["radial_reach_m"]),
                linear_velocity_abs_m_s=endpoint_b[
                    "linear_velocity_abs_m_s"
                ],
            )
            return plugin.evaluate(
                time_s=time_s,
                input_face=input_face,
                driven_face=driven_face,
            )
        raise RuntimeError(
            f"unsupported clock interface geometry: {type(plugin)!r}"
        )

    def step_conditional_interface(interface_id: str) -> Any:
        observation = interface_observation(interface_id)
        decision = conditional_interfaces[interface_id].step(observation)
        conditional_interface_evaluation_counts[interface_id] += 1
        state_path = conditional_interface_state_paths[interface_id]
        if not state_path or state_path[-1] != decision.state:
            state_path.append(decision.state)
        return decision

    def transmit_conditional_effort(
        interface_id: str,
        decision: Any,
        requested_effort_si: float,
    ) -> float:
        transmitted = decision.transmit_scalar(requested_effort_si)
        if not decision.transmission_active:
            conditional_interface_max_inactive_efforts[
                interface_id
            ] = max(
                conditional_interface_max_inactive_efforts[interface_id],
                abs(transmitted),
            )
        current_conditional_interface_states[interface_id] = {
            "interface_id": interface_id,
            "state": decision.state,
            "transmission_active": decision.transmission_active,
            "transmitted_effort_si": transmitted,
            "activation_coordinate_si": (
                decision.observation.activation_coordinate_si
            ),
            "constraints_satisfied": (
                decision.observation.constraints_satisfied
            ),
        }
        return transmitted

    def update_support_interfaces() -> None:
        if execution.branch != "physical":
            return
        for shaft in spec.shafts:
            runtime = support_runtimes[shaft.shaft_id]
            parent = runtime["parent"]
            child = runtime["child"]
            parent_point = _tuple3(
                parent.TransformPointLocalToParent(
                    runtime["parent_anchor_local"]
                )
            )
            child_point = _tuple3(
                child.TransformPointLocalToParent(
                    runtime["child_anchor_local"]
                )
            )
            parent_axis = _tuple3(
                parent.TransformDirectionLocalToParent(
                    runtime["parent_axis_local"]
                )
            )
            child_axis = _tuple3(
                child.TransformDirectionLocalToParent(
                    runtime["child_axis_local"]
                )
            )
            axis_misalignment = math.acos(
                max(
                    -1.0,
                    min(1.0, abs(_dot(parent_axis, child_axis))),
                )
            )

            def point_velocity(
                body: Any,
                point_abs_m: Sequence[float],
            ) -> tuple[float, float, float]:
                return _add(
                    _tuple3(body.GetPosDt()),
                    _cross(
                        _angular_velocity(body),
                        _sub(point_abs_m, _tuple3(body.GetPos())),
                    ),
                )

            support_state = runtime["gate"].evaluate(
                time_s=time_s,
                anchor_a_abs_m=parent_point,
                anchor_b_abs_m=child_point,
                velocity_a_abs_m_s=point_velocity(
                    parent,
                    parent_point,
                ),
                velocity_b_abs_m_s=point_velocity(
                    child,
                    child_point,
                ),
                requested_effort_si=_norm(
                    _tuple3(runtime["bushing"].GetForce())
                ),
                axis_misalignment_rad=axis_misalignment,
            )
            current_conditional_interface_states[
                runtime["interface_id"]
            ] = support_state
            current_support_drifts[
                shaft.shaft_id
            ] = _support_drift_from_gate_state(support_state)
            active = bool(support_state["transmission_active"])
            runtime["bushing"].SetStiffnessMatrix(
                runtime[
                    "stiffness_matrix"
                    if active
                    else "zero_stiffness_matrix"
                ]
            )
            runtime["bushing"].SetDampingMatrix(
                runtime[
                    "damping_matrix"
                    if active
                    else "zero_damping_matrix"
                ]
            )

    def update_couplings(external_torque_abs: tuple[float, float, float]) -> None:
        nonlocal current_applied_torque, current_setting_clutch_error
        nonlocal current_alarm_setting_clutch_error
        nonlocal current_additional_alarm_torque_nm
        update_support_interfaces()
        speeds = shaft_speeds()
        torques = {shaft.body_id: [0.0, 0.0, 0.0] for shaft in spec.shafts}

        def add_axis_torque(body_id: str, scalar: float) -> None:
            axis = shaft_by_body[body_id].axis_abs
            for component in range(3):
                torques[body_id][component] += axis[component] * scalar

        if _norm(external_torque_abs) > 1.0e-12:
            current_wrench_ledger().record_task_input_event(
                target_body=current_active_input_body_id,
                source_role="declared_input_port",
                active_interval=f"phase:{current_phase_id}",
                vector_si=external_torque_abs,
                law_id=(
                    "validator_bounded_controller_phase_v1"
                    if current_phase_id
                    in {
                        "turn_hour_hand_to_alarm_target",
                        "restore_hour_hand_to_current_time",
                    }
                    else "validator_load_phase_v1"
                ),
            )
        for component in range(3):
            torques[current_active_input_body_id][component] += external_torque_abs[component]
        for shaft in spec.shafts:
            add_axis_torque(shaft.body_id, -shaft_drag_nm_s_rad * speeds[shaft.body_id])
        time_detent_active = (
            holding_program
            or isolated_alarm_program
            or (
                scheduled_alarm_program
                and current_phase_id in {
                    'initial',
                    'alarm_target_ready',
                }
            )
            or (
                hour_alarm_setting_program
                and current_phase_id == 'initial'
            )
        )
        if time_detent_active:
            # An unwound clock is retained by its click/escapement.  This
            # finite detent arrests the otherwise neutral global gear-train
            # coordinate without prescribing any hand angle during powered
            # running or manual setting.
            detent_torque = _clamp(
                -8.0 * shaft_angles[spec.input_body_id]
                - 0.50 * speeds[spec.input_body_id],
                1.00,
            )
            add_axis_torque(spec.input_body_id, detent_torque)
        current_additional_alarm_torque_nm = 0.0
        if scheduled_alarm_program:
            scheduled = spec.scheduled_alarm
            assert scheduled is not None
            if current_phase_id == 'ring_with_timekeeping':
                current_additional_alarm_torque_nm = (
                    spec.effective_nominal_alarm_input_torque_nm
                )
                add_axis_torque(
                    spec.effective_alarm_input_body_id,
                    current_additional_alarm_torque_nm,
                )
            else:
                alarm_detent_torque = _clamp(
                    -8.0
                    * (
                        shaft_angles[spec.effective_alarm_input_body_id]
                        - current_alarm_hold_angle_rad
                    )
                    - 0.50 * speeds[spec.effective_alarm_input_body_id],
                    1.00,
                )
                add_axis_torque(
                    spec.effective_alarm_input_body_id,
                    alarm_detent_torque,
                )
            if not hour_set_alarm_program:
                setting_body = scheduled.setting_body_id
                setting_angle = shaft_angles[setting_body]
                setting_speed = speeds[setting_body]
                if current_phase_id == 'set_alarm_for_one_minute':
                    overshoot = max(
                        0.0,
                        setting_angle - scheduled.setting_target_angle_rad,
                    )
                    if overshoot > 0.0 or setting_speed > 0.0:
                        add_axis_torque(
                            setting_body,
                            _clamp(
                                -45.0 * overshoot
                                - 0.30 * max(0.0, setting_speed),
                                0.60,
                            ),
                        )
                elif current_phase_id not in {'initial'}:
                    add_axis_torque(
                        setting_body,
                        _clamp(
                            -28.0
                            * (
                                setting_angle
                                - scheduled.setting_target_angle_rad
                            )
                            - 0.25 * setting_speed,
                            0.60,
                        ),
                    )

        if hour_set_alarm_program:
            scheduled = spec.scheduled_alarm
            assert scheduled is not None
            assert spec.alarm_setting_clutch is not None
            target_detent_limit_nm = min(
                spec.alarm_setting_clutch.max_torque_nm,
                0.002,
            )
            target_body = scheduled.setting_body_id
            target_angle = shaft_angles[target_body]
            target_speed = speeds[target_body]
            target_coordinate = output_orientation * target_angle
            target_speed_coordinate = output_orientation * target_speed
            if current_phase_id == 'turn_hour_hand_to_alarm_target':
                target_limit_error = (
                    target_coordinate - scheduled.setting_target_angle_rad
                    if target_coordinate
                    > scheduled.setting_target_angle_rad
                    else target_coordinate
                    if target_coordinate < 0.0
                    else 0.0
                )
                if target_limit_error != 0.0:
                    add_axis_torque(
                        target_body,
                        _clamp(
                            5.0 * target_limit_error
                            + 0.01 * target_speed_coordinate,
                            target_detent_limit_nm,
                        ),
                    )
            elif current_phase_id not in {'initial'}:
                target_hold_coordinate = (
                    output_orientation * current_alarm_target_hold_angle_rad
                )
                add_axis_torque(
                    target_body,
                    _clamp(
                        5.0
                        * (
                            target_coordinate
                            - target_hold_coordinate
                        )
                        + 0.01 * target_speed_coordinate,
                        target_detent_limit_nm,
                    ),
                )

        for mesh in spec.gear_meshes:
            error = (
                shaft_angles[mesh.driven_body_id]
                + mesh.ratio * shaft_angles[mesh.driver_body_id]
                - mesh.phase_offset_rad
            )
            speed_error = (
                speeds[mesh.driven_body_id]
                + mesh.ratio * speeds[mesh.driver_body_id]
            )
            current_mesh_errors[mesh.mesh_id] = error
            if execution.branch == 'physical':
                interface_id = f"gear-{mesh.mesh_id}"
                decision = step_conditional_interface(interface_id)
                if decision.capture_reference:
                    conditional_interface_reference_errors[
                        interface_id
                    ] = error
                gated_error = (
                    error
                    - conditional_interface_reference_errors[interface_id]
                )
                # Reflect compliance through the ratio before applying equal
                # energy-conjugate torques.  Without this normalization a
                # high-ratio pinion stage becomes r^2 times stiffer at the
                # driver and can inject numerical energy into a free train.
                reflected_scale = max(1.0, mesh.ratio * mesh.ratio)
                requested_driven_torque = _clamp(
                    -(mesh.mesh_stiffness_nm_rad / reflected_scale) * gated_error
                    - (mesh.mesh_damping_nm_s_rad / reflected_scale) * speed_error,
                    torque_limit_nm,
                )
                driven_torque = transmit_conditional_effort(
                    interface_id,
                    decision,
                    requested_driven_torque,
                )
                add_axis_torque(mesh.driven_body_id, driven_torque)
                add_axis_torque(mesh.driver_body_id, mesh.ratio * driven_torque)

        for transfer in spec.shaft_transfers:
            error = (
                shaft_angles[transfer.driven_body_id]
                - transfer.ratio * shaft_angles[transfer.driver_body_id]
            )
            speed_error = (
                speeds[transfer.driven_body_id]
                - transfer.ratio * speeds[transfer.driver_body_id]
            )
            current_mesh_errors[transfer.transfer_id] = error
            interface_id = f"transfer-{transfer.transfer_id}"
            gated_error = error
            decision = None
            if execution.branch == 'physical':
                decision = step_conditional_interface(interface_id)
                if decision.capture_reference:
                    conditional_interface_reference_errors[
                        interface_id
                    ] = error
                gated_error = (
                    error
                    - conditional_interface_reference_errors[interface_id]
                )
            requested_driven_torque = _clamp(
                -transfer.stiffness_nm_rad * gated_error
                - transfer.damping_nm_s_rad * speed_error,
                transfer.max_torque_nm,
            )
            driven_torque = (
                transmit_conditional_effort(
                    interface_id,
                    decision,
                    requested_driven_torque,
                )
                if decision is not None
                else requested_driven_torque
            )
            add_axis_torque(transfer.driven_body_id, driven_torque)
            add_axis_torque(
                transfer.driver_body_id,
                -transfer.ratio * driven_torque,
            )

        for follower in spec.cam_followers:
            cam_angle = shaft_angles[follower.cam_body_id]
            cam_speed = speeds[follower.cam_body_id]
            target = follower.center_angle_rad + follower.amplitude_rad * math.sin(
                follower.lobe_count * cam_angle
            )
            derivative = (
                follower.amplitude_rad
                * follower.lobe_count
                * math.cos(follower.lobe_count * cam_angle)
            )
            target_speed = derivative * cam_speed
            error = shaft_angles[follower.follower_body_id] - target
            speed_error = speeds[follower.follower_body_id] - target_speed
            current_follower_errors[follower.follower_id] = error
            interface_id = f"cam-{follower.follower_id}"
            gated_error = error
            decision = None
            if execution.branch == 'physical':
                decision = step_conditional_interface(interface_id)
                if decision.capture_reference:
                    conditional_interface_reference_errors[
                        interface_id
                    ] = error
                gated_error = (
                    error
                    - conditional_interface_reference_errors[interface_id]
                )
            requested_follower_torque = _clamp(
                -follower.stiffness_nm_rad * gated_error
                - follower.damping_nm_s_rad * speed_error,
                torque_limit_nm,
            )
            follower_torque = (
                transmit_conditional_effort(
                    interface_id,
                    decision,
                    requested_follower_torque,
                )
                if decision is not None
                else requested_follower_torque
            )
            add_axis_torque(follower.follower_body_id, follower_torque)
            add_axis_torque(follower.cam_body_id, -follower_torque * derivative)

        current_setting_clutch_error = 0.0
        clutch_decision = None
        clutch_error = 0.0
        clutch_speed_error = 0.0
        clutch_interface_id = None
        if spec.setting_clutch is not None:
            clutch = spec.setting_clutch
            clutch_interface_id = f"clutch-{clutch.clutch_id}"
            clutch_error = (
                shaft_angles[clutch.driven_body_id]
                - clutch.ratio * shaft_angles[clutch.input_body_id]
            )
            clutch_speed_error = (
                speeds[clutch.driven_body_id]
                - clutch.ratio * speeds[clutch.input_body_id]
            )
            if execution.branch == 'physical':
                clutch_decision = step_conditional_interface(
                    clutch_interface_id
                )
                if clutch_decision.capture_reference:
                    conditional_interface_reference_errors[
                        clutch_interface_id
                    ] = clutch_error
        if current_setting_clutch_engaged:
            if spec.setting_clutch is None:
                raise MechanismValidationError(
                    'CLOCK_SETTING_CLUTCH_REQUIRED',
                    'manual time-setting program requires a declared finite clutch',
                )
            clutch = spec.setting_clutch
            current_setting_clutch_error = clutch_error
            gated_error = (
                clutch_error
                - conditional_interface_reference_errors[
                    str(clutch_interface_id)
                ]
                if execution.branch == 'physical'
                else clutch_error
            )
            requested_driven_torque = _clamp(
                -clutch.stiffness_nm_rad * gated_error
                - clutch.damping_nm_s_rad * clutch_speed_error,
                clutch.max_torque_nm,
            )
            driven_torque = (
                transmit_conditional_effort(
                    str(clutch_interface_id),
                    clutch_decision,
                    requested_driven_torque,
                )
                if clutch_decision is not None
                else requested_driven_torque
            )
            add_axis_torque(clutch.driven_body_id, driven_torque)
            add_axis_torque(clutch.input_body_id, -clutch.ratio * driven_torque)
        elif clutch_decision is not None:
            transmit_conditional_effort(
                str(clutch_interface_id),
                clutch_decision,
                0.0,
            )

        current_alarm_setting_clutch_error = 0.0
        alarm_clutch_decision = None
        alarm_clutch_error = 0.0
        alarm_clutch_speed_error = 0.0
        alarm_clutch_interface_id = None
        if spec.alarm_setting_clutch is not None:
            alarm_clutch = spec.alarm_setting_clutch
            alarm_clutch_interface_id = (
                f"clutch-{alarm_clutch.clutch_id}"
            )
            alarm_clutch_error = (
                shaft_angles[alarm_clutch.driven_body_id]
                - alarm_clutch.ratio
                * shaft_angles[alarm_clutch.input_body_id]
            )
            alarm_clutch_speed_error = (
                speeds[alarm_clutch.driven_body_id]
                - alarm_clutch.ratio
                * speeds[alarm_clutch.input_body_id]
            )
            if execution.branch == 'physical':
                alarm_clutch_decision = step_conditional_interface(
                    alarm_clutch_interface_id
                )
                if alarm_clutch_decision.capture_reference:
                    conditional_interface_reference_errors[
                        alarm_clutch_interface_id
                    ] = alarm_clutch_error
        if current_alarm_setting_clutch_engaged:
            if spec.alarm_setting_clutch is None:
                raise MechanismValidationError(
                    'CLOCK_ALARM_SETTING_CLUTCH_REQUIRED',
                    'hour-hand alarm setting requires a declared finite clutch',
                )
            alarm_clutch = spec.alarm_setting_clutch
            current_alarm_setting_clutch_error = (
                output_orientation * alarm_clutch_error
            )
            gated_error = (
                alarm_clutch_error
                - conditional_interface_reference_errors[
                    str(alarm_clutch_interface_id)
                ]
                if execution.branch == 'physical'
                else alarm_clutch_error
            )
            requested_target_torque = _clamp(
                -alarm_clutch.stiffness_nm_rad * gated_error
                - alarm_clutch.damping_nm_s_rad
                * alarm_clutch_speed_error,
                alarm_clutch.max_torque_nm,
            )
            target_torque = (
                transmit_conditional_effort(
                    str(alarm_clutch_interface_id),
                    alarm_clutch_decision,
                    requested_target_torque,
                )
                if alarm_clutch_decision is not None
                else requested_target_torque
            )
            add_axis_torque(
                alarm_clutch.driven_body_id,
                target_torque,
            )
            add_axis_torque(
                alarm_clutch.input_body_id,
                -alarm_clutch.ratio * target_torque,
            )
        elif alarm_clutch_decision is not None:
            transmit_conditional_effort(
                str(alarm_clutch_interface_id),
                alarm_clutch_decision,
                0.0,
            )

        for shaft in spec.shafts:
            vector = torques[shaft.body_id]
            current_wrench_ledger().record_component(
                f"shaft-wrench-accumulator-{shaft.body_id}",
                component_source=(
                    "declared_phase_input_plus_finite_support_transmission_"
                    "and_resistance_components"
                ),
            )
            torque_loads[shaft.body_id].SetTorque(chrono.ChVector3d(*vector), False)
            if execution.branch == 'physical':
                support_state = current_conditional_interface_states.get(
                    f"support-{shaft.shaft_id}"
                )
                try:
                    current_bushing_reactions[shaft.shaft_id] = (
                        _norm(_tuple3(bushings[shaft.shaft_id].GetForce()))
                        if support_state
                        and support_state["transmission_active"]
                        else 0.0
                    )
                except Exception as exc:
                    raise MechanismValidationError(
                        "CLOCK_BUSHING_REACTION_UNAVAILABLE",
                        "physical clock support reaction could not be measured",
                    ) from exc
        current_applied_torque = external_torque_abs

    def state() -> dict[str, Any]:
        speeds = shaft_speeds()
        output_raw = shaft_angles[spec.output_body_id]
        output_speed_raw = speeds[spec.output_body_id]
        poses = {body_id: _pose(body) for body_id, body in bodies.items()}
        linear_velocities = {
            body_id: _tuple3(body.GetPosDt())
            for body_id, body in bodies.items()
        }
        angular_velocities = {
            body_id: _angular_velocity(body)
            for body_id, body in bodies.items()
        }
        off_axis = {
            shaft.shaft_id: _off_axis_angle(
                poses[shaft.body_id]['rotation_abs_quat_wxyz'],
                shaft.axis_abs,
            )
            for shaft in spec.shafts
        }
        return {
            'shaft_speeds_rad_s': speeds,
            **{
                f"{shaft.body_id}_angle_rad": shaft_angles[
                    shaft.body_id
                ]
                for shaft in spec.shafts
            },
            **{
                f"{shaft.body_id}_speed_rad_s": speeds[shaft.body_id]
                for shaft in spec.shafts
            },
            'multi_output_primary_angle_rad': (
                output_orientation * output_raw
            ),
            'multi_output_primary_speed_rad_s': (
                output_orientation * output_speed_raw
            ),
            'clock_output_angle_rad': output_orientation * output_raw,
            'clock_output_speed_rad_s': output_orientation * output_speed_raw,
            'hour_hand_angle_rad': output_orientation * shaft_angles[spec.hour_body_id],
            'hour_hand_speed_rad_s': output_orientation * speeds[spec.hour_body_id],
            'second_hand_angle_rad': output_orientation * shaft_angles[spec.second_body_id],
            'second_hand_speed_rad_s': output_orientation * speeds[spec.second_body_id],
            'clock_input_angle_rad': shaft_angles[current_active_input_body_id],
            'clock_input_speed_rad_s': speeds[current_active_input_body_id],
            'barrel_input_angle_rad': shaft_angles[spec.input_body_id],
            'setting_input_angle_rad': shaft_angles[spec.setting_input_body_id],
            'alarm_setting_target_angle_rad': (
                output_orientation
                * shaft_angles[spec.scheduled_alarm.setting_body_id]
                if spec.scheduled_alarm is not None
                else 0.0
            ),
            'alarm_input_angle_rad': shaft_angles[spec.effective_alarm_input_body_id],
            'alarm_angle_rad': shaft_angles[spec.alarm_body_id],
            'alarm_speed_rad_s': speeds[spec.alarm_body_id],
            'body_poses': poses,
            'body_linear_velocities_abs_m_s': linear_velocities,
            'body_angular_velocities_abs_rad_s': angular_velocities,
            'off_axis_rotations_rad': off_axis,
        }

    samples: list[dict[str, Any]] = []
    executed_phases: list[dict[str, Any]] = []
    time_s = 0.0
    next_sample_s = 0.0
    catastrophic_sample_count = 0
    current_phase_index = -1
    current_phase_elapsed_s = 0.0
    current_phase_start_second_angle_rad = shaft_angles[spec.second_body_id]
    current_phase_termination_reason = 'duration_limit'
    dynamics_step_count = 0
    resume_history: list[dict[str, Any]] = []

    def dynamics_resume_state() -> dict[str, Any]:
        snapshot = state()
        return {
            'schema_version': 'clock_gear_resume_v1',
            'time_s': time_s,
            'next_sample_s': next_sample_s,
            'current_phase_index': current_phase_index,
            'current_phase_id': current_phase_id,
            'current_phase_elapsed_s': current_phase_elapsed_s,
            'current_phase_start_second_angle_rad': (
                current_phase_start_second_angle_rad
            ),
            'current_phase_termination_reason': (
                current_phase_termination_reason
            ),
            'dynamics_step_count': dynamics_step_count,
            'catastrophic_sample_count': catastrophic_sample_count,
            'shaft_angles_rad': dict(shaft_angles),
            'body_poses': snapshot['body_poses'],
            'body_linear_velocities_abs_m_s': (
                snapshot['body_linear_velocities_abs_m_s']
            ),
            'body_angular_velocities_abs_rad_s': (
                snapshot['body_angular_velocities_abs_rad_s']
            ),
            'current_alarm_hold_angle_rad': current_alarm_hold_angle_rad,
            'current_alarm_target_hold_angle_rad': (
                current_alarm_target_hold_angle_rad
            ),
            'current_setting_clutch_engaged': (
                current_setting_clutch_engaged
            ),
            'current_alarm_setting_clutch_engaged': (
                current_alarm_setting_clutch_engaged
            ),
            'current_additional_alarm_torque_nm': (
                current_additional_alarm_torque_nm
            ),
            'conditional_interface_states': {
                interface_id: controller.state
                for interface_id, controller in conditional_interfaces.items()
            },
            'bounded_support_runtimes': {
                runtime['interface_id']: {
                    'state': runtime['gate'].controller.state,
                    'evaluation_count': runtime['gate'].evaluation_count,
                    'state_path': list(runtime['gate'].state_path),
                    'max_inactive_abs_transmitted_effort_si': (
                        runtime['gate'].max_inactive_abs_transmitted_effort_si
                    ),
                    'ever_engaged': runtime['gate'].ever_engaged,
                    'current_state': (
                        None
                        if runtime['gate'].current_state is None
                        else dict(runtime['gate'].current_state)
                    ),
                }
                for runtime in support_runtimes.values()
            },
            'conditional_interface_evaluation_counts': dict(
                conditional_interface_evaluation_counts
            ),
            'conditional_interface_state_paths': {
                interface_id: list(path)
                for interface_id, path
                in conditional_interface_state_paths.items()
            },
            'conditional_interface_max_inactive_efforts': dict(
                conditional_interface_max_inactive_efforts
            ),
            'current_conditional_interface_states': {
                interface_id: dict(value)
                for interface_id, value
                in current_conditional_interface_states.items()
            },
            'conditional_interface_reference_errors': dict(
                conditional_interface_reference_errors
            ),
            'resume_history': list(resume_history),
            'wrench_ledger_state': (
                current_wrench_ledger().checkpoint_state()
            ),
            'solver_warm_start_cache_serialized': False,
        }

    def record(phase_id: str) -> None:
        snapshot = state()
        sample = {
                'time_s': time_s,
                'phase_id': phase_id,
                'applied_torque_abs_nm': list(current_applied_torque),
                'generalized_angle_rad': snapshot['clock_output_angle_rad'],
                'generalized_speed_rad_s': snapshot['clock_output_speed_rad_s'],
                'multi_output_primary_angle_rad': (
                    snapshot['multi_output_primary_angle_rad']
                ),
                'multi_output_primary_speed_rad_s': (
                    snapshot['multi_output_primary_speed_rad_s']
                ),
                **{
                    f"{shaft.body_id}_angle_rad": snapshot[
                        f"{shaft.body_id}_angle_rad"
                    ]
                    for shaft in spec.shafts
                },
                **{
                    f"{shaft.body_id}_speed_rad_s": snapshot[
                        f"{shaft.body_id}_speed_rad_s"
                    ]
                    for shaft in spec.shafts
                },
                'clock_output_angle_rad': snapshot['clock_output_angle_rad'],
                'clock_output_speed_rad_s': snapshot['clock_output_speed_rad_s'],
                'clock_input_angle_rad': snapshot['clock_input_angle_rad'],
                'clock_input_speed_rad_s': snapshot['clock_input_speed_rad_s'],
                'barrel_input_angle_rad': snapshot['barrel_input_angle_rad'],
                'setting_input_angle_rad': snapshot['setting_input_angle_rad'],
                'alarm_setting_target_angle_rad': (
                    snapshot['alarm_setting_target_angle_rad']
                ),
                'alarm_input_angle_rad': snapshot['alarm_input_angle_rad'],
                'hour_hand_angle_rad': snapshot['hour_hand_angle_rad'],
                'hour_hand_speed_rad_s': snapshot['hour_hand_speed_rad_s'],
                'second_hand_angle_rad': snapshot['second_hand_angle_rad'],
                'second_hand_speed_rad_s': snapshot['second_hand_speed_rad_s'],
                'alarm_angle_rad': snapshot['alarm_angle_rad'],
                'alarm_speed_rad_s': snapshot['alarm_speed_rad_s'],
                'shaft_angles_rad': dict(shaft_angles),
                'shaft_speeds_rad_s': snapshot['shaft_speeds_rad_s'],
                'mesh_errors_rad': dict(current_mesh_errors),
                'max_mesh_error_rad': max(abs(value) for value in current_mesh_errors.values()),
                'cam_follower_errors_rad': dict(current_follower_errors),
                'setting_clutch_error_rad': current_setting_clutch_error,
                'setting_clutch_engaged': current_setting_clutch_engaged,
                'alarm_setting_clutch_error_rad': (
                    current_alarm_setting_clutch_error
                ),
                'alarm_setting_clutch_engaged': (
                    current_alarm_setting_clutch_engaged
                ),
                'active_input_body_id': current_active_input_body_id,
                'active_input_point_abs_m': list(current_active_input_point_abs_m),
                'additional_alarm_torque_nm': current_additional_alarm_torque_nm,
                'support_drifts_m': dict(current_support_drifts),
                'max_support_drift_m': max(current_support_drifts.values()),
                'bushing_reaction_forces_n': dict(current_bushing_reactions),
                'max_bushing_reaction_force_n': max(current_bushing_reactions.values()),
                'off_axis_rotations_rad': snapshot['off_axis_rotations_rad'],
                'max_off_axis_rotation_rad': max(snapshot['off_axis_rotations_rad'].values()),
                'body_poses': snapshot['body_poses'],
                'body_linear_velocities_abs_m_s': (
                    snapshot['body_linear_velocities_abs_m_s']
                ),
                'body_angular_velocities_abs_rad_s': (
                    snapshot['body_angular_velocities_abs_rad_s']
                ),
                'contact_count': _contact_count(system),
                'contact_pair_counts': _contact_pair_counts(chrono, system),
            }
        if execution.branch == 'physical':
            if (
                set(current_conditional_interface_states)
                != expected_conditional_interface_ids
            ):
                raise RuntimeError(
                    "clock conditional interfaces were not all evaluated"
                )
            sample["conditional_interface_states"] = [
                dict(current_conditional_interface_states[interface_id])
                for interface_id in sorted(
                    current_conditional_interface_states
                )
            ]
        samples.append(sample)
        _write_trace_checkpoint(
            execution,
            samples,
            executed_phases=executed_phases,
            resume_state=dynamics_resume_state(),
        )

    loaded_checkpoint = _load_trace_checkpoint_for_resume(execution)
    (
        prior_checkpoint,
        checkpoint_restart_policy,
    ) = _clock_checkpoint_resume(loaded_checkpoint)
    if prior_checkpoint is None:
        current_active_input_body_id = phase_input_body_id(current_phase_id)
        (
            current_active_input_point_abs_m,
            current_active_input_axis_abs,
        ) = input_frame_for_body(current_active_input_body_id)
        update_couplings((0.0, 0.0, 0.0))
        record('initial')
        next_sample_s = execution.load_case.sample_interval_s
        first_dynamics_step = True
        start_phase_index = 0
        resumed_open_phase = False
    else:
        checkpoint = prior_checkpoint['checkpoint']
        resume_state = checkpoint['resume_state']

        def finite_resume_number(field_name: str) -> float:
            value = resume_state.get(field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise MechanismValidationError(
                    'CLOCK_CHECKPOINT_STATE_INVALID',
                    f'clock checkpoint field {field_name!r} is not numeric',
                )
            number = float(value)
            if not math.isfinite(number):
                raise MechanismValidationError(
                    'CLOCK_CHECKPOINT_STATE_INVALID',
                    f'clock checkpoint field {field_name!r} is not finite',
                )
            return number

        def exact_mapping(
            field_name: str,
            expected_keys: set[str],
        ) -> Mapping[str, Any]:
            value = resume_state.get(field_name)
            if not isinstance(value, Mapping) or set(value) != expected_keys:
                raise MechanismValidationError(
                    'CLOCK_CHECKPOINT_STATE_INVALID',
                    f'clock checkpoint field {field_name!r} has changed keys',
                )
            return value

        restored_samples = prior_checkpoint.get('samples')
        restored_executed_phases = prior_checkpoint.get('executed_phases')
        if (
            not isinstance(restored_samples, list)
            or not restored_samples
            or not all(isinstance(item, Mapping) for item in restored_samples)
            or not isinstance(restored_executed_phases, list)
            or not all(
                isinstance(item, Mapping)
                for item in restored_executed_phases
            )
        ):
            raise MechanismValidationError(
                'CLOCK_CHECKPOINT_STATE_INVALID',
                'clock checkpoint is missing its saved trace prefix',
            )
        samples = [dict(item) for item in restored_samples]
        executed_phases = [
            dict(item) for item in restored_executed_phases
        ]
        time_s = finite_resume_number('time_s')
        next_sample_s = finite_resume_number('next_sample_s')
        current_phase_index_value = resume_state.get('current_phase_index')
        dynamics_step_count_value = resume_state.get('dynamics_step_count')
        catastrophic_sample_count_value = resume_state.get(
            'catastrophic_sample_count'
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (
                current_phase_index_value,
                dynamics_step_count_value,
                catastrophic_sample_count_value,
            )
        ):
            raise MechanismValidationError(
                'CLOCK_CHECKPOINT_STATE_INVALID',
                'clock checkpoint counters are invalid',
            )
        current_phase_index = int(current_phase_index_value)
        dynamics_step_count = int(dynamics_step_count_value)
        catastrophic_sample_count = int(catastrophic_sample_count_value)
        if (
            current_phase_index < -1
            or current_phase_index >= len(execution.load_case.phases)
            or dynamics_step_count < 0
            or catastrophic_sample_count < 0
        ):
            raise MechanismValidationError(
                'CLOCK_CHECKPOINT_STATE_INVALID',
                'clock checkpoint counters are outside their valid range',
            )
        current_phase_id = str(resume_state.get('current_phase_id') or '')
        expected_phase_id = (
            'initial'
            if current_phase_index == -1
            else execution.load_case.phases[current_phase_index].phase_id
        )
        if current_phase_id != expected_phase_id:
            raise MechanismValidationError(
                'CLOCK_CHECKPOINT_STATE_INVALID',
                'clock checkpoint phase does not match its phase index',
            )
        current_phase_elapsed_s = finite_resume_number(
            'current_phase_elapsed_s'
        )
        current_phase_start_second_angle_rad = finite_resume_number(
            'current_phase_start_second_angle_rad'
        )
        current_phase_termination_reason = str(
            resume_state.get('current_phase_termination_reason') or ''
        )
        if current_phase_elapsed_s < 0.0 or not current_phase_termination_reason:
            raise MechanismValidationError(
                'CLOCK_CHECKPOINT_STATE_INVALID',
                'clock checkpoint phase progress is invalid',
            )

        shaft_state = exact_mapping(
            'shaft_angles_rad',
            set(shaft_angles),
        )
        shaft_angles = {
            body_id: float(shaft_state[body_id])
            for body_id in shaft_angles
        }
        if not all(math.isfinite(value) for value in shaft_angles.values()):
            raise MechanismValidationError(
                'CLOCK_CHECKPOINT_STATE_INVALID',
                'clock checkpoint shaft angles are invalid',
            )
        body_ids = set(bodies)
        poses = exact_mapping('body_poses', body_ids)
        linear_velocities = exact_mapping(
            'body_linear_velocities_abs_m_s', body_ids
        )
        angular_velocities = exact_mapping(
            'body_angular_velocities_abs_rad_s', body_ids
        )
        for body_id, body in bodies.items():
            pose = poses[body_id]
            if not isinstance(pose, Mapping):
                raise MechanismValidationError(
                    'CLOCK_CHECKPOINT_STATE_INVALID',
                    'clock checkpoint body pose is invalid',
                )
            position = pose.get('position_abs_m')
            rotation = pose.get('rotation_abs_quat_wxyz')
            linear = linear_velocities[body_id]
            angular = angular_velocities[body_id]
            if not all(
                isinstance(value, (list, tuple)) and len(value) == size
                for value, size in (
                    (position, 3),
                    (rotation, 4),
                    (linear, 3),
                    (angular, 3),
                )
            ):
                raise MechanismValidationError(
                    'CLOCK_CHECKPOINT_STATE_INVALID',
                    'clock checkpoint rigid-body vectors are invalid',
                )
            body.SetPos(chrono.ChVector3d(*(float(item) for item in position)))
            body.SetRot(
                chrono.ChQuaterniond(*(float(item) for item in rotation))
            )
            body.SetPosDt(
                chrono.ChVector3d(*(float(item) for item in linear))
            )
            if not hasattr(body, 'SetAngVelParent'):
                raise MechanismValidationError(
                    'CLOCK_CHECKPOINT_STATE_INVALID',
                    'PyChrono cannot restore parent-frame angular velocity',
                )
            body.SetAngVelParent(
                chrono.ChVector3d(*(float(item) for item in angular))
            )

        current_alarm_hold_angle_rad = finite_resume_number(
            'current_alarm_hold_angle_rad'
        )
        current_alarm_target_hold_angle_rad = finite_resume_number(
            'current_alarm_target_hold_angle_rad'
        )
        current_additional_alarm_torque_nm = finite_resume_number(
            'current_additional_alarm_torque_nm'
        )
        for field_name in (
            'current_setting_clutch_engaged',
            'current_alarm_setting_clutch_engaged',
        ):
            if not isinstance(resume_state.get(field_name), bool):
                raise MechanismValidationError(
                    'CLOCK_CHECKPOINT_STATE_INVALID',
                    f'clock checkpoint field {field_name!r} is invalid',
                )
        current_setting_clutch_engaged = resume_state[
            'current_setting_clutch_engaged'
        ]
        current_alarm_setting_clutch_engaged = resume_state[
            'current_alarm_setting_clutch_engaged'
        ]

        interface_ids = set(conditional_interfaces)
        controller_states = exact_mapping(
            'conditional_interface_states', interface_ids
        )
        evaluation_counts = exact_mapping(
            'conditional_interface_evaluation_counts', interface_ids
        )
        state_paths = exact_mapping(
            'conditional_interface_state_paths', interface_ids
        )
        inactive_efforts = exact_mapping(
            'conditional_interface_max_inactive_efforts', interface_ids
        )
        reference_errors = exact_mapping(
            'conditional_interface_reference_errors', interface_ids
        )
        current_states = exact_mapping(
            'current_conditional_interface_states',
            expected_conditional_interface_ids,
        )
        for interface_id, controller in conditional_interfaces.items():
            controller.state = str(controller_states[interface_id])
            controller.last_decision = None
        conditional_interface_evaluation_counts = {
            interface_id: int(evaluation_counts[interface_id])
            for interface_id in interface_ids
        }
        conditional_interface_state_paths = {
            interface_id: [str(item) for item in state_paths[interface_id]]
            for interface_id in interface_ids
        }
        conditional_interface_max_inactive_efforts = {
            interface_id: float(inactive_efforts[interface_id])
            for interface_id in interface_ids
        }
        conditional_interface_reference_errors = {
            interface_id: float(reference_errors[interface_id])
            for interface_id in interface_ids
        }
        current_conditional_interface_states = {
            interface_id: dict(current_states[interface_id])
            for interface_id in expected_conditional_interface_ids
        }

        support_by_interface = {
            runtime['interface_id']: runtime['gate']
            for runtime in support_runtimes.values()
        }
        support_states = exact_mapping(
            'bounded_support_runtimes', set(support_by_interface)
        )
        for interface_id, runtime in support_by_interface.items():
            saved = support_states[interface_id]
            if not isinstance(saved, Mapping):
                raise MechanismValidationError(
                    'CLOCK_CHECKPOINT_STATE_INVALID',
                    'clock checkpoint support state is invalid',
                )
            runtime.controller.state = str(saved.get('state') or '')
            runtime.controller.last_decision = None
            runtime.evaluation_count = int(saved.get('evaluation_count'))
            runtime.state_path = [
                str(item) for item in saved.get('state_path') or []
            ]
            runtime.max_inactive_abs_transmitted_effort_si = float(
                saved.get('max_inactive_abs_transmitted_effort_si')
            )
            runtime.ever_engaged = bool(saved.get('ever_engaged'))
            runtime.current_state = (
                None
                if saved.get('current_state') is None
                else dict(saved['current_state'])
            )

        saved_history = resume_state.get('resume_history')
        if not isinstance(saved_history, list):
            raise MechanismValidationError(
                'CLOCK_CHECKPOINT_STATE_INVALID',
                'clock checkpoint resume history is invalid',
            )
        resume_history = [dict(item) for item in saved_history]
        resume_history.append(
            {
                'resume_index': len(resume_history) + 1,
                'simulated_time_s': time_s,
                'phase_id': current_phase_id,
                'sample_count': len(samples),
            }
        )
        restored_wrench_phase = restore_current_wrench_checkpoint(
            resume_state['wrench_ledger_state']
        )
        expected_wrench_phase = (
            None if current_phase_index == -1 else current_phase_id
        )
        if restored_wrench_phase != expected_wrench_phase:
            raise MechanismValidationError(
                'CLOCK_CHECKPOINT_STATE_INVALID',
                'clock checkpoint wrench phase does not match dynamics phase',
            )
        current_active_input_body_id = phase_input_body_id(current_phase_id)
        (
            current_active_input_point_abs_m,
            current_active_input_axis_abs,
        ) = input_frame_for_body(current_active_input_body_id)
        first_dynamics_step = False
        resumed_open_phase = current_phase_index >= 0
        start_phase_index = max(0, current_phase_index)
    for phase_index, phase in enumerate(
        execution.load_case.phases[start_phase_index:],
        start=start_phase_index,
    ):
        resuming_this_phase = (
            prior_checkpoint is not None
            and resumed_open_phase
            and phase_index == start_phase_index
        )
        current_phase_index = phase_index
        current_phase_id = phase.phase_id
        if resuming_this_phase:
            if current_wrench_phase() != phase.phase_id:
                raise MechanismValidationError(
                    'CLOCK_CHECKPOINT_STATE_INVALID',
                    'clock checkpoint resumed with the wrong wrench phase',
                )
        else:
            transition_current_wrench_phase(phase.phase_id)
        current_setting_clutch_engaged = manual_setting_program or (
            zero_resume_program
            and phase.phase_id in {'calibration_offset', 'calibration_return'}
        )
        current_alarm_setting_clutch_engaged = (
            hour_set_alarm_program
            and phase.phase_id == 'turn_hour_hand_to_alarm_target'
        )
        if (
            hour_set_alarm_program
            and phase.phase_id == 'alarm_setting_latched'
            and not resuming_this_phase
            and spec.scheduled_alarm is not None
        ):
            current_alarm_target_hold_angle_rad = shaft_angles[
                spec.scheduled_alarm.setting_body_id
            ]
        current_active_input_body_id = phase_input_body_id(phase.phase_id)
        (
            current_active_input_point_abs_m,
            current_active_input_axis_abs,
        ) = input_frame_for_body(current_active_input_body_id)
        if (
            scheduled_alarm_program
            and phase.phase_id == 'scheduled_alarm_settle'
            and not resuming_this_phase
        ):
            # The alarm barrel click arrests the spring at its current wound
            # phase; it does not command an instantaneous rewind to zero.
            current_alarm_hold_angle_rad = shaft_angles[
                spec.effective_alarm_input_body_id
            ]
        if phase.wrench_kind == 'force':
            raise MechanismValidationError(
                'CLOCK_FORCE_LOAD_UNSUPPORTED',
                'clock backend accepts winding torque, not direct force on hands or hammer',
            )
        direction = phase.direction_abs or (0.0, 0.0, 0.0)
        applied_torque_abs = tuple(value * phase.magnitude_si for value in direction)
        phase_limit_s = _phase_limit_seconds(
            branch=execution.branch,
            program=execution.load_case.program,
            phase_id=phase.phase_id,
            declared_duration_s=phase.duration_s,
            reference_duration_s=(
                reference_schedule[phase_index]['duration_s']
                if reference_schedule is not None
                else None
            ),
        )
        if (
            hour_set_alarm_program
            and phase.phase_id in {
                'turn_hour_hand_to_alarm_target',
                'restore_hour_hand_to_current_time',
            }
        ):
            phase_limit_s = max(phase_limit_s, 1.20)
        elapsed = (
            current_phase_elapsed_s if resuming_this_phase else 0.0
        )
        phase_start_second_angle = (
            current_phase_start_second_angle_rad
            if resuming_this_phase
            else shaft_angles[spec.second_body_id]
        )
        phase_termination_reason = (
            current_phase_termination_reason
            if resuming_this_phase
            else 'duration_limit'
        )
        current_phase_elapsed_s = elapsed
        current_phase_start_second_angle_rad = phase_start_second_angle
        current_phase_termination_reason = phase_termination_reason
        update_couplings((0.0, 0.0, 0.0))
        if (
            not resuming_this_phase
            and (
                not samples
                or samples[-1]['time_s'] < time_s - 1.0e-12
            )
        ):
            record(phase.phase_id)
        while elapsed < phase_limit_s - 1.0e-12:
            step_torque_abs = applied_torque_abs
            if (
                hour_set_alarm_program
                and spec.scheduled_alarm is not None
                and phase.phase_id in {
                    'turn_hour_hand_to_alarm_target',
                    'restore_hour_hand_to_current_time',
                }
            ):
                desired_hour_coordinate = (
                    spec.scheduled_alarm.setting_target_angle_rad
                    if phase.phase_id == 'turn_hour_hand_to_alarm_target'
                    else 0.0
                )
                current_hour_coordinate = (
                    output_orientation * shaft_angles[spec.hour_body_id]
                )
                current_hour_speed = (
                    output_orientation
                    * shaft_speeds()[spec.hour_body_id]
                )
                commanded_hour_torque = _clamp(
                    -100.0
                    * (
                        desired_hour_coordinate
                        - current_hour_coordinate
                    )
                    + 0.30 * current_hour_speed,
                    phase.magnitude_si,
                )
                hour_axis = shaft_by_body[
                    spec.hour_body_id
                ].axis_abs
                step_torque_abs = tuple(
                    component * commanded_hour_torque
                    for component in hour_axis
                )
            update_couplings(step_torque_abs)
            if (
                scheduled_alarm_program
                and hour_set_alarm_program
                and execution.branch in {'physical', 'reduced'}
                and phase.phase_id == 'advance_to_alarm_time'
                and spec.scheduled_alarm is not None
                and abs(
                    output_orientation
                    * shaft_angles[spec.hour_body_id]
                    - output_orientation
                    * shaft_angles[
                        spec.scheduled_alarm.setting_body_id
                    ]
                )
                <= spec.scheduled_alarm.setting_angle_tolerance_rad
                and abs(
                    output_orientation
                    * shaft_angles[spec.hour_body_id]
                )
                >= 0.5
                * spec.scheduled_alarm.target_hour_hand_travel_rad
                and abs(
                    shaft_angles[spec.second_body_id]
                    - phase_start_second_angle
                )
                >= (
                    spec.scheduled_alarm.target_second_hand_travel_rad
                    - spec.scheduled_alarm.target_second_hand_tolerance_rad
                )
            ):
                phase_termination_reason = (
                    'selected_hour_hand_position_reached'
                )
                break
            if (
                scheduled_alarm_program
                and not hour_set_alarm_program
                and execution.branch in {'physical', 'reduced'}
                and phase.phase_id == 'advance_to_alarm_time'
                and spec.scheduled_alarm is not None
                and abs(
                    shaft_angles[spec.second_body_id] - phase_start_second_angle
                )
                >= spec.scheduled_alarm.target_second_hand_travel_rad
            ):
                phase_termination_reason = 'scheduled_alarm_target_reached'
                break
            if (
                zero_resume_program
                and execution.branch in {'physical', 'reduced'}
                and phase.phase_id == 'calibration_return'
                and elapsed >= 0.45
                and abs(
                    output_orientation * shaft_angles[spec.output_body_id]
                    - spec.home_output_angle_rad
                )
                <= 0.02
            ):
                # A manual return is an external, torque-limited target task:
                # stop applying reverse torque once the executed dial reaches
                # zero.  The following released phase still proves that
                # inertia or clutch drag does not move it outside the
                # unchanged family tolerance.
                phase_termination_reason = 'zero_calibration_target_reached'
                break
            step = min(execution.load_case.step_s, phase_limit_s - elapsed)
            if first_dynamics_step:
                _collision_progress(
                    'first_dynamics_step_begin', capsule_id=execution.plan.capsule_id
                )
            system.DoStepDynamics(step)
            dynamics_step_count += 1
            if first_dynamics_step:
                _collision_progress(
                    'first_dynamics_step_complete', capsule_id=execution.plan.capsule_id
                )
                first_dynamics_step = False
            elapsed += step
            time_s += step
            current_phase_elapsed_s = elapsed
            speeds = shaft_speeds()
            for body_id, speed in speeds.items():
                shaft_angles[body_id] += speed * step
            update_couplings(step_torque_abs)
            if time_s + 1.0e-12 >= next_sample_s:
                record(phase.phase_id)
                while next_sample_s <= time_s + 1.0e-12:
                    next_sample_s += execution.load_case.sample_interval_s
            if dynamics_step_count % 1000 == 0:
                _collision_progress(
                    'dynamics_progress',
                    capsule_id=execution.plan.capsule_id,
                    step_count=dynamics_step_count,
                    phase_id=phase.phase_id,
                    simulated_time_s=time_s,
                )
        current_phase_termination_reason = phase_termination_reason
        if not samples or samples[-1]['time_s'] < time_s - 1.0e-12:
            record(phase.phase_id)
        executed_phase = {
                'phase_id': phase.phase_id,
                'wrench_kind': phase.wrench_kind,
                'direction_abs': None if phase.direction_abs is None else list(phase.direction_abs),
                'magnitude_si': phase.magnitude_si,
                'duration_s': (
                    elapsed
                    if phase.stop_condition is not None
                    or reference_schedule is None
                    else phase_limit_s
                ),
                'active_input_body_id': current_active_input_body_id,
                'active_input_point_abs_m': list(current_active_input_point_abs_m),
                'termination_reason': phase_termination_reason,
                'setting_clutch_engaged': current_setting_clutch_engaged,
                'actuation_channels': [
                    {
                        'body_id': current_active_input_body_id,
                        'kind': phase.wrench_kind,
                        'magnitude_si': phase.magnitude_si,
                    },
                    *(
                        [
                            {
                                'body_id': spec.effective_alarm_input_body_id,
                                'kind': 'torque',
                                'magnitude_si': spec.effective_nominal_alarm_input_torque_nm,
                            }
                        ]
                        if scheduled_alarm_program
                        and phase.phase_id == 'ring_with_timekeeping'
                        else []
                    ),
                ],
            }
        if phase.stop_condition is not None:
            actual_coordinate = (
                abs(
                    shaft_angles[spec.second_body_id]
                    - phase_start_second_angle
                )
                if phase.stop_condition.startswith('progress_')
                else (
                    output_orientation
                    * shaft_angles[spec.output_body_id]
                )
            )
            executed_phase['termination'] = (
                registered_adaptive_termination_evidence(
                    phase,
                    actual_coordinate_si=actual_coordinate,
                    reached=(
                        phase_termination_reason != 'duration_limit'
                    ),
                )
            )
        executed_phases.append(executed_phase)
    update_couplings((0.0, 0.0, 0.0))
    set_current_wrench_phase(None)
    total_mass = sum(body.mass_kg for body in spec.bodies if not body.fixed)
    conditional_interface_runtime = (
        conditional_runtime_summary(
            [runtime["gate"] for runtime in support_runtimes.values()],
            physics_step_count=dynamics_step_count,
            additional_interface_summaries=[
                {
                    "interface_id": interface_id,
                    "evaluation_count": (
                        conditional_interface_evaluation_counts[
                            interface_id
                        ]
                    ),
                    "state_path": list(
                        conditional_interface_state_paths[interface_id]
                    ),
                    "max_inactive_abs_transmitted_effort_si": (
                        conditional_interface_max_inactive_efforts[
                            interface_id
                        ]
                    ),
                }
                for interface_id in sorted(conditional_interfaces)
            ],
        )
        if execution.branch == "physical"
        else {
            "status": "NOT_REQUIRED",
            "conditional_interface_count": 0,
        }
    )
    chrono_version = str(chrono.GetChronoVersion()) if hasattr(chrono, 'GetChronoVersion') else None
    return {
        'schema_version': '0.1',
        'status': 'COMPLETED',
        'validator_source_sha256': lane_execution_source_sha256(),
        'branch': execution.branch,
        'adapter_id': execution.adapter.adapter_id,
        'mechanism_plugin_id': execution.mechanism_plugin.plugin_id,
        'package_id': execution.plan.package_id,
        'design_version': execution.plan.design_version,
        'package_binding_sha256': execution.plan.package_binding_sha256,
        'capsule_id': execution.plan.capsule_id,
        'load_case_id': execution.load_case.load_case_id,
        'plan_sha256': canonical_sha256(execution.plan.to_dict()),
        'load_case_sha256': canonical_sha256(execution.load_case.to_dict()),
        'engine': {
            'name': 'PyChrono',
            'version': chrono_version,
            'system': 'ChSystemNSC',
            'solver': 'ChSolverGMRES' if execution.branch == 'physical' else 'ChSolverPSOR',
            'mechanism_plugin_id': execution.mechanism_plugin.plugin_id,
            'worker_backend': execution.mechanism_plugin.worker_backend,
            'step_s': execution.load_case.step_s,
            'gravity_abs_m_s2': list(execution.plan.gravity_abs_m_s2),
            'chrono_link_count': actual_link_count,
            'ideal_links': ideal_links,
            'internal_ideal_joint_count': len(ideal_links),
            'shaft_support': (
                'finite_ChLoadBodyBodyBushingGeneric'
                if execution.branch == 'physical'
                else 'ChLinkLockRevolute'
            ),
            'gear_coupling': (
                'finite_compliant_mesh_torque'
                if execution.branch == 'physical'
                else 'ChLinkLockGear'
            ),
            'seconds_transfer': 'finite_torsional_shaft_transfer',
            'cam_coupling': 'finite_analytical_follower_torque',
            'setting_coupling': (
                'not_applicable'
                if is_multi_output
                else 'phase_selectable_finite_torque_limited_clutch'
                if zero_resume_program
                else 'engaged_finite_torque_limited_clutch'
                if manual_setting_program
                else 'disengaged_pull_to_set_clutch'
            ),
            'unwound_barrel_detent': (
                'finite_click_and_escapement_holding_torque'
                if (
                    holding_program
                    or isolated_alarm_program
                    or scheduled_alarm_program
                    or hour_alarm_setting_program
                )
                else 'not_active'
            ),
        },
        'branch_semantics': {
            'same_mass_kg': total_mass,
            'same_inertia_xx_kg_m2': inertias,
            'same_initial_pose': initial_poses,
            'same_external_load_sha256': canonical_sha256(executed_phases),
            'replaced_contacts_disabled': execution.branch == 'reduced',
            'physical_internal_ideal_joints_forbidden': True,
            'family_spec_sha256': canonical_sha256(spec.to_dict()),
            'collision_geometry_sha256': canonical_sha256(collision_manifest),
            'physical_interface_policy': collision_policy_disclosure(
                physical_interfaces, branch=execution.branch
            ),
            'functional_interface_geometry_binding': (
                functional_interface_geometry_binding
            ),
            'conditional_interface_runtime': (
                conditional_interface_runtime
            ),
            'input_actuation_policy': (
                'external_torque_on_declared_input_shaft_only'
                if is_multi_output
                else 'external_torque_on_setting_crown_only'
                if manual_setting_program
                else 'calibrate_on_setting_crown_then_release_and_run_time_barrel'
                if zero_resume_program
                else 'turn_hour_hand_to_latch_alarm_target_only'
                if hour_alarm_setting_program
                else 'turn_hour_hand_to_latch_target_then_run_time_barrel_and_release_independent_alarm_at_selected_position'
                if scheduled_alarm_program and hour_set_alarm_program
                else 'set_alarm_on_setting_crown_then_run_time_barrel_and_trigger_independent_alarm_barrel'
                if scheduled_alarm_program
                else 'external_torque_on_independent_alarm_barrel_only'
                if isolated_alarm_program
                else 'external_torque_on_barrel_only'
            ),
            'active_input_body_ids': sorted(
                {item['active_input_body_id'] for item in executed_phases}
            ),
            'actuation_schedule': executed_phases,
            'alarm_train_isolated': spec.alarm_isolated,
            'direct_hand_or_hammer_actuation': False,
            'direct_hour_second_or_hammer_actuation': (
                hour_set_alarm_program
            ),
            'direct_output_shaft_actuation': False,
        },
        'collision_geometry': {
            'mode': 'intake-bound STEP per-solid body-local collision components',
            **trace_collision_thresholds(collision_manifest),
            'schema': collision_manifest['schema'],
            'source': collision_manifest['source'],
            'assembly_file': collision_manifest['assembly_file'],
            'assembly_sha256': collision_manifest['assembly_sha256'],
            'assembly_contract_sha256': collision_manifest['assembly_contract_sha256'],
            'package_binding_sha256': collision_manifest['package_binding_sha256'],
            'component_count': collision_manifest['component_count'],
            'bodies': collision_audit,
            'contact_enabled': execution.branch == 'physical',
            'physical_interface_policy': collision_policy_disclosure(
                physical_interfaces, branch=execution.branch
            ),
            'collision_filter_policy': (
                'all_body_pairs_enabled_with_validator_step_bound_'
                'analytical_interface_components_omitted'
                if execution.branch == 'physical'
                else 'all_collision_disabled_for_reduced_branch'
            ),
            'validator_owned_suppressed_component_count': sum(
                len(item)
                for item in runtime_suppressed_by_body.values()
            ) if execution.branch == 'physical' else 0,
            'whole_body_box_proxy_used': False,
            'whole_body_convex_hull_used': False,
            'convex_hull_used': False,
            'shaft_count': len(spec.shafts),
            'gear_mesh_count': len(spec.gear_meshes),
            'shaft_transfer_count': len(spec.shaft_transfers),
            'cam_follower_count': len(spec.cam_followers),
            'friction_assumption': spec.friction,
        },
        'axis_abs': list(spec.input_axis_abs),
        'joint_origin_abs_m': list(spec.input_point_abs_m),
        'input_point_abs_m': list(spec.input_point_abs_m),
        'observable_point_initial_abs_m': list(execution.plan.observable_point_abs_m),
        'characteristic_length_m': execution.plan.characteristic_length_m,
        'executed_phases': executed_phases,
        'executed_load_sha256': canonical_sha256(executed_phases),
        'resume_provenance': {
            'schema_version': 'mechanism_resume_provenance_v1',
            'resume_count': len(resume_history),
            'segments': resume_history,
            'state_binding_verified': True,
            'solver_warm_start_cache_serialized': False,
            'checkpoint_restart_policy': checkpoint_restart_policy,
            'continuation_semantics': (
                'a bound partial checkpoint restores the complete saved '
                'clock dynamics and wrench state, then continues the '
                'interrupted phase'
            ),
        },
        'sample_count': len(samples),
        'samples': samples,
        'strict_non_claims': {
            'exact_tooth_impact_or_backlash_certified': False,
            'escapement_timekeeping_accuracy_certified': False,
            'acoustic_alarm_level_certified': False,
            'manufacturing_certification': False,
            'strength_fatigue_or_wear': False,
            'measured_real_friction_or_efficiency': False,
            'dynamic_response_certified': False,
        },
    }
