from __future__ import annotations

import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from .bounded_support import (
    BoundedSupportRuntime,
    TRANSVERSE_ANCHOR_DRIFT,
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
from .conditional_interface import build_functional_interface_geometry_binding
from .detent_slider_spec import (
    PrismaticDetentLatchSpec,
    prismatic_detent_latch_from_dict,
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
    set_current_wrench_phase,
    standard_wrench_declaration,
    transition_current_wrench_phase,
)

if TYPE_CHECKING:
    from .worker import ExecutionRequest


GUIDE_TRANSVERSE_STIFFNESS_N_M = 2_400_000.0
GUIDE_TRANSVERSE_DAMPING_N_S_M = 1_900.0
GUIDE_FREE_AXIS_DAMPING_N_S_M = 5.0
GUIDE_ROTATIONAL_STIFFNESS_NM_RAD = 1_200.0
GUIDE_ROTATIONAL_DAMPING_NM_S_RAD = 18.0
LIMIT_STIFFNESS_N_M = 180_000.0
LIMIT_DAMPING_N_S_M = 180.0


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _add(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, float, float]:
    return tuple(float(a) + float(b) for a, b in zip(left, right))


def _sub(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, float, float]:
    return tuple(float(a) - float(b) for a, b in zip(left, right))


def _scale(
    value: Sequence[float],
    factor: float,
) -> tuple[float, float, float]:
    return tuple(float(item) * factor for item in value)


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


def _unsigned_axis_angle(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    return math.acos(max(-1.0, min(1.0, abs(_dot(first, second)))))


def _nearest_detent(
    position_m: float,
    positions_m: Sequence[float],
) -> tuple[float, float]:
    nearest = min(positions_m, key=lambda item: abs(float(item) - position_m))
    return float(nearest), abs(float(nearest) - position_m)


def executed_detent_slider_phase(
    phase: Any,
    *,
    elapsed_s: float,
    spec: PrismaticDetentLatchSpec,
    stop_reached: bool = False,
    actual_coordinate_si: float = 0.0,
) -> dict[str, Any]:
    """Closed executed-phase record: old keys plus strict adaptive termination."""

    external_hold = phase.phase_id == "axial_load_hold"
    payload: dict[str, Any] = {
        "phase_id": phase.phase_id,
        "wrench_kind": phase.wrench_kind,
        "direction_abs": (
            None
            if phase.direction_abs is None
            else list(phase.direction_abs)
        ),
        "magnitude_si": phase.magnitude_si,
        "duration_s": elapsed_s,
        "termination_reason": (
            "target_reached" if stop_reached else "duration_limit"
        ),
        "actuation_channels": (
            [
                {
                    "body_id": spec.output_body_id,
                    "kind": "external_resistance_load",
                    "magnitude_si": phase.magnitude_si,
                }
            ]
            if external_hold
            else [
                {
                    "body_id": spec.input_body_id,
                    "kind": phase.wrench_kind,
                    "magnitude_si": phase.magnitude_si,
                }
            ]
        ),
    }
    if phase.end_magnitude_si is not None:
        payload["end_magnitude_si"] = phase.end_magnitude_si
    if phase.stop_condition is not None:
        payload["termination"] = registered_adaptive_termination_evidence(
            phase,
            actual_coordinate_si=actual_coordinate_si,
            reached=stop_reached,
        )
    return payload


def simulate_detent_slider(
    execution: ExecutionRequest,
    workspace: Path,
) -> dict[str, Any]:
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
        _inertia_box,
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

    if execution.mechanism_plugin.worker_backend != "prismatic_detent_latch_v1":
        raise MechanismValidationError(
            "MECHANISM_WORKER_BACKEND_UNAVAILABLE",
            "detent slider runtime received a different mechanism backend",
        )
    spec = prismatic_detent_latch_from_dict(execution.adapter.family_spec)
    collision = family_collision_geometry(
        execution.adapter,
        expected_bodies=spec.bodies,
    )
    if collision["package_binding_sha256"] != execution.plan.package_binding_sha256:
        raise MechanismValidationError(
            "FAMILY_COLLIDER_PACKAGE_BINDING_CHANGED",
            "detent slider collision geometry does not match the package",
        )
    system = _make_nsc_system(
        chrono,
        collision_envelope_m=float(collision["collision_envelope_m"]),
        collision_margin_m=float(collision["collision_margin_m"]),
    )
    if execution.branch == "physical":
        # The physical branch combines unilateral STEP contact with finite
        # bushing stiffness/damping matrices.  ADMM supports both terms while
        # preserving contact separation; a linear MINRES solve incorrectly
        # kept released contact bilateral.
        solver = chrono.ChSolverADMM()
        if hasattr(solver, "SetMaxIterations"):
            solver.SetMaxIterations(500)
        if hasattr(solver, "SetTolerance"):
            solver.SetTolerance(1.0e-10)
        system.SetSolver(solver)
    system.SetGravitationalAcceleration(
        chrono.ChVector3d(*execution.plan.gravity_abs_m_s2)
    )
    material = _make_material(chrono, spec.friction)
    bodies: dict[str, Any] = {}
    collision_audit: list[dict[str, Any]] = []
    for index, body_spec in enumerate(spec.bodies):
        body, audit = make_family_body(
            chrono,
            body_spec=body_spec,
            geometry=collision,
            workspace=workspace,
            material=material,
            inertia_xx=_inertia_box(body_spec.mass_kg, body_spec.size_m),
            branch=execution.branch,
        )
        if execution.branch == "physical":
            body.GetCollisionModel().SetFamily(index + 1)
            audit["collision_family"] = index + 1
        _collision_progress(
            "system_add_body_begin",
            body_id=body_spec.body_id,
        )
        system.AddBody(body)
        _collision_progress(
            "system_add_body_complete",
            body_id=body_spec.body_id,
        )
        bodies[body_spec.body_id] = body
        collision_audit.append(audit)

    fixed = bodies[spec.fixed_body_id]
    blade = bodies[spec.output_body_id]
    latch = bodies[spec.input_body_id]
    blade_joint = spec.blade_prismatic_joint
    latch_joint = spec.latch_press_joint
    blade_axis = blade_joint.axis_abs
    press_axis = latch_joint.axis_abs

    load_container = chrono.ChLoadContainer()
    system.Add(load_container)
    plugin_id = execution.mechanism_plugin.plugin_id
    input_force = create_body_force(
        chrono,
        latch,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        False,
        chrono.ChVector3d(*execution.plan.input_point_abs_m),
        False,
        declaration=standard_wrench_declaration(
            plugin_id=plugin_id,
            profile="task_input",
            wrench_kind="force",
            load_id="declared-latch-input-force",
            owner=plugin_id,
            target_body=spec.input_body_id,
            source_role="declared_input_port",
            target_role="declared_input",
            geometry_gate="load_case_phase_and_input_frame",
            active_interval="declared_load_case_phases",
            release_required=True,
        ),
    )
    blade_external_force = create_body_force(
        chrono,
        blade,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        False,
        chrono.ChVector3d(*blade_joint.origin_abs_m),
        False,
        declaration=standard_wrench_declaration(
            plugin_id=plugin_id,
            profile="probe",
            wrench_kind="force",
            load_id="output-resistance-probe",
            owner=plugin_id,
            target_body=spec.output_body_id,
            source_role="validator_resistance_probe",
            target_role="task_output",
            geometry_gate="axial_load_hold_phase_only",
            active_interval="declared_adversarial_hold_phase",
            release_required=True,
        ),
    )
    blade_limit_force = create_body_force(
        chrono,
        blade,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        False,
        chrono.ChVector3d(*blade_joint.origin_abs_m),
        False,
        declaration=standard_wrench_declaration(
            plugin_id=plugin_id,
            profile="limit",
            wrench_kind="force",
            load_id="blade-finite-travel-limit",
            owner=plugin_id,
            target_body=spec.output_body_id,
            source_role="finite_travel_stop",
            target_role="task_output",
            geometry_gate="blade_position_outside_declared_range",
            active_interval="geometry_gated_simulation_interval",
            release_required=True,
        ),
    )
    latch_limit_force = create_body_force(
        chrono,
        latch,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        False,
        chrono.ChVector3d(*latch_joint.origin_abs_m),
        False,
        declaration=standard_wrench_declaration(
            plugin_id=plugin_id,
            profile="limit",
            wrench_kind="force",
            load_id="latch-finite-press-limit",
            owner=plugin_id,
            target_body=spec.input_body_id,
            source_role="finite_press_stop_and_preload",
            target_role="declared_input",
            geometry_gate="latch_press_range_and_support_gate",
            active_interval="geometry_gated_simulation_interval",
            release_required=True,
        ),
    )
    carriage_reaction_force = create_body_force(
        chrono,
        blade,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        False,
        chrono.ChVector3d(*latch_joint.origin_abs_m),
        False,
        declaration=standard_wrench_declaration(
            plugin_id=plugin_id,
            profile="transmission",
            wrench_kind="force",
            load_id="latch-carriage-reaction",
            owner=plugin_id,
            target_body=spec.output_body_id,
            source_role="latch_press_interface",
            target_role="task_output",
            geometry_gate="latch_support_and_press_geometry_gate",
            active_interval="geometry_gated_simulation_interval",
            release_required=True,
        ),
    )
    for load in (
        input_force,
        blade_external_force,
        blade_limit_force,
        latch_limit_force,
        carriage_reaction_force,
    ):
        add_ledgered_load(load_container, load)

    support_specs = (
        (
            blade_joint,
            fixed,
            blade,
            "finite_prismatic_guide_capture_v1",
            0.0,
            GUIDE_FREE_AXIS_DAMPING_N_S_M,
        ),
        (
            latch_joint,
            blade,
            latch,
            "finite_spring_return_press_capture_v1",
            latch_joint.spring_stiffness_n_m,
            latch_joint.damping_n_s_m,
        ),
    )
    physical_interfaces: list[dict[str, Any]] = []
    support_runtimes: list[dict[str, Any]] = []
    ideal_links: list[str] = []
    if execution.branch == "physical":
        for joint, parent, child, template_id, axial_stiffness, axial_damping in support_specs:
            gate = bounded_support_gate(
                body_a_id=joint.parent_body_id,
                body_b_id=joint.child_body_id,
                endpoint_a_component_refs=body_component_refs(
                    collision,
                    joint.parent_body_id,
                ),
                endpoint_b_component_refs=body_component_refs(
                    collision,
                    joint.child_body_id,
                ),
                maximum_drift_m=spec.max_support_drift_m,
                coordinate_mode=TRANSVERSE_ANCHOR_DRIFT,
                maximum_axis_misalignment_rad=spec.max_off_axis_rotation_rad,
                template_id=template_id,
                proof_source="validator_prismatic_detent_support_envelope_v1",
            )
            physical_interfaces.append(
                interface_declaration(
                    interface_id=f"support-{joint.joint_id}",
                    classification=COMPLIANT_INTERFACE_PHYSICAL,
                    body_a_id=joint.parent_body_id,
                    body_b_id=joint.child_body_id,
                    model="finite_linear_bushing",
                    source="trusted_prismatic_detent_latch_runtime_v1",
                    free_coordinates=("translation_along_axis",),
                    parameters={
                        "transverse_stiffness_n_m": GUIDE_TRANSVERSE_STIFFNESS_N_M,
                        "transverse_damping_n_s_m": GUIDE_TRANSVERSE_DAMPING_N_S_M,
                        "axial_stiffness_n_m": axial_stiffness,
                        "axial_damping_n_s_m": axial_damping,
                        "axial_preload_n": joint.spring_preload_n,
                        "rotational_stiffness_nm_rad": GUIDE_ROTATIONAL_STIFFNESS_NM_RAD,
                        "rotational_damping_nm_s_rad": GUIDE_ROTATIONAL_DAMPING_NM_S_RAD,
                    },
                    transmission_gate=gate,
                )
            )
            stiffness = chrono.ChMatrix66d()
            damping = chrono.ChMatrix66d()
            stiffness.SetZero()
            damping.SetZero()
            for axis_index, value in enumerate(
                (
                    GUIDE_TRANSVERSE_STIFFNESS_N_M,
                    GUIDE_TRANSVERSE_STIFFNESS_N_M,
                    axial_stiffness,
                    GUIDE_ROTATIONAL_STIFFNESS_NM_RAD,
                    GUIDE_ROTATIONAL_STIFFNESS_NM_RAD,
                    GUIDE_ROTATIONAL_STIFFNESS_NM_RAD,
                )
            ):
                stiffness.SetItem(axis_index, axis_index, value)
            for axis_index, value in enumerate(
                (
                    GUIDE_TRANSVERSE_DAMPING_N_S_M,
                    GUIDE_TRANSVERSE_DAMPING_N_S_M,
                    axial_damping,
                    GUIDE_ROTATIONAL_DAMPING_NM_S_RAD,
                    GUIDE_ROTATIONAL_DAMPING_NM_S_RAD,
                    GUIDE_ROTATIONAL_DAMPING_NM_S_RAD,
                )
            ):
                damping.SetItem(axis_index, axis_index, value)
            frame = chrono.ChFramed(
                chrono.ChVector3d(*joint.origin_abs_m),
                _quat_from_z_to_axis(chrono, joint.axis_abs),
            )
            bushing = chrono.ChLoadBodyBodyBushingGeneric(
                child,
                parent,
                frame,
                stiffness,
                damping,
            )
            load_container.Add(bushing)
            zero_stiffness = chrono.ChMatrix66d()
            zero_damping = chrono.ChMatrix66d()
            zero_stiffness.SetZero()
            zero_damping.SetZero()
            origin = chrono.ChVector3d(*joint.origin_abs_m)
            support_runtimes.append(
                {
                    "joint": joint,
                    "parent": parent,
                    "child": child,
                    "bushing": bushing,
                    "stiffness_matrix": stiffness,
                    "damping_matrix": damping,
                    "zero_stiffness_matrix": zero_stiffness,
                    "zero_damping_matrix": zero_damping,
                    "parent_anchor_local": parent.TransformPointParentToLocal(origin),
                    "child_anchor_local": child.TransformPointParentToLocal(origin),
                    "parent_axis_local": parent.TransformDirectionParentToLocal(
                        chrono.ChVector3d(*joint.axis_abs)
                    ),
                    "child_axis_local": child.TransformDirectionParentToLocal(
                        chrono.ChVector3d(*joint.axis_abs)
                    ),
                    "gate": BoundedSupportRuntime(
                        interface_id=f"support-{joint.joint_id}",
                        gate=gate,
                    ),
                }
            )
        physical_interfaces.append(
            interface_declaration(
                interface_id="contact-discrete-detent",
                classification=CONTACT_PHYSICAL,
                body_a_id=spec.detent_interface.rack_body_id,
                body_b_id=spec.detent_interface.tab_body_id,
                model="finite_STEP_surface_contact",
                source="trusted_prismatic_detent_latch_runtime_v1",
                parameters={
                    "rack_component_refs": list(
                        spec.detent_interface.geometry_contract.rack_component_refs
                    ),
                    "seat_component_refs": list(
                        spec.detent_interface.geometry_contract.seat_component_refs
                    ),
                    "tab_component_ref": (
                        spec.detent_interface.geometry_contract.tab_component_ref
                    ),
                    "contact_component_a_refs": list(
                        spec.detent_interface.geometry_contract.rack_component_refs
                        + spec.detent_interface.geometry_contract.seat_component_refs
                    ),
                    "contact_component_b_refs": [
                        spec.detent_interface.geometry_contract.tab_component_ref
                    ],
                    "contact_evidence_max_separation_m": min(
                        spec.detent_interface.geometry_contract.contact_tolerance_m,
                        float(collision["runtime_interpenetration_tolerance_m"]),
                    ),
                    "no_analytical_lock_force": True,
                },
            )
        )
    else:
        blade_link = chrono.ChLinkLockPrismatic()
        blade_link.Initialize(
            blade,
            fixed,
            chrono.ChFramed(
                chrono.ChVector3d(*blade_joint.origin_abs_m),
                _quat_from_z_to_axis(chrono, blade_axis),
            ),
        )
        latch_link = chrono.ChLinkLockPrismatic()
        latch_link.Initialize(
            latch,
            blade,
            chrono.ChFramed(
                chrono.ChVector3d(*latch_joint.origin_abs_m),
                _quat_from_z_to_axis(chrono, press_axis),
            ),
        )
        system.AddLink(blade_link)
        system.AddLink(latch_link)
        ideal_links.extend(("ChLinkLockPrismatic", "ChLinkLockPrismatic"))

    actual_link_count = (
        int(system.GetNumLinks()) if hasattr(system, "GetNumLinks") else None
    )
    if execution.branch == "physical" and actual_link_count not in {0, None}:
        raise RuntimeError("physical detent slider unexpectedly contains ideal links")

    initial_blade_position = _tuple3(blade.GetPos())
    initial_latch_position = _tuple3(latch.GetPos())
    input_local = latch.TransformPointParentToLocal(
        chrono.ChVector3d(*execution.plan.input_point_abs_m)
    )
    blade_origin_local = blade.TransformPointParentToLocal(
        chrono.ChVector3d(*blade_joint.origin_abs_m)
    )
    latch_origin_local = latch.TransformPointParentToLocal(
        chrono.ChVector3d(*latch_joint.origin_abs_m)
    )
    samples: list[dict[str, Any]] = []
    executed_phases: list[dict[str, Any]] = []
    current_support_states: dict[str, dict[str, Any]] = {}
    current_input_force = (0.0, 0.0, 0.0)
    current_external_force = (0.0, 0.0, 0.0)
    time_s = 0.0
    dynamics_step_count = 0
    reference_schedule = _reference_schedule(execution)

    def blade_position_m() -> float:
        return _dot(_sub(_tuple3(blade.GetPos()), initial_blade_position), blade_axis)

    def blade_speed_m_s() -> float:
        return _dot(_tuple3(blade.GetPosDt()), blade_axis)

    def latch_position_m() -> float:
        return _dot(_sub(_tuple3(latch.GetPos()), initial_latch_position), blade_axis)

    def latch_press_depth_m() -> float:
        expected_from_blade = _add(
            initial_latch_position,
            _scale(blade_axis, blade_position_m()),
        )
        return _dot(_sub(_tuple3(latch.GetPos()), expected_from_blade), press_axis)

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

    def update_supports() -> None:
        if execution.branch != "physical":
            return
        for runtime in support_runtimes:
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
            state = runtime["gate"].evaluate(
                time_s=time_s,
                anchor_a_abs_m=parent_point,
                anchor_b_abs_m=child_point,
                velocity_a_abs_m_s=point_velocity(parent, parent_point),
                velocity_b_abs_m_s=point_velocity(child, child_point),
                requested_effort_si=_norm(
                    _tuple3(runtime["bushing"].GetForce())
                ),
                free_axis_abs=parent_axis,
                axis_misalignment_rad=_unsigned_axis_angle(
                    parent_axis,
                    child_axis,
                ),
            )
            current_support_states[runtime["joint"].joint_id] = state
            active = bool(state["transmission_active"])
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

    def set_limit_forces() -> None:
        progress = blade_position_m()
        speed = blade_speed_m_s()
        blade_scalar = 0.0
        if progress < blade_joint.minimum_position_m:
            blade_scalar = (
                LIMIT_STIFFNESS_N_M
                * (blade_joint.minimum_position_m - progress)
                - LIMIT_DAMPING_N_S_M * min(speed, 0.0)
            )
        elif progress > blade_joint.maximum_position_m:
            blade_scalar = (
                -LIMIT_STIFFNESS_N_M
                * (progress - blade_joint.maximum_position_m)
                - LIMIT_DAMPING_N_S_M * max(speed, 0.0)
            )
        blade_limit_force.SetApplicationPoint(
            blade.TransformPointLocalToParent(blade_origin_local),
            False,
        )
        blade_limit_force.SetForce(
            chrono.ChVector3d(*_scale(blade_axis, blade_scalar)),
            False,
        )

        press = latch_press_depth_m()
        relative_speed = _dot(
            _sub(_tuple3(latch.GetPosDt()), _tuple3(blade.GetPosDt())),
            press_axis,
        )
        preload_active = (
            execution.branch != "physical"
            or bool(
                current_support_states.get(
                    latch_joint.joint_id,
                    {},
                ).get("transmission_active")
            )
        )
        latch_scalar = (
            -latch_joint.spring_preload_n
            if preload_active
            else 0.0
        )
        if press < latch_joint.minimum_position_m:
            latch_scalar += (
                LIMIT_STIFFNESS_N_M
                * (latch_joint.minimum_position_m - press)
                - LIMIT_DAMPING_N_S_M * min(relative_speed, 0.0)
            )
        elif press > latch_joint.maximum_position_m:
            latch_scalar += (
                -LIMIT_STIFFNESS_N_M
                * (press - latch_joint.maximum_position_m)
                - LIMIT_DAMPING_N_S_M * max(relative_speed, 0.0)
            )
        latch_force_abs = _scale(press_axis, latch_scalar)
        latch_limit_force.SetApplicationPoint(
            latch.TransformPointLocalToParent(latch_origin_local),
            False,
        )
        latch_limit_force.SetForce(
            chrono.ChVector3d(*latch_force_abs),
            False,
        )
        carriage_reaction_force.SetApplicationPoint(
            chrono.ChVector3d(*latch_joint.origin_abs_m),
            False,
        )
        carriage_reaction_force.SetForce(
            chrono.ChVector3d(*_scale(latch_force_abs, -1.0)),
            False,
        )

    def detent_state() -> tuple[bool, float, float]:
        nearest, error = _nearest_detent(
            blade_position_m(),
            spec.detent_interface.positions_m,
        )
        engaged = (
            latch_press_depth_m() < spec.detent_interface.release_press_m
            and error <= spec.detent_interface.capture_tolerance_m
        )
        return engaged, nearest, error

    def record(phase_id: str) -> None:
        engaged, nearest, detent_error = detent_state()
        body_poses = {
            body_id: _pose(body)
            for body_id, body in bodies.items()
        }
        support_drifts = {
            joint_id: float(state["support_drift_m"])
            for joint_id, state in current_support_states.items()
        }
        blade_progress = blade_position_m()
        max_support_drift = max(support_drifts.values(), default=0.0)
        sample = {
            "time_s": time_s,
            "phase_id": phase_id,
            "generalized_progress_m": blade_progress,
            "lateral_shift_m": max_support_drift,
            "blade_carriage_position_m": blade_progress,
            "blade_carriage_speed_m_s": blade_speed_m_s(),
            "thumb_latch_position_m": latch_position_m(),
            "thumb_press_depth_m": latch_press_depth_m(),
            "detent_engaged": engaged,
            "locked_detent_position_m": nearest if engaged else None,
            "detent_position_error_m": detent_error,
            "expected_contact_count": 1 if engaged else 0,
            "max_support_drift_m": max_support_drift,
            "support_drifts_m": support_drifts,
            "max_off_axis_rotation_rad": max(
                _off_axis_angle(
                    body_poses[body_id]["rotation_abs_quat_wxyz"],
                    (
                        blade_axis
                        if body_id == spec.output_body_id
                        else press_axis
                    ),
                )
                for body_id in (spec.output_body_id, spec.input_body_id)
            ),
            "applied_force_abs_n": list(current_input_force),
            "external_load_abs_n": list(current_external_force),
            "body_poses": body_poses,
            "body_linear_velocities_abs_m_s": {
                body_id: list(_tuple3(body.GetPosDt()))
                for body_id, body in bodies.items()
            },
            "body_angular_velocities_abs_rad_s": {
                body_id: list(_angular_velocity(body))
                for body_id, body in bodies.items()
            },
            "contact_count": _contact_count(system),
            "contact_pair_counts": _contact_pair_counts(chrono, system),
            "conditional_interface_states": [
                dict(current_support_states[joint_id])
                for joint_id in sorted(current_support_states)
            ],
        }
        samples.append(sample)
        _write_trace_checkpoint(
            execution,
            samples,
            executed_phases=executed_phases,
        )

    update_supports()
    record("initial")
    next_sample_s = execution.load_case.sample_interval_s
    first_step = True
    for phase_index, phase in enumerate(execution.load_case.phases):
        transition_current_wrench_phase(phase.phase_id)
        if phase.wrench_kind == "torque":
            raise MechanismValidationError(
                "DETENT_SLIDER_TORQUE_UNSUPPORTED",
                "detent slider accepts force programs only",
            )
        direction = phase.direction_abs or (0.0, 0.0, 0.0)
        declared_force = _scale(direction, phase.magnitude_si)
        external_hold = phase.phase_id == "axial_load_hold"
        phase_limit_s = (
            reference_schedule[phase_index]["duration_s"]
            if reference_schedule is not None
            else phase.duration_s
        )
        elapsed = 0.0
        stop_reached = False
        while elapsed < phase_limit_s - 1.0e-12:
            current_input_force = (
                (0.0, 0.0, 0.0)
                if external_hold
                else declared_force
            )
            current_external_force = (
                declared_force
                if external_hold
                else (0.0, 0.0, 0.0)
            )
            input_force.SetApplicationPoint(
                latch.TransformPointLocalToParent(input_local),
                False,
            )
            input_force.SetForce(
                chrono.ChVector3d(*current_input_force),
                False,
            )
            blade_external_force.SetApplicationPoint(
                blade.TransformPointLocalToParent(blade_origin_local),
                False,
            )
            blade_external_force.SetForce(
                chrono.ChVector3d(*current_external_force),
                False,
            )
            set_limit_forces()
            update_supports()
            step = min(execution.load_case.step_s, phase_limit_s - elapsed)
            if first_step:
                _collision_progress(
                    "first_dynamics_step_begin",
                    capsule_id=execution.plan.capsule_id,
                )
            system.DoStepDynamics(step)
            dynamics_step_count += 1
            if first_step:
                _collision_progress(
                    "first_dynamics_step_complete",
                    capsule_id=execution.plan.capsule_id,
                )
                first_step = False
            elapsed += step
            time_s += step
            if time_s + 1.0e-12 >= next_sample_s:
                record(phase.phase_id)
                while next_sample_s <= time_s + 1.0e-12:
                    next_sample_s += execution.load_case.sample_interval_s
            if phase.stop_condition is not None:
                reached = registered_stop_comparator_reached(
                    phase.stop_condition,
                    target_coordinate_si=float(phase.target_coordinate),
                    actual_coordinate_si=blade_position_m(),
                )
                if reached:
                    stop_reached = True
                    break
        if not samples or samples[-1]["time_s"] < time_s - 1.0e-12:
            record(phase.phase_id)
        executed_phases.append(
            executed_detent_slider_phase(
                phase,
                elapsed_s=elapsed,
                spec=spec,
                stop_reached=stop_reached,
                actual_coordinate_si=blade_position_m(),
            )
        )
        input_force.SetForce(chrono.ChVector3d(0.0, 0.0, 0.0), False)

    for load in (
        input_force,
        blade_external_force,
        blade_limit_force,
        latch_limit_force,
        carriage_reaction_force,
    ):
        load.SetForce(chrono.ChVector3d(0.0, 0.0, 0.0), False)
    set_current_wrench_phase(None)
    update_supports()
    conditional_summary = (
        conditional_runtime_summary(
            [runtime["gate"] for runtime in support_runtimes],
            physics_step_count=dynamics_step_count,
        )
        if execution.branch == "physical"
        else {
            "status": "NOT_REQUIRED",
            "conditional_interface_count": 0,
        }
    )
    functional_interface_geometry_binding = (
        build_functional_interface_geometry_binding(
            physical_interfaces,
            collision,
        )
        if execution.branch == "physical"
        else None
    )
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
            "version": (
                str(chrono.GetChronoVersion())
                if hasattr(chrono, "GetChronoVersion")
                else None
            ),
            "system": "ChSystemNSC",
            "solver": (
                "ChSolverADMM"
                if execution.branch == "physical"
                else "ChSolverPSOR"
            ),
            "mechanism_plugin_id": execution.mechanism_plugin.plugin_id,
            "worker_backend": execution.mechanism_plugin.worker_backend,
            "step_s": execution.load_case.step_s,
            "gravity_abs_m_s2": list(execution.plan.gravity_abs_m_s2),
            "chrono_link_count": actual_link_count,
            "ideal_links": ideal_links,
            "internal_ideal_joint_count": len(ideal_links),
            "blade_support": (
                "finite_ChLoadBodyBodyBushingGeneric"
                if execution.branch == "physical"
                else "ChLinkLockPrismatic"
            ),
            "latch_support": (
                "finite_spring_return_ChLoadBodyBodyBushingGeneric"
                if execution.branch == "physical"
                else "ChLinkLockPrismatic"
            ),
            "latch_spring_preload_n": latch_joint.spring_preload_n,
            "detent_lock": (
                "STEP_surface_contact_only"
                if execution.branch == "physical"
                else "not_simulated_use_deterministic_ideal_reference"
            ),
        },
        "axis_abs": list(execution.plan.axis_abs),
        "joint_origin_abs_m": list(execution.plan.joint_origin_abs_m),
        "input_point_abs_m": list(execution.plan.input_point_abs_m),
        "observable_point_initial_abs_m": list(
            execution.plan.observable_point_abs_m
        ),
        "characteristic_length_m": execution.plan.characteristic_length_m,
        "collision_geometry": {
            "mode": collision["schema"],
            "source": collision["source"],
            "package_binding_sha256": collision["package_binding_sha256"],
            "assembly_sha256": collision["assembly_sha256"],
            **trace_collision_thresholds(collision),
            "contact_enabled": execution.branch == "physical",
            "component_count": collision["component_count"],
            "body_count": collision["body_count"],
            "bodies": collision_audit,
            "whole_body_box_proxy_used": False,
            "whole_body_convex_hull_used": False,
            "convex_hull_used": False,
            "physical_interface_declarations": physical_interfaces,
            "physical_interface_policy": collision_policy_disclosure(
                physical_interfaces,
                branch=execution.branch,
            ),
        },
        "conditional_interface_runtime": conditional_summary,
        "executed_phases": executed_phases,
        "executed_load_sha256": canonical_sha256(executed_phases),
        "sample_count": len(samples),
        "samples": samples,
        "branch_semantics": {
            "direct_blade_carriage_actuation": False,
            "direct_output_actuation": False,
            "replaced_contacts_disabled": execution.branch == "reduced",
            "external_axial_resistance_load_is_not_actuation": True,
            "input_body_id": spec.input_body_id,
            "output_body_id": spec.output_body_id,
            "physical_uses_ideal_joint": execution.branch != "physical",
            "detent_force_is_analytical": execution.branch != "physical",
            "physical_interface_policy": collision_policy_disclosure(
                physical_interfaces,
                branch=execution.branch,
            ),
            "functional_interface_geometry_binding": (
                functional_interface_geometry_binding
            ),
            "conditional_interface_runtime": conditional_summary,
        },
        "strict_non_claims": {
            "manufacturing_certification": False,
            "blade_sharpness_or_cut_rating": False,
            "wear_life": False,
            "measured_real_friction": False,
            "dynamic_response_certified": False,
        },
    }
