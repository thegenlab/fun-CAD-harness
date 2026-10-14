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
from .click_pen_spec import click_pen_cam_latch_from_dict
from .collision_policy import (
    COMPLIANT_INTERFACE_PHYSICAL,
    CONTACT_PHYSICAL,
    collision_policy_disclosure,
    interface_declaration,
)
from .conditional_interface import build_functional_interface_geometry_binding
from .contact_intent import (
    EXPECTED_REAL_CONTACT,
    FORBIDDEN_CONTACT,
    build_runtime_component_contact_intent_policy,
    contact_intent_declaration,
)
from .io import canonical_sha256
from .models import MechanismValidationError
from .wrench_ledger import (
    add_ledgered_load,
    create_body_force,
    set_current_wrench_phase,
    standard_wrench_declaration,
    transition_current_wrench_phase,
)

if TYPE_CHECKING:
    from .worker import ExecutionRequest


GUIDE_TRANSVERSE_STIFFNESS_N_M = 500_000.0
GUIDE_TRANSVERSE_DAMPING_N_S_M = 400.0
GUIDE_ROTATIONAL_STIFFNESS_NM_RAD = 300.0
GUIDE_ROTATIONAL_DAMPING_NM_S_RAD = 6.0
FREE_AXIS_DAMPING_N_S_M = 1.5
# The small plastic indexing cam is free about its guide axis.  Keep only a
# light viscous loss here; 2 mN m s/rad made an ordinary 0.2 s button press
# consume nearly the entire stroke in artificial guide drag.
FREE_TORSION_DAMPING_NM_S_RAD = 0.00025
LIMIT_STIFFNESS_N_M = 60_000.0
LIMIT_DAMPING_N_S_M = 80.0


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


def _axis_alignment_angle(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    return math.acos(max(-1.0, min(1.0, abs(_dot(first, second)))))


def _pair_count(sample: dict[str, Any], first: str, second: str) -> int:
    key = "|".join(sorted((first, second)))
    return int((sample.get("contact_pair_counts") or {}).get(key, 0))


def simulate_click_pen(
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
        _signed_axis_angle,
        _tuple3,
        _write_trace_checkpoint,
        lane_execution_source_sha256,
    )

    if execution.mechanism_plugin.worker_backend != "click_pen_cam_latch_v1":
        raise MechanismValidationError(
            "MECHANISM_WORKER_BACKEND_UNAVAILABLE",
            "click pen runtime received a different mechanism backend",
        )
    spec = click_pen_cam_latch_from_dict(execution.adapter.family_spec)
    collision = family_collision_geometry(
        execution.adapter,
        expected_bodies=spec.bodies,
    )
    if collision["package_binding_sha256"] != execution.plan.package_binding_sha256:
        raise MechanismValidationError(
            "FAMILY_COLLIDER_PACKAGE_BINDING_CHANGED",
            "click pen collision geometry does not match the package",
        )
    system = _make_nsc_system(
        chrono,
        collision_envelope_m=float(collision["collision_envelope_m"]),
        collision_margin_m=float(collision["collision_margin_m"]),
    )
    if execution.branch == "physical":
        solver = chrono.ChSolverADMM()
        if hasattr(solver, "SetMaxIterations"):
            solver.SetMaxIterations(1000)
        if hasattr(solver, "SetTolerance"):
            solver.SetTolerance(1.0e-12)
        if hasattr(solver, "SetTolerancePrimal"):
            solver.SetTolerancePrimal(1.0e-10)
        if hasattr(solver, "SetToleranceDual"):
            solver.SetToleranceDual(1.0e-10)
        iterative = solver.AsIterative() if hasattr(solver, "AsIterative") else None
        if iterative is not None and hasattr(iterative, "EnableWarmStart"):
            iterative.EnableWarmStart(True)
        system.SetSolver(solver)
        if hasattr(system, "SetMaxPenetrationRecoverySpeed"):
            system.SetMaxPenetrationRecoverySpeed(1.0)
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
        _collision_progress("system_add_body_begin", body_id=body_spec.body_id)
        system.AddBody(body)
        _collision_progress("system_add_body_complete", body_id=body_spec.body_id)
        bodies[body_spec.body_id] = body
        collision_audit.append(audit)

    fixed = bodies[spec.fixed_body_id]
    button = bodies[spec.input_body_id]
    refill = bodies[spec.output_body_id]
    cam = bodies[spec.cam_body_id]
    button_joint = spec.button_prismatic_joint
    refill_joint = spec.refill_prismatic_joint
    cam_joint = spec.cam_cylindrical_joint
    button_axis = button_joint.axis_abs
    refill_axis = refill_joint.axis_abs
    cam_axis = cam_joint.axis_abs
    cam_press_axis = tuple(-item for item in cam_axis)

    load_container = chrono.ChLoadContainer()
    system.Add(load_container)
    plugin_id = execution.mechanism_plugin.plugin_id
    input_force = create_body_force(
        chrono,
        button,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        False,
        chrono.ChVector3d(*execution.plan.input_point_abs_m),
        False,
        declaration=standard_wrench_declaration(
            plugin_id=plugin_id,
            profile="task_input",
            wrench_kind="force",
            load_id="declared-button-input-force",
            owner=plugin_id,
            target_body=spec.input_body_id,
            source_role="declared_input_port",
            target_role="declared_input",
            geometry_gate="load_case_phase_and_button_input_frame",
            active_interval="declared_load_case_phases",
            release_required=True,
        ),
    )
    add_ledgered_load(load_container, input_force)
    limit_loads: dict[str, Any] = {}
    for joint, body in (
        (button_joint, button),
        (refill_joint, refill),
        (cam_joint, cam),
    ):
        target_body_id = (
            spec.input_body_id
            if joint.joint_id == button_joint.joint_id
            else spec.output_body_id
            if joint.joint_id == refill_joint.joint_id
            else spec.cam_body_id
        )
        load = create_body_force(
            chrono,
            body,
            chrono.ChVector3d(0.0, 0.0, 0.0),
            False,
            chrono.ChVector3d(*joint.origin_abs_m),
            False,
            declaration=standard_wrench_declaration(
                plugin_id=plugin_id,
                profile="limit",
                wrench_kind="force",
                load_id=f"{joint.joint_id}-finite-limit",
                owner=plugin_id,
                target_body=target_body_id,
                source_role="finite_guide_stop_and_preload",
                target_role=(
                    "declared_input"
                    if target_body_id == spec.input_body_id
                    else "task_output"
                    if target_body_id == spec.output_body_id
                    else "internal_moving"
                ),
                geometry_gate=f"guide_range_and_support:{joint.joint_id}",
                active_interval="geometry_gated_simulation_interval",
                release_required=True,
            ),
        )
        add_ledgered_load(load_container, load)
        limit_loads[joint.joint_id] = load

    support_definitions = (
        {
            "joint": button_joint,
            "parent": fixed,
            "child": button,
            "axis": button_axis,
            "template": "finite_spring_return_prismatic_v1",
            "axial_stiffness": button_joint.spring_stiffness_n_m,
            "axial_damping": button_joint.damping_n_s_m,
            "torsional_stiffness": GUIDE_ROTATIONAL_STIFFNESS_NM_RAD,
            "torsional_damping": GUIDE_ROTATIONAL_DAMPING_NM_S_RAD,
            "free_coordinates": ("translation_along_axis",),
        },
        {
            "joint": refill_joint,
            "parent": fixed,
            "child": refill,
            "axis": refill_axis,
            "template": "finite_compression_spring_prismatic_v1",
            "axial_stiffness": refill_joint.spring_stiffness_n_m,
            "axial_damping": refill_joint.damping_n_s_m,
            "torsional_stiffness": GUIDE_ROTATIONAL_STIFFNESS_NM_RAD,
            "torsional_damping": GUIDE_ROTATIONAL_DAMPING_NM_S_RAD,
            "free_coordinates": ("translation_along_axis",),
        },
        {
            "joint": cam_joint,
            "parent": fixed,
            "child": cam,
            "axis": cam_axis,
            "template": "finite_cylindrical_indexing_guide_v1",
            "axial_stiffness": 0.0,
            "axial_damping": FREE_AXIS_DAMPING_N_S_M,
            "torsional_stiffness": 0.0,
            "torsional_damping": FREE_TORSION_DAMPING_NM_S_RAD,
            "free_coordinates": (
                "translation_along_axis",
                "rotation_about_axis",
            ),
        },
    )
    physical_interfaces: list[dict[str, Any]] = []
    support_runtimes: list[dict[str, Any]] = []
    ideal_links: list[str] = []
    if execution.branch == "physical":
        for definition in support_definitions:
            joint = definition["joint"]
            parent = definition["parent"]
            child = definition["child"]
            axis = definition["axis"]
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
                template_id=definition["template"],
                proof_source="validator_click_pen_support_envelope_v1",
            )
            physical_interfaces.append(
                interface_declaration(
                    interface_id=f"support-{joint.joint_id}",
                    classification=COMPLIANT_INTERFACE_PHYSICAL,
                    body_a_id=joint.parent_body_id,
                    body_b_id=joint.child_body_id,
                    model="finite_linear_bushing",
                    source="trusted_click_pen_cam_latch_runtime_v1",
                    free_coordinates=definition["free_coordinates"],
                    parameters={
                        "transverse_stiffness_n_m": (
                            GUIDE_TRANSVERSE_STIFFNESS_N_M
                        ),
                        "transverse_damping_n_s_m": (
                            GUIDE_TRANSVERSE_DAMPING_N_S_M
                        ),
                        "axial_stiffness_n_m": definition["axial_stiffness"],
                        "axial_damping_n_s_m": definition["axial_damping"],
                        "rotational_stiffness_nm_rad": (
                            GUIDE_ROTATIONAL_STIFFNESS_NM_RAD
                        ),
                        "rotational_damping_nm_s_rad": (
                            GUIDE_ROTATIONAL_DAMPING_NM_S_RAD
                        ),
                        "torsional_stiffness_nm_rad": (
                            definition["torsional_stiffness"]
                        ),
                        "torsional_damping_nm_s_rad": (
                            definition["torsional_damping"]
                        ),
                    },
                    transmission_gate=gate,
                )
            )
            stiffness = chrono.ChMatrix66d()
            damping = chrono.ChMatrix66d()
            stiffness.SetZero()
            damping.SetZero()
            stiffness_values = (
                GUIDE_TRANSVERSE_STIFFNESS_N_M,
                GUIDE_TRANSVERSE_STIFFNESS_N_M,
                definition["axial_stiffness"],
                GUIDE_ROTATIONAL_STIFFNESS_NM_RAD,
                GUIDE_ROTATIONAL_STIFFNESS_NM_RAD,
                definition["torsional_stiffness"],
            )
            damping_values = (
                GUIDE_TRANSVERSE_DAMPING_N_S_M,
                GUIDE_TRANSVERSE_DAMPING_N_S_M,
                definition["axial_damping"],
                GUIDE_ROTATIONAL_DAMPING_NM_S_RAD,
                GUIDE_ROTATIONAL_DAMPING_NM_S_RAD,
                definition["torsional_damping"],
            )
            for matrix_index, value in enumerate(stiffness_values):
                stiffness.SetItem(matrix_index, matrix_index, value)
            for matrix_index, value in enumerate(damping_values):
                damping.SetItem(matrix_index, matrix_index, value)
            frame = chrono.ChFramed(
                chrono.ChVector3d(*joint.origin_abs_m),
                _quat_from_z_to_axis(chrono, axis),
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
                        chrono.ChVector3d(*axis)
                    ),
                    "child_axis_local": child.TransformDirectionParentToLocal(
                        chrono.ChVector3d(*axis)
                    ),
                    "gate": BoundedSupportRuntime(
                        interface_id=f"support-{joint.joint_id}",
                        gate=gate,
                    ),
                }
            )
        geometry = spec.click_state_machine.geometry_contract
        contact_tolerance = min(
            geometry.home_contact_tolerance_m,
            float(collision["runtime_interpenetration_tolerance_m"]),
        )
        for declaration in (
            (
                "contact-button-cam-drive",
                spec.input_body_id,
                spec.cam_body_id,
                geometry.button_contact_component_refs,
                geometry.cam_button_face_component_refs,
            ),
            (
                "contact-cam-refill-follower",
                spec.cam_body_id,
                spec.output_body_id,
                (geometry.cam_follower_component_ref,),
                (geometry.refill_carrier_component_ref,),
            ),
            (
                "contact-cam-barrel-index",
                spec.fixed_body_id,
                spec.cam_body_id,
                geometry.barrel_guide_component_refs,
                geometry.cam_lobe_component_refs,
            ),
        ):
            interface_id, body_a, body_b, refs_a, refs_b = declaration
            physical_interfaces.append(
                interface_declaration(
                    interface_id=interface_id,
                    classification=CONTACT_PHYSICAL,
                    body_a_id=body_a,
                    body_b_id=body_b,
                    model="finite_STEP_surface_contact",
                    source="trusted_click_pen_cam_latch_runtime_v1",
                    parameters={
                        "contact_component_a_refs": list(refs_a),
                        "contact_component_b_refs": list(refs_b),
                        "contact_evidence_max_separation_m": contact_tolerance,
                        "no_analytical_cam_torque": True,
                        "no_analytical_latch_force": True,
                    },
                )
            )
    else:
        for joint, parent, child in (
            (button_joint, fixed, button),
            (refill_joint, fixed, refill),
        ):
            link = chrono.ChLinkLockPrismatic()
            link.Initialize(
                child,
                parent,
                chrono.ChFramed(
                    chrono.ChVector3d(*joint.origin_abs_m),
                    _quat_from_z_to_axis(chrono, joint.axis_abs),
                ),
            )
            system.AddLink(link)
            ideal_links.append("ChLinkLockPrismatic")
        cylindrical = chrono.ChLinkLockCylindrical()
        cylindrical.Initialize(
            cam,
            fixed,
            chrono.ChFramed(
                chrono.ChVector3d(*cam_joint.origin_abs_m),
                _quat_from_z_to_axis(chrono, cam_axis),
            ),
        )
        system.AddLink(cylindrical)
        ideal_links.append("ChLinkLockCylindrical")

    actual_link_count = (
        int(system.GetNumLinks()) if hasattr(system, "GetNumLinks") else None
    )
    if execution.branch == "physical" and actual_link_count not in {0, None}:
        raise RuntimeError("physical click pen unexpectedly contains ideal links")

    initial_positions = {
        body_id: _tuple3(body.GetPos())
        for body_id, body in bodies.items()
    }
    initial_cam_quaternion = tuple(_pose(cam)["rotation_abs_quat_wxyz"])
    input_local = button.TransformPointParentToLocal(
        chrono.ChVector3d(*execution.plan.input_point_abs_m)
    )
    origin_locals = {
        button_joint.joint_id: button.TransformPointParentToLocal(
            chrono.ChVector3d(*button_joint.origin_abs_m)
        ),
        refill_joint.joint_id: refill.TransformPointParentToLocal(
            chrono.ChVector3d(*refill_joint.origin_abs_m)
        ),
        cam_joint.joint_id: cam.TransformPointParentToLocal(
            chrono.ChVector3d(*cam_joint.origin_abs_m)
        ),
    }
    samples: list[dict[str, Any]] = []
    executed_phases: list[dict[str, Any]] = []
    current_support_states: dict[str, dict[str, Any]] = {}
    current_input_force = (0.0, 0.0, 0.0)
    time_s = 0.0
    dynamics_step_count = 0
    reference_schedule = _reference_schedule(execution)
    unwrapped_cam_angle = 0.0
    previous_raw_cam_angle = 0.0

    def coordinate(body_id: str, axis: Sequence[float]) -> float:
        return _dot(
            _sub(_tuple3(bodies[body_id].GetPos()), initial_positions[body_id]),
            axis,
        )

    def speed(body_id: str, axis: Sequence[float]) -> float:
        return _dot(_tuple3(bodies[body_id].GetPosDt()), axis)

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

    def button_cam_contact_diagnostics() -> dict[str, Any]:
        """Retain the actual solver contact frame and reaction for cam debugging."""

        container = system.GetContactContainer()
        if container is None or int(container.GetNumContacts()) <= 0:
            return {"contact_count": 0, "contacts": []}

        class ContactReporter(chrono.ReportContactCallback):
            def __init__(self) -> None:
                super().__init__()
                self.contacts: list[dict[str, Any]] = []
                self.errors: list[str] = []

            def OnReportContact(
                self,
                point_a: Any,
                point_b: Any,
                plane: Any,
                distance: float,
                effective_radius: float,
                reaction_force: Any,
                reaction_torque: Any,
                contact_a: Any,
                contact_b: Any,
                constraint_offset: int,
            ) -> bool:
                name_a = str(contact_a.GetPhysicsItem().GetName())
                name_b = str(contact_b.GetPhysicsItem().GetName())
                if {name_a, name_b} != {
                    spec.input_body_id,
                    spec.cam_body_id,
                }:
                    return True
                try:
                    self.contacts.append(
                        {
                            "body_a_id": name_a,
                            "body_b_id": name_b,
                            "point_a_abs_m": list(_tuple3(point_a)),
                            "point_b_abs_m": list(_tuple3(point_b)),
                            "distance_m": float(distance),
                            "effective_radius_m": float(effective_radius),
                            "contact_plane_abs_from_contact": [
                                [
                                    float(
                                        _tuple3(
                                            (
                                                plane.GetAxisX(),
                                                plane.GetAxisY(),
                                                plane.GetAxisZ(),
                                            )[column]
                                        )[row]
                                    )
                                    for column in range(3)
                                ]
                                for row in range(3)
                            ],
                            "reaction_force_contact_n": list(
                                _tuple3(reaction_force)
                            ),
                            "reaction_torque_contact_nm": list(
                                _tuple3(reaction_torque)
                            ),
                            "constraint_offset": int(constraint_offset),
                        }
                    )
                except Exception as exc:
                    self.errors.append(f"{type(exc).__name__}: {exc}")
                return True

        reporter = ContactReporter()
        container.ReportAllContacts(reporter)
        return {
            "contact_count": len(reporter.contacts),
            "contacts": reporter.contacts,
            "reporting_errors": reporter.errors,
        }

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
                axis_misalignment_rad=_axis_alignment_angle(
                    parent_axis,
                    child_axis,
                ),
            )
            current_support_states[runtime["joint"].joint_id] = state
            active = bool(state["transmission_active"])
            runtime["bushing"].SetStiffnessMatrix(
                runtime[
                    "stiffness_matrix" if active else "zero_stiffness_matrix"
                ]
            )
            runtime["bushing"].SetDampingMatrix(
                runtime[
                    "damping_matrix" if active else "zero_damping_matrix"
                ]
            )

    def limited_spring_scalar(
        position_m: float,
        velocity_m_s: float,
        minimum_m: float,
        maximum_m: float,
        preload_n: float,
    ) -> float:
        scalar = -preload_n
        if position_m < minimum_m:
            scalar += (
                LIMIT_STIFFNESS_N_M * (minimum_m - position_m)
                - LIMIT_DAMPING_N_S_M * min(velocity_m_s, 0.0)
            )
        elif position_m > maximum_m:
            scalar += (
                -LIMIT_STIFFNESS_N_M * (position_m - maximum_m)
                - LIMIT_DAMPING_N_S_M * max(velocity_m_s, 0.0)
            )
        return scalar

    def set_limit_forces() -> None:
        button_position = coordinate(spec.input_body_id, button_axis)
        refill_position = coordinate(spec.output_body_id, refill_axis)
        cam_position = coordinate(spec.cam_body_id, cam_press_axis)
        values = (
            (
                button_joint,
                button,
                button_axis,
                button_position,
                speed(spec.input_body_id, button_axis),
                button_joint.spring_preload_n,
            ),
            (
                refill_joint,
                refill,
                refill_axis,
                refill_position,
                speed(spec.output_body_id, refill_axis),
                refill_joint.spring_preload_n,
            ),
            (
                cam_joint,
                cam,
                cam_press_axis,
                cam_position,
                speed(spec.cam_body_id, cam_press_axis),
                0.0,
            ),
        )
        for joint, body, axis, position, velocity, preload in values:
            active = (
                execution.branch != "physical"
                or bool(
                    current_support_states.get(joint.joint_id, {}).get(
                        "transmission_active"
                    )
                )
            )
            scalar = limited_spring_scalar(
                position,
                velocity,
                joint.minimum_position_m,
                joint.maximum_position_m,
                preload if active else 0.0,
            )
            load = limit_loads[joint.joint_id]
            load.SetApplicationPoint(
                body.TransformPointLocalToParent(origin_locals[joint.joint_id]),
                False,
            )
            load.SetForce(
                chrono.ChVector3d(*_scale(axis, scalar)),
                False,
            )

    def current_cam_angle() -> float:
        nonlocal previous_raw_cam_angle, unwrapped_cam_angle
        quaternion = tuple(_pose(cam)["rotation_abs_quat_wxyz"])
        raw = (
            _signed_axis_angle(quaternion, cam_axis)
            - _signed_axis_angle(initial_cam_quaternion, cam_axis)
        )
        delta = raw - previous_raw_cam_angle
        while delta > math.pi:
            delta -= 2.0 * math.pi
        while delta < -math.pi:
            delta += 2.0 * math.pi
        unwrapped_cam_angle += delta
        previous_raw_cam_angle = raw
        return unwrapped_cam_angle

    def record(phase_id: str) -> None:
        button_depth = coordinate(spec.input_body_id, button_axis)
        refill_extension = coordinate(spec.output_body_id, refill_axis)
        cam_depth = coordinate(spec.cam_body_id, cam_press_axis)
        cam_angle = current_cam_angle()
        body_poses = {
            body_id: _pose(body)
            for body_id, body in bodies.items()
        }
        support_drifts = {
            joint_id: float(state["support_drift_m"])
            for joint_id, state in current_support_states.items()
        }
        contact_pairs = _contact_pair_counts(chrono, system)
        latch_contacts = int(
            contact_pairs.get(
                "|".join(sorted((spec.fixed_body_id, spec.cam_body_id))),
                0,
            )
        )
        target_error = abs(
            refill_extension - spec.click_state_machine.first_click_target_m
        )
        extended_latched = (
            target_error <= spec.click_state_machine.latch_contact_tolerance_m
            and latch_contacts > 0
        )
        sample = {
            "time_s": time_s,
            "phase_id": phase_id,
            "generalized_progress_m": refill_extension,
            "lateral_shift_m": max(support_drifts.values(), default=0.0),
            "button_press_depth_m": button_depth,
            "button_speed_m_s": speed(spec.input_body_id, button_axis),
            "refill_extension_m": refill_extension,
            "refill_speed_m_s": speed(spec.output_body_id, refill_axis),
            "cam_axial_position_m": cam_depth,
            "cam_index_angle_rad": cam_angle,
            "cam_index_count": cam_angle / cam_joint.index_angle_rad,
            "extended_latched": extended_latched,
            "latch_contact_count": latch_contacts,
            "expected_contact_count": 1 if extended_latched else 0,
            "max_support_drift_m": max(support_drifts.values(), default=0.0),
            "support_drifts_m": support_drifts,
            "max_off_axis_rotation_rad": max(
                _off_axis_angle(
                    body_poses[body_id]["rotation_abs_quat_wxyz"],
                    (
                        cam_axis
                        if body_id == spec.cam_body_id
                        else (
                            button_axis
                            if body_id == spec.input_body_id
                            else refill_axis
                        )
                    ),
                )
                for body_id in (
                    spec.input_body_id,
                    spec.output_body_id,
                    spec.cam_body_id,
                )
            ),
            "applied_force_abs_n": list(current_input_force),
            "external_load_abs_n": [0.0, 0.0, 0.0],
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
            "contact_pair_counts": contact_pairs,
            "button_cam_contact_diagnostics": (
                button_cam_contact_diagnostics()
                if execution.branch == "physical"
                else {"contact_count": 0, "contacts": []}
            ),
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
                "CLICK_PEN_TORQUE_UNSUPPORTED",
                "click pen accepts button force programs only",
            )
        direction = phase.direction_abs or (0.0, 0.0, 0.0)
        declared_force = _scale(direction, phase.magnitude_si)
        phase_limit_s = (
            reference_schedule[phase_index]["duration_s"]
            if reference_schedule is not None
            else phase.duration_s
        )
        elapsed = 0.0
        record(phase.phase_id)
        while elapsed < phase_limit_s - 1.0e-12:
            current_input_force = declared_force
            input_force.SetApplicationPoint(
                button.TransformPointLocalToParent(input_local),
                False,
            )
            input_force.SetForce(
                chrono.ChVector3d(*current_input_force),
                False,
            )
            set_limit_forces()
            update_supports()
            step = min(execution.load_case.step_s, phase_limit_s - elapsed)
            if execution.branch == "physical" and any(
                position
                >= maximum - 0.0015
                for position, maximum in (
                    (
                        coordinate(spec.input_body_id, button_axis),
                        button_joint.maximum_position_m,
                    ),
                    (
                        coordinate(spec.output_body_id, refill_axis),
                        refill_joint.maximum_position_m,
                    ),
                    (
                        coordinate(spec.cam_body_id, cam_press_axis),
                        cam_joint.maximum_position_m,
                    ),
                )
            ):
                # The sawtooth index changes contact sets at the finite travel
                # stops.  Substep only inside the last 1.5 mm so the compliant
                # limit remains stable without making the whole load expensive.
                step = min(step, 0.0001)
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
        if not samples or samples[-1]["time_s"] < time_s - 1.0e-12:
            record(phase.phase_id)
        executed_phases.append(
            {
                "phase_id": phase.phase_id,
                "wrench_kind": phase.wrench_kind,
                "direction_abs": (
                    None
                    if phase.direction_abs is None
                    else list(phase.direction_abs)
                ),
                "magnitude_si": phase.magnitude_si,
                "duration_s": elapsed,
                "termination_reason": "duration_limit",
                "actuation_channels": [
                    {
                        "body_id": spec.input_body_id,
                        "kind": phase.wrench_kind,
                        "magnitude_si": phase.magnitude_si,
                    }
                ],
            }
        )
        input_force.SetForce(chrono.ChVector3d(0.0, 0.0, 0.0), False)

    input_force.SetForce(chrono.ChVector3d(0.0, 0.0, 0.0), False)
    set_current_wrench_phase(None)
    for load in limit_loads.values():
        load.SetForce(chrono.ChVector3d(0.0, 0.0, 0.0), False)
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
    geometry_binding = (
        build_functional_interface_geometry_binding(
            physical_interfaces,
            collision,
        )
        if execution.branch == "physical"
        else None
    )
    component_contact_policy = None
    if execution.branch == "physical":
        geometry = spec.click_state_machine.geometry_contract
        active_phase_ids = [
            str(item["phase_id"])
            for item in executed_phases
            if item.get("phase_id")
        ]
        zero_input = execution.load_case.program == "zero_input_click_pen"
        contact_threshold = float(collision["interpenetration_tolerance_m"])
        contact_tolerance = min(
            geometry.home_contact_tolerance_m,
            float(collision["runtime_interpenetration_tolerance_m"]),
        )

        def contact_declaration(
            *,
            interface_id: str,
            component_a_refs: Sequence[str],
            component_b_refs: Sequence[str],
            expected_during_zero_input: bool,
        ) -> dict[str, Any]:
            expected = not zero_input or expected_during_zero_input
            return contact_intent_declaration(
                intent_id=f"interface-{interface_id}",
                kind=EXPECTED_REAL_CONTACT if expected else FORBIDDEN_CONTACT,
                component_a_refs=component_a_refs,
                component_b_refs=component_b_refs,
                source="trusted_click_pen_cam_latch_runtime_v1",
                interface_id=interface_id,
                active_phase_ids=active_phase_ids,
                max_penetration_m=contact_threshold if expected else None,
                max_separation_m=contact_tolerance if expected else None,
                binding_scope="validator_owned_exact_component_subset",
            )

        component_contact_policy = build_runtime_component_contact_intent_policy(
            physical_interfaces,
            collision,
            collision_audit,
            executed_phases,
            explicit_declarations=(
                contact_declaration(
                    interface_id="contact-button-cam-drive",
                    component_a_refs=geometry.button_contact_component_refs,
                    component_b_refs=geometry.cam_button_face_component_refs,
                    expected_during_zero_input=False,
                ),
                contact_declaration(
                    interface_id="contact-cam-refill-follower",
                    component_a_refs=(geometry.cam_follower_component_ref,),
                    component_b_refs=(geometry.refill_carrier_component_ref,),
                    expected_during_zero_input=True,
                ),
                contact_declaration(
                    interface_id="contact-cam-barrel-index",
                    component_a_refs=geometry.barrel_guide_component_refs,
                    component_b_refs=geometry.cam_lobe_component_refs,
                    expected_during_zero_input=False,
                ),
            ),
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
            "button_support": (
                "finite_spring_return_ChLoadBodyBodyBushingGeneric"
                if execution.branch == "physical"
                else "ChLinkLockPrismatic"
            ),
            "refill_support": (
                "finite_compression_spring_ChLoadBodyBodyBushingGeneric"
                if execution.branch == "physical"
                else "ChLinkLockPrismatic"
            ),
            "cam_support": (
                "finite_cylindrical_ChLoadBodyBodyBushingGeneric"
                if execution.branch == "physical"
                else "ChLinkLockCylindrical"
            ),
            "cam_indexing": (
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
            "direct_refill_actuation": False,
            "direct_output_actuation": False,
            "button_is_only_actuated_body": True,
            "replaced_contacts_disabled": execution.branch == "reduced",
            "input_body_id": spec.input_body_id,
            "output_body_id": spec.output_body_id,
            "physical_uses_ideal_joint": execution.branch != "physical",
            "cam_index_torque_is_analytical": execution.branch != "physical",
            "latch_force_is_analytical": execution.branch != "physical",
            "physical_interface_policy": collision_policy_disclosure(
                physical_interfaces,
                branch=execution.branch,
            ),
            "functional_interface_geometry_binding": geometry_binding,
            "conditional_interface_runtime": conditional_summary,
            "component_contact_intent_policy": component_contact_policy,
        },
        "strict_non_claims": {
            "manufacturing_certification": False,
            "ink_flow_or_writing_quality": False,
            "wear_life": False,
            "measured_real_friction": False,
            "dynamic_response_certified": False,
        },
    }
