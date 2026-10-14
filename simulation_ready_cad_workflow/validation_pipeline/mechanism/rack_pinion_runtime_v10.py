from __future__ import annotations

import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .bounded_support import (
    BoundedSupportRuntime,
    RELATIVE_ANCHOR_DRIFT,
    TRANSVERSE_ANCHOR_DRIFT,
    body_component_refs,
    bounded_support_gate,
    conditional_runtime_summary,
)
from .collision_policy import (
    ANALYTICAL_TRANSMISSION,
    COMPLIANT_INTERFACE_PHYSICAL,
    CONTACT_PHYSICAL,
    collision_policy_disclosure,
    interface_declaration,
)
from .conditional_interface import (
    ConditionalCompliantInterface,
    RackPinionGeometry,
    RackPinionPose,
    build_functional_interface_geometry_binding,
    conditional_transmission_gate,
)
from .io import canonical_sha256
from .models import MechanismValidationError
from .task_motion_evidence import (
    registered_adaptive_termination_evidence,
    registered_stop_comparator_reached,
)
from .rack_pinion_loads_v3 import (
    RACK_PINION_BRAKE_DAMPING_NM_S_RAD,
    RACK_PINION_ACTUATOR_LIMIT_SCALE,
    RACK_PINION_KINETIC_BRAKE_FRACTION,
    RACK_PINION_OPERATOR_DAMPING_NM_S_RAD,
    RACK_PINION_STATIC_BRAKE_FRACTION,
)
from .rack_pinion_spec import RackPinionSpec, rack_pinion_from_dict

# Normally-engaged holding brake lock loop: omega_n * step_s for the anchored
# rotational spring so the explicit per-step torque update stays stable.
RACK_PINION_BRAKE_LOCK_OMEGA_DT = 0.05
DEADBOLT_AUXILIARY_TORQUE_RAMP_S = 0.25


def _bounded_operator_torque_axis_nm(
    commanded_torque_axis_nm: float,
    pinion_speed_rad_s: float,
    nominal_input_torque_nm: float,
    operator_damping_n_m_s_rad: float = (
        RACK_PINION_OPERATOR_DAMPING_NM_S_RAD
    ),
) -> float:
    """Return the bounded operator torque applied about the pinion axis."""

    if abs(commanded_torque_axis_nm) <= 1.0e-9:
        return 0.0
    torque_axis_nm = (
        commanded_torque_axis_nm
        - operator_damping_n_m_s_rad * pinion_speed_rad_s
    )
    # This is an operator-input model, not a bidirectional velocity servo.
    # Damping may reduce the declared manual effort to zero, but must never
    # reverse it when a stiff transmission makes the sampled speed oscillate.
    # A reversal drives the rack back into its home stop on alternating steps
    # and can cancel essentially all net travel while still reporting a
    # non-zero task input.
    if torque_axis_nm * commanded_torque_axis_nm <= 0.0:
        return 0.0
    actuator_limit_nm = (
        RACK_PINION_ACTUATOR_LIMIT_SCALE * nominal_input_torque_nm
    )
    return max(-actuator_limit_nm, min(actuator_limit_nm, torque_axis_nm))


def _debug_dump_contacts(
    chrono: Any, system: Any, workspace: Path, time_s: float
) -> None:
    """Env-gated (SR_CAD_CONTACT_DEBUG) contact detail dump for diagnosis."""

    import json as _json

    container = system.GetContactContainer()
    rows: list[dict[str, Any]] = []
    if container is not None and int(container.GetNumContacts()) > 0:

        class Reporter(chrono.ReportContactCallback):
            def OnReportContact(
                self,
                point_a: Any,
                point_b: Any,
                _plane: Any,
                distance: float,
                _radius: float,
                force: Any,
                _torque: Any,
                contact_a: Any,
                contact_b: Any,
                _offset: int,
            ) -> bool:
                rows.append(
                    {
                        "a": str(contact_a.GetPhysicsItem().GetName()),
                        "b": str(contact_b.GetPhysicsItem().GetName()),
                        "point_a": [point_a.x, point_a.y, point_a.z],
                        "point_b": [point_b.x, point_b.y, point_b.z],
                        "distance": float(distance),
                        "force_plane": [force.x, force.y, force.z],
                    }
                )
                return True

        container.ReportAllContacts(Reporter())
    debug_path = workspace / "contact_debug.jsonl"
    with debug_path.open("a", encoding="utf-8") as handle:
        handle.write(
            _json.dumps({"time_s": time_s, "contacts": rows}) + "\n"
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


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _cross(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _add(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return tuple(a + b for a, b in zip(left, right))


def _sub(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return tuple(a - b for a, b in zip(left, right))


def _scale(value: Sequence[float], factor: float) -> tuple[float, float, float]:
    return tuple(item * factor for item in value)


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(_dot(value, value))


def _component_ids_for_body_refs(
    body_id: str,
    component_refs: Sequence[str],
) -> tuple[str, ...]:
    prefix = f"{body_id}/"
    component_ids: list[str] = []
    for component_ref in component_refs:
        if not component_ref.startswith(prefix):
            raise MechanismValidationError(
                "RUNTIME_COMPONENT_SUPPRESSION_BINDING_INVALID",
                f"component ref {component_ref!r} is not owned by {body_id!r}",
            )
        component_id = component_ref[len(prefix) :]
        if not component_id or component_id in component_ids:
            raise MechanismValidationError(
                "RUNTIME_COMPONENT_SUPPRESSION_BINDING_INVALID",
                f"component ref {component_ref!r} is empty or duplicated",
            )
        component_ids.append(component_id)
    return tuple(component_ids)


def _rotate_vector_by_quaternion(
    vector: Sequence[float],
    quaternion_wxyz: Sequence[float],
) -> tuple[float, float, float]:
    w, x, y, z = quaternion_wxyz
    quaternion_vector = (x, y, z)
    twice_cross = _scale(_cross(quaternion_vector, vector), 2.0)
    return _add(
        _add(vector, _scale(twice_cross, w)),
        _cross(quaternion_vector, twice_cross),
    )


def _world_point_to_body_local(
    point_abs: Sequence[float],
    body_spec: Any,
) -> tuple[float, float, float]:
    w, x, y, z = body_spec.rotation_abs_quat_wxyz
    return _rotate_vector_by_quaternion(
        _sub(point_abs, body_spec.center_abs_m),
        (w, -x, -y, -z),
    )


def _world_direction_to_body_local(
    direction_abs: Sequence[float],
    body_spec: Any,
) -> tuple[float, float, float]:
    w, x, y, z = body_spec.rotation_abs_quat_wxyz
    return _rotate_vector_by_quaternion(
        direction_abs,
        (w, -x, -y, -z),
    )


def _component_vertices(
    component: Mapping[str, Any],
    *,
    workspace: Path,
) -> tuple[tuple[float, float, float], ...]:
    mesh = component.get("mesh")
    if not isinstance(mesh, str) or not mesh:
        raise MechanismValidationError(
            "RACK_PINION_INTERFACE_COMPONENT_MESH_MISSING",
            "rack-pinion interface selection requires per-solid STEP meshes",
        )
    path = Path(mesh)
    if not path.is_absolute():
        path = workspace / path
    vertices: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            fields = line.split()
            if len(fields) != 4:
                continue
            vertices.append(
                (float(fields[1]), float(fields[2]), float(fields[3]))
            )
    if not vertices:
        raise MechanismValidationError(
            "RACK_PINION_INTERFACE_COMPONENT_MESH_EMPTY",
            f"rack-pinion STEP component has no mesh vertices: {path}",
        )
    return tuple(vertices)


def _projected_span(
    vertices: Sequence[Sequence[float]],
    axis: Sequence[float],
) -> float:
    coordinates = [_dot(vertex, axis) for vertex in vertices]
    return max(coordinates) - min(coordinates)


def _rack_pinion_interface_component_refs(
    collision_manifest: Mapping[str, Any],
    *,
    workspace: Path,
    spec: RackPinionSpec,
    pinion_center_local_m: Sequence[float],
    pinion_axis_local: Sequence[float],
    rack_pitch_line_local_m: Sequence[float],
    rack_axis_local: Sequence[float],
    rack_face_axis_local: Sequence[float],
) -> tuple[tuple[str, ...], tuple[str, ...], float]:
    bodies = {
        str(item.get("body_id")): item
        for item in collision_manifest.get("bodies") or ()
        if isinstance(item, Mapping)
    }
    pinion_body = bodies.get(spec.pinion_body_id)
    rack_body = bodies.get(spec.rack_body_id)
    if not isinstance(pinion_body, Mapping) or not isinstance(
        rack_body, Mapping
    ):
        raise MechanismValidationError(
            "RACK_PINION_INTERFACE_BODY_GEOMETRY_MISSING",
            "rack-pinion interface bodies are missing from STEP collision geometry",
        )

    pinion_refs: list[str] = []
    pinion_face_widths: list[float] = []
    for component in pinion_body.get("components") or ():
        if not isinstance(component, Mapping):
            continue
        vertices = _component_vertices(component, workspace=workspace)
        radial_maximum = 0.0
        for vertex in vertices:
            delta = _sub(vertex, pinion_center_local_m)
            axial = _dot(delta, pinion_axis_local)
            radial = _sub(
                delta,
                _scale(pinion_axis_local, axial),
            )
            radial_maximum = max(radial_maximum, _norm(radial))
        if not (
            0.90 * spec.pinion_pitch_radius_m
            <= radial_maximum
            <= 1.35 * spec.pinion_pitch_radius_m
        ):
            continue
        face_width = _projected_span(vertices, pinion_axis_local)
        if face_width <= 0.0:
            continue
        component_id = str(
            component.get("component_id") or component.get("solid_id") or ""
        )
        if component_id:
            component_ref = f"{spec.pinion_body_id}/{component_id}"
            if component_ref in pinion_refs:
                existing = pinion_refs.index(component_ref)
                pinion_face_widths[existing] = max(
                    pinion_face_widths[existing], face_width
                )
            else:
                pinion_refs.append(component_ref)
                pinion_face_widths.append(face_width)

    if not pinion_refs:
        raise MechanismValidationError(
            "RACK_PINION_PINION_TEETH_NOT_BOUND",
            "validator could not bind STEP pinion tooth components",
        )
    face_width_m = max(pinion_face_widths)
    rack_refs: list[str] = []
    for component in rack_body.get("components") or ():
        if not isinstance(component, Mapping):
            continue
        vertices = _component_vertices(component, workspace=workspace)
        rack_span = _projected_span(vertices, rack_axis_local)
        face_span = _projected_span(vertices, rack_face_axis_local)
        center = tuple(
            sum(vertex[index] for vertex in vertices) / len(vertices)
            for index in range(3)
        )
        center_delta = _sub(center, rack_pitch_line_local_m)
        along_rack = _dot(center_delta, rack_axis_local)
        normal_delta = _sub(
            center_delta,
            _scale(rack_axis_local, along_rack),
        )
        if (
            rack_span <= 0.55 * spec.pinion_pitch_radius_m
            and 0.65 * face_width_m <= face_span <= 1.35 * face_width_m
            and _norm(normal_delta)
            <= 0.55 * spec.pinion_pitch_radius_m
        ):
            component_id = str(
                component.get("component_id")
                or component.get("solid_id")
                or ""
            )
            if component_id:
                component_ref = f"{spec.rack_body_id}/{component_id}"
                if component_ref not in rack_refs:
                    rack_refs.append(component_ref)

    if not rack_refs:
        raise MechanismValidationError(
            "RACK_PINION_RACK_TEETH_NOT_BOUND",
            "validator could not bind STEP rack tooth components",
        )
    return tuple(pinion_refs), tuple(rack_refs), face_width_m


def _rack_frame_quaternion(chrono: Any, spec: RackPinionSpec) -> Any:
    # ChLinkMateRackPinion uses rack-frame X as its rack direction.
    mesh_axis = spec.rack_axis_abs
    axis_x = chrono.ChVector3d(*mesh_axis)
    mesh_normal_axis = _scale(spec.pinion_axis_abs, -1.0)
    axis_z = chrono.ChVector3d(*mesh_normal_axis)
    axis_y = chrono.ChVector3d(*_cross(mesh_normal_axis, mesh_axis))
    matrix = chrono.ChMatrix33d()
    matrix.SetFromDirectionAxes(axis_x, axis_y, axis_z)
    return matrix.GetQuaternion()


def _local_frame(body: Any, absolute_position: Any, absolute_rotation: Any, chrono: Any) -> Any:
    local_position = body.TransformPointParentToLocal(absolute_position)
    local_rotation = body.GetRot().GetConjugate() * absolute_rotation
    return chrono.ChFramed(local_position, local_rotation)


def simulate_rack_pinion(
    execution: ExecutionRequest,
    workspace: Path,
) -> dict[str, Any]:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(os.path.join(conda_prefix, "Library", "bin"))
    import pychrono as chrono

    from .worker import (
        _angular_velocity,
        _configure_prismatic_travel_limit,
        _configure_revolute_angle_limit,
        _contact_count,
        _contact_pair_counts,
        _inertia_box,
        _make_material,
        _make_nsc_system,
        _pose,
        _quat_from_z_to_axis,
        _reference_schedule,
        _tuple3,
        _write_trace_checkpoint,
        _collision_progress,
        lane_execution_source_sha256,
    )

    if execution.mechanism_plugin.worker_backend != "rack_pinion_v1":
        raise MechanismValidationError(
            "MECHANISM_WORKER_BACKEND_UNAVAILABLE",
            "rack-pinion runtime received a different mechanism backend",
        )
    spec = rack_pinion_from_dict(execution.adapter.family_spec)
    from .family_collision import (
        family_collision_geometry,
        make_family_body,
        trace_collision_thresholds,
    )

    collision_manifest = family_collision_geometry(
        execution.adapter,
        expected_bodies=spec.bodies,
    )
    if collision_manifest["package_binding_sha256"] != execution.plan.package_binding_sha256:
        raise MechanismValidationError(
            "FAMILY_COLLIDER_PACKAGE_BINDING_CHANGED",
            "rack-pinion collision geometry does not match the executed package",
        )
    body_specs = {body.body_id: body for body in spec.bodies}
    pinion_body_spec = body_specs[spec.pinion_body_id]
    rack_body_spec = body_specs[spec.rack_body_id]
    pinion_center_local = _world_point_to_body_local(
        spec.pinion_origin_abs_m,
        pinion_body_spec,
    )
    pinion_axis_local = _world_direction_to_body_local(
        spec.pinion_axis_abs,
        pinion_body_spec,
    )
    rack_pitch_line_local = _world_point_to_body_local(
        execution.plan.observable_point_abs_m,
        rack_body_spec,
    )
    rack_axis_local = _world_direction_to_body_local(
        spec.rack_axis_abs,
        rack_body_spec,
    )
    rack_face_axis_local = _world_direction_to_body_local(
        spec.pinion_axis_abs,
        rack_body_spec,
    )
    (
        pinion_interface_refs,
        rack_interface_refs,
        interface_face_width_m,
    ) = _rack_pinion_interface_component_refs(
        collision_manifest,
        workspace=workspace,
        spec=spec,
        pinion_center_local_m=pinion_center_local,
        pinion_axis_local=pinion_axis_local,
        rack_pitch_line_local_m=rack_pitch_line_local,
        rack_axis_local=rack_axis_local,
        rack_face_axis_local=rack_face_axis_local,
    )
    raw_family_spec = (
        execution.adapter.family_spec
        if isinstance(execution.adapter.family_spec, Mapping)
        else {}
    )
    auxiliary_raw = raw_family_spec.get("auxiliary_revolute")
    runtime_suppressed_component_ids: dict[str, tuple[str, ...]] = {}
    if execution.branch == "physical" and isinstance(auxiliary_raw, Mapping):
        # The deadbolt adapter already declares a validator-owned local
        # collision exception for the STEP-bound pinion/tooth endpoints. Honor
        # that declaration in the physical collision model so hard contact
        # does not duplicate the geometry-gated finite penalty transmission.
        # The complete STEP components remain in saved-pose legality audits.
        runtime_suppressed_component_ids = {
            spec.pinion_body_id: _component_ids_for_body_refs(
                spec.pinion_body_id,
                pinion_interface_refs,
            ),
            spec.rack_body_id: _component_ids_for_body_refs(
                spec.rack_body_id,
                rack_interface_refs,
            ),
        }
    reference_schedule = _reference_schedule(execution)
    system = _make_nsc_system(
        chrono, collision_envelope_m=12e-6, collision_margin_m=2e-6
    )
    if execution.branch == "physical":
        solver = chrono.ChSolverMINRES()
        if hasattr(solver, "SetMaxIterations"):
            solver.SetMaxIterations(300)
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
            runtime_suppressed_component_ids=(
                runtime_suppressed_component_ids.get(body_spec.body_id, ())
            ),
        )
        if execution.branch == "physical":
            family = body_index + 1
            body.GetCollisionModel().SetFamily(family)
            collision_families[body_spec.body_id] = family
            audit["collision_family"] = family
        _collision_progress("system_add_body_begin", body_id=body_spec.body_id)
        system.AddBody(body)
        _collision_progress("system_add_body_complete", body_id=body_spec.body_id)
        bodies[body_spec.body_id] = body
        collision_audit.append(audit)
        inertias[body_spec.body_id] = list(inertia)
        initial_poses[body_spec.body_id] = _pose(body)

    ground = bodies[spec.fixed_body_id]
    pinion = bodies[spec.pinion_body_id]
    rack = bodies[spec.rack_body_id]
    auxiliary_revolute: dict[str, Any] | None = None
    auxiliary_body = None
    auxiliary_frame_abs = None
    auxiliary_input_point_local = None
    if isinstance(auxiliary_raw, Mapping):
        body_id = str(auxiliary_raw.get("body_id") or "")
        if body_id not in bodies or body_id in {
            spec.fixed_body_id,
            spec.pinion_body_id,
            spec.rack_body_id,
        }:
            raise MechanismValidationError(
                "RACK_PINION_AUXILIARY_BODY_INVALID",
                "auxiliary revolute body must be one distinct moving body",
            )
        axis = tuple(float(item) for item in auxiliary_raw.get("axis_abs") or ())
        origin = tuple(
            float(item) for item in auxiliary_raw.get("origin_abs_m") or ()
        )
        input_point = tuple(
            float(item) for item in auxiliary_raw.get("input_point_abs_m") or ()
        )
        if len(axis) != 3 or len(origin) != 3 or len(input_point) != 3:
            raise MechanismValidationError(
                "RACK_PINION_AUXILIARY_GEOMETRY_INVALID",
                "auxiliary revolute axis, origin, and input point must be vectors",
            )
        axis_norm = _norm(axis)
        if abs(axis_norm - 1.0) > 1.0e-9:
            raise MechanismValidationError(
                "RACK_PINION_AUXILIARY_GEOMETRY_INVALID",
                "auxiliary revolute axis must be normalized",
            )
        auxiliary_revolute = {
            "body_id": body_id,
            "axis_abs": axis,
            "origin_abs_m": origin,
            "input_point_abs_m": input_point,
            "lock_progress_threshold_m": float(
                auxiliary_raw.get("lock_progress_threshold_m")
            ),
            "lock_stiffness_n_m_rad": float(
                auxiliary_raw.get("lock_stiffness_n_m_rad")
            ),
            "lock_damping_n_m_s_rad": float(
                auxiliary_raw.get("lock_damping_n_m_s_rad")
            ),
            "maximum_angle_rad": float(
                auxiliary_raw.get("maximum_angle_rad")
            ),
            "limit_stiffness_n_m_rad": float(
                auxiliary_raw.get("limit_stiffness_n_m_rad")
            ),
            "limit_damping_n_m_s_rad": float(
                auxiliary_raw.get("limit_damping_n_m_s_rad")
            ),
        }
        if any(
            not math.isfinite(value) or value <= 0.0
            for key, value in auxiliary_revolute.items()
            if key
            in {
                "lock_stiffness_n_m_rad",
                "lock_damping_n_m_s_rad",
                "maximum_angle_rad",
                "limit_stiffness_n_m_rad",
                "limit_damping_n_m_s_rad",
            }
        ):
            raise MechanismValidationError(
                "RACK_PINION_AUXILIARY_PARAMETERS_INVALID",
                "auxiliary revolute stiffness, damping, and limit must be positive",
            )
        auxiliary_body = bodies[body_id]
        auxiliary_input_point_local = (
            auxiliary_body.TransformPointParentToLocal(
                chrono.ChVector3d(*input_point)
            )
        )
        auxiliary_frame_abs = chrono.ChFramed(
            chrono.ChVector3d(*origin),
            _quat_from_z_to_axis(chrono, axis),
        )
    support_parent = auxiliary_body if auxiliary_body is not None else ground
    support_parent_body_id = (
        auxiliary_revolute["body_id"]
        if auxiliary_revolute is not None
        else spec.fixed_body_id
    )
    support_gates = {
        "support-pinion-bearing": bounded_support_gate(
            body_a_id=support_parent_body_id,
            body_b_id=spec.pinion_body_id,
            endpoint_a_component_refs=body_component_refs(
                collision_manifest,
                support_parent_body_id,
            ),
            endpoint_b_component_refs=body_component_refs(
                collision_manifest,
                spec.pinion_body_id,
            ),
            maximum_drift_m=spec.max_support_drift_m,
            coordinate_mode=RELATIVE_ANCHOR_DRIFT,
            maximum_axis_misalignment_rad=spec.max_off_axis_rotation_rad,
            template_id="finite_revolute_pin_capture_v1",
            proof_source="validator_rack_pinion_support_envelope_v1",
        ),
        "support-rack-guide": bounded_support_gate(
            body_a_id=support_parent_body_id,
            body_b_id=spec.rack_body_id,
            endpoint_a_component_refs=body_component_refs(
                collision_manifest,
                support_parent_body_id,
            ),
            endpoint_b_component_refs=body_component_refs(
                collision_manifest,
                spec.rack_body_id,
            ),
            maximum_drift_m=spec.max_support_drift_m,
            coordinate_mode=TRANSVERSE_ANCHOR_DRIFT,
            maximum_axis_misalignment_rad=spec.max_off_axis_rotation_rad,
            template_id="finite_prismatic_guide_retention_v1",
            proof_source="validator_rack_pinion_support_envelope_v1",
        ),
    }
    if auxiliary_revolute is not None:
        support_gates["support-auxiliary-revolute"] = bounded_support_gate(
            body_a_id=spec.fixed_body_id,
            body_b_id=auxiliary_revolute["body_id"],
            endpoint_a_component_refs=body_component_refs(
                collision_manifest,
                spec.fixed_body_id,
            ),
            endpoint_b_component_refs=body_component_refs(
                collision_manifest,
                auxiliary_revolute["body_id"],
            ),
            maximum_drift_m=spec.max_support_drift_m,
            coordinate_mode=RELATIVE_ANCHOR_DRIFT,
            maximum_axis_misalignment_rad=spec.max_off_axis_rotation_rad,
            template_id="finite_revolute_pin_capture_v1",
            proof_source="validator_deadbolt_auxiliary_hinge_v1",
        )
    physical_interfaces = [
        interface_declaration(
            interface_id="support-pinion-bearing",
            classification=COMPLIANT_INTERFACE_PHYSICAL,
            body_a_id=support_parent_body_id,
            body_b_id=spec.pinion_body_id,
            model="finite_revolute_bushing",
            source="trusted_rack_pinion_runtime_v1",
            free_coordinates=("rotation_about_axis",),
            transmission_gate=support_gates["support-pinion-bearing"],
        ),
        interface_declaration(
            interface_id="support-rack-guide",
            classification=COMPLIANT_INTERFACE_PHYSICAL,
            body_a_id=support_parent_body_id,
            body_b_id=spec.rack_body_id,
            model="finite_prismatic_guide_bushing",
            source="trusted_rack_pinion_runtime_v1",
            free_coordinates=("translation_along_axis",),
            transmission_gate=support_gates["support-rack-guide"],
        ),
    ]
    if auxiliary_revolute is not None:
        physical_interfaces.append(
            interface_declaration(
                interface_id="support-auxiliary-revolute",
                classification=COMPLIANT_INTERFACE_PHYSICAL,
                body_a_id=spec.fixed_body_id,
                body_b_id=auxiliary_revolute["body_id"],
                model="finite_revolute_bushing",
                source="trusted_deadbolt_auxiliary_hinge_runtime_v1",
                free_coordinates=("rotation_about_axis",),
                transmission_gate=support_gates[
                    "support-auxiliary-revolute"
                ],
            )
        )
        reaches_strike = execution.load_case.program in {
            "task_deadbolt_lock",
            "task_deadbolt_locked_hold",
            "task_deadbolt_unlock_open",
        } and any(
            phase.stop_condition == "position_at_least"
            and phase.target_position_si is not None
            and float(phase.target_position_si)
            >= spec.outbound_progress_m - spec.output_progress_tolerance_m
            for phase in execution.load_case.phases
        )
        if reaches_strike:
            frame_refs = set(
                body_component_refs(
                    collision_manifest,
                    spec.fixed_body_id,
                )
            )
            bolt_refs = set(
                body_component_refs(
                    collision_manifest,
                    spec.rack_body_id,
                )
            )
            strike_component_ref = f"{spec.fixed_body_id}/solid_2"
            bolt_nose_component_ref = (
                f"{spec.rack_body_id}/bolt_body_component_002"
            )
            if (
                strike_component_ref not in frame_refs
                or bolt_nose_component_ref not in bolt_refs
            ):
                raise MechanismValidationError(
                    "DEADBOLT_STRIKE_COMPONENT_BINDING_INVALID",
                    "deadbolt strike contact components are unavailable",
                )
            physical_interfaces.append(
                interface_declaration(
                    interface_id="contact-deadbolt-strike",
                    classification=CONTACT_PHYSICAL,
                    body_a_id=spec.fixed_body_id,
                    body_b_id=spec.rack_body_id,
                    model="finite_bolt_strike_contact",
                    source="trusted_deadbolt_strike_contact_v1",
                    parameters={
                        "contact_component_a_refs": [
                            strike_component_ref
                        ],
                        "contact_component_b_refs": [
                            bolt_nose_component_ref
                        ],
                        "contact_evidence_max_separation_m": 0.001,
                    },
                )
            )
    # Supports retain complete body-pair collision. The deadbolt analytical
    # transmission removes only its validator-bound endpoint components from
    # runtime hard contact; complete STEP geometry remains audit-visible.
    initial_rack_position = _tuple3(rack.GetPos())
    initial_rack_position_parent_local = _tuple3(
        support_parent.TransformPointParentToLocal(rack.GetPos())
    )
    support_rack_axis_parent_local = _tuple3(
        support_parent.TransformDirectionParentToLocal(
            chrono.ChVector3d(*spec.rack_axis_abs)
        )
    )
    support_pinion_axis_parent_local = _tuple3(
        support_parent.TransformDirectionParentToLocal(
            chrono.ChVector3d(*spec.pinion_axis_abs)
        )
    )
    pinion_frame_abs = chrono.ChFramed(
        chrono.ChVector3d(*spec.pinion_origin_abs_m),
        _quat_from_z_to_axis(chrono, spec.pinion_axis_abs),
    )
    rack_rotation_abs = _rack_frame_quaternion(chrono, spec)
    rack_mesh_origin_abs = chrono.ChVector3d(*execution.plan.observable_point_abs_m)
    rack_frame_abs = chrono.ChFramed(rack_mesh_origin_abs, rack_rotation_abs)
    rack_guide_frame_abs = chrono.ChFramed(
        rack.GetPos(), _quat_from_z_to_axis(chrono, spec.rack_axis_abs)
    )
    initial_pitch_delta = _sub(
        spec.pinion_origin_abs_m,
        execution.plan.observable_point_abs_m,
    )
    initial_pitch_delta = _sub(
        initial_pitch_delta,
        _scale(
            spec.rack_axis_abs,
            _dot(initial_pitch_delta, spec.rack_axis_abs),
        ),
    )
    required_pitch_line_offset_m = _norm(initial_pitch_delta)
    pitch_line_offset_tolerance_m = max(
        0.0005,
        1.5 * spec.max_support_drift_m,
    )
    axis_tolerance_rad = max(
        math.radians(0.25),
        0.5 * spec.max_off_axis_rotation_rad,
    )
    rack_pinion_gate = conditional_transmission_gate(
        geometry_plugin_id="rack_pinion_mesh_geometry_v1",
        body_a_id=spec.pinion_body_id,
        body_b_id=spec.rack_body_id,
        endpoint_a_component_refs=pinion_interface_refs,
        endpoint_b_component_refs=rack_interface_refs,
        activation_coordinate="face_overlap_m",
        enter_threshold_si=0.0005,
        release_threshold_si=0.0001,
        proof_source="validator_rack_pinion_step_geometry_v1",
        geometry_parameters={
            "required_pitch_line_offset_m": (
                required_pitch_line_offset_m
            ),
            "pitch_line_offset_tolerance_m": (
                pitch_line_offset_tolerance_m
            ),
            "axis_orthogonality_tolerance_rad": axis_tolerance_rad,
            "face_axis_parallel_tolerance_rad": axis_tolerance_rad,
            "pinion_face_width_m": interface_face_width_m,
            "rack_face_width_m": interface_face_width_m,
        },
    )
    physical_interfaces.append(
        interface_declaration(
            interface_id="transmission-pinion-rack",
            classification=ANALYTICAL_TRANSMISSION,
            body_a_id=spec.pinion_body_id,
            body_b_id=spec.rack_body_id,
            model="finite_conditional_penalty_gear_mesh",
            source="trusted_rack_pinion_runtime_v1",
            transmission_gate=rack_pinion_gate,
        )
    )
    functional_interface_geometry_binding = (
        build_functional_interface_geometry_binding(
            physical_interfaces,
            collision_manifest,
        )
    )
    rack_pinion_geometry = RackPinionGeometry(
        required_pitch_line_offset_m=required_pitch_line_offset_m,
        pitch_line_offset_tolerance_m=pitch_line_offset_tolerance_m,
        axis_orthogonality_tolerance_rad=axis_tolerance_rad,
        face_axis_parallel_tolerance_rad=axis_tolerance_rad,
    )
    rack_pinion_interface = ConditionalCompliantInterface(
        rack_pinion_gate
    )

    ideal_links: list[str] = []
    ideal_limit_disclosure: dict[str, Any] = {}
    if execution.branch == "reduced":
        revolute = chrono.ChLinkLockRevolute()
        revolute.Initialize(pinion, ground, pinion_frame_abs)
        # The rack travel stop owns the coupled range. Avoid a duplicate
        # angular stop whose native scalar sign depends on link body ordering.
        ideal_limit_disclosure["pinion_angle"] = {
            "coordinate": "coupled_to_rack_progress",
            "min_rad": spec.home_progress_m / spec.pinion_pitch_radius_m,
            "max_rad": spec.outbound_progress_m / spec.pinion_pitch_radius_m,
            "source": "rack_travel_limit_via_ChLinkMateRackPinion",
        }
        system.AddLink(revolute)
        ideal_links.append("ChLinkLockRevolute")

        prismatic = chrono.ChLinkLockPrismatic()
        prismatic.Initialize(rack, ground, rack_guide_frame_abs)
        ideal_limit_disclosure["rack_progress"] = _configure_prismatic_travel_limit(
            prismatic,
            spec.outbound_progress_m,
            min_progress_m=spec.home_progress_m,
        )
        system.AddLink(prismatic)
        ideal_links.append("ChLinkLockPrismatic")

        rack_pinion = chrono.ChLinkMateRackPinion()
        rack_pinion.Initialize(pinion, rack, pinion_frame_abs)
        rack_pinion.SetPinionFrame(
            _local_frame(
                pinion,
                chrono.ChVector3d(*spec.pinion_origin_abs_m),
                pinion_frame_abs.GetRot(),
                chrono,
            )
        )
        rack_pinion.SetRackFrame(
            _local_frame(rack, rack_mesh_origin_abs, rack_rotation_abs, chrono)
        )
        rack_pinion.SetPinionRadius(spec.pinion_pitch_radius_m)
        rack_pinion.SetPressureAngle(spec.pressure_angle_rad)
        rack_pinion.SetEnforcePhase(True)
        system.AddLink(rack_pinion)
        ideal_links.append("ChLinkMateRackPinion")

    link_count = int(system.GetNumLinks()) if hasattr(system, "GetNumLinks") else None
    if execution.branch == "physical" and link_count not in {0, None}:
        raise RuntimeError("rack-pinion physical branch unexpectedly contains an ideal link")
    if execution.branch == "reduced" and link_count != 3:
        raise RuntimeError("rack-pinion reduced branch did not create its three ideal links")

    load_container = chrono.ChLoadContainer()
    system.Add(load_container)
    plugin_id = execution.mechanism_plugin.plugin_id
    input_torque = create_body_torque(
        chrono,
        pinion,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        False,
        declaration=standard_wrench_declaration(
            plugin_id=plugin_id,
            profile="task_input_controller_150",
            wrench_kind="torque",
            load_id="declared-pinion-input-torque",
            owner=plugin_id,
            target_body=spec.pinion_body_id,
            source_role="declared_input_port",
            target_role="declared_input",
            geometry_gate="load_case_phase_and_pinion_input_frame",
            active_interval="declared_load_case_phases",
            release_required=True,
        ),
    )
    add_ledgered_load(load_container, input_torque)
    auxiliary_input_force = None
    auxiliary_input_torque = None
    auxiliary_lock_torque = None
    endpoint_detent_force = None
    if auxiliary_revolute is not None and auxiliary_body is not None:
        auxiliary_input_force = create_body_force(
            chrono,
            auxiliary_body,
            chrono.ChVector3d(0.0, 0.0, 0.0),
            False,
            chrono.ChVector3d(*auxiliary_revolute["input_point_abs_m"]),
            False,
            declaration=standard_wrench_declaration(
                plugin_id=plugin_id,
                profile="task_input",
                wrench_kind="force",
                load_id="declared-auxiliary-door-input-force",
                owner=plugin_id,
                target_body=auxiliary_revolute["body_id"],
                source_role="declared_input_port",
                target_role="declared_input",
                geometry_gate="load_case_phase_and_auxiliary_input_frame",
                active_interval="declared_load_case_phases",
                release_required=True,
            ),
        )
        auxiliary_input_torque = create_body_torque(
            chrono,
            auxiliary_body,
            chrono.ChVector3d(0.0, 0.0, 0.0),
            False,
            declaration=standard_wrench_declaration(
                plugin_id=plugin_id,
                profile="task_input_controller_150",
                wrench_kind="torque",
                load_id="declared-auxiliary-door-input-torque",
                owner=plugin_id,
                target_body=auxiliary_revolute["body_id"],
                source_role="declared_input_port",
                target_role="declared_input",
                geometry_gate="load_case_phase_and_auxiliary_input_frame",
                active_interval="declared_load_case_phases",
                release_required=True,
            ),
        )
        auxiliary_lock_torque = create_body_torque(
            chrono,
            auxiliary_body,
            chrono.ChVector3d(0.0, 0.0, 0.0),
            False,
            declaration=standard_wrench_declaration(
                plugin_id=plugin_id,
                profile="support",
                wrench_kind="torque",
                load_id="deadbolt-strike-door-lock-torque",
                owner=plugin_id,
                target_body=auxiliary_revolute["body_id"],
                source_role="finite_bolt_strike_lock_contact",
                target_role="task_output",
                geometry_gate="rack_progress_at_lock_threshold",
                active_interval="geometry_gated_locked_interval",
            ),
        )
        endpoint_detent_force = create_body_force(
            chrono,
            rack,
            chrono.ChVector3d(0.0, 0.0, 0.0),
            False,
            rack.GetPos(),
            False,
            declaration=standard_wrench_declaration(
                plugin_id=plugin_id,
                profile="support",
                wrench_kind="force",
                load_id="deadbolt-rack-endpoint-detent-force",
                owner=plugin_id,
                target_body=spec.rack_body_id,
                source_role="finite_deadbolt_endpoint_detent",
                target_role="task_output",
                geometry_gate="rack_progress_near_home_or_locked_endpoint",
                active_interval="zero_knob_input_endpoint_capture_interval",
            ),
        )
        for load in (
            auxiliary_input_force,
            auxiliary_input_torque,
            auxiliary_lock_torque,
            endpoint_detent_force,
        ):
            add_ledgered_load(load_container, load)

    pinion_bushing = None
    rack_bushing = None
    auxiliary_bushing = None
    support_runtimes: list[dict[str, Any]] = []
    gear_force = None
    gear_torque = None
    stop_force = None
    brake_torque = create_body_torque(
        chrono,
        pinion,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        False,
        declaration=standard_wrench_declaration(
            plugin_id=plugin_id,
            profile="resistance",
            wrench_kind="torque",
            load_id="pinion-passive-brake",
            owner=plugin_id,
            target_body=spec.pinion_body_id,
            source_role="finite_passive_brake",
            target_role="declared_input",
            geometry_gate="brake_static_or_kinetic_regime",
            active_interval="simulation_when_brake_response_is_nonzero",
        ),
    )
    add_ledgered_load(load_container, brake_torque)
    reduced_brake_force = None
    if execution.branch == "reduced":
        reduced_brake_force = create_body_force(
            chrono,
            rack,
            chrono.ChVector3d(0.0, 0.0, 0.0),
            False,
            rack.GetPos(),
            False,
            declaration=standard_wrench_declaration(
                plugin_id=plugin_id,
                profile="resistance",
                wrench_kind="force",
                load_id="rack-virtual-work-passive-brake",
                owner=plugin_id,
                target_body=spec.rack_body_id,
                source_role="finite_passive_brake",
                target_role="task_output",
                geometry_gate="reduced_branch_virtual_work_mapping",
                active_interval="simulation_when_brake_response_is_nonzero",
            ),
        )
        add_ledgered_load(load_container, reduced_brake_force)
    if execution.branch == "physical":
        pinion_stiffness = chrono.ChMatrix66d()
        pinion_damping = chrono.ChMatrix66d()
        rack_stiffness = chrono.ChMatrix66d()
        rack_damping = chrono.ChMatrix66d()
        for matrix in (pinion_stiffness, pinion_damping, rack_stiffness, rack_damping):
            matrix.SetZero()
        # Model the pinion bearings and rack guide as finite, steel-supported
        # interfaces.  The previous placeholder values were softer than the
        # gate frame by several orders of magnitude, so the first real tooth
        # contact could throw both bodies out of their supports even though
        # the authoritative STEP geometry started without overlap.
        for index, value in enumerate(
            (50_000_000.0, 50_000_000.0, 50_000_000.0, 80_000.0, 80_000.0, 0.0)
        ):
            pinion_stiffness.SetItem(index, index, value)
        for index, value in enumerate(
            (18_000.0, 18_000.0, 18_000.0, 120.0, 120.0, 0.08)
        ):
            pinion_damping.SetItem(index, index, value)
        for index, value in enumerate(
            (25_000_000.0, 25_000_000.0, 0.0, 50_000.0, 50_000.0, 50_000.0)
        ):
            rack_stiffness.SetItem(index, index, value)
        for index, value in enumerate(
            (32_000.0, 32_000.0, 3.0, 500.0, 500.0, 500.0)
        ):
            rack_damping.SetItem(index, index, value)
        pinion_bushing = chrono.ChLoadBodyBodyBushingGeneric(
            pinion,
            support_parent,
            pinion_frame_abs,
            pinion_stiffness,
            pinion_damping,
        )
        rack_bushing = chrono.ChLoadBodyBodyBushingGeneric(
            rack,
            support_parent,
            rack_guide_frame_abs,
            rack_stiffness,
            rack_damping,
        )
        load_container.Add(pinion_bushing)
        load_container.Add(rack_bushing)
        auxiliary_stiffness = None
        auxiliary_damping = None
        if auxiliary_revolute is not None and auxiliary_body is not None:
            auxiliary_stiffness = chrono.ChMatrix66d()
            auxiliary_damping = chrono.ChMatrix66d()
            auxiliary_stiffness.SetZero()
            auxiliary_damping.SetZero()
            for index, value in enumerate(
                (
                    50_000_000.0,
                    50_000_000.0,
                    50_000_000.0,
                    80_000.0,
                    80_000.0,
                    0.0,
                )
            ):
                auxiliary_stiffness.SetItem(index, index, value)
            for index, value in enumerate(
                (18_000.0, 18_000.0, 18_000.0, 120.0, 120.0, 1.0)
            ):
                auxiliary_damping.SetItem(index, index, value)
            auxiliary_bushing = chrono.ChLoadBodyBodyBushingGeneric(
                auxiliary_body,
                ground,
                auxiliary_frame_abs,
                auxiliary_stiffness,
                auxiliary_damping,
            )
            load_container.Add(auxiliary_bushing)
        for (
            interface_id,
            child,
            parent,
            frame_abs,
            axis_abs,
            coordinate_mode,
            bushing,
            stiffness,
            damping,
        ) in (
            (
                "support-pinion-bearing",
                pinion,
                support_parent,
                chrono.ChVector3d(*spec.pinion_origin_abs_m),
                spec.pinion_axis_abs,
                RELATIVE_ANCHOR_DRIFT,
                pinion_bushing,
                pinion_stiffness,
                pinion_damping,
            ),
            (
                "support-rack-guide",
                rack,
                support_parent,
                rack.GetPos(),
                spec.rack_axis_abs,
                TRANSVERSE_ANCHOR_DRIFT,
                rack_bushing,
                rack_stiffness,
                rack_damping,
            ),
            *(
                (
                    (
                        "support-auxiliary-revolute",
                        auxiliary_body,
                        ground,
                        chrono.ChVector3d(
                            *auxiliary_revolute["origin_abs_m"]
                        ),
                        auxiliary_revolute["axis_abs"],
                        RELATIVE_ANCHOR_DRIFT,
                        auxiliary_bushing,
                        auxiliary_stiffness,
                        auxiliary_damping,
                    ),
                )
                if auxiliary_revolute is not None
                and auxiliary_body is not None
                and auxiliary_bushing is not None
                and auxiliary_stiffness is not None
                and auxiliary_damping is not None
                else ()
            ),
        ):
            zero_stiffness = chrono.ChMatrix66d()
            zero_damping = chrono.ChMatrix66d()
            zero_stiffness.SetZero()
            zero_damping.SetZero()
            support_runtimes.append(
                {
                    "interface_id": interface_id,
                    "coordinate_mode": coordinate_mode,
                    "parent": parent,
                    "child": child,
                    "bushing": bushing,
                    "stiffness_matrix": stiffness,
                    "damping_matrix": damping,
                    "zero_stiffness_matrix": zero_stiffness,
                    "zero_damping_matrix": zero_damping,
                    "parent_anchor_local": (
                        parent.TransformPointParentToLocal(frame_abs)
                    ),
                    "child_anchor_local": (
                        child.TransformPointParentToLocal(frame_abs)
                    ),
                    "parent_axis_local": (
                        parent.TransformDirectionParentToLocal(
                            chrono.ChVector3d(*axis_abs)
                        )
                    ),
                    "child_axis_local": (
                        child.TransformDirectionParentToLocal(
                            chrono.ChVector3d(*axis_abs)
                        )
                    ),
                    "gate": BoundedSupportRuntime(
                        interface_id=interface_id,
                        gate=support_gates[interface_id],
                    ),
                }
            )
        gear_force = create_body_force(
            chrono,
            rack,
            chrono.ChVector3d(0.0, 0.0, 0.0),
            False,
            rack.GetPos(),
            False,
            declaration=standard_wrench_declaration(
                plugin_id=plugin_id,
                profile="transmission",
                wrench_kind="force",
                load_id="rack-gear-transmission-force",
                owner=plugin_id,
                target_body=spec.rack_body_id,
                source_role="pinion_gear_interface",
                target_role="task_output",
                geometry_gate="conditional_pinion_rack_geometry_gate",
                active_interval="geometry_gated_physical_interval",
            ),
        )
        gear_torque = create_body_torque(
            chrono,
            pinion,
            chrono.ChVector3d(0.0, 0.0, 0.0),
            False,
            declaration=standard_wrench_declaration(
                plugin_id=plugin_id,
                profile="transmission",
                wrench_kind="torque",
                load_id="pinion-gear-transmission-reaction",
                owner=plugin_id,
                target_body=spec.pinion_body_id,
                source_role="rack_gear_interface",
                target_role="declared_input",
                geometry_gate="conditional_pinion_rack_geometry_gate",
                active_interval="geometry_gated_physical_interval",
            ),
        )
        stop_force = create_body_force(
            chrono,
            rack,
            chrono.ChVector3d(0.0, 0.0, 0.0),
            False,
            rack.GetPos(),
            False,
            declaration=standard_wrench_declaration(
                plugin_id=plugin_id,
                profile="limit",
                wrench_kind="force",
                load_id="rack-finite-travel-stop",
                owner=plugin_id,
                target_body=spec.rack_body_id,
                source_role="finite_travel_stop",
                target_role="task_output",
                geometry_gate="rack_progress_outside_declared_travel",
                active_interval="geometry_gated_physical_interval",
            ),
        )
        add_ledgered_load(load_container, gear_force)
        add_ledgered_load(load_container, gear_torque)
        add_ledgered_load(load_container, stop_force)

    pinion_angle_rad = 0.0
    current_coupling_error_m = 0.0
    current_support_drifts_m = {"pinion_bearing": 0.0, "rack_guide": 0.0}
    current_bushing_reactions_n = {"pinion_bearing": 0.0, "rack_guide": 0.0}
    if auxiliary_revolute is not None:
        current_support_drifts_m["auxiliary_revolute"] = 0.0
        current_bushing_reactions_n["auxiliary_revolute"] = 0.0
    current_commanded_input_torque_axis_nm = 0.0
    current_input_torque_axis_nm = 0.0
    current_auxiliary_force_abs_n = (0.0, 0.0, 0.0)
    current_auxiliary_input_torque_axis_nm = 0.0
    current_auxiliary_lock_torque_axis_nm = 0.0
    current_endpoint_detent_force_axis_n = 0.0
    auxiliary_angle_rad = 0.0
    current_brake_torque_axis_nm = 0.0
    brake_last_progress_m = spec.home_progress_m
    gear_reference_offset_m = 0.0
    conditional_interface_evaluation_count = 0
    conditional_interface_state_path: list[str] = []
    max_inactive_abs_transmitted_effort_si = 0.0
    current_conditional_interface_state: dict[str, Any] | None = None
    current_support_interface_states: dict[str, dict[str, Any]] = {}
    raw_family_spec = (
        execution.adapter.family_spec
        if isinstance(execution.adapter.family_spec, Mapping)
        else {}
    )
    gear_stiffness_n_m = float(
        raw_family_spec.get("gear_penalty_stiffness_n_m", 3_000_000.0)
    )
    gear_damping_n_s_m = float(
        raw_family_spec.get("gear_penalty_damping_n_s_m", 3_000.0)
    )
    if (
        not math.isfinite(gear_stiffness_n_m)
        or gear_stiffness_n_m <= 0.0
        or not math.isfinite(gear_damping_n_s_m)
        or gear_damping_n_s_m <= 0.0
    ):
        raise MechanismValidationError(
            "RACK_PINION_GEAR_PENALTY_INVALID",
            "gear penalty stiffness and damping must be finite and positive",
        )
    stop_stiffness_n_m = 750_000.0
    stop_damping_n_s_m = 9_000.0
    endpoint_detent_capture_m = 0.003
    endpoint_detent_stiffness_n_m = 100_000.0
    endpoint_detent_damping_n_s_m = 1_000.0
    endpoint_detent_force_limit_n = 500.0
    auxiliary_closed_catch_range_rad = math.radians(5.0)
    auxiliary_closed_catch_stiffness_n_m_rad = 200.0
    auxiliary_closed_catch_damping_n_m_s_rad = 20.0
    static_brake_nm = (
        spec.nominal_input_torque_nm * RACK_PINION_STATIC_BRAKE_FRACTION
    )
    kinetic_brake_nm = (
        spec.nominal_input_torque_nm * RACK_PINION_KINETIC_BRAKE_FRACTION
    )
    # Physical-branch holding brake: an anchored rotational spring-damper on
    # the pinion, sized from the pinion's axis inertia so the explicit
    # per-step torque update stays numerically stable (omega_n * dt = 0.05).
    pinion_axis_inertia_kg_m2 = sum(
        axis * axis * inertia
        for axis, inertia in zip(
            spec.pinion_axis_abs, inertias[spec.pinion_body_id]
        )
    )
    brake_lock_omega_rad_s = (
        RACK_PINION_BRAKE_LOCK_OMEGA_DT / execution.load_case.step_s
    )
    brake_lock_stiffness_nm_rad = (
        pinion_axis_inertia_kg_m2 * brake_lock_omega_rad_s**2
    )
    brake_lock_damping_nm_s_rad = 2.0 * math.sqrt(
        brake_lock_stiffness_nm_rad * pinion_axis_inertia_kg_m2
    )
    brake_anchor_angle_rad: float | None = None
    rack_mass = next(item.mass_kg for item in spec.bodies if item.body_id == spec.rack_body_id)

    def support_rack_axis_abs() -> tuple[float, float, float]:
        return _tuple3(
            support_parent.TransformDirectionLocalToParent(
                chrono.ChVector3d(*support_rack_axis_parent_local)
            )
        )

    def support_pinion_axis_abs() -> tuple[float, float, float]:
        return _tuple3(
            support_parent.TransformDirectionLocalToParent(
                chrono.ChVector3d(*support_pinion_axis_parent_local)
            )
        )

    def body_point_velocity(
        body: Any,
        point_abs: Sequence[float],
    ) -> tuple[float, float, float]:
        return _add(
            _tuple3(body.GetPosDt()),
            _cross(
                _angular_velocity(body),
                _sub(point_abs, _tuple3(body.GetPos())),
            ),
        )

    def state() -> tuple[float, float, float, float, float, float]:
        rack_position = _tuple3(rack.GetPos())
        rack_position_parent_local = _tuple3(
            support_parent.TransformPointParentToLocal(rack.GetPos())
        )
        rack_delta_parent_local = _sub(
            rack_position_parent_local,
            initial_rack_position_parent_local,
        )
        progress = spec.home_progress_m + _dot(
            rack_delta_parent_local,
            support_rack_axis_parent_local,
        )
        rack_axis_abs_now = support_rack_axis_abs()
        rack_velocity_relative = _sub(
            _tuple3(rack.GetPosDt()),
            body_point_velocity(support_parent, rack_position),
        )
        speed = _dot(rack_velocity_relative, rack_axis_abs_now)
        pinion_speed = _dot(
            _sub(
                _angular_velocity(pinion),
                _angular_velocity(support_parent),
            ),
            support_pinion_axis_abs(),
        )
        coupling_error = (
            progress
            - spec.home_progress_m
            + spec.pinion_pitch_radius_m * pinion_angle_rad
            - gear_reference_offset_m
        )
        return progress, speed, pinion_angle_rad, pinion_speed, coupling_error, _norm(
            _sub(
                rack_delta_parent_local,
                _scale(
                    support_rack_axis_parent_local,
                    progress - spec.home_progress_m,
                ),
            )
        )

    def interface_observation() -> Any:
        pinion_center = _tuple3(
            pinion.TransformPointLocalToParent(
                chrono.ChVector3d(*pinion_center_local)
            )
        )
        pinion_axis = _tuple3(
            pinion.GetRot().Rotate(
                chrono.ChVector3d(*pinion_axis_local)
            )
        )
        rack_pitch_line = _tuple3(
            rack.TransformPointLocalToParent(
                chrono.ChVector3d(*rack_pitch_line_local)
            )
        )
        rack_axis = _tuple3(
            rack.GetRot().Rotate(
                chrono.ChVector3d(*rack_axis_local)
            )
        )
        rack_face_axis = _tuple3(
            rack.GetRot().Rotate(
                chrono.ChVector3d(*rack_face_axis_local)
            )
        )
        return rack_pinion_geometry.evaluate(
            time_s=time_s,
            pose=RackPinionPose(
                pinion_center_abs_m=pinion_center,
                pinion_axis_abs=pinion_axis,
                pinion_face_width_m=interface_face_width_m,
                rack_pitch_line_point_abs_m=rack_pitch_line,
                rack_axis_abs=rack_axis,
                rack_face_axis_abs=rack_face_axis,
                rack_face_width_m=interface_face_width_m,
                pinion_linear_velocity_abs_m_s=_tuple3(
                    pinion.GetPosDt()
                ),
                rack_linear_velocity_abs_m_s=_tuple3(
                    rack.GetPosDt()
                ),
            ),
        )

    def update_physical_supports() -> None:
        nonlocal brake_anchor_angle_rad
        nonlocal brake_last_progress_m
        nonlocal current_brake_torque_axis_nm
        nonlocal current_coupling_error_m
        nonlocal gear_reference_offset_m
        nonlocal conditional_interface_evaluation_count
        nonlocal max_inactive_abs_transmitted_effort_si
        nonlocal current_conditional_interface_state
        nonlocal current_auxiliary_lock_torque_axis_nm
        nonlocal current_endpoint_detent_force_axis_n
        (
            progress,
            rack_speed,
            pinion_angle_now,
            pinion_speed,
            coupling_error,
            lateral,
        ) = state()
        current_coupling_error_m = coupling_error
        if execution.branch == "physical":
            support_active_by_id: dict[str, bool] = {}
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
                axis_misalignment = math.acos(
                    max(
                        -1.0,
                        min(1.0, abs(_dot(parent_axis, child_axis))),
                    )
                )

                support_state = runtime["gate"].evaluate(
                    time_s=time_s,
                    anchor_a_abs_m=parent_point,
                    anchor_b_abs_m=child_point,
                    velocity_a_abs_m_s=body_point_velocity(
                        parent,
                        parent_point,
                    ),
                    velocity_b_abs_m_s=body_point_velocity(
                        child,
                        child_point,
                    ),
                    requested_effort_si=_norm(
                        _tuple3(runtime["bushing"].GetForce())
                    ),
                    free_axis_abs=(
                        parent_axis
                        if runtime["coordinate_mode"]
                        == TRANSVERSE_ANCHOR_DRIFT
                        else None
                    ),
                    axis_misalignment_rad=axis_misalignment,
                )
                # The trace audit requires an explicit N*m support-torque
                # channel on every compliant physical interface; report the
                # bushing's reaction torque alongside its force effort.
                bushing_torque_nm = _norm(
                    _tuple3(runtime["bushing"].GetTorque())
                )
                support_state["requested_support_torque_nm"] = (
                    bushing_torque_nm
                )
                support_state["transmitted_support_torque_nm"] = (
                    bushing_torque_nm
                    if support_state["transmission_active"]
                    else 0.0
                )
                current_support_interface_states[
                    runtime["interface_id"]
                ] = support_state
                active = bool(support_state["transmission_active"])
                support_active_by_id[runtime["interface_id"]] = active
                if runtime["interface_id"] == "support-auxiliary-revolute":
                    current_support_drifts_m["auxiliary_revolute"] = _norm(
                        _sub(child_point, parent_point)
                    )
                    current_bushing_reactions_n["auxiliary_revolute"] = (
                        _norm(_tuple3(runtime["bushing"].GetForce()))
                        if active
                        else 0.0
                    )
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
            observation = interface_observation()
            decision = rack_pinion_interface.step(observation)
            conditional_interface_evaluation_count += 1
            if (
                not conditional_interface_state_path
                or conditional_interface_state_path[-1] != decision.state
            ):
                conditional_interface_state_path.append(decision.state)
            if decision.capture_reference:
                gear_reference_offset_m += coupling_error
                coupling_error = 0.0
                current_coupling_error_m = 0.0
            error_speed = rack_speed + spec.pinion_pitch_radius_m * pinion_speed
            requested_force_scalar = (
                -gear_stiffness_n_m * coupling_error
                - gear_damping_n_s_m * error_speed
            )
            force_scalar = decision.transmit_scalar(
                requested_force_scalar
            )
            if not decision.transmission_active:
                max_inactive_abs_transmitted_effort_si = max(
                    max_inactive_abs_transmitted_effort_si,
                    abs(force_scalar),
                )
            current_conditional_interface_state = {
                "interface_id": "transmission-pinion-rack",
                "state": decision.state,
                "transmission_active": decision.transmission_active,
                "transmitted_effort_si": force_scalar,
                "activation_coordinate_si": (
                    observation.activation_coordinate_si
                ),
                "constraints_satisfied": (
                    observation.constraints_satisfied
                ),
            }
            torque_scalar = force_scalar * spec.pinion_pitch_radius_m
            rack_axis_abs_now = support_rack_axis_abs()
            pinion_axis_abs_now = support_pinion_axis_abs()
            gear_force.SetApplicationPoint(rack.GetPos(), False)
            gear_force.SetForce(
                chrono.ChVector3d(
                    *_scale(rack_axis_abs_now, force_scalar)
                ),
                False,
            )
            gear_torque.SetTorque(
                chrono.ChVector3d(
                    *_scale(pinion_axis_abs_now, torque_scalar)
                ),
                False,
            )
            stop_scalar = 0.0
            if progress < spec.home_progress_m:
                stop_scalar = (
                    -stop_stiffness_n_m * (progress - spec.home_progress_m)
                    - stop_damping_n_s_m * min(rack_speed, 0.0)
                )
            elif progress > spec.outbound_progress_m:
                stop_scalar = (
                    -stop_stiffness_n_m * (progress - spec.outbound_progress_m)
                    - stop_damping_n_s_m * max(rack_speed, 0.0)
                )
            stop_force.SetApplicationPoint(rack.GetPos(), False)
            stop_force.SetForce(
                chrono.ChVector3d(*_scale(rack_axis_abs_now, stop_scalar)),
                False,
            )
            endpoint_detent_scalar = 0.0
            if (
                endpoint_detent_force is not None
                and abs(current_commanded_input_torque_axis_nm) <= 1.0e-9
            ):
                endpoint_target_m: float | None = None
                if progress <= spec.home_progress_m + endpoint_detent_capture_m:
                    endpoint_target_m = spec.home_progress_m
                elif progress >= (
                    spec.outbound_progress_m - endpoint_detent_capture_m
                ):
                    endpoint_target_m = spec.outbound_progress_m
                if endpoint_target_m is not None:
                    endpoint_detent_scalar = (
                        -endpoint_detent_stiffness_n_m
                        * (progress - endpoint_target_m)
                        - endpoint_detent_damping_n_s_m * rack_speed
                    )
                    endpoint_detent_scalar = max(
                        -endpoint_detent_force_limit_n,
                        min(
                            endpoint_detent_force_limit_n,
                            endpoint_detent_scalar,
                        ),
                    )
            current_endpoint_detent_force_axis_n = endpoint_detent_scalar
            if endpoint_detent_force is not None:
                endpoint_detent_force.SetApplicationPoint(
                    rack.GetPos(), False
                )
                endpoint_detent_force.SetForce(
                    chrono.ChVector3d(
                        *_scale(rack_axis_abs_now, endpoint_detent_scalar)
                    ),
                    False,
                )
            current_support_drifts_m["pinion_bearing"] = float(
                current_support_interface_states[
                    "support-pinion-bearing"
                ].get("support_drift_m", 0.0)
            )
            current_support_drifts_m["rack_guide"] = float(
                current_support_interface_states[
                    "support-rack-guide"
                ].get("support_drift_m", lateral)
            )
            current_bushing_reactions_n["pinion_bearing"] = _norm(
                _tuple3(pinion_bushing.GetForce())
            ) if support_active_by_id["support-pinion-bearing"] else 0.0
            current_bushing_reactions_n["rack_guide"] = _norm(
                _tuple3(rack_bushing.GetForce())
            ) if support_active_by_id["support-rack-guide"] else 0.0
            if (
                auxiliary_revolute is not None
                and auxiliary_body is not None
                and auxiliary_bushing is not None
                and auxiliary_lock_torque is not None
            ):
                auxiliary_speed = _dot(
                    _angular_velocity(auxiliary_body),
                    auxiliary_revolute["axis_abs"],
                )
                lock_torque_axis = 0.0
                if (
                    abs(current_auxiliary_input_torque_axis_nm) <= 1.0e-9
                    and 0.0 <= auxiliary_angle_rad
                    <= auxiliary_closed_catch_range_rad
                ):
                    lock_torque_axis += (
                        -auxiliary_closed_catch_stiffness_n_m_rad
                        * auxiliary_angle_rad
                        - auxiliary_closed_catch_damping_n_m_s_rad
                        * auxiliary_speed
                    )
                if progress >= auxiliary_revolute[
                    "lock_progress_threshold_m"
                ]:
                    lock_torque_axis = (
                        -auxiliary_revolute["lock_stiffness_n_m_rad"]
                        * auxiliary_angle_rad
                        - auxiliary_revolute["lock_damping_n_m_s_rad"]
                        * auxiliary_speed
                    )
                if auxiliary_angle_rad < 0.0:
                    lock_torque_axis += (
                        -auxiliary_revolute["limit_stiffness_n_m_rad"]
                        * auxiliary_angle_rad
                        - auxiliary_revolute["limit_damping_n_m_s_rad"]
                        * min(auxiliary_speed, 0.0)
                    )
                elif auxiliary_angle_rad > auxiliary_revolute[
                    "maximum_angle_rad"
                ]:
                    lock_torque_axis += (
                        -auxiliary_revolute["limit_stiffness_n_m_rad"]
                        * (
                            auxiliary_angle_rad
                            - auxiliary_revolute["maximum_angle_rad"]
                        )
                        - auxiliary_revolute["limit_damping_n_m_s_rad"]
                        * max(auxiliary_speed, 0.0)
                    )
                current_auxiliary_lock_torque_axis_nm = lock_torque_axis
                auxiliary_lock_torque.SetTorque(
                    chrono.ChVector3d(
                        *_scale(
                            auxiliary_revolute["axis_abs"],
                            lock_torque_axis,
                        )
                    ),
                    False,
                )
        else:
            current_support_drifts_m["pinion_bearing"] = 0.0
            current_support_drifts_m["rack_guide"] = lateral
            if auxiliary_revolute is not None:
                current_support_drifts_m["auxiliary_revolute"] = 0.0

        # x = -r*theta, so gravity generalized torque is -F_g*r.
        gravity_force_axis_n = rack_mass * _dot(
            execution.plan.gravity_abs_m_s2,
            support_rack_axis_abs(),
        )
        net_without_brake = current_input_torque_axis_nm - (
            gravity_force_axis_n * spec.pinion_pitch_radius_m
        )
        if abs(current_commanded_input_torque_axis_nm) <= 1.0e-9:
            if execution.branch == "physical":
                # A normally-engaged holding brake is resistive: it locks the
                # pinion at the angle where the crank was released and never
                # injects energy.  The earlier feed-forward counter-torque
                # (-gravity generalized torque) assumed the mesh was always
                # taut; with real backlash, or with the gate resting on its
                # hard stops, that torque spins the unloaded pinion until the
                # teeth slam and eject both bodies from their supports.
                if brake_anchor_angle_rad is None:
                    brake_anchor_angle_rad = pinion_angle_now
                requested_brake_nm = (
                    -brake_lock_stiffness_nm_rad
                    * (pinion_angle_now - brake_anchor_angle_rad)
                    - brake_lock_damping_nm_s_rad * pinion_speed
                )
                brake_scalar = max(
                    -static_brake_nm,
                    min(static_brake_nm, requested_brake_nm),
                )
                if abs(requested_brake_nm) > static_brake_nm:
                    # Kinetic slip: the shoe slides so the lock spring stays
                    # at its static capacity instead of winding up.
                    brake_anchor_angle_rad = (
                        pinion_angle_now
                        + brake_scalar / brake_lock_stiffness_nm_rad
                    )
            else:
                # The reduced branch couples pinion and rack through an ideal
                # taut constraint, so the gravity feed-forward estimate is
                # exact there.  Constraint impulses can make instantaneous
                # body angular velocity chatter, so estimate pinion speed
                # from the trusted rack displacement through x = -r*theta.
                finite_rack_speed_m_s = (
                    (progress - brake_last_progress_m)
                    / execution.load_case.step_s
                )
                finite_pinion_speed_rad_s = (
                    -finite_rack_speed_m_s / spec.pinion_pitch_radius_m
                )
                requested_brake_nm = -net_without_brake - (
                    RACK_PINION_BRAKE_DAMPING_NM_S_RAD
                    * finite_pinion_speed_rad_s
                )
                brake_scalar = max(
                    -static_brake_nm,
                    min(static_brake_nm, requested_brake_nm),
                )
        else:
            brake_anchor_angle_rad = None
            tendency = pinion_speed if abs(pinion_speed) >= 0.02 else net_without_brake
            brake_scalar = -math.copysign(kinetic_brake_nm, tendency) if tendency else 0.0
        current_brake_torque_axis_nm = brake_scalar
        brake_last_progress_m = progress
        if execution.branch == "physical":
            brake_torque.SetTorque(
                chrono.ChVector3d(
                    *_scale(support_pinion_axis_abs(), brake_scalar)
                ),
                False,
            )
        else:
            # The ideal rack-pinion mate owns the reduced branch kinematics.
            # Apply the brake's virtual-work-equivalent force at the rack so
            # its holding capacity is not lost in the ideal constraint solve.
            brake_torque.SetTorque(chrono.ChVector3d(0.0, 0.0, 0.0), False)
            reduced_brake_force.SetApplicationPoint(rack.GetPos(), False)
            reduced_brake_force.SetForce(
                chrono.ChVector3d(
                    *_scale(
                        spec.rack_axis_abs,
                        -brake_scalar / spec.pinion_pitch_radius_m,
                    )
                ),
                False,
            )

    samples: list[dict[str, Any]] = []
    executed_phases: list[dict[str, Any]] = []
    time_s = 0.0
    next_sample_s = 0.0

    def record(phase_id: str, applied_torque_abs: tuple[float, float, float]) -> None:
        progress, speed, angle, pinion_speed, coupling_error, lateral = state()
        initial_rack = initial_poses[spec.rack_body_id]["position_abs_m"]
        rack_position = _tuple3(rack.GetPos())
        rack_rotation = _pose(rack)["rotation_abs_quat_wxyz"]
        rotation_dot = abs(
            sum(a * b for a, b in zip(
                rack_rotation,
                initial_poses[spec.rack_body_id]["rotation_abs_quat_wxyz"],
            ))
        )
        off_axis_rotation = 2.0 * math.acos(max(-1.0, min(1.0, rotation_dot)))
        sample = {
            "time_s": time_s,
            "phase_id": phase_id,
            "applied_torque_abs_nm": list(applied_torque_abs),
            "applied_force_abs_n": list(current_auxiliary_force_abs_n),
            "auxiliary_input_torque_axis_nm": (
                current_auxiliary_input_torque_axis_nm
            ),
            "auxiliary_lock_torque_axis_nm": (
                current_auxiliary_lock_torque_axis_nm
            ),
            "endpoint_detent_force_axis_n": (
                current_endpoint_detent_force_axis_n
            ),
            "door_rotation_rad": (
                auxiliary_angle_rad
                if auxiliary_revolute is not None
                else 0.0
            ),
            "door_speed_rad_s": (
                _dot(
                    _angular_velocity(auxiliary_body),
                    auxiliary_revolute["axis_abs"],
                )
                if auxiliary_revolute is not None
                and auxiliary_body is not None
                else 0.0
            ),
            "moving_pose": _pose(rack),
            "body_poses": {
                body_id: _pose(body)
                for body_id, body in bodies.items()
            },
            "observable_point_abs_m": list(rack_position),
            "observable_displacement_abs_m": list(
                _sub(rack_position, initial_rack)
            ),
            "generalized_progress_m": progress,
            "generalized_speed_m_s": speed,
            "axis_speed_m_s": speed,
            "lateral_shift_m": lateral,
            "rotation_from_start_rad": off_axis_rotation,
            "pinion_angle_rad": angle,
            "pinion_speed_rad_s": pinion_speed,
            "brake_torque_axis_nm": current_brake_torque_axis_nm,
            "rack_progress_m": progress,
            "rack_speed_m_s": speed,
            "coupling_error_m": coupling_error,
            "support_drifts_m": dict(current_support_drifts_m),
            "max_support_drift_m": max(
                current_support_drifts_m.values()
            ),
            "bushing_reaction_forces_n": dict(
                current_bushing_reactions_n
            ),
            "max_bushing_reaction_force_n": max(
                current_bushing_reactions_n.values()
            ),
            "off_axis_rotation_rad": off_axis_rotation,
            "contact_count": _contact_count(system),
            "contact_pair_counts": _contact_pair_counts(
                chrono, system
            ),
        }
        if execution.branch == "physical":
            if current_conditional_interface_state is None:
                raise RuntimeError(
                    "rack-pinion conditional interface was not evaluated"
                )
            sample["conditional_interface_states"] = [
                *(
                    dict(current_support_interface_states[interface_id])
                    for interface_id in sorted(
                        current_support_interface_states
                    )
                ),
                dict(current_conditional_interface_state),
            ]
        samples.append(sample)
        if os.environ.get("SR_CAD_CONTACT_DEBUG"):
            _debug_dump_contacts(chrono, system, workspace, time_s)
        _write_trace_checkpoint(
            execution,
            samples,
            executed_phases=executed_phases,
        )

    update_physical_supports()
    record("initial", (0.0, 0.0, 0.0))
    next_sample_s = execution.load_case.sample_interval_s
    first_dynamics_step = True
    dynamics_step_count = 0
    for phase_index, phase in enumerate(execution.load_case.phases):
        transition_current_wrench_phase(phase.phase_id)
        if phase.wrench_kind == "force" and auxiliary_revolute is None:
            raise MechanismValidationError(
                "RACK_PINION_FORCE_LOAD_UNSUPPORTED",
                "rack-pinion backend accepts crank torque, not direct rack force",
            )
        direction = phase.direction_abs or (0.0, 0.0, 0.0)
        commanded_torque_abs = (
            tuple(value * phase.magnitude_si for value in direction)
            if phase.wrench_kind == "torque"
            else (0.0, 0.0, 0.0)
        )
        commanded_force_abs = (
            tuple(value * phase.magnitude_si for value in direction)
            if phase.wrench_kind == "force"
            else (0.0, 0.0, 0.0)
        )
        auxiliary_torque_component = (
            _dot(commanded_torque_abs, auxiliary_revolute["axis_abs"])
            if auxiliary_revolute is not None
            else 0.0
        )
        pinion_torque_component = _dot(
            commanded_torque_abs, spec.pinion_axis_abs
        )
        auxiliary_torque_phase = bool(
            auxiliary_revolute is not None
            and abs(auxiliary_torque_component)
            > abs(pinion_torque_component) + 1.0e-12
        )
        current_commanded_input_torque_axis_nm = _dot(
            commanded_torque_abs, spec.pinion_axis_abs
        ) if not auxiliary_torque_phase else 0.0
        commanded_auxiliary_input_torque_axis_nm = (
            auxiliary_torque_component if auxiliary_torque_phase else 0.0
        )
        current_auxiliary_input_torque_axis_nm = (
            0.0
            if auxiliary_torque_phase
            else commanded_auxiliary_input_torque_axis_nm
        )
        current_auxiliary_force_abs_n = commanded_force_abs
        applied_torque_abs = commanded_torque_abs
        phase_limit_s = (
            reference_schedule[phase_index]["duration_s"]
            if reference_schedule is not None
            else phase.duration_s
        )
        elapsed = 0.0
        stop_reached = False
        while elapsed < phase_limit_s - 1e-12:
            if auxiliary_torque_phase:
                current_auxiliary_input_torque_axis_nm = (
                    commanded_auxiliary_input_torque_axis_nm
                    * min(
                        1.0,
                        elapsed / DEADBOLT_AUXILIARY_TORQUE_RAMP_S,
                    )
                )
            pinion_speed_for_operator = state()[3]
            current_input_torque_axis_nm = _bounded_operator_torque_axis_nm(
                current_commanded_input_torque_axis_nm,
                pinion_speed_for_operator,
                spec.nominal_input_torque_nm,
                (
                    0.05
                    if auxiliary_revolute is not None
                    else RACK_PINION_OPERATOR_DAMPING_NM_S_RAD
                ),
            )
            applied_torque_abs = _scale(
                spec.pinion_axis_abs, current_input_torque_axis_nm
            )
            input_torque.SetTorque(chrono.ChVector3d(*applied_torque_abs), False)
            if (
                auxiliary_input_force is not None
                and auxiliary_body is not None
                and auxiliary_input_point_local is not None
            ):
                auxiliary_input_force.SetApplicationPoint(
                    auxiliary_body.TransformPointLocalToParent(
                        auxiliary_input_point_local
                    ),
                    False,
                )
                auxiliary_input_force.SetForce(
                    chrono.ChVector3d(*current_auxiliary_force_abs_n),
                    False,
                )
            if (
                auxiliary_input_torque is not None
                and auxiliary_revolute is not None
            ):
                auxiliary_input_torque.SetTorque(
                    chrono.ChVector3d(
                        *_scale(
                            auxiliary_revolute["axis_abs"],
                            current_auxiliary_input_torque_axis_nm,
                        )
                    ),
                    False,
                )
            update_physical_supports()
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
            pinion_angle_rad += _dot(
                _sub(
                    _angular_velocity(pinion),
                    _angular_velocity(support_parent),
                ),
                support_pinion_axis_abs(),
            ) * step
            if auxiliary_revolute is not None and auxiliary_body is not None:
                auxiliary_angle_rad += _dot(
                    _angular_velocity(auxiliary_body),
                    auxiliary_revolute["axis_abs"],
                ) * step
            if time_s + 1e-12 >= next_sample_s:
                record(phase.phase_id, applied_torque_abs)
                while next_sample_s <= time_s + 1e-12:
                    next_sample_s += execution.load_case.sample_interval_s
            if execution.branch == "physical" and phase.stop_condition is not None:
                progress = state()[0]
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
            record(phase.phase_id, applied_torque_abs)
        input_torque.SetTorque(
            chrono.ChVector3d(0.0, 0.0, 0.0),
            False,
        )
        if auxiliary_input_force is not None:
            auxiliary_input_force.SetForce(
                chrono.ChVector3d(0.0, 0.0, 0.0), False
            )
            current_auxiliary_force_abs_n = (0.0, 0.0, 0.0)
        if auxiliary_input_torque is not None:
            auxiliary_input_torque.SetTorque(
                chrono.ChVector3d(0.0, 0.0, 0.0), False
            )
            current_auxiliary_input_torque_axis_nm = 0.0
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
                executed_phase["termination"] = (
                    registered_adaptive_termination_evidence(
                        phase,
                        actual_coordinate_si=state()[0],
                        reached=stop_reached,
                    )
                )
        executed_phases.append(executed_phase)

    input_torque.SetTorque(chrono.ChVector3d(0.0, 0.0, 0.0), False)
    if auxiliary_input_force is not None:
        auxiliary_input_force.SetForce(
            chrono.ChVector3d(0.0, 0.0, 0.0), False
        )
    if auxiliary_input_torque is not None:
        auxiliary_input_torque.SetTorque(
            chrono.ChVector3d(0.0, 0.0, 0.0), False
        )
    set_current_wrench_phase(None)
    total_mass = sum(item.mass_kg for item in spec.bodies if not item.fixed)
    chrono_version = str(chrono.GetChronoVersion()) if hasattr(chrono, "GetChronoVersion") else None
    conditional_interface_runtime = conditional_runtime_summary(
        [runtime["gate"] for runtime in support_runtimes],
        physics_step_count=dynamics_step_count,
        additional_interface_summaries=[
            {
                "interface_id": "transmission-pinion-rack",
                "evaluation_count": (
                    conditional_interface_evaluation_count
                ),
                "max_inactive_abs_transmitted_effort_si": (
                    max_inactive_abs_transmitted_effort_si
                ),
                "state_path": list(
                    conditional_interface_state_path
                ),
            }
        ],
    ) if execution.branch == "physical" else None
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
            "solver": "ChSolverMINRES" if execution.branch == "physical" else "ChSolverPSOR",
            "mechanism_plugin_id": execution.mechanism_plugin.plugin_id,
            "worker_backend": execution.mechanism_plugin.worker_backend,
            "step_s": execution.load_case.step_s,
            "gravity_abs_m_s2": list(execution.plan.gravity_abs_m_s2),
            "chrono_link_count": link_count,
            "ideal_links": ideal_links,
            "internal_ideal_joint_count": len(ideal_links),
            "ideal_limits": ideal_limit_disclosure,
            "rack_pinion_coupling": (
                "finite_penalty_gear_mesh" if execution.branch == "physical"
                else "ChLinkMateRackPinion"
            ),
        },
        "branch_semantics": {
            "same_mass_kg": total_mass,
            "same_inertia_xx_kg_m2": inertias,
            "same_initial_pose": initial_poses,
            "same_external_load_sha256": canonical_sha256(executed_phases),
            "replaced_contacts_disabled": execution.branch == "reduced",
            # Generic indirect-actuation role declaration consumed by the
            # requirement-obligation layer (validator.actuation.indirect_only
            # and fixed_chassis_and_moving_groups): the crank torque is applied
            # to the pinion only and the rack is never driven directly.
            "direct_output_actuation_used": False,
            "task_input_body_id": spec.pinion_body_id,
            "unique_task_output_body_id": spec.rack_body_id,
            "runtime_bindings": {
                "fixed_body_id": spec.fixed_body_id,
                "input_body_id": spec.pinion_body_id,
                "output_body_id": spec.rack_body_id,
            },
            "support_mode": (
                "finite_bushings_and_finite_gear_mesh"
                if execution.branch == "physical"
                else "ideal_revolute_prismatic_rack_pinion"
            ),
            "physical_internal_ideal_joints_forbidden": True,
            "family_spec_sha256": canonical_sha256(spec.to_dict()),
            "collision_geometry_sha256": canonical_sha256(collision_manifest),
            "physical_interface_policy": collision_policy_disclosure(
                physical_interfaces, branch=execution.branch
            ),
            "functional_interface_geometry_binding": (
                functional_interface_geometry_binding
            ),
            "conditional_interface_runtime": (
                conditional_interface_runtime
                if execution.branch == "physical"
                else {
                    "status": "NOT_REQUIRED",
                    "conditional_interface_count": 0,
                }
            ),
            "pinion_rack_exact_collision_remains_enabled": execution.branch == "physical",
            "holding_brake_model": {
                "static_limit_nm": static_brake_nm,
                "kinetic_nm": kinetic_brake_nm,
                "engaged_damping_nm_s_rad": RACK_PINION_BRAKE_DAMPING_NM_S_RAD,
                "release_condition": "manual_crank_torque_applied",
                "operator_speed_damping_nm_s_rad": (
                    0.05
                    if auxiliary_revolute is not None
                    else RACK_PINION_OPERATOR_DAMPING_NM_S_RAD
                ),
                "reduced_representation": "virtual_work_equivalent_rack_force",
                "source": "validator_normally_engaged_manual_gate_brake_v2",
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
                physical_interfaces, branch=execution.branch
            ),
            "collision_filter_policy": (
                "all_inter_body_collision_pairs_enabled"
                if execution.branch == "physical"
                else "all_collision_disabled_for_reduced_branch"
            ),
            "whole_body_box_proxy_used": False,
            "whole_body_convex_hull_used": False,
            "convex_hull_used": False,
            "joint_count": 3,
            "loop_count": 1,
            "pinion_pitch_radius_m": spec.pinion_pitch_radius_m,
            "progress_limits_m": [spec.home_progress_m, spec.outbound_progress_m],
            "friction_assumption": spec.friction,
        },
        "axis_abs": list(spec.rack_axis_abs),
        "joint_origin_abs_m": list(spec.pinion_origin_abs_m),
        "input_point_abs_m": list(spec.pinion_origin_abs_m),
        "observable_point_initial_abs_m": list(initial_rack_position),
        "characteristic_length_m": execution.plan.characteristic_length_m,
        "executed_phases": executed_phases,
        "executed_load_sha256": canonical_sha256(executed_phases),
        "sample_count": len(samples),
        "samples": samples,
        "strict_non_claims": {
            "manufacturing_certification": False,
            "strength_or_fatigue": False,
            "measured_real_friction_or_brake_torque": False,
            "dynamic_response_certified": False,
            "material_density_inertia_certified": False,
        },
    }
