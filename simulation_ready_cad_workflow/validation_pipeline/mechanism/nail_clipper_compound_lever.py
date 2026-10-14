"""Built-in adapter for a handheld nail-clipper compound lever.

This adapter owns three bodies, two revolutes, a restoring torsion
spring, and a parsed point-plane coupling that transmits lever torque
into jaw motion. Task torque is applied only to ``actuating_lever``.
Direct jaw actuation is forbidden. ``simulate`` returns measured
samples and never a PASS verdict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .compound_spec import CompoundBodySpec
from .family_plugins import (
    CompliantSupportSpec,
    GeneralizedCoordinateSpec,
    MechanismAdapter,
)
from .io import canonical_sha256, lane_execution_source_sha256
from .joint_realization import RevoluteProofRequirement
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    LoadCase,
    LoadPhase,
    MechanismPlan,
    MechanismValidationError,
    finite,
    identifier,
    vector3,
)

if TYPE_CHECKING:
    from simulation_ready_cad_workflow.validation_pipeline.mechanism.adapters import AdapterSpec
    from simulation_ready_cad_workflow.validation_pipeline.mechanism.worker import ExecutionRequest


_REQUIRED_BODY_IDS = (
    "stationary_cutting_jaw",
    "actuating_lever",
    "moving_cutting_jaw",
)
_INPUT_BODY_ID = "actuating_lever"
_OUTPUT_BODY_ID = "moving_cutting_jaw"
_FIXED_BODY_ID = "stationary_cutting_jaw"
_JAW_JOINT_ID = "jaw_revolute"
_LEVER_JOINT_ID = "lever_revolute"


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MechanismValidationError(
            "FAMILY_SPEC_INVALID",
            f"{label} must be an object",
        )
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise MechanismValidationError(
            "FAMILY_SPEC_INVALID",
            f"{label} must be a non-empty array",
        )
    return value


def _joint_by_child(
    joints: Sequence[ClipperRevoluteSpec],
    child_body_id: str,
    label: str,
) -> ClipperRevoluteSpec:
    matches = [item for item in joints if item.child_body_id == child_body_id]
    if len(matches) != 1:
        raise MechanismValidationError(
            "FAMILY_SPEC_INVALID",
            f"{label} must identify exactly one revolute by child_body_id",
        )
    return matches[0]


def _coupling_partner(spec: NailClipperCompoundLeverSpec) -> str:
    jaw_joint = _joint_by_child(
        spec.revolute_joints, spec.output_body_id, "jaw revolute"
    )
    lever_joint = _joint_by_child(
        spec.revolute_joints, spec.input_body_id, "lever revolute"
    )
    if (
        jaw_joint.parent_body_id == _FIXED_BODY_ID
        and lever_joint.parent_body_id == spec.output_body_id
    ):
        # Lever is hinged on the jaw; the handle contact is against the
        # stationary body and closes the jaw.
        return _FIXED_BODY_ID
    if (
        jaw_joint.parent_body_id == _FIXED_BODY_ID
        and lever_joint.parent_body_id == _FIXED_BODY_ID
    ):
        # Both members pivot on the stationary body; coupling is the
        # lever-jaw contact.
        return spec.output_body_id
    raise MechanismValidationError(
        "FAMILY_SPEC_INVALID",
        "revolute topology must be jaw-on-stationary plus lever-on-jaw "
        "or both members on stationary",
    )


@dataclass(frozen=True)
class ClipperBodySpec:
    body_id: str
    group_ids: tuple[str, ...]
    fixed: bool
    mass_kg: float
    center_abs_m: tuple[float, float, float]
    size_m: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        groups = list(self.group_ids)
        return {
            "body_id": self.body_id,
            "group_ids": groups,
            "fixed": self.fixed,
            "mass_kg": self.mass_kg,
            "center_abs_m": list(self.center_abs_m),
            "size_m": list(self.size_m),
        }


@dataclass(frozen=True)
class ClipperRevoluteSpec:
    joint_id: str
    parent_body_id: str
    child_body_id: str
    origin_abs_m: tuple[float, float, float]
    axis_abs: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint_id": self.joint_id,
            "parent_body_id": self.parent_body_id,
            "child_body_id": self.child_body_id,
            "origin_abs_m": list(self.origin_abs_m),
            "axis_abs": list(self.axis_abs),
        }


@dataclass(frozen=True)
class ClipperSpringSpec:
    body_a_id: str
    body_b_id: str
    origin_abs_m: tuple[float, float, float]
    axis_abs: tuple[float, float, float]
    stiffness_n_m_rad: float
    damping_n_m_s_rad: float
    rest_angle_rad: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "body_a_id": self.body_a_id,
            "body_b_id": self.body_b_id,
            "origin_abs_m": list(self.origin_abs_m),
            "axis_abs": list(self.axis_abs),
            "stiffness_n_m_rad": self.stiffness_n_m_rad,
            "damping_n_m_s_rad": self.damping_n_m_s_rad,
            "rest_angle_rad": self.rest_angle_rad,
        }


@dataclass(frozen=True)
class NailClipperCompoundLeverSpec:
    mechanism_kind: str
    bodies: tuple[ClipperBodySpec, ...]
    revolute_joints: tuple[ClipperRevoluteSpec, ...]
    restoring_spring: ClipperSpringSpec
    input_body_id: str
    output_body_id: str
    contact_point_abs_m: tuple[float, float, float]
    cutting_edge_radius_m: float
    transmission_ratio: float
    coupling_partner_body_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mechanism_kind": self.mechanism_kind,
            "bodies": [item.to_dict() for item in self.bodies],
            "revolute_joints": [item.to_dict() for item in self.revolute_joints],
            "restoring_spring": self.restoring_spring.to_dict(),
            "input_body_id": self.input_body_id,
            "output_body_id": self.output_body_id,
            "contact_point_abs_m": list(self.contact_point_abs_m),
            "cutting_edge_radius_m": self.cutting_edge_radius_m,
            "transmission_ratio": self.transmission_ratio,
            "coupling_partner_body_id": self.coupling_partner_body_id,
        }


def _compound_bodies(
    spec: NailClipperCompoundLeverSpec,
) -> tuple[CompoundBodySpec, ...]:
    return tuple(
        CompoundBodySpec(
            body_id=item.body_id,
            group_ids=item.group_ids,
            fixed=item.fixed,
            mass_kg=item.mass_kg,
            center_abs_m=item.center_abs_m,
            size_m=item.size_m,
        )
        for item in spec.bodies
    )


def _inertia_box(
    mass_kg: float,
    size_m: Sequence[float],
) -> tuple[float, float, float]:
    width, depth, height = (float(item) for item in size_m)
    return (
        mass_kg * (depth * depth + height * height) / 12.0,
        mass_kg * (width * width + height * height) / 12.0,
        mass_kg * (width * width + depth * depth) / 12.0,
    )


def _transition_task_input_phase(
    input_torque: Any,
    chrono: Any,
    phase_id: str,
    *,
    prior_phase_open: bool,
    transition: Any,
) -> None:
    if prior_phase_open:
        input_torque.SetTorque(chrono.ChVector3d(0.0, 0.0, 0.0), False)
    transition(phase_id)


def _completed_trace_payload(
    execution: Any,
    samples: Sequence[dict[str, Any]],
    executed_phases: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    schedule = [dict(item) for item in executed_phases]
    return {
        "schema_version": "0.1",
        "status": "COMPLETED",
        "validator_source_sha256": lane_execution_source_sha256(),
        "branch": execution.artifact_branch,
        "adapter_id": execution.adapter.adapter_id,
        "mechanism_plugin_id": execution.mechanism_plugin.plugin_id,
        "plugin_id": execution.mechanism_plugin.plugin_id,
        "package_binding_sha256": execution.plan.package_binding_sha256,
        "capsule_id": execution.plan.capsule_id,
        "load_case_id": execution.load_case.load_case_id,
        "plan_sha256": canonical_sha256(execution.plan.to_dict()),
        "load_case_sha256": canonical_sha256(execution.load_case.to_dict()),
        "executed_phases": schedule,
        "executed_load_sha256": canonical_sha256(schedule),
        "samples": list(samples),
    }


def _require_fixed_duration_phases(load_case: LoadCase) -> None:
    adaptive = [
        phase.phase_id
        for phase in load_case.phases
        if phase.stop_condition is not None
    ]
    if adaptive:
        raise MechanismValidationError(
            "NAIL_CLIPPER_ADAPTIVE_STOP_UNSUPPORTED",
            "nail-clipper runtime has no adaptive termination evidence for: "
            + ", ".join(adaptive),
        )


def _build_nail_clipper_topology(
    chrono: Any,
    execution: Any,
    workspace: Path,
    spec: NailClipperCompoundLeverSpec,
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    from .compound_runtime import (
        _compound_physical_solver,
        compound_bushing_parameters,
    )
    from .family_collision import make_family_body

    system = chrono.ChSystemNSC()
    if execution.branch == "physical":
        system.SetSolver(_compound_physical_solver(chrono))
    system.SetGravitationalAcceleration(chrono.ChVector3d(0.0, 0.0, -9.81))
    material = chrono.ChContactMaterialNSC()
    material.SetFriction(0.2)
    expected_bodies = _compound_bodies(spec)
    bodies: dict[str, Any] = {}
    inertias: dict[str, list[float]] = {}
    initial_poses: dict[str, dict[str, Any]] = {}
    for index, body_spec in enumerate(expected_bodies):
        inertia = _inertia_box(body_spec.mass_kg, body_spec.size_m)
        body, _audit = make_family_body(
            chrono,
            body_spec=body_spec,
            geometry=geometry,
            workspace=workspace,
            material=material,
            inertia_xx=inertia,
            branch=execution.branch,
        )
        if execution.branch == "physical":
            body.GetCollisionModel().SetFamily(index + 1)
        system.Add(body)
        bodies[body_spec.body_id] = body
        inertias[body_spec.body_id] = list(inertia)
        initial_poses[body_spec.body_id] = _pose(body)

    load_container = chrono.ChLoadContainer()
    system.Add(load_container)
    ideal_links: list[str] = []
    compound_bushings: list[dict[str, Any]] = []
    if execution.branch == "physical":
        parameters = compound_bushing_parameters()
        for joint in spec.revolute_joints:
            stiffness = chrono.ChMatrix66d()
            damping = chrono.ChMatrix66d()
            stiffness.SetZero()
            damping.SetZero()
            stiffness_values = (
                parameters["translational_stiffness_n_m"],
                parameters["translational_stiffness_n_m"],
                parameters["translational_stiffness_n_m"],
                parameters["axis_stiffness_n_m_rad"],
                parameters["axis_stiffness_n_m_rad"],
                0.0,
            )
            damping_values = (
                parameters["translational_damping_n_s_m"],
                parameters["translational_damping_n_s_m"],
                parameters["translational_damping_n_s_m"],
                parameters["axis_damping_n_m_s_rad"],
                parameters["axis_damping_n_m_s_rad"],
                parameters["free_axis_damping_n_m_s_rad"],
            )
            for matrix_index, value in enumerate(stiffness_values):
                stiffness.SetItem(matrix_index, matrix_index, value)
            for matrix_index, value in enumerate(damping_values):
                damping.SetItem(matrix_index, matrix_index, value)
            bushing = chrono.ChLoadBodyBodyBushingGeneric(
                bodies[joint.child_body_id],
                bodies[joint.parent_body_id],
                chrono.ChFramed(
                    chrono.ChVector3d(*joint.origin_abs_m),
                    _quat_from_z_to_axis(chrono, joint.axis_abs),
                ),
                stiffness,
                damping,
            )
            load_container.Add(bushing)
            compound_bushings.append(
                {
                    "joint_id": joint.joint_id,
                    "chrono_type": "ChLoadBodyBodyBushingGeneric",
                    **parameters,
                    "ideal_constraint": False,
                }
            )
    else:
        for joint in spec.revolute_joints:
            link = chrono.ChLinkLockRevolute()
            link.Initialize(
                bodies[joint.child_body_id],
                bodies[joint.parent_body_id],
                chrono.ChFramed(
                    chrono.ChVector3d(*joint.origin_abs_m),
                    _quat_from_z_to_axis(chrono, joint.axis_abs),
                ),
            )
            system.AddLink(link)
            ideal_links.append("ChLinkLockRevolute")
        lever_joint = _joint_by_child(
            spec.revolute_joints, spec.input_body_id, "lever revolute"
        )
        plane_normal = _in_plane_normal(
            lever_joint.axis_abs,
            lever_joint.origin_abs_m,
            spec.contact_point_abs_m,
        )
        coupler = chrono.ChLinkLockPointPlane()
        coupler.Initialize(
            bodies[spec.input_body_id],
            bodies[spec.coupling_partner_body_id],
            chrono.ChFramed(
                chrono.ChVector3d(*spec.contact_point_abs_m),
                _quat_from_z_to_axis(chrono, plane_normal),
            ),
        )
        system.AddLink(coupler)
        ideal_links.append("ChLinkLockPointPlane")

    link_count = int(system.GetNumLinks()) if hasattr(system, "GetNumLinks") else None
    if execution.branch == "physical" and link_count not in {0, None}:
        raise MechanismValidationError(
            "NAIL_CLIPPER_PHYSICAL_IDEAL_LINK_FORBIDDEN",
            "physical nail-clipper topology contains an ideal link",
        )
    return {
        "system": system,
        "load_container": load_container,
        "bodies": bodies,
        "inertias": inertias,
        "initial_poses": initial_poses,
        "ideal_links": ideal_links,
        "compound_bushings": compound_bushings,
        "link_count": link_count,
        "total_moving_mass_kg": sum(
            item.mass_kg for item in expected_bodies if not item.fixed
        ),
    }


def _completed_nail_clipper_trace(
    execution: Any,
    spec: NailClipperCompoundLeverSpec,
    samples: Sequence[dict[str, Any]],
    executed_phases: Sequence[dict[str, Any]],
    topology: Mapping[str, Any],
    constitutive_supports: Sequence[Mapping[str, Any]],
    *,
    chrono_version: str | None,
) -> dict[str, Any]:
    payload = _completed_trace_payload(
        execution,
        samples,
        executed_phases,
    )
    payload["engine"] = {
        "name": "PyChrono",
        "version": chrono_version,
        "system": "ChSystemNSC",
        "mechanism_plugin_id": execution.mechanism_plugin.plugin_id,
        "worker_backend": execution.mechanism_plugin.worker_backend,
        "collision_backend": "BULLET",
        "step_s": execution.load_case.step_s,
        "chrono_link_count": topology["link_count"],
        "ideal_links": list(topology["ideal_links"]),
        "internal_ideal_joint_count": len(topology["ideal_links"]),
        "compound_bushings": list(topology["compound_bushings"]),
        "compliant_supports": [],
        "constitutive_supports": [dict(item) for item in constitutive_supports],
    }
    payload["branch_semantics"] = {
        "same_mass_kg": topology["total_moving_mass_kg"],
        "same_inertia_xx_kg_m2": dict(topology["inertias"]),
        "same_initial_pose": dict(topology["initial_poses"]),
        "same_external_load_sha256": payload["executed_load_sha256"],
        "replaced_contacts_disabled": execution.branch != "physical",
        "support_mode": (
            "finite_revolute_bushings_and_native_step_contact"
            if execution.branch == "physical"
            else "ideal_revolute_and_point_plane_links"
        ),
        "physical_internal_ideal_joints_forbidden": True,
        "runtime_body_roles": {
            "fixed_body_ids": [
                item.body_id for item in spec.bodies if item.fixed
            ],
            "input_body_id": spec.input_body_id,
            "output_body_id": spec.output_body_id,
        },
        "runtime_bindings": {
            "fixed_body_id": next(
                item.body_id for item in spec.bodies if item.fixed
            ),
            "input_body_id": spec.input_body_id,
            "output_body_id": spec.output_body_id,
        },
        "task_input_body_id": spec.input_body_id,
        "unique_task_output_body_id": spec.output_body_id,
        "direct_output_actuation": False,
        "direct_output_actuation_used": False,
    }
    return payload


class NailClipperIdealModel:
    model_id = "nail_clipper_compound_lever_geometry_v1"

    def __init__(self, spec: NailClipperCompoundLeverSpec) -> None:
        self._spec = spec

    def evaluate_samples(
        self,
        plan: MechanismPlan,
        samples: Sequence[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        del plan
        evaluated: list[dict[str, Any]] = []
        for sample in samples:
            if not isinstance(sample, Mapping):
                raise MechanismValidationError(
                    "IDEAL_REFERENCE_INPUT_INVALID",
                    "ideal samples must be objects",
                )
            time_s = finite(sample.get("time_s"), "sample.time_s")
            phase_id = sample.get("phase_id")
            if not isinstance(phase_id, str) or not phase_id:
                raise MechanismValidationError(
                    "IDEAL_REFERENCE_INPUT_INVALID",
                    "physical sample phase_id is missing",
                )
            input_angle = finite(
                sample.get("input_angle_rad"),
                "sample.input_angle_rad",
            )
            output_angle = input_angle * self._spec.transmission_ratio
            evaluated.append(
                {
                    "time_s": time_s,
                    "phase_id": phase_id,
                    "input_angle_rad": input_angle,
                    "output_angle_rad": output_angle,
                    "generalized_angle_rad": output_angle,
                    "signed_displacement_m": (
                        output_angle * self._spec.cutting_edge_radius_m
                    ),
                    "generalized_progress_m": (
                        output_angle * self._spec.cutting_edge_radius_m
                    ),
                }
            )
        return evaluated, {
            "domain": "nail_clipper_compound_lever",
            "transmission_ratio": self._spec.transmission_ratio,
            "cutting_edge_radius_m": self._spec.cutting_edge_radius_m,
        }


class NailClipperCompoundLeverAdapter(MechanismAdapter):
    plugin_id = "nail_clipper_compound_lever_v1"
    geometry_proof_kinds = frozenset({"revolute.pin_bore.v1"})
    initial_geometry_evidence_keys = frozenset({"joint_realization"})
    legacy_project_adapter_migrations = (
        {
            "basename": "nail_clipper_compound_lever.py",
            "sha256": (
                "65166c438501231c57f58288495894a22"
                "bee9e7ad68fecb484da18343dc2a4a8"
            ),
            "size_bytes": 12612,
            "legacy_mechanism_kind": "project_nail_clipper_compound_lever",
            "installed_mechanism_kind": "nail_clipper_compound_lever",
            "template_id": "joint.nail_clipper_compound_lever",
            "variant_id": "clipper.compound_lever",
        },
    )
    families = frozenset({"compound"})
    family_spec_kinds = frozenset({"nail_clipper_compound_lever"})
    extra_family_spec_fields = frozenset(
        {
            "bodies",
            "revolute_joints",
            "restoring_spring",
            "input_body_id",
            "output_body_id",
            "contact_point_abs_m",
            "cutting_edge_radius_m",
            "mechanism_kind",
            "coupling_partner_body_id",
            "step_collider_binding",
        }
    )
    coordinate = GeneralizedCoordinateSpec(
        position_field="angle_rad",
        velocity_field="angular_velocity_rad_s",
        position_unit="rad",
        velocity_unit="rad/s",
        input_wrench_type="torque",
        input_unit="N*m",
    )
    worker_backend = "nail_clipper_compound_lever_v1"
    whole_machine_backend = "nail_clipper_compound_lever_v1"
    wrench_policy_profiles = frozenset(
        {"task_input", "support", "constitutive", "transmission", "limit"}
    )
    # The clipper lane path is this adapter's simulate/ideal helpers plus
    # shared comparison/loads. Those modules do not read
    # ``plan.acceptance_contract`` or ``plan.semantic_binding``.
    lane_fingerprint_dropped_keys = (
        "acceptance_contract",
        "semantic_binding",
    )

    def supports(self, plan: MechanismPlan, adapter: "AdapterSpec") -> bool:
        return (
            super().supports(plan, adapter)
            and isinstance(adapter.family_spec, dict)
            and adapter.family_spec.get("mechanism_kind")
            == "nail_clipper_compound_lever"
        )

    def load_cases(self, plan: MechanismPlan) -> tuple[LoadCase, ...]:
        close_direction = tuple(-component for component in plan.axis_abs)
        return (
            LoadCase(
                load_case_id="task_output_angle_roundtrip",
                family="compound",
                program="task_output_angle_roundtrip",
                purpose="nominal",
                moving_mass_kg=0.08,
                step_s=0.0005,
                sample_interval_s=0.05,
                phases=(
                    LoadPhase(
                        "close_press",
                        "torque",
                        close_direction,
                        0.35,
                        0.45,
                    ),
                    LoadPhase("close_hold", "none", None, 0.0, 0.10),
                    LoadPhase("release_settle", "none", None, 0.0, 0.55),
                ),
            ),
        )

    def parse_family_spec(self, raw: Mapping[str, Any]) -> NailClipperCompoundLeverSpec:
        payload = _require_mapping(raw, "family_spec")
        kind = identifier(payload.get("mechanism_kind"), "family_spec.mechanism_kind")
        if kind != "nail_clipper_compound_lever":
            raise MechanismValidationError(
                "FAMILY_SPEC_INVALID",
                "mechanism_kind must be nail_clipper_compound_lever",
            )
        bodies = tuple(
            self._parse_body(item, index)
            for index, item in enumerate(
                _require_list(payload.get("bodies"), "family_spec.bodies")
            )
        )
        body_ids = tuple(item.body_id for item in bodies)
        if body_ids != _REQUIRED_BODY_IDS:
            raise MechanismValidationError(
                "FAMILY_SPEC_INVALID",
                "bodies must be stationary_cutting_jaw, actuating_lever, "
                "and moving_cutting_jaw in that order",
            )
        if sum(1 for item in bodies if item.fixed) != 1 or not bodies[0].fixed:
            raise MechanismValidationError(
                "FAMILY_SPEC_INVALID",
                "stationary_cutting_jaw must be the only fixed body",
            )
        joints = tuple(
            self._parse_joint(item, index, set(body_ids))
            for index, item in enumerate(
                _require_list(
                    payload.get("revolute_joints"),
                    "family_spec.revolute_joints",
                )
            )
        )
        if len(joints) != 2:
            raise MechanismValidationError(
                "FAMILY_SPEC_INVALID",
                "nail clipper requires exactly two revolute joints",
            )
        spring = self._parse_spring(payload.get("restoring_spring"), set(body_ids))
        input_body_id = identifier(
            payload.get("input_body_id"),
            "family_spec.input_body_id",
        )
        output_body_id = identifier(
            payload.get("output_body_id"),
            "family_spec.output_body_id",
        )
        if input_body_id != _INPUT_BODY_ID or output_body_id != _OUTPUT_BODY_ID:
            raise MechanismValidationError(
                "FAMILY_SPEC_INVALID",
                "input_body_id must be actuating_lever and output_body_id "
                "must be moving_cutting_jaw",
            )
        contact_point = vector3(
            payload.get("contact_point_abs_m"),
            "family_spec.contact_point_abs_m",
            allow_zero=True,
        )
        cutting_radius = finite(
            payload.get("cutting_edge_radius_m"),
            "family_spec.cutting_edge_radius_m",
        )
        if cutting_radius <= 0.0:
            raise MechanismValidationError(
                "FAMILY_SPEC_INVALID",
                "cutting_edge_radius_m must be positive",
            )
        lever_joint = _joint_by_child(joints, input_body_id, "lever revolute")
        jaw_joint = _joint_by_child(joints, output_body_id, "jaw revolute")
        if (
            jaw_joint.joint_id != _JAW_JOINT_ID
            or lever_joint.joint_id != _LEVER_JOINT_ID
        ):
            raise MechanismValidationError(
                "FAMILY_SPEC_INVALID",
                "nail clipper revolute joint IDs must be jaw_revolute and "
                "lever_revolute",
            )
        spec = NailClipperCompoundLeverSpec(
            mechanism_kind=kind,
            bodies=bodies,
            revolute_joints=joints,
            restoring_spring=spring,
            input_body_id=input_body_id,
            output_body_id=output_body_id,
            contact_point_abs_m=contact_point,
            cutting_edge_radius_m=cutting_radius,
            transmission_ratio=self._transmission_ratio(
                joints, contact_point, input_body_id, output_body_id
            ),
            coupling_partner_body_id="",
        )
        partner = _coupling_partner(spec)
        return NailClipperCompoundLeverSpec(
            mechanism_kind=spec.mechanism_kind,
            bodies=spec.bodies,
            revolute_joints=spec.revolute_joints,
            restoring_spring=spec.restoring_spring,
            input_body_id=spec.input_body_id,
            output_body_id=spec.output_body_id,
            contact_point_abs_m=spec.contact_point_abs_m,
            cutting_edge_radius_m=spec.cutting_edge_radius_m,
            transmission_ratio=spec.transmission_ratio,
            coupling_partner_body_id=partner,
        )

    def validate_family_spec(self, adapter: AdapterSpec) -> None:
        self.parse_family_spec(adapter.family_spec or {})

    def required_geometry_proofs(
        self,
        adapter: AdapterSpec,
        *,
        capsule_id: str,
    ) -> tuple[RevoluteProofRequirement, ...]:
        del capsule_id
        spec = self.parse_family_spec(adapter.family_spec or {})
        return tuple(
            RevoluteProofRequirement(
                joint_id=joint.joint_id,
                proof_kind="revolute.pin_bore.v1",
                parent_body_id=joint.parent_body_id,
                child_body_id=joint.child_body_id,
                origin_abs_m=joint.origin_abs_m,
                axis_abs=joint.axis_abs,
            )
            for joint in spec.revolute_joints
        )

    def materialize(
        self,
        plan: MechanismPlan,
        adapter: AdapterSpec,
        *,
        workspace_root: str | Path,
        output_dir: str | Path,
    ) -> AdapterSpec:
        from .family_collision import materialize_family_collision_adapter

        spec = self.parse_family_spec(adapter.family_spec or {})
        return materialize_family_collision_adapter(
            plan,
            adapter,
            _compound_bodies(spec),
            workspace_root=workspace_root,
            output_dir=output_dir,
        )

    def expected_reduced_ideal_links(
        self,
        plan: MechanismPlan,
        adapter: AdapterSpec,
    ) -> tuple[str, ...]:
        del plan
        spec = self.parse_family_spec(adapter.family_spec or {})
        return tuple(
            "ChLinkLockRevolute" for _ in spec.revolute_joints
        ) + ("ChLinkLockPointPlane",)

    def constitutive_supports(
        self,
        plan: MechanismPlan,
        adapter: AdapterSpec,
    ) -> tuple[CompliantSupportSpec, ...]:
        del plan
        spec = self.parse_family_spec(adapter.family_spec or {})
        spring = spec.restoring_spring
        return (
            CompliantSupportSpec(
                support_id="jaw_restoring_torsion_spring_v1",
                chrono_type="ExplicitChLoadBodyForceTorque",
                translational_stiffness_n_m=(0.0, 0.0, 0.0),
                translational_damping_n_s_m=(0.0, 0.0, 0.0),
                rotational_stiffness_n_m_rad=(0.0, 0.0, spring.stiffness_n_m_rad),
                rotational_damping_n_m_s_rad=(0.0, 0.0, spring.damping_n_m_s_rad),
                free_coordinate="none",
                source="project_nail_clipper_restoring_spring_v1",
            ),
        )

    def wrench_body_roles(
        self,
        family_spec: Mapping[str, Any],
    ) -> tuple[frozenset[str], frozenset[str]]:
        spec = self.parse_family_spec(family_spec)
        return (frozenset({spec.input_body_id}), frozenset({spec.output_body_id}))

    def task_channel_layout(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any],
        body_ids: tuple[str, ...],
    ) -> tuple[
        set[str],
        set[str],
        dict[str, str],
        dict[str, tuple[float, float, float]],
    ]:
        del plan, body_ids
        spec = self.parse_family_spec(family_spec)
        jaw_joint = _joint_by_child(
            spec.revolute_joints, spec.output_body_id, "jaw revolute"
        )
        lever_joint = _joint_by_child(
            spec.revolute_joints, spec.input_body_id, "lever revolute"
        )
        return (
            {spec.input_body_id},
            {spec.output_body_id},
            {
                spec.input_body_id: "input_angle_rad",
                spec.output_body_id: "output_angle_rad",
            },
            {
                spec.input_body_id: lever_joint.axis_abs,
                spec.output_body_id: jaw_joint.axis_abs,
            },
        )

    def ideal_model(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any] | None,
    ) -> NailClipperIdealModel:
        del plan
        if not isinstance(family_spec, Mapping):
            raise MechanismValidationError(
                "IDEAL_REFERENCE_UNSUPPORTED",
                "nail clipper ideal model requires a parsed family spec",
            )
        return NailClipperIdealModel(self.parse_family_spec(family_spec))

    def ideal_comparison_protocol(
        self,
        plan: MechanismPlan,
        family_spec: Mapping[str, Any] | None,
    ) -> tuple[tuple[str, ...], float, str, str]:
        del plan
        if not isinstance(family_spec, Mapping):
            raise MechanismValidationError(
                "IDEAL_REFERENCE_UNSUPPORTED",
                "nail clipper ideal comparison requires a family spec",
            )
        self.parse_family_spec(family_spec)
        return (
            ("generalized_angle_rad", "signed_displacement_m"),
            0.02,
            "mixed",
            "project_nail_clipper_geometry_ideal_v1",
        )

    def physical_behavior(
        self,
        plan: MechanismPlan,
        load_case: LoadCase,
        trace: dict[str, Any],
        family_spec: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
        del plan, load_case
        if not isinstance(family_spec, Mapping):
            raise MechanismValidationError(
                "PHYSICAL_IDEAL_COMPARISON_UNSUPPORTED",
                "nail clipper physical behavior requires a family spec",
            )
        spec = self.parse_family_spec(family_spec)
        samples = trace.get("samples")
        if not isinstance(samples, list) or not samples:
            raise MechanismValidationError(
                "PHYSICAL_TRACE_INVALID",
                "physical trace samples are missing",
            )
        output_angles = [
            finite(item.get("generalized_angle_rad"), "sample.generalized_angle_rad")
            for item in samples
            if isinstance(item, Mapping)
        ]
        displacements = [
            finite(item.get("signed_displacement_m"), "sample.signed_displacement_m")
            for item in samples
            if isinstance(item, Mapping)
        ]
        if not output_angles or not displacements:
            raise MechanismValidationError(
                "PHYSICAL_TRACE_INVALID",
                "physical samples must expose generalized_angle_rad and signed displacement",
            )
        start = output_angles[0]
        metrics = {
            "peak_output_angle_rad": max(abs(value) for value in output_angles),
            "peak_signed_displacement_m": max(abs(value) for value in displacements),
            "terminal_return_error_rad": abs(output_angles[-1] - start),
            "transmission_ratio": spec.transmission_ratio,
        }
        return metrics, {}, []

    def simulate(
        self,
        execution: ExecutionRequest,
        workspace: Path,
    ) -> dict[str, Any]:
        import pychrono as chrono

        from simulation_ready_cad_workflow.validation_pipeline.mechanism.wrench_ledger import (
            add_ledgered_load,
            create_body_torque,
            set_current_wrench_phase,
            standard_wrench_declaration,
            transition_current_wrench_phase,
        )

        spec = self.parse_family_spec(execution.adapter.family_spec or {})
        from .family_collision import family_collision_geometry

        expected_bodies = _compound_bodies(spec)
        geometry = family_collision_geometry(
            execution.adapter,
            expected_bodies=expected_bodies,
        )
        if (
            geometry["package_binding_sha256"]
            != execution.plan.package_binding_sha256
        ):
            raise MechanismValidationError(
                "FAMILY_COLLIDER_PACKAGE_BINDING_CHANGED",
                "clipper collision geometry does not match the executed package",
            )
        load_case = execution.load_case
        _require_fixed_duration_phases(load_case)
        topology = _build_nail_clipper_topology(
            chrono,
            execution,
            workspace,
            spec,
            geometry,
        )
        system = topology["system"]
        load_container = topology["load_container"]
        bodies = topology["bodies"]
        lever_joint = _joint_by_child(
            spec.revolute_joints, spec.input_body_id, "lever revolute"
        )
        jaw_joint = _joint_by_child(
            spec.revolute_joints, spec.output_body_id, "jaw revolute"
        )
        lever = bodies[spec.input_body_id]
        jaw = bodies[spec.output_body_id]
        input_torque = create_body_torque(
            chrono,
            lever,
            chrono.ChVector3d(0.0, 0.0, 0.0),
            False,
            declaration=standard_wrench_declaration(
                plugin_id=self.plugin_id,
                profile="task_input",
                wrench_kind="torque",
                load_id="declared-input-torque",
                owner=self.plugin_id,
                target_body=spec.input_body_id,
                source_role="declared_task_input",
                target_role="declared_input",
                geometry_gate="actuating_lever_axis",
                active_interval="active_load_phase",
            ),
        )
        spring_torque = create_body_torque(
            chrono,
            jaw,
            chrono.ChVector3d(0.0, 0.0, 0.0),
            False,
            declaration=standard_wrench_declaration(
                plugin_id=self.plugin_id,
                profile="constitutive",
                wrench_kind="torque",
                load_id="jaw-restoring-spring",
                owner=self.plugin_id,
                target_body=spec.output_body_id,
                source_role="constitutive_restoring_spring",
                target_role="task_output",
                geometry_gate="jaw_revolute_axis",
                active_interval="simulation_when_spring_response_is_nonzero",
            ),
        )
        add_ledgered_load(load_container, input_torque)
        add_ledgered_load(load_container, spring_torque)
        initial_jaw_quat = _quat_tuple(jaw)
        initial_lever_quat = _quat_tuple(lever)
        edge_world = chrono.ChVector3d(*spec.contact_point_abs_m)
        edge_local = jaw.TransformPointParentToLocal(edge_world)
        initial_edge = _vector_tuple(edge_world)
        tangent = _in_plane_tangent(
            jaw_joint.axis_abs, jaw_joint.origin_abs_m, initial_edge
        )
        samples: list[dict[str, Any]] = []
        executed_phases: list[dict[str, Any]] = []
        time_s = 0.0

        def record(phase_id: str) -> None:
            input_angle = _signed_axis_angle(
                lever, initial_lever_quat, lever_joint.axis_abs
            )
            output_angle = _signed_axis_angle(
                jaw, initial_jaw_quat, jaw_joint.axis_abs
            )
            current_edge = _vector_tuple(jaw.TransformPointLocalToParent(edge_local))
            displacement = _dot(_sub(current_edge, initial_edge), tangent)
            samples.append(
                {
                    "time_s": time_s,
                    "phase_id": phase_id,
                    "input_angle_rad": input_angle,
                    "generalized_angle_rad": output_angle,
                    "signed_displacement_m": displacement,
                    "generalized_progress_m": displacement,
                    "body_poses": {
                        body_id: _pose(body) for body_id, body in bodies.items()
                    },
                }
            )

        record("initial")
        next_sample_s = load_case.sample_interval_s
        prior_phase_open = False
        for phase in load_case.phases:
            _transition_task_input_phase(
                input_torque,
                chrono,
                phase.phase_id,
                prior_phase_open=prior_phase_open,
                transition=transition_current_wrench_phase,
            )
            prior_phase_open = True
            elapsed = 0.0
            while elapsed < phase.duration_s - 1e-12:
                if phase.wrench_kind == "torque":
                    direction = phase.direction_abs or (0.0, 0.0, 0.0)
                    torque = tuple(
                        component * phase.magnitude_si for component in direction
                    )
                else:
                    torque = (0.0, 0.0, 0.0)
                input_torque.SetTorque(chrono.ChVector3d(*torque), False)
                output_angle = _signed_axis_angle(
                    jaw, initial_jaw_quat, jaw_joint.axis_abs
                )
                output_rate = _axis_rate(jaw, jaw_joint.axis_abs)
                spring = spec.restoring_spring
                restoring = (
                    -spring.stiffness_n_m_rad
                    * (output_angle - spring.rest_angle_rad)
                    - spring.damping_n_m_s_rad * output_rate
                )
                spring_vector = tuple(
                    component * restoring for component in spring.axis_abs
                )
                spring_torque.SetTorque(chrono.ChVector3d(*spring_vector), False)
                step = min(load_case.step_s, phase.duration_s - elapsed)
                system.DoStepDynamics(step)
                elapsed += step
                time_s += step
                if time_s + 1e-12 >= next_sample_s:
                    record(phase.phase_id)
                    while next_sample_s <= time_s + 1e-12:
                        next_sample_s += load_case.sample_interval_s
            if not samples or samples[-1]["time_s"] < time_s - 1e-12:
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
                }
            )
        if prior_phase_open:
            input_torque.SetTorque(chrono.ChVector3d(0.0, 0.0, 0.0), False)
            set_current_wrench_phase(None)
        constitutive_supports = [
            item.to_dict()
            for item in self.constitutive_supports(
                execution.plan,
                execution.adapter,
            )
        ]
        return _completed_nail_clipper_trace(
            execution,
            spec,
            samples,
            executed_phases,
            topology,
            constitutive_supports,
            chrono_version=(
                str(chrono.GetChronoVersion())
                if hasattr(chrono, "GetChronoVersion")
                else None
            ),
        )

    def _parse_body(self, raw: Any, index: int) -> ClipperBodySpec:
        item = _require_mapping(raw, f"family_spec.bodies[{index}]")
        groups_raw = item.get("group_ids")
        if not isinstance(groups_raw, list) or not groups_raw:
            raise MechanismValidationError(
                "FAMILY_SPEC_INVALID",
                f"bodies[{index}].group_ids must be a non-empty array",
            )
        fixed_raw = item.get("fixed")
        if not isinstance(fixed_raw, bool):
            raise MechanismValidationError(
                "FAMILY_SPEC_INVALID",
                f"bodies[{index}].fixed must be a boolean",
            )
        mass = finite(item.get("mass_kg"), f"bodies[{index}].mass_kg")
        if mass <= 0.0:
            raise MechanismValidationError(
                "FAMILY_SPEC_INVALID",
                f"bodies[{index}].mass_kg must be positive",
            )
        return ClipperBodySpec(
            body_id=identifier(item.get("body_id"), f"bodies[{index}].body_id"),
            group_ids=tuple(
                identifier(group, f"bodies[{index}].group_ids[{group_index}]")
                for group_index, group in enumerate(groups_raw)
            ),
            fixed=bool(fixed_raw),
            mass_kg=mass,
            center_abs_m=vector3(
                item.get("center_abs_m"),
                f"bodies[{index}].center_abs_m",
                allow_zero=True,
            ),
            size_m=vector3(item.get("size_m"), f"bodies[{index}].size_m"),
        )

    def _parse_joint(
        self,
        raw: Any,
        index: int,
        body_ids: set[str],
    ) -> ClipperRevoluteSpec:
        item = _require_mapping(raw, f"family_spec.revolute_joints[{index}]")
        parent_id = identifier(
            item.get("parent_body_id"),
            f"revolute_joints[{index}].parent_body_id",
        )
        child_id = identifier(
            item.get("child_body_id"),
            f"revolute_joints[{index}].child_body_id",
        )
        if parent_id not in body_ids or child_id not in body_ids or parent_id == child_id:
            raise MechanismValidationError(
                "FAMILY_SPEC_INVALID",
                f"revolute_joints[{index}] body binding is invalid",
            )
        return ClipperRevoluteSpec(
            joint_id=identifier(
                item.get("joint_id"),
                f"revolute_joints[{index}].joint_id",
            ),
            parent_body_id=parent_id,
            child_body_id=child_id,
            origin_abs_m=vector3(
                item.get("origin_abs_m"),
                f"revolute_joints[{index}].origin_abs_m",
                allow_zero=True,
            ),
            axis_abs=vector3(
                item.get("axis_abs"),
                f"revolute_joints[{index}].axis_abs",
                unit=True,
            ),
        )

    def _parse_spring(self, raw: Any, body_ids: set[str]) -> ClipperSpringSpec:
        item = _require_mapping(raw, "family_spec.restoring_spring")
        body_a = identifier(item.get("body_a_id"), "restoring_spring.body_a_id")
        body_b = identifier(item.get("body_b_id"), "restoring_spring.body_b_id")
        if body_a not in body_ids or body_b not in body_ids or body_a == body_b:
            raise MechanismValidationError(
                "FAMILY_SPEC_INVALID",
                "restoring_spring body binding is invalid",
            )
        stiffness = finite(
            item.get("stiffness_n_m_rad"),
            "restoring_spring.stiffness_n_m_rad",
        )
        damping = finite(
            item.get("damping_n_m_s_rad"),
            "restoring_spring.damping_n_m_s_rad",
        )
        if stiffness <= 0.0 or damping < 0.0:
            raise MechanismValidationError(
                "FAMILY_SPEC_INVALID",
                "restoring_spring stiffness must be positive and damping nonnegative",
            )
        return ClipperSpringSpec(
            body_a_id=body_a,
            body_b_id=body_b,
            origin_abs_m=vector3(
                item.get("origin_abs_m"),
                "restoring_spring.origin_abs_m",
                allow_zero=True,
            ),
            axis_abs=vector3(
                item.get("axis_abs"),
                "restoring_spring.axis_abs",
                unit=True,
            ),
            stiffness_n_m_rad=stiffness,
            damping_n_m_s_rad=damping,
            rest_angle_rad=finite(
                item.get("rest_angle_rad"),
                "restoring_spring.rest_angle_rad",
            ),
        )

    def _transmission_ratio(
        self,
        joints: Sequence[ClipperRevoluteSpec],
        contact_point: tuple[float, float, float],
        input_body_id: str,
        output_body_id: str,
    ) -> float:
        jaw_joint = _joint_by_child(joints, output_body_id, "jaw revolute")
        lever_joint = _joint_by_child(joints, input_body_id, "lever revolute")
        jaw_arm = _distance(jaw_joint.origin_abs_m, contact_point)
        lever_arm = _distance(lever_joint.origin_abs_m, contact_point)
        if jaw_arm <= 1e-9 or lever_arm <= 1e-9:
            raise MechanismValidationError(
                "FAMILY_SPEC_INVALID",
                "joint origins and contact point must define nonzero lever arms",
            )
        return lever_arm / jaw_arm


def _distance(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right)) ** 0.5


def _sub(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _dot(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _cross(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _in_plane_normal(
    axis: tuple[float, float, float],
    origin: tuple[float, float, float],
    point: tuple[float, float, float],
) -> tuple[float, float, float]:
    radial = _sub(point, origin)
    radial = _sub(radial, tuple(axis[i] * _dot(radial, axis) for i in range(3)))
    length = _distance(radial, (0.0, 0.0, 0.0))
    if length <= 1e-9:
        raise MechanismValidationError(
            "FAMILY_SPEC_INVALID",
            "contact point must be offset from the joint origin in the joint plane",
        )
    return (radial[0] / length, radial[1] / length, radial[2] / length)


def _in_plane_tangent(
    axis: tuple[float, float, float],
    origin: tuple[float, float, float],
    point: tuple[float, float, float],
) -> tuple[float, float, float]:
    radial = _in_plane_normal(axis, origin, point)
    tangent = _cross(axis, radial)
    length = _distance(tangent, (0.0, 0.0, 0.0))
    if length <= 1e-9:
        raise MechanismValidationError(
            "FAMILY_SPEC_INVALID",
            "joint axis and contact radius must define a nonzero in-plane tangent",
        )
    return (tangent[0] / length, tangent[1] / length, tangent[2] / length)


def _vector_tuple(vector: Any) -> tuple[float, float, float]:
    return (float(vector.x), float(vector.y), float(vector.z))


def _quat_from_z_to_axis(chrono: Any, axis: tuple[float, float, float]) -> Any:
    dot = max(-1.0, min(1.0, axis[2]))
    if dot < -1.0 + 1e-12:
        return chrono.ChQuaterniond(0.0, 1.0, 0.0, 0.0)
    cross = (-axis[1], axis[0], 0.0)
    scale = math.sqrt(2.0 * (1.0 + dot))
    return chrono.ChQuaterniond(
        0.5 * scale,
        cross[0] / scale,
        cross[1] / scale,
        cross[2] / scale,
    )


def _quat_tuple(body: Any) -> tuple[float, float, float, float]:
    quat = body.GetRot()
    return (float(quat.e0), float(quat.e1), float(quat.e2), float(quat.e3))


def _pose(body: Any) -> dict[str, list[float]]:
    pos = body.GetPos()
    quat = body.GetRot()
    return {
        "position_abs_m": [float(pos.x), float(pos.y), float(pos.z)],
        "rotation_abs_quat_wxyz": [
            float(quat.e0),
            float(quat.e1),
            float(quat.e2),
            float(quat.e3),
        ],
    }


def _signed_axis_angle(
    body: Any,
    initial_quat: tuple[float, float, float, float],
    axis: tuple[float, float, float],
) -> float:
    current = _quat_tuple(body)
    relative = _quat_product(_quat_conjugate(initial_quat), current)
    angle = 2.0 * math.atan2(
        math.sqrt(relative[1] ** 2 + relative[2] ** 2 + relative[3] ** 2),
        relative[0],
    )
    vector = (relative[1], relative[2], relative[3])
    sign = -1.0 if sum(a * b for a, b in zip(vector, axis)) < 0.0 else 1.0
    return sign * angle


def _axis_rate(body: Any, axis: tuple[float, float, float]) -> float:
    omega = body.GetAngVelParent()
    return (
        float(omega.x) * axis[0]
        + float(omega.y) * axis[1]
        + float(omega.z) * axis[2]
    )


def _quat_conjugate(
    quat: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return (quat[0], -quat[1], -quat[2], -quat[3])


def _quat_product(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )
