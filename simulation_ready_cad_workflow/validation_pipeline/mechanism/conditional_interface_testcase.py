from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from .collision_policy import (
    ANALYTICAL_TRANSMISSION,
    collision_policy_disclosure,
    interface_declaration,
)
from .conditional_interface import (
    AxialGearPose,
    AxialSlidingSpurGearGeometry,
    ConditionalCompliantInterface,
    assess_initial_functional_geometry,
    audit_conditional_interface_runtime_evidence,
    build_functional_interface_geometry_binding,
    conditional_transmission_gate,
    validate_functional_interface_geometry_binding,
)
from .io import write_json_atomic
from .wrench_ledger import (
    WrenchLedger,
    WrenchPolicy,
    activate_wrench_ledger,
    add_ledgered_load,
    create_body_force,
    create_body_torque,
    set_current_wrench_phase,
    standard_wrench_declaration,
    transition_current_wrench_phase,
)


_TESTCASE_WRENCH_PLUGIN_ID = "conditional_interface_testcase_v1"
_TESTCASE_WRENCH_REGISTRATION = {
    "schema_version": "simulation_ready_cad_plugin_registration_v1",
    "plugin_id": _TESTCASE_WRENCH_PLUGIN_ID,
    "status": "ACTIVE",
    "registration_source": "validator_owned_testcase",
}
_TESTCASE_WRENCH_POLICIES = (
    WrenchPolicy(
        policy_id=f"{_TESTCASE_WRENCH_PLUGIN_ID}.task_input_testcase.force",
        wrench_kind="force",
        category="task_input",
        law_id="validator_testcase_multichannel_v1",
        target_roles=("declared_input",),
        task_input=True,
    ),
    WrenchPolicy(
        policy_id=f"{_TESTCASE_WRENCH_PLUGIN_ID}.task_input_testcase.torque",
        wrench_kind="torque",
        category="task_input",
        law_id="validator_testcase_multichannel_v1",
        target_roles=("declared_input",),
        task_input=True,
    ),
    WrenchPolicy(
        policy_id=f"{_TESTCASE_WRENCH_PLUGIN_ID}.transmission.torque",
        wrench_kind="torque",
        category="analytical_transmission",
        law_id="geometry_gated_transmission_v1",
        target_roles=("declared_input", "task_output"),
        transmission=True,
        capability_required=True,
    ),
    WrenchPolicy(
        policy_id=f"{_TESTCASE_WRENCH_PLUGIN_ID}.resistance.torque",
        wrench_kind="torque",
        category="validator_resistance",
        law_id="validator_resistance_v1",
        target_roles=("declared_input", "task_output"),
    ),
)


def _clamp(value: float, magnitude: float) -> float:
    return max(-magnitude, min(magnitude, value))


def _unique_state_path(samples: Sequence[dict[str, Any]]) -> list[str]:
    states: list[str] = []
    for sample in samples:
        state = str(sample["interface_state"])
        if not states or states[-1] != state:
            states.append(state)
    return states


def _runtime_summary(
    samples: Sequence[dict[str, Any]],
    *,
    physics_step_count: int,
) -> dict[str, Any]:
    inactive_efforts = [
        abs(float(sample["transmitted_mesh_torque_nm"]))
        for sample in samples
        if not sample["transmission_active"]
    ]
    return {
        "schema_version": "conditional_interface_runtime_v1",
        "evaluated_every_physics_step": True,
        "physics_step_count": physics_step_count,
        "interfaces": [
            {
                "interface_id": "sliding-driver-to-driven",
                "evaluation_count": physics_step_count,
                "max_inactive_abs_transmitted_effort_si": max(
                    inactive_efforts, default=0.0
                ),
                "state_path": _unique_state_path(samples),
            }
        ],
    }


def _testcase_contract() -> tuple[
    AxialSlidingSpurGearGeometry,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    geometry = AxialSlidingSpurGearGeometry(
        driver_pitch_radius_m=0.020,
        driven_pitch_radius_m=0.030,
        radial_tolerance_m=0.0002,
        axis_parallel_tolerance_rad=math.radians(0.25),
    )
    gate = conditional_transmission_gate(
        geometry_plugin_id=geometry.plugin_id,
        body_a_id="driver_gear",
        body_b_id="driven_gear",
        endpoint_a_component_refs=("driver_gear/solid_driver_teeth",),
        endpoint_b_component_refs=("driven_gear/solid_driven_teeth",),
        activation_coordinate="face_overlap_m",
        enter_threshold_si=0.0005,
        release_threshold_si=0.0001,
        proof_source="validator_step_geometry_runtime_v1",
        geometry_parameters={
            "driver_pitch_radius_m": geometry.driver_pitch_radius_m,
            "driven_pitch_radius_m": geometry.driven_pitch_radius_m,
            "radial_tolerance_m": geometry.radial_tolerance_m,
            "axis_parallel_tolerance_rad": (
                geometry.axis_parallel_tolerance_rad
            ),
        },
    )
    declaration = interface_declaration(
        interface_id="sliding-driver-to-driven",
        classification=ANALYTICAL_TRANSMISSION,
        body_a_id="driver_gear",
        body_b_id="driven_gear",
        model="finite_conditional_spur_gear_penalty_v1",
        source="conditional_interface_testcase_v1",
        parameters={
            "mesh_stiffness_nm_rad": 0.60,
            "mesh_damping_nm_s_rad": 0.025,
            "max_mesh_torque_nm": 0.08,
        },
        transmission_gate=gate,
    )
    policy = collision_policy_disclosure((declaration,), branch="physical")
    collision_geometry = {
        "package_binding_sha256": "a" * 64,
        "assembly_sha256": "b" * 64,
        "bodies": [
            {
                "body_id": "driver_gear",
                "fixed": False,
                "components": [
                    {
                        "solid_id": "solid_driver_teeth",
                        "sha256": "c" * 64,
                    }
                ],
            },
            {
                "body_id": "driven_gear",
                "fixed": False,
                "components": [
                    {
                        "solid_id": "solid_driven_teeth",
                        "sha256": "d" * 64,
                    }
                ],
            },
        ],
    }
    geometry_binding = build_functional_interface_geometry_binding(
        (declaration,), collision_geometry
    )
    validate_functional_interface_geometry_binding(
        geometry_binding,
        interfaces=(declaration,),
        collision_geometry=collision_geometry,
    )
    return geometry, gate, declaration, policy, geometry_binding


def run_sliding_gear_engagement_testcase(
    *,
    duration_s: float = 1.20,
    step_s: float = 0.0005,
    sample_interval_s: float = 0.005,
) -> dict[str, Any]:
    """Run a deterministic physical-law fixture for late gear engagement.

    This intentionally contains no ideal gear constraint. A finite
    spring-damper tooth law is evaluated throughout the run, while the
    generic conditional interface is the only authority allowed to pass its
    torque to the bodies.
    """

    geometry, gate, declaration, policy, geometry_binding = (
        _testcase_contract()
    )
    controller = ConditionalCompliantInterface(gate)

    driver_z_initial_m = 0.018
    driver_z_final_m = 0.0
    slide_start_s = 0.10
    slide_end_s = 0.70
    slide_speed_m_s = (
        driver_z_final_m - driver_z_initial_m
    ) / (slide_end_s - slide_start_s)
    driver_face_width_m = 0.008
    driven_face_width_m = 0.008
    driver_pose_initial = AxialGearPose(
        center_abs_m=(0.050, 0.0, driver_z_initial_m),
        axis_abs=(0.0, 0.0, 1.0),
        face_width_m=driver_face_width_m,
        linear_velocity_abs_m_s=(0.0, 0.0, 0.0),
    )
    driven_pose = AxialGearPose(
        center_abs_m=(0.0, 0.0, 0.0),
        axis_abs=(0.0, 0.0, 1.0),
        face_width_m=driven_face_width_m,
    )
    initial_observation = geometry.evaluate(
        time_s=0.0,
        driver=driver_pose_initial,
        driven=driven_pose,
    )
    cad_assessment = assess_initial_functional_geometry(
        gate, initial_observation
    )

    driver_angle = 0.0
    driven_angle = 0.0
    driver_speed = 0.0
    driven_speed = 0.0
    driver_inertia = 0.0020
    driven_inertia = 0.0045
    ratio = geometry.driver_pitch_radius_m / geometry.driven_pitch_radius_m
    input_torque_nm = 0.018
    shaft_drag_nm_s_rad = 0.002
    mesh_stiffness_nm_rad = 0.60
    mesh_damping_nm_s_rad = 0.025
    mesh_torque_limit_nm = 0.08
    engagement_reference = None
    first_engagement_time_s = None
    samples: list[dict[str, Any]] = []
    max_inactive_transmitted_torque_nm = 0.0
    max_inactive_driven_angle_rad = 0.0
    transmitted_work_j = 0.0
    sample_stride = max(1, int(round(sample_interval_s / step_s)))
    step_count = int(math.ceil(duration_s / step_s))

    for index in range(step_count + 1):
        time_s = min(duration_s, index * step_s)
        if time_s < slide_start_s:
            driver_z_m = driver_z_initial_m
            driver_slide_speed_m_s = 0.0
        elif time_s < slide_end_s:
            driver_z_m = (
                driver_z_initial_m
                + slide_speed_m_s * (time_s - slide_start_s)
            )
            driver_slide_speed_m_s = slide_speed_m_s
        else:
            driver_z_m = driver_z_final_m
            driver_slide_speed_m_s = 0.0
        driver_pose = AxialGearPose(
            center_abs_m=(0.050, 0.0, driver_z_m),
            axis_abs=(0.0, 0.0, 1.0),
            face_width_m=driver_face_width_m,
            linear_velocity_abs_m_s=(0.0, 0.0, driver_slide_speed_m_s),
        )
        observation = geometry.evaluate(
            time_s=time_s,
            driver=driver_pose,
            driven=driven_pose,
        )
        decision = controller.step(observation)
        if decision.capture_reference:
            engagement_reference = driven_angle + ratio * driver_angle
            if first_engagement_time_s is None:
                first_engagement_time_s = time_s

        if engagement_reference is None:
            requested_mesh_torque_nm = 0.0
            phase_error_rad = 0.0
        else:
            phase_error_rad = (
                driven_angle
                + ratio * driver_angle
                - engagement_reference
            )
            speed_error_rad_s = driven_speed + ratio * driver_speed
            requested_mesh_torque_nm = _clamp(
                -mesh_stiffness_nm_rad * phase_error_rad
                - mesh_damping_nm_s_rad * speed_error_rad_s,
                mesh_torque_limit_nm,
            )
        transmitted_mesh_torque_nm = decision.transmit_scalar(
            requested_mesh_torque_nm
        )
        driver_mesh_torque_nm = ratio * transmitted_mesh_torque_nm
        driver_acceleration = (
            input_torque_nm
            + driver_mesh_torque_nm
            - shaft_drag_nm_s_rad * driver_speed
        ) / driver_inertia
        driven_acceleration = (
            transmitted_mesh_torque_nm
            - shaft_drag_nm_s_rad * driven_speed
        ) / driven_inertia

        if index < step_count:
            driver_speed += driver_acceleration * step_s
            driven_speed += driven_acceleration * step_s
            driver_angle += driver_speed * step_s
            driven_angle += driven_speed * step_s
            transmitted_work_j += abs(
                transmitted_mesh_torque_nm * driven_speed * step_s
            )

        if not decision.transmission_active:
            max_inactive_transmitted_torque_nm = max(
                max_inactive_transmitted_torque_nm,
                abs(transmitted_mesh_torque_nm),
            )
            if first_engagement_time_s is None:
                max_inactive_driven_angle_rad = max(
                    max_inactive_driven_angle_rad,
                    abs(driven_angle),
                )
        if (
            index % sample_stride == 0
            or index == step_count
            or decision.transitioned
        ):
            samples.append(
                {
                    "time_s": time_s,
                    "driver_axial_position_m": driver_z_m,
                    "driver_angle_rad": driver_angle,
                    "driver_speed_rad_s": driver_speed,
                    "driven_angle_rad": driven_angle,
                    "driven_speed_rad_s": driven_speed,
                    "interface_state": decision.state,
                    "transmission_active": decision.transmission_active,
                    "interface_reason": decision.reason,
                    "face_overlap_m": observation.activation_coordinate_si,
                    "phase_error_rad": phase_error_rad,
                    "requested_mesh_torque_nm": requested_mesh_torque_nm,
                    "transmitted_mesh_torque_nm": transmitted_mesh_torque_nm,
                    "capture_reference": decision.capture_reference,
                    "conditional_interface_states": [
                        {
                            "interface_id": "sliding-driver-to-driven",
                            "state": decision.state,
                            "transmission_active": (
                                decision.transmission_active
                            ),
                            "transmitted_effort_si": (
                                transmitted_mesh_torque_nm
                            ),
                            "activation_coordinate_si": (
                                observation.activation_coordinate_si
                            ),
                            "constraints_satisfied": (
                                observation.constraints_satisfied
                            ),
                        }
                    ],
                }
            )

    post_engagement_driven_span_rad = (
        max(sample["driven_angle_rad"] for sample in samples)
        - min(sample["driven_angle_rad"] for sample in samples)
    )
    state_path = _unique_state_path(samples)
    runtime_summary = _runtime_summary(
        samples, physics_step_count=step_count + 1
    )
    runtime_audit = audit_conditional_interface_runtime_evidence(
        (declaration,), samples, runtime_summary
    )
    checks = {
        "cad_initial_pose_is_warning_not_error": (
            cad_assessment["status"] == "WARNING"
            and cad_assessment["hard_error"] is False
        ),
        "inactive_effort_is_exact_zero": (
            max_inactive_transmitted_torque_nm == 0.0
        ),
        "driven_gear_stationary_before_engagement": (
            max_inactive_driven_angle_rad <= 1e-12
        ),
        "late_geometry_engagement_observed": (
            first_engagement_time_s is not None
            and "APPROACHING" in state_path
            and "ENGAGED" in state_path
        ),
        "driven_gear_rotates_after_engagement": (
            post_engagement_driven_span_rad >= 0.01
            and transmitted_work_j > 0.0
        ),
        "whole_body_pair_collision_remains_enabled": (
            declaration["whole_body_pair_collision_enabled"] is True
        ),
        "no_ideal_gear_joint_used": True,
        "runtime_evidence_audit_passes": runtime_audit["passed"],
    }
    passed = all(checks.values())
    return {
        "schema_version": "conditional_sliding_gear_testcase_v1",
        "testcase_id": "conditional_sliding_gear_v1",
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "description": (
            "driver gear descends axially on a sleeve, remains dynamically "
            "decoupled while separated, then transmits finite compliant torque"
        ),
        "branch": "physical_law_fixture",
        "cad_initial_assessment": cad_assessment,
        "physical_interface_policy": policy,
        "functional_interface_geometry_binding": geometry_binding,
        "conditional_interface_runtime": runtime_summary,
        "conditional_interface_runtime_audit": runtime_audit,
        "state_path": state_path,
        "first_engagement_time_s": first_engagement_time_s,
        "max_inactive_transmitted_torque_nm": (
            max_inactive_transmitted_torque_nm
        ),
        "max_inactive_driven_angle_rad": max_inactive_driven_angle_rad,
        "post_engagement_driven_span_rad": post_engagement_driven_span_rad,
        "transmitted_work_j": transmitted_work_j,
        "checks": checks,
        "samples": samples,
    }


def _chrono_vector_tuple(value: Any) -> tuple[float, float, float]:
    return (float(value.x), float(value.y), float(value.z))


def _run_pychrono_sliding_gear_engagement_testcase_active(
    *,
    duration_s: float = 1.20,
    step_s: float = 0.0005,
    sample_interval_s: float = 0.005,
) -> dict[str, Any]:
    """Exercise the same gate against live PyChrono rigid bodies.

    The shaft and sliding-sleeve supports are finite 6-DOF bushings. No
    revolute, prismatic, rack-pinion, or ideal gear link is created. The
    fixture has no certificate authority over collider fidelity; its purpose
    is to prove that runtime geometry transitions control real body torques.
    """

    import pychrono as chrono

    geometry, gate, declaration, policy, geometry_binding = (
        _testcase_contract()
    )
    controller = ConditionalCompliantInterface(gate)
    system = chrono.ChSystemNSC()
    system.SetGravitationalAcceleration(chrono.ChVector3d(0.0, 0.0, 0.0))
    solver = chrono.ChSolverMINRES()
    if hasattr(solver, "SetMaxIterations"):
        solver.SetMaxIterations(200)
    if hasattr(solver, "SetTolerance"):
        solver.SetTolerance(1e-11)
    system.SetSolver(solver)

    ground = chrono.ChBody()
    ground.SetName("ground")
    ground.SetFixed(True)
    system.AddBody(ground)

    driver = chrono.ChBody()
    driver.SetName("driver_gear")
    driver.SetMass(0.20)
    driver.SetInertiaXX(chrono.ChVector3d(0.00030, 0.00030, 0.00020))
    driver.SetPos(chrono.ChVector3d(0.050, 0.0, 0.018))
    system.AddBody(driver)

    driven = chrono.ChBody()
    driven.SetName("driven_gear")
    driven.SetMass(0.30)
    driven.SetInertiaXX(chrono.ChVector3d(0.00070, 0.00070, 0.00045))
    driven.SetPos(chrono.ChVector3d(0.0, 0.0, 0.0))
    system.AddBody(driven)

    loads = chrono.ChLoadContainer()
    system.Add(loads)

    def support_bushing(
        body: Any,
        stiffness: Sequence[float],
        damping: Sequence[float],
    ) -> Any:
        matrix_k = chrono.ChMatrix66d()
        matrix_d = chrono.ChMatrix66d()
        matrix_k.SetZero()
        matrix_d.SetZero()
        for index, value in enumerate(stiffness):
            matrix_k.SetItem(index, index, value)
        for index, value in enumerate(damping):
            matrix_d.SetItem(index, index, value)
        bushing = chrono.ChLoadBodyBodyBushingGeneric(
            body,
            ground,
            chrono.ChFramed(body.GetPos(), chrono.QUNIT),
            matrix_k,
            matrix_d,
        )
        loads.Add(bushing)
        return bushing

    support_bushing(
        driver,
        (1_000_000.0, 1_000_000.0, 0.0, 500.0, 500.0, 0.0),
        (500.0, 500.0, 0.0, 3.0, 3.0, 0.0),
    )
    support_bushing(
        driven,
        (1_000_000.0, 1_000_000.0, 1_000_000.0, 500.0, 500.0, 0.0),
        (500.0, 500.0, 500.0, 3.0, 3.0, 0.0),
    )

    axial_actuator = create_body_force(
        chrono,
        driver,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        False,
        driver.GetPos(),
        False,
        declaration=standard_wrench_declaration(
            plugin_id=_TESTCASE_WRENCH_PLUGIN_ID,
            profile="task_input_testcase",
            wrench_kind="force",
            load_id="driver-axial-actuator",
            owner=_TESTCASE_WRENCH_PLUGIN_ID,
            target_body="driver_gear",
            source_role="declared_input_port",
            target_role="declared_input",
            geometry_gate="testcase_axial_engagement_schedule",
            active_interval="testcase_slide_interval",
            release_required=True,
        ),
    )
    driver_input_torque = create_body_torque(
        chrono,
        driver,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        False,
        declaration=standard_wrench_declaration(
            plugin_id=_TESTCASE_WRENCH_PLUGIN_ID,
            profile="task_input_testcase",
            wrench_kind="torque",
            load_id="driver-input-torque",
            owner=_TESTCASE_WRENCH_PLUGIN_ID,
            target_body="driver_gear",
            source_role="declared_input_port",
            target_role="declared_input",
            geometry_gate="testcase_powered_interval",
            active_interval="testcase_powered_interval",
            release_required=True,
        ),
    )
    driver_transmission_torque = create_body_torque(
        chrono,
        driver,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        False,
        declaration=standard_wrench_declaration(
            plugin_id=_TESTCASE_WRENCH_PLUGIN_ID,
            profile="transmission",
            wrench_kind="torque",
            load_id="driver-transmission-reaction",
            owner=_TESTCASE_WRENCH_PLUGIN_ID,
            target_body="driver_gear",
            source_role="sliding_gear_interface",
            target_role="declared_input",
            geometry_gate="conditional_sliding_gear_geometry_gate",
            active_interval="geometry_gated_engaged_interval",
            release_required=True,
        ),
    )
    driven_transmission_torque = create_body_torque(
        chrono,
        driven,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        False,
        declaration=standard_wrench_declaration(
            plugin_id=_TESTCASE_WRENCH_PLUGIN_ID,
            profile="transmission",
            wrench_kind="torque",
            load_id="driven-transmission-torque",
            owner=_TESTCASE_WRENCH_PLUGIN_ID,
            target_body="driven_gear",
            source_role="sliding_gear_interface",
            target_role="task_output",
            geometry_gate="conditional_sliding_gear_geometry_gate",
            active_interval="geometry_gated_engaged_interval",
            release_required=True,
        ),
    )
    driver_drag_torque = create_body_torque(
        chrono,
        driver,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        False,
        declaration=standard_wrench_declaration(
            plugin_id=_TESTCASE_WRENCH_PLUGIN_ID,
            profile="resistance",
            wrench_kind="torque",
            load_id="driver-passive-drag",
            owner=_TESTCASE_WRENCH_PLUGIN_ID,
            target_body="driver_gear",
            source_role="finite_shaft_drag",
            target_role="declared_input",
            geometry_gate="nonzero_driver_speed",
            active_interval="simulation_when_drag_is_nonzero",
            release_required=True,
        ),
    )
    driven_drag_torque = create_body_torque(
        chrono,
        driven,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        False,
        declaration=standard_wrench_declaration(
            plugin_id=_TESTCASE_WRENCH_PLUGIN_ID,
            profile="resistance",
            wrench_kind="torque",
            load_id="driven-passive-drag",
            owner=_TESTCASE_WRENCH_PLUGIN_ID,
            target_body="driven_gear",
            source_role="finite_shaft_drag",
            target_role="task_output",
            geometry_gate="nonzero_driven_speed",
            active_interval="simulation_when_drag_is_nonzero",
            release_required=True,
        ),
    )
    for load in (
        axial_actuator,
        driver_input_torque,
        driver_transmission_torque,
        driven_transmission_torque,
        driver_drag_torque,
        driven_drag_torque,
    ):
        add_ledgered_load(loads, load)

    driver_z_initial_m = 0.018
    driver_z_final_m = 0.0
    slide_start_s = 0.10
    slide_end_s = 0.70
    slide_speed_m_s = (
        driver_z_final_m - driver_z_initial_m
    ) / (slide_end_s - slide_start_s)
    face_width_m = 0.008
    driven_pose_initial = AxialGearPose(
        center_abs_m=(0.0, 0.0, 0.0),
        axis_abs=(0.0, 0.0, 1.0),
        face_width_m=face_width_m,
    )
    initial_observation = geometry.evaluate(
        time_s=0.0,
        driver=AxialGearPose(
            center_abs_m=(0.050, 0.0, driver_z_initial_m),
            axis_abs=(0.0, 0.0, 1.0),
            face_width_m=face_width_m,
        ),
        driven=driven_pose_initial,
    )
    cad_assessment = assess_initial_functional_geometry(
        gate, initial_observation
    )

    ratio = geometry.driver_pitch_radius_m / geometry.driven_pitch_radius_m
    input_torque_nm = 0.018
    shaft_drag_nm_s_rad = 0.002
    mesh_stiffness_nm_rad = 0.60
    mesh_damping_nm_s_rad = 0.025
    mesh_torque_limit_nm = 0.08
    driver_angle = 0.0
    driven_angle = 0.0
    engagement_reference = None
    first_engagement_time_s = None
    max_inactive_transmitted_torque_nm = 0.0
    max_pre_engagement_driven_angle_rad = 0.0
    max_support_drift_m = 0.0
    transmitted_work_j = 0.0
    samples: list[dict[str, Any]] = []
    sample_stride = max(1, int(round(sample_interval_s / step_s)))
    step_count = int(math.ceil(duration_s / step_s))

    transition_current_wrench_phase("testcase_powered_interval")
    for index in range(step_count + 1):
        time_s = min(duration_s, index * step_s)
        driver_position = _chrono_vector_tuple(driver.GetPos())
        driven_position = _chrono_vector_tuple(driven.GetPos())
        driver_velocity = _chrono_vector_tuple(driver.GetPosDt())
        driven_velocity = _chrono_vector_tuple(driven.GetPosDt())
        driver_angular_velocity = _chrono_vector_tuple(
            driver.GetAngVelParent()
        )
        driven_angular_velocity = _chrono_vector_tuple(
            driven.GetAngVelParent()
        )
        driver_speed = driver_angular_velocity[2]
        driven_speed = driven_angular_velocity[2]

        observation = geometry.evaluate(
            time_s=time_s,
            driver=AxialGearPose(
                center_abs_m=driver_position,
                axis_abs=(0.0, 0.0, 1.0),
                face_width_m=face_width_m,
                linear_velocity_abs_m_s=driver_velocity,
            ),
            driven=AxialGearPose(
                center_abs_m=driven_position,
                axis_abs=(0.0, 0.0, 1.0),
                face_width_m=face_width_m,
                linear_velocity_abs_m_s=driven_velocity,
            ),
        )
        decision = controller.step(observation)
        if decision.capture_reference:
            engagement_reference = driven_angle + ratio * driver_angle
            if first_engagement_time_s is None:
                first_engagement_time_s = time_s

        if engagement_reference is None:
            phase_error_rad = 0.0
            requested_mesh_torque_nm = 0.0
        else:
            phase_error_rad = (
                driven_angle
                + ratio * driver_angle
                - engagement_reference
            )
            speed_error_rad_s = driven_speed + ratio * driver_speed
            requested_mesh_torque_nm = _clamp(
                -mesh_stiffness_nm_rad * phase_error_rad
                - mesh_damping_nm_s_rad * speed_error_rad_s,
                mesh_torque_limit_nm,
            )
        transmitted_mesh_torque_nm = decision.transmit_scalar(
            requested_mesh_torque_nm
        )

        if time_s < slide_start_s:
            target_z_m = driver_z_initial_m
            target_speed_m_s = 0.0
        elif time_s < slide_end_s:
            target_z_m = (
                driver_z_initial_m
                + slide_speed_m_s * (time_s - slide_start_s)
            )
            target_speed_m_s = slide_speed_m_s
        else:
            target_z_m = driver_z_final_m
            target_speed_m_s = 0.0
        axial_force_n = _clamp(
            2_500.0 * (target_z_m - driver_position[2])
            + 45.0 * (target_speed_m_s - driver_velocity[2]),
            12.0,
        )
        axial_actuator.SetForce(
            chrono.ChVector3d(0.0, 0.0, axial_force_n), False
        )
        axial_actuator.SetApplicationPoint(driver.GetPos(), False)
        driver_input_torque.SetTorque(
            chrono.ChVector3d(0.0, 0.0, input_torque_nm),
            False,
        )
        driver_transmission_torque.SetTorque(
            chrono.ChVector3d(
                0.0, 0.0, ratio * transmitted_mesh_torque_nm
            ),
            False,
        )
        driven_transmission_torque.SetTorque(
            chrono.ChVector3d(0.0, 0.0, transmitted_mesh_torque_nm),
            False,
        )
        driver_drag_torque.SetTorque(
            chrono.ChVector3d(
                0.0, 0.0, -shaft_drag_nm_s_rad * driver_speed
            ),
            False,
        )
        driven_drag_torque.SetTorque(
            chrono.ChVector3d(
                0.0, 0.0, -shaft_drag_nm_s_rad * driven_speed
            ),
            False,
        )

        if not decision.transmission_active:
            max_inactive_transmitted_torque_nm = max(
                max_inactive_transmitted_torque_nm,
                abs(transmitted_mesh_torque_nm),
            )
            if first_engagement_time_s is None:
                max_pre_engagement_driven_angle_rad = max(
                    max_pre_engagement_driven_angle_rad,
                    abs(driven_angle),
                )
        max_support_drift_m = max(
            max_support_drift_m,
            abs(driver_position[0] - 0.050),
            abs(driver_position[1]),
            abs(driven_position[0]),
            abs(driven_position[1]),
            abs(driven_position[2]),
        )
        if (
            index % sample_stride == 0
            or index == step_count
            or decision.transitioned
        ):
            samples.append(
                {
                    "time_s": time_s,
                    "driver_axial_position_m": driver_position[2],
                    "driver_angle_rad": driver_angle,
                    "driver_speed_rad_s": driver_speed,
                    "driven_angle_rad": driven_angle,
                    "driven_speed_rad_s": driven_speed,
                    "interface_state": decision.state,
                    "transmission_active": decision.transmission_active,
                    "interface_reason": decision.reason,
                    "face_overlap_m": observation.activation_coordinate_si,
                    "phase_error_rad": phase_error_rad,
                    "requested_mesh_torque_nm": requested_mesh_torque_nm,
                    "transmitted_mesh_torque_nm": transmitted_mesh_torque_nm,
                    "axial_actuator_force_n": axial_force_n,
                    "capture_reference": decision.capture_reference,
                    "conditional_interface_states": [
                        {
                            "interface_id": "sliding-driver-to-driven",
                            "state": decision.state,
                            "transmission_active": (
                                decision.transmission_active
                            ),
                            "transmitted_effort_si": (
                                transmitted_mesh_torque_nm
                            ),
                            "activation_coordinate_si": (
                                observation.activation_coordinate_si
                            ),
                            "constraints_satisfied": (
                                observation.constraints_satisfied
                            ),
                        }
                    ],
                }
            )

        if index < step_count:
            system.DoStepDynamics(step_s)
            next_driver_speed = float(driver.GetAngVelParent().z)
            next_driven_speed = float(driven.GetAngVelParent().z)
            driver_angle += 0.5 * (driver_speed + next_driver_speed) * step_s
            driven_angle += 0.5 * (driven_speed + next_driven_speed) * step_s
            transmitted_work_j += abs(
                transmitted_mesh_torque_nm * next_driven_speed * step_s
            )

    zero = chrono.ChVector3d(0.0, 0.0, 0.0)
    axial_actuator.SetForce(zero, False)
    for load in (
        driver_input_torque,
        driver_transmission_torque,
        driven_transmission_torque,
        driver_drag_torque,
        driven_drag_torque,
    ):
        load.SetTorque(zero, False)
    set_current_wrench_phase(None)

    state_path = _unique_state_path(samples)
    runtime_summary = _runtime_summary(
        samples, physics_step_count=step_count + 1
    )
    runtime_audit = audit_conditional_interface_runtime_evidence(
        (declaration,), samples, runtime_summary
    )
    post_engagement_driven_span_rad = (
        max(sample["driven_angle_rad"] for sample in samples)
        - min(sample["driven_angle_rad"] for sample in samples)
    )
    checks = {
        "cad_initial_pose_is_warning_not_error": (
            cad_assessment["status"] == "WARNING"
            and cad_assessment["hard_error"] is False
        ),
        "inactive_effort_is_exact_zero": (
            max_inactive_transmitted_torque_nm == 0.0
        ),
        "driven_gear_stationary_before_engagement": (
            max_pre_engagement_driven_angle_rad <= 1e-9
        ),
        "late_geometry_engagement_observed": (
            first_engagement_time_s is not None
            and "INACTIVE" in state_path
            and "APPROACHING" in state_path
            and "ENGAGED" in state_path
        ),
        "driven_gear_rotates_after_engagement": (
            post_engagement_driven_span_rad >= 0.01
            and transmitted_work_j > 0.0
        ),
        "finite_supports_hold_non_free_coordinates": (
            max_support_drift_m <= 0.0002
        ),
        "whole_body_pair_collision_remains_enabled": (
            declaration["whole_body_pair_collision_enabled"] is True
        ),
        "no_ideal_joint_or_gear_link_used": (
            int(system.GetNumLinks()) == 0
        ),
        "runtime_evidence_audit_passes": runtime_audit["passed"],
    }
    passed = all(checks.values())
    return {
        "schema_version": "conditional_sliding_gear_pychrono_testcase_v1",
        "testcase_id": "conditional_sliding_gear_v1",
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "backend": "pychrono_live_rigid_bodies",
        "scope": (
            "runtime transmission gate smoke; not collider-fidelity or "
            "whole-machine certificate evidence"
        ),
        "cad_initial_assessment": cad_assessment,
        "physical_interface_policy": policy,
        "functional_interface_geometry_binding": geometry_binding,
        "conditional_interface_runtime": runtime_summary,
        "conditional_interface_runtime_audit": runtime_audit,
        "state_path": state_path,
        "first_engagement_time_s": first_engagement_time_s,
        "max_inactive_transmitted_torque_nm": (
            max_inactive_transmitted_torque_nm
        ),
        "max_pre_engagement_driven_angle_rad": (
            max_pre_engagement_driven_angle_rad
        ),
        "post_engagement_driven_span_rad": post_engagement_driven_span_rad,
        "max_support_drift_m": max_support_drift_m,
        "transmitted_work_j": transmitted_work_j,
        "ideal_link_count": int(system.GetNumLinks()),
        "checks": checks,
        "samples": samples,
    }


def run_pychrono_sliding_gear_engagement_testcase(
    *,
    duration_s: float = 1.20,
    step_s: float = 0.0005,
    sample_interval_s: float = 0.005,
) -> dict[str, Any]:
    program_phases = (
        {
            "phase_id": "testcase_powered_interval",
            "duration_s": float(duration_s),
        },
    )
    ledger = WrenchLedger(
        plugin_id=_TESTCASE_WRENCH_PLUGIN_ID,
        capability_registration=_TESTCASE_WRENCH_REGISTRATION,
        policies=_TESTCASE_WRENCH_POLICIES,
        declared_input_body_ids=("driver_gear",),
        declared_output_body_ids=("driven_gear",),
        branch="physical_testcase",
        load_case_id="conditional_sliding_gear_v1",
        program_phases=program_phases,
    )
    with activate_wrench_ledger(ledger):
        try:
            payload = _run_pychrono_sliding_gear_engagement_testcase_active(
                duration_s=duration_s,
                step_s=step_s,
                sample_interval_s=sample_interval_s,
            )
            executed_phases = [dict(item) for item in program_phases]
            ledger.bind_executed_schedule(executed_phases)
            payload["executed_phases"] = executed_phases
            payload["wrench_audit"] = ledger.verify_for_certification()
            payload["executed_load_sha256"] = payload["wrench_audit"][
                "executed_load_sha256"
            ]
            return payload
        finally:
            set_current_wrench_phase(None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the conditional sliding-gear harness testcase"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON output path",
    )
    parser.add_argument(
        "--backend",
        choices=("deterministic", "pychrono"),
        default="deterministic",
        help="execution backend",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.backend == "pychrono":
        result = run_pychrono_sliding_gear_engagement_testcase()
    else:
        result = run_sliding_gear_engagement_testcase()
    if args.output is not None:
        write_json_atomic(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
