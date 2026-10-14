from __future__ import annotations

"""Trusted behavior gates for the bicycle ground-roll task."""

import math
from typing import Any, Mapping

from .adapters import adapter_from_dict, get_adapter
from .bicycle_ground_roll_spec import (
    BICYCLE_GROUND_ROLL_PLUGIN_ID,
    BicycleGroundRollSpec,
    bicycle_ground_roll_from_dict,
)
from .io import canonical_sha256
from .models import LoadCase, MechanismPlan, MechanismValidationError


def _fail(code: str, message: str) -> MechanismValidationError:
    return MechanismValidationError(code, message)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise _fail("BICYCLE_GROUND_ROLL_TRACE_NUMBER_INVALID", f"{label} must be finite")
    return float(value)


def _spec(plan: MechanismPlan) -> BicycleGroundRollSpec:
    adapter = (
        adapter_from_dict(plan.adapter_spec)
        if plan.adapter_spec is not None
        else get_adapter(str(plan.adapter_id))
    )
    raw = adapter.family_spec
    if isinstance(raw, Mapping):
        raw = {key: value for key, value in raw.items() if key != "collision_geometry"}
    return bicycle_ground_roll_from_dict(raw)


_REQUIRED_FIELDS = (
    "rear_wheel_angle_rad",
    "front_wheel_angle_rad",
    "chassis_position_m",
    "rear_wheel_speed_rad_s",
    "front_wheel_speed_rad_s",
    "chassis_speed_m_s",
    "rolling_slip_ratio",
    "rear_axle_drift_m",
    "front_axle_drift_m",
    "chassis_guide_transverse_drift_m",
    "max_off_axis_rotation_rad",
    "max_penetration_m",
    "max_ground_separation_m",
)


def _expected_phase_at_time(load_case: LoadCase, time_s: float) -> str | None:
    """Return the validator-owned phase for a saved post-step timestamp."""

    if math.isclose(time_s, 0.0, rel_tol=0.0, abs_tol=1.0e-12):
        return "initial"
    boundary_s = 0.0
    for phase in load_case.phases:
        boundary_s += phase.duration_s
        # The runtime records a phase after advancing dynamics.  A sample on a
        # phase boundary therefore belongs to the phase that just completed.
        if time_s <= boundary_s + 1.0e-9:
            return phase.phase_id
    return None


def _validate_trace(
    trace: Mapping[str, Any],
    *,
    branch: str,
    plan: MechanismPlan,
    load_case: LoadCase,
    spec: BicycleGroundRollSpec,
) -> list[dict[str, Any]]:
    expected = {
        "status": "COMPLETED",
        "branch": branch,
        "mechanism_plugin_id": BICYCLE_GROUND_ROLL_PLUGIN_ID,
        "package_binding_sha256": plan.package_binding_sha256,
        "capsule_id": plan.capsule_id,
        "load_case_id": load_case.load_case_id,
        "plan_sha256": canonical_sha256(plan.to_dict()),
        "load_case_sha256": canonical_sha256(load_case.to_dict()),
    }
    mismatches = [key for key, value in expected.items() if trace.get(key) != value]
    if mismatches:
        raise _fail(
            "BICYCLE_GROUND_ROLL_TRACE_IDENTITY_MISMATCH",
            f"{branch} trace differs at: {', '.join(mismatches)}",
        )
    samples = trace.get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        raise _fail("BICYCLE_GROUND_ROLL_TRACE_SAMPLES_MISSING", f"{branch} trace has too few samples")
    prior = -math.inf
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise _fail("BICYCLE_GROUND_ROLL_TRACE_SAMPLE_INVALID", f"{branch}.samples[{index}] is invalid")
        time_s = _finite(sample.get("time_s"), f"{branch}.samples[{index}].time_s")
        if time_s <= prior:
            raise _fail("BICYCLE_GROUND_ROLL_TRACE_TIME_INVALID", "saved times must be strictly increasing")
        prior = time_s
        expected_phase = _expected_phase_at_time(load_case, time_s)
        if expected_phase is None or sample.get("phase_id") != expected_phase:
            raise _fail(
                "BICYCLE_GROUND_ROLL_TRACE_PHASE_WINDOW_INVALID",
                f"{branch}.samples[{index}] at {time_s!r} s must be in {expected_phase!r}",
            )
        for field in _REQUIRED_FIELDS:
            _finite(sample.get(field), f"{branch}.samples[{index}].{field}")
        body_poses = sample.get("body_poses")
        if not isinstance(body_poses, Mapping) or not {
            spec.ground_body_id,
            spec.chassis_body_id,
            spec.steering_body_id,
            spec.rear_wheel_body_id,
            spec.front_wheel_body_id,
        }.issubset(body_poses):
            raise _fail(
                "BICYCLE_GROUND_ROLL_BODY_POSES_MISSING",
                f"{branch}.samples[{index}] lacks all five body poses",
            )
        for field in ("rear_ground_contact_count", "front_ground_contact_count", "expected_contact_count"):
            value = sample.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise _fail("BICYCLE_GROUND_ROLL_CONTACT_EVIDENCE_INVALID", f"{branch}.samples[{index}].{field} is invalid")
        if sample["expected_contact_count"] != min(
            sample["rear_ground_contact_count"],
            sample["front_ground_contact_count"],
        ):
            raise _fail(
                "BICYCLE_GROUND_ROLL_CONTACT_EVIDENCE_INVALID",
                f"{branch}.samples[{index}].expected_contact_count must be the paired wheel-ground count",
            )
    return samples


def _metrics(samples: list[dict[str, Any]], spec: BicycleGroundRollSpec) -> dict[str, Any]:
    powered = [sample for sample in samples if sample["phase_id"] == "rear_wheel_powered_drive"]
    if len(powered) < 3:
        raise _fail("BICYCLE_GROUND_ROLL_POWERED_WINDOW_MISSING", "powered window has fewer than three samples")
    baseline_index = max(0, samples.index(powered[0]) - 1)
    baseline = samples[baseline_index]
    front_rotation = max(
        abs(float(sample["front_wheel_angle_rad"]) - float(baseline["front_wheel_angle_rad"]))
        for sample in powered
    )
    chassis_travel = max(
        abs(float(sample["chassis_position_m"]) - float(baseline["chassis_position_m"]))
        for sample in powered
    )
    significant = [
        sample for sample in powered
        if abs(float(sample["chassis_position_m"]) - float(baseline["chassis_position_m"])) >= 0.10
    ]
    return {
        "powered_front_rotation_rad": front_rotation,
        "powered_chassis_travel_m": chassis_travel,
        "powered_rear_rotation_rad": max(
            abs(float(sample["rear_wheel_angle_rad"]) - float(baseline["rear_wheel_angle_rad"]))
            for sample in powered
        ),
        "max_rolling_slip_ratio": max(
            (float(sample["rolling_slip_ratio"]) for sample in significant),
            default=math.inf,
        ),
        "max_axle_drift_m": max(
            max(float(sample["rear_axle_drift_m"]), float(sample["front_axle_drift_m"]))
            for sample in samples
        ),
        "max_guide_drift_m": max(float(sample["chassis_guide_transverse_drift_m"]) for sample in samples),
        "max_off_axis_rotation_rad": max(float(sample["max_off_axis_rotation_rad"]) for sample in samples),
        "max_ground_penetration_m": max(float(sample["max_penetration_m"]) for sample in samples),
        "max_ground_separation_m": max(
            float(sample["max_ground_separation_m"]) for sample in powered
        ),
        "rear_ground_contact_coverage": sum(
            int(sample["rear_ground_contact_count"]) > 0 for sample in powered
        ) / len(powered),
        "front_ground_contact_coverage": sum(
            int(sample["front_ground_contact_count"]) > 0 for sample in powered
        ) / len(powered),
        "powered_contact_coverage_ratio": sum(
            int(sample["expected_contact_count"]) > 0 for sample in powered
        ) / len(powered),
        "rear_ground_contact_observed": any(int(sample["rear_ground_contact_count"]) > 0 for sample in powered),
        "front_ground_contact_observed": any(int(sample["front_ground_contact_count"]) > 0 for sample in powered),
    }


def _semantic_failures(trace: Mapping[str, Any], spec: BicycleGroundRollSpec) -> list[str]:
    failures: list[str] = []
    engine = trace.get("engine")
    semantics = trace.get("branch_semantics")
    geometry = trace.get("functional_geometry")
    if (
        not isinstance(engine, Mapping)
        or engine.get("internal_ideal_joint_count") != 0
        or engine.get("declared_support_links")
        != [
            "ChLinkLockPrismatic",
            "ChLinkLockRevolute",
            "ChLinkLockRevolute",
            "ChLinkLockLock",
        ]
        or engine.get("declared_axle_revolute_joint_count") != 2
        or engine.get("declared_upright_guide_count") != 1
        or engine.get("declared_straight_steering_lock_count") != 1
        or engine.get("contact_bypass_link_count") != 0
        or engine.get("contact_model") != "native_finite_coulomb_wheel_ground_contact"
    ):
        failures.append("bicycle_ground_roll_axle_or_contact_model_invalid")
    if (
        not isinstance(semantics, Mapping)
        or semantics.get("replaced_contacts_disabled") is not False
        or semantics.get("pose_setters_used") is not False
        or semantics.get("direct_front_wheel_actuation_used") is not False
        or semantics.get("front_wheel_passive") is not True
        or semantics.get("chassis_fixed") is not False
        or semantics.get("chassis_fixed_during_contact_settle") is not True
        or semantics.get("chassis_released_and_velocities_zeroed_before_power")
        is not True
        or semantics.get("rolling_constraint_used") is not False
        or semantics.get("declared_axle_revolute_joints_used") is not True
        or semantics.get("contact_bypass_ideal_links_forbidden") is not True
        or semantics.get("steering_fixture_mode")
        != "declared_ground_roll_task_lock_straight_ahead"
        or semantics.get("task_input_body_id") != spec.rear_wheel_body_id
        or semantics.get("unique_task_output_body_id") != spec.front_wheel_body_id
    ):
        failures.append("bicycle_ground_roll_branch_semantics_invalid")
    if (
        not isinstance(geometry, Mapping)
        or geometry.get("passed") is not True
        or geometry.get("mechanism_kind") != "bicycle_ground_roll"
        or geometry.get("package_binding_sha256")
        != trace.get("package_binding_sha256")
        or geometry.get("wheel_geometry_bound_to_step") is not True
        or geometry.get("ground_plane_bound_to_step") is not True
        or geometry.get("visible_upright_fixture_bound") is not True
    ):
        failures.append("bicycle_ground_roll_functional_geometry_not_bound")
    audit = trace.get("wrench_audit")
    if (
        not isinstance(audit, Mapping)
        or audit.get("passed") is not True
        or audit.get("direct_output_drive_detected") is not False
        or set(audit.get("task_input_bodies_observed") or ()) != {spec.rear_wheel_body_id}
    ):
        failures.append("bicycle_ground_roll_actuation_provenance_invalid")
    records = audit.get("records") if isinstance(audit, Mapping) else None
    if not isinstance(records, list) or any(
        isinstance(record, Mapping)
        and record.get("target_body") == spec.front_wheel_body_id
        and record.get("task_input") is True
        for record in records
    ):
        failures.append("bicycle_ground_roll_front_wheel_direct_actuation_detected")
    return failures


def _behavior_failures(
    metrics: Mapping[str, Any],
    spec: BicycleGroundRollSpec,
    *,
    require_physical_contacts: bool = True,
) -> list[str]:
    failures: list[str] = []
    if metrics["powered_front_rotation_rad"] < spec.minimum_front_rotation_rad:
        failures.append("bicycle_ground_roll_front_rotation_shortfall")
    if metrics["powered_chassis_travel_m"] < spec.minimum_chassis_travel_m:
        failures.append("bicycle_ground_roll_chassis_travel_shortfall")
    if metrics["max_rolling_slip_ratio"] > spec.maximum_rolling_slip_ratio:
        failures.append("bicycle_ground_roll_slip_exceeded")
    if metrics["max_axle_drift_m"] > spec.maximum_axle_drift_m:
        failures.append("bicycle_ground_roll_axle_drift_exceeded")
    if metrics["max_guide_drift_m"] > spec.maximum_guide_drift_m:
        failures.append("bicycle_ground_roll_guide_drift_exceeded")
    if metrics["max_off_axis_rotation_rad"] > spec.maximum_off_axis_rotation_rad:
        failures.append("bicycle_ground_roll_off_axis_rotation_exceeded")
    if metrics["max_ground_penetration_m"] > spec.maximum_ground_penetration_m:
        failures.append("bicycle_ground_roll_ground_penetration_exceeded")
    if metrics["max_ground_separation_m"] > spec.maximum_ground_separation_m:
        failures.append("bicycle_ground_roll_ground_separation_exceeded")
    if require_physical_contacts:
        if (
            metrics["powered_contact_coverage_ratio"]
            < spec.minimum_powered_contact_coverage
        ):
            failures.append("bicycle_ground_roll_contact_coverage_incomplete")
    return failures


def _thresholds(spec: BicycleGroundRollSpec) -> dict[str, Any]:
    return {
        "minimum_front_rotation_rad": spec.minimum_front_rotation_rad,
        "minimum_chassis_travel_m": spec.minimum_chassis_travel_m,
        "maximum_rolling_slip_ratio": spec.maximum_rolling_slip_ratio,
        "maximum_axle_drift_m": spec.maximum_axle_drift_m,
        "maximum_guide_drift_m": spec.maximum_guide_drift_m,
        "maximum_off_axis_rotation_rad": spec.maximum_off_axis_rotation_rad,
        "maximum_ground_penetration_m": spec.maximum_ground_penetration_m,
        "maximum_ground_separation_m": spec.maximum_ground_separation_m,
        "minimum_powered_contact_coverage": spec.minimum_powered_contact_coverage,
        "source": "public_frozen_bicycle_ground_roll_v1",
    }


def _requirement_checks(
    trace: Mapping[str, Any],
    spec: BicycleGroundRollSpec,
    metrics: Mapping[str, Any],
    *,
    physical_failures: list[str],
    strict_non_claims: Mapping[str, bool],
) -> dict[str, dict[str, Any]]:
    engine = trace.get("engine")
    engine = engine if isinstance(engine, Mapping) else {}
    semantics = trace.get("branch_semantics")
    semantics = semantics if isinstance(semantics, Mapping) else {}
    geometry = trace.get("functional_geometry")
    geometry = geometry if isinstance(geometry, Mapping) else {}
    collision = trace.get("collision_geometry")
    collision = collision if isinstance(collision, Mapping) else {}
    wrench = trace.get("wrench_audit")
    wrench = wrench if isinstance(wrench, Mapping) else {}
    collision_bodies = [
        item
        for item in collision.get("bodies", [])
        if isinstance(item, Mapping) and item.get("body_id")
    ]
    expected_body_ids = {body.body_id for body in spec.bodies}
    declared_group_ids = [
        group_id for body in spec.bodies for group_id in body.group_ids
    ]
    collision_body_ids = {str(body["body_id"]) for body in collision_bodies}
    collision_components_complete = bool(collision_bodies) and all(
        isinstance(body.get("components"), list) and bool(body["components"])
        for body in collision_bodies
    )
    step_geometry_passed = bool(
        geometry.get("passed") is True
        and geometry.get("mechanism_kind") == "bicycle_ground_roll"
        and geometry.get("package_binding_sha256")
        == trace.get("package_binding_sha256")
        and geometry.get("wheel_geometry_bound_to_step") is True
        and geometry.get("ground_plane_bound_to_step") is True
        and geometry.get("visible_upright_fixture_bound") is True
    )
    single_source_passed = bool(
        step_geometry_passed
        and collision.get("package_binding_sha256")
        == trace.get("package_binding_sha256")
        and isinstance(collision.get("assembly_sha256"), str)
        and len(str(collision["assembly_sha256"])) == 64
        and geometry.get("assembly_sha256")
        == collision.get("assembly_sha256")
        and isinstance(collision.get("assembly_contract_sha256"), str)
        and len(str(collision["assembly_contract_sha256"])) == 64
        and collision_components_complete
    )
    ownership_passed = bool(
        collision_body_ids == expected_body_ids
        and len(collision_bodies) == len(expected_body_ids)
        and collision_components_complete
        and len(expected_body_ids) == len(spec.bodies)
        and len(set(declared_group_ids)) == len(declared_group_ids)
    )
    native_contact_passed = bool(
        engine.get("contact_model")
        == "native_finite_coulomb_wheel_ground_contact"
        and engine.get("contact_bypass_link_count") == 0
        and engine.get("declared_axle_revolute_joint_count") == 2
        and semantics.get("replaced_contacts_disabled") is False
        and semantics.get("pose_setters_used") is False
        and semantics.get("rolling_constraint_used") is False
        and semantics.get("declared_axle_revolute_joints_used") is True
        and semantics.get("contact_bypass_ideal_links_forbidden") is True
        and metrics.get("rear_ground_contact_observed") is True
        and metrics.get("front_ground_contact_observed") is True
    )
    passive_front_passed = bool(
        semantics.get("direct_front_wheel_actuation_used") is False
        and semantics.get("front_wheel_passive") is True
        and semantics.get("chassis_fixed") is False
        and semantics.get("chassis_fixed_during_contact_settle") is True
        and semantics.get("chassis_released_and_velocities_zeroed_before_power")
        is True
        and semantics.get("task_input_body_id") == spec.rear_wheel_body_id
        and semantics.get("unique_task_output_body_id")
        == spec.front_wheel_body_id
        and wrench.get("passed") is True
        and wrench.get("direct_output_drive_detected") is False
        and set(wrench.get("task_input_bodies_observed") or ())
        == {spec.rear_wheel_body_id}
    )
    guided_boundary_passed = bool(
        semantics.get("upright_fixture_mode")
        == "visible_declared_prismatic_test_stand_guide_free_in_travel_axis"
        and semantics.get("steering_fixture_mode")
        == "declared_ground_roll_task_lock_straight_ahead"
        and strict_non_claims.get("unassisted_balance_certified") is False
        and strict_non_claims.get("tire_deformation_or_wear_certified") is False
        and strict_non_claims.get("strength_or_fatigue_certified") is False
        and strict_non_claims.get("steering_dynamics_certified") is False
    )

    def check(passed: bool, *roles: str) -> dict[str, Any]:
        return {"passed": passed, "evidence_roles": list(roles)}

    return {
        "validator.mechanical.single_source_cad": check(
            single_source_passed,
            "validation.assembly_contract",
            "validation.original_step_collision_geometry",
        ),
        "validator.mechanical.rigid_group_ownership": check(
            ownership_passed,
            "validation.assembly_contract",
            "validation.body_pose_trace",
        ),
        "validator.bicycle.native_ground_contact": check(
            native_contact_passed,
            "validation.contact_trace",
        ),
        "validator.bicycle.front_wheel_passive": check(
            passive_front_passed,
            "validation.body_pose_trace",
            "validation.wrench_audit",
        ),
        "validator.bicycle.ground_roll_step_geometry": check(
            step_geometry_passed,
            "validation.initial_functional_geometry",
            "validation.original_step_collision_geometry",
        ),
        "validator.bicycle.ground_roll_behavior": check(
            not physical_failures,
            "validation.contact_trace",
            "validation.saved_pose_geometry_audit",
        ),
        "policy.bicycle.guided_test_boundaries": check(
            guided_boundary_passed,
            "validation.terminal_verdict",
        ),
    }


def physical_bicycle_ground_roll_behavior(
    plan: MechanismPlan,
    load_case: LoadCase,
    trace: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    spec = _spec(plan)
    samples = _validate_trace(trace, branch="physical", plan=plan, load_case=load_case, spec=spec)
    metrics = _metrics(samples, spec)
    return metrics, _thresholds(spec), [*_semantic_failures(trace, spec), *_behavior_failures(metrics, spec)]


def compare_bicycle_ground_roll_trace_pair(
    plan: MechanismPlan,
    load_case: LoadCase,
    physical_trace: dict[str, Any],
    reduced_trace: dict[str, Any],
) -> dict[str, Any]:
    try:
        spec = _spec(plan)
        physical_samples = _validate_trace(
            physical_trace, branch="physical", plan=plan, load_case=load_case, spec=spec
        )
        reduced_samples = _validate_trace(
            reduced_trace, branch="reduced", plan=plan, load_case=load_case, spec=spec
        )
        physical_metrics = _metrics(physical_samples, spec)
        reduced_metrics = _metrics(reduced_samples, spec)
    except MechanismValidationError as exc:
        return {
            "schema_version": "0.1",
            "capsule_id": plan.capsule_id,
            "load_case_id": load_case.load_case_id,
            "purpose": load_case.purpose,
            "program": load_case.program,
            "required_for_certificate": load_case.required_for_certificate,
            "verdict": "PHYSICS_INCONCLUSIVE",
            "failure_reasons": [exc.code],
            "detail": str(exc),
        }
    semantic = _semantic_failures(physical_trace, spec)
    physical_failures = _behavior_failures(physical_metrics, spec)
    reduced_failures = _behavior_failures(
        reduced_metrics,
        spec,
        require_physical_contacts=False,
    )
    verdict = (
        "PHYSICS_INCONCLUSIVE" if semantic else
        "PHYSICAL_FAILURE" if physical_failures else
        "CLAIM_MISMATCH" if reduced_failures else
        "PASS"
    )
    strict_non_claims = {
        "ideal_no_slip_constraint_is_physical_evidence": False,
        "unassisted_balance_certified": False,
        "tire_deformation_or_wear_certified": False,
        "strength_or_fatigue_certified": False,
        "steering_dynamics_certified": False,
        "front_wheel_direct_actuation_allowed": False,
    }
    return {
        "schema_version": "0.1",
        "capsule_id": plan.capsule_id,
        "load_case_id": load_case.load_case_id,
        "purpose": load_case.purpose,
        "program": load_case.program,
        "required_for_certificate": load_case.required_for_certificate,
        "verdict": verdict,
        "failure_reasons": [*semantic, *physical_failures, *reduced_failures],
        "thresholds": _thresholds(spec),
        "physical_metrics": physical_metrics,
        "reduced_metrics": reduced_metrics,
        "strict_non_claims": strict_non_claims,
        "requirement_checks": _requirement_checks(
            physical_trace,
            spec,
            physical_metrics,
            physical_failures=physical_failures,
            strict_non_claims=strict_non_claims,
        ),
    }
