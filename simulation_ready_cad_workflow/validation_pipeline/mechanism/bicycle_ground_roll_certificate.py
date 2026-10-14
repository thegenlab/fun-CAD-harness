from __future__ import annotations

"""Certificate records for the native-contact bicycle rolling task."""

from typing import Any, Mapping

from .adapters import AdapterSpec
from .bicycle_ground_roll_spec import (
    BicycleGroundRollSpec,
    bicycle_ground_roll_from_dict,
)
from .io import canonical_sha256
from .models import LoadCase


def _spec(adapter: AdapterSpec) -> BicycleGroundRollSpec:
    raw = adapter.family_spec
    if isinstance(raw, Mapping):
        raw = {key: item for key, item in raw.items() if key != "collision_geometry"}
    return bicycle_ground_roll_from_dict(raw)


def bicycle_ground_roll_load_envelope(
    load_cases: tuple[LoadCase, ...],
    comparisons: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    torque_magnitudes = [
        phase.magnitude_si
        for load_case in load_cases
        for phase in load_case.phases
        if phase.wrench_kind == "torque"
    ]
    masses = [load_case.moving_mass_kg for load_case in load_cases]
    metrics = [
        item["physical_metrics"]
        for item in comparisons
        if isinstance(item.get("physical_metrics"), dict)
    ]

    def maximum(field: str) -> float | None:
        return max(float(item[field]) for item in metrics) if metrics else None

    return {
        "moving_mass_kg": {"min": min(masses), "max": max(masses)},
        "load_case_ids": [item.load_case_id for item in load_cases],
        "purposes": sorted({item.purpose for item in load_cases}),
        "gravity_abs_m_s2": None,
        "applied_rear_wheel_torque_nm": {
            "min_nonzero": min(torque_magnitudes) if torque_magnitudes else 0.0,
            "max": max(torque_magnitudes) if torque_magnitudes else 0.0,
        },
        "max_observed_powered_front_rotation_rad": maximum(
            "powered_front_rotation_rad"
        ),
        "max_observed_powered_chassis_travel_m": maximum(
            "powered_chassis_travel_m"
        ),
        "max_observed_rolling_slip_ratio": maximum("max_rolling_slip_ratio"),
        "max_observed_axle_drift_m": maximum("max_axle_drift_m"),
        "max_observed_guide_drift_m": maximum("max_guide_drift_m"),
        "max_observed_off_axis_rotation_rad": maximum(
            "max_off_axis_rotation_rad"
        ),
        "max_observed_ground_penetration_m": maximum(
            "max_ground_penetration_m"
        ),
        "max_observed_ground_separation_m": maximum(
            "max_ground_separation_m"
        ),
        "minimum_observed_rear_contact_coverage": (
            min(float(item["rear_ground_contact_coverage"]) for item in metrics)
            if metrics
            else None
        ),
        "minimum_observed_front_contact_coverage": (
            min(float(item["front_ground_contact_coverage"]) for item in metrics)
            if metrics
            else None
        ),
        "rear_ground_contact_observed": bool(metrics) and all(
            item.get("rear_ground_contact_observed") is True for item in metrics
        ),
        "front_ground_contact_observed": bool(metrics) and all(
            item.get("front_ground_contact_observed") is True for item in metrics
        ),
    }


def bicycle_ground_roll_certificate_model(adapter: AdapterSpec) -> dict[str, Any]:
    spec = _spec(adapter)
    return {
        "adapter_id": adapter.adapter_id,
        "mechanism_id": spec.mechanism_id,
        "mechanism_kind": spec.mechanism_kind,
        "topology_sha256": canonical_sha256(spec.to_dict()),
        "body_count": len(spec.bodies),
        "ground_body_id": spec.ground_body_id,
        "chassis_body_id": spec.chassis_body_id,
        "steering_body_id": spec.steering_body_id,
        "rear_wheel_body_id": spec.rear_wheel_body_id,
        "front_wheel_body_id": spec.front_wheel_body_id,
        "physical_support": "declared_revolute_axle_bearings_visible_prismatic_upright_guide_and_straight_steering_task_lock",
        "physical_transmission": "native_finite_coulomb_wheel_ground_contact",
        "ideal_reference": "straight_line_no_slip_reference_only",
        "rear_wheel_only_task_actuation": True,
        "front_wheel_passive": True,
        "chassis_translation_required": True,
        "unassisted_balance_certified": False,
    }


def bicycle_ground_roll_replacement_topology(
    adapter: AdapterSpec,
) -> dict[str, Any]:
    spec = _spec(adapter)
    return {
        "mechanism_id": spec.mechanism_id,
        "mechanism_kind": spec.mechanism_kind,
        "family_spec_sha256": canonical_sha256(spec.to_dict()),
        "body_ids": [item.body_id for item in spec.bodies],
        "ground_body_id": spec.ground_body_id,
        "chassis_body_id": spec.chassis_body_id,
        "steering_body_id": spec.steering_body_id,
        "rear_wheel_body_id": spec.rear_wheel_body_id,
        "front_wheel_body_id": spec.front_wheel_body_id,
        "ground_plane_height_m": spec.ground_plane_height_m,
        "rear_wheel_radius_m": spec.rear_wheel_radius_m,
        "front_wheel_radius_m": spec.front_wheel_radius_m,
        "ideal_reference_kind": "straight_line_no_slip_reference",
        "contact_or_dynamic_response_claimed": False,
    }
