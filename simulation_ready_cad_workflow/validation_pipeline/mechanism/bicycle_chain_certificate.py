from __future__ import annotations

from typing import Any

from .adapters import AdapterSpec
from .bicycle_chain_spec import BicycleChainSpec, bicycle_chain_from_dict
from .io import canonical_sha256
from .models import LoadCase


def _bicycle_spec(adapter: AdapterSpec) -> BicycleChainSpec:
    family_spec = adapter.family_spec
    if isinstance(family_spec, dict):
        family_spec = {
            key: value
            for key, value in family_spec.items()
            if key != "collision_geometry"
        }
    return bicycle_chain_from_dict(family_spec)


def bicycle_chain_load_envelope(
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

    def angle_range(
        minimum_field: str,
        maximum_field: str,
        source: str,
    ) -> dict[str, Any] | None:
        if not metrics:
            return None
        return {
            "min": min(float(item[minimum_field]) for item in metrics),
            "max": max(float(item[maximum_field]) for item in metrics),
            "source": source,
        }

    return {
        "moving_mass_kg": {
            "min": min(masses),
            "max": max(masses),
        },
        "load_case_ids": [
            load_case.load_case_id for load_case in load_cases
        ],
        "purposes": sorted(
            {load_case.purpose for load_case in load_cases}
        ),
        "gravity_abs_m_s2": None,
        "applied_torque_nm": {
            "min_nonzero": (
                min(torque_magnitudes) if torque_magnitudes else 0.0
            ),
            "max": max(torque_magnitudes) if torque_magnitudes else 0.0,
        },
        "input_crank_angle_rad": angle_range(
            "input_angle_min_rad",
            "input_angle_max_rad",
            "observed_physical_bicycle_input_crank_angle",
        ),
        "rear_drive_wheel_angle_rad": angle_range(
            "output_angle_min_rad",
            "output_angle_max_rad",
            "observed_physical_bicycle_rear_drive_wheel_angle",
        ),
        "max_observed_support_drift_m": maximum(
            "max_support_drift_m"
        ),
        "max_observed_off_axis_rotation_rad": maximum(
            "max_off_axis_rotation_rad"
        ),
        "max_observed_chain_coordinate_error_m": maximum(
            "max_chain_coordinate_error_m"
        ),
        "max_observed_chain_differential_tension_n": maximum(
            "max_chain_differential_tension_n"
        ),
        "max_observed_chain_axis_angle_error_rad": maximum(
            "max_chain_axis_angle_error_rad"
        ),
        "max_observed_chain_axial_alignment_error_m": maximum(
            "max_chain_axial_alignment_error_m"
        ),
        "max_observed_support_force_n": maximum(
            "max_support_force_n"
        ),
        "max_observed_support_torque_nm": maximum(
            "max_support_torque_nm"
        ),
        "max_observed_powered_output_excursion_rad": maximum(
            "powered_output_excursion_rad"
        ),
        "max_observed_powered_ratio_pointwise_error": maximum(
            "max_powered_ratio_pointwise_error"
        ),
        "max_observed_powered_ratio_incremental_error": maximum(
            "max_powered_ratio_incremental_error"
        ),
        "max_observed_final_output_home_error_rad": maximum(
            "final_output_home_error_rad"
        ),
    }


def bicycle_chain_certificate_model(
    adapter: AdapterSpec,
) -> dict[str, Any]:
    spec = _bicycle_spec(adapter)
    return {
        "adapter_id": adapter.adapter_id,
        "mechanism_id": spec.mechanism_id,
        "mechanism_kind": spec.mechanism_kind,
        "topology_sha256": canonical_sha256(spec.to_dict()),
        "body_count": len(spec.bodies),
        "fixed_body_id": spec.fixed_body_id,
        "input_body_id": spec.input_body_id,
        "output_body_id": spec.output_body_id,
        "input_sprocket_id": spec.input_sprocket.sprocket_id,
        "output_sprocket_id": spec.output_sprocket.sprocket_id,
        "input_pitch_radius_m": spec.input_pitch_radius_m,
        "output_pitch_radius_m": spec.output_pitch_radius_m,
        "declared_signed_output_per_input": spec.signed_output_per_input,
        "physical_support": (
            "finite_bearing_force_and_alignment_torque"
        ),
        "physical_transmission": (
            "bounded_geometry_gated_compliant_differential_tension"
        ),
        "ideal_reference": (
            "signed_open_chain_pitch_mapping_non_contact_reference_only"
        ),
        "moving_chain_path_evidence_required": True,
        "fixed_spatial_sprocket_flex_zones_required": True,
        "direct_output_actuation": False,
        "individual_link_tooth_contact_simulated": False,
    }


def bicycle_chain_replacement_topology(
    adapter: AdapterSpec,
) -> dict[str, Any]:
    spec = _bicycle_spec(adapter)
    return {
        "mechanism_id": spec.mechanism_id,
        "mechanism_kind": spec.mechanism_kind,
        "family_spec_sha256": canonical_sha256(spec.to_dict()),
        "body_ids": [item.body_id for item in spec.bodies],
        "fixed_body_id": spec.fixed_body_id,
        "input_body_id": spec.input_body_id,
        "output_body_id": spec.output_body_id,
        "input_sprocket_id": spec.input_sprocket.sprocket_id,
        "output_sprocket_id": spec.output_sprocket.sprocket_id,
        "input_pitch_radius_m": spec.input_pitch_radius_m,
        "output_pitch_radius_m": spec.output_pitch_radius_m,
        "declared_signed_output_per_input": spec.signed_output_per_input,
        "ideal_reference_kind": "signed_open_chain_pitch_mapping",
        "contact_or_dynamic_response_claimed": False,
    }
