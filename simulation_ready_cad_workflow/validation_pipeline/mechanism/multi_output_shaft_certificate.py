from __future__ import annotations

from typing import Any

from .adapters import AdapterSpec
from .io import canonical_sha256
from .models import LoadCase
from .multi_output_shaft_spec import multi_output_shaft_from_dict


def multi_output_shaft_load_envelope(
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
    body_ids = sorted(
        {
            body_id
            for metric in metrics
            for body_id in (
                metric.get("shaft_angle_ranges_rad") or {}
            )
        }
    )
    shaft_ranges = {
        body_id: {
            "min": min(
                metric["shaft_angle_ranges_rad"][body_id]["min"]
                for metric in metrics
                if body_id in metric.get("shaft_angle_ranges_rad", {})
            ),
            "max": max(
                metric["shaft_angle_ranges_rad"][body_id]["max"]
                for metric in metrics
                if body_id in metric.get("shaft_angle_ranges_rad", {})
            ),
            "source": "observed_physical_multi_output_shaft_angle",
        }
        for body_id in body_ids
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
            "max": (
                max(torque_magnitudes) if torque_magnitudes else 0.0
            ),
        },
        "shaft_angle_ranges_rad": shaft_ranges,
        "max_observed_support_drift_m": (
            max(item["max_support_drift_m"] for item in metrics)
            if metrics
            else None
        ),
        "max_observed_off_axis_rotation_rad": (
            max(item["max_off_axis_rotation_rad"] for item in metrics)
            if metrics
            else None
        ),
        "max_observed_mesh_error_rad": (
            max(item["max_mesh_error_rad"] for item in metrics)
            if metrics
            else None
        ),
    }


def multi_output_shaft_certificate_model(
    adapter: AdapterSpec,
) -> dict[str, Any]:
    spec = multi_output_shaft_from_dict(adapter.family_spec)
    return {
        "adapter_id": adapter.adapter_id,
        "mechanism_id": spec.mechanism_id,
        "mechanism_kind": spec.mechanism_kind,
        "topology_sha256": canonical_sha256(spec.to_dict()),
        "body_count": len(spec.bodies),
        "shaft_count": len(spec.shafts),
        "gear_mesh_count": len(spec.gear_meshes),
        "fixed_body_id": spec.fixed_body_id,
        "input_body_id": spec.input_body_id,
        "output_body_ids": list(spec.output_body_ids),
        "physical_support": (
            "finite_conditional_ChLoadBodyBodyBushingGeneric"
        ),
        "physical_transmission": (
            "finite_geometry_gated_compliant_right_angle_gear_torque"
        ),
        "direct_output_shaft_actuation": False,
        "reduced_support": "declared_ChLinkLockRevolute_topology",
        "reduced_transmission": "declared_ChLinkLockGear_topology",
        "exact_tooth_impact_or_backlash_certified": False,
    }


def multi_output_shaft_replacement_topology(
    adapter: AdapterSpec,
) -> dict[str, Any]:
    spec = multi_output_shaft_from_dict(adapter.family_spec)
    return {
        "mechanism_id": spec.mechanism_id,
        "family_spec_sha256": canonical_sha256(spec.to_dict()),
        "body_ids": [item.body_id for item in spec.bodies],
        "shaft_ids": [item.shaft_id for item in spec.shafts],
        "gear_mesh_ids": [item.mesh_id for item in spec.gear_meshes],
        "input_body_id": spec.input_body_id,
        "output_body_ids": list(spec.output_body_ids),
        "declared_output_ratios": {
            body_id: spec.declared_ratio_for_body(body_id)
            for body_id in spec.output_body_ids
        },
    }
