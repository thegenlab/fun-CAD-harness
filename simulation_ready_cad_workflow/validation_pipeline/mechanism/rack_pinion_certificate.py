from __future__ import annotations

from typing import Any

from .adapters import AdapterSpec
from .io import canonical_sha256
from .models import LoadCase
from .rack_pinion_spec import rack_pinion_from_dict


def rack_pinion_load_envelope(
    load_cases: tuple[LoadCase, ...], comparisons: tuple[dict[str, Any], ...]
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
    progress = (
        {
            "min": min(item["min_progress_m"] for item in metrics),
            "max": max(item["max_progress_m"] for item in metrics),
            "source": "observed_physical_branch_rack_progress",
        }
        if metrics
        else None
    )
    return {
        "moving_mass_kg": {"min": min(masses), "max": max(masses)},
        "load_case_ids": [item.load_case_id for item in load_cases],
        "purposes": sorted({item.purpose for item in load_cases}),
        "gravity_abs_m_s2": None,
        "applied_torque_nm": {
            "min_nonzero": min(torque_magnitudes) if torque_magnitudes else 0.0,
            "max": max(torque_magnitudes) if torque_magnitudes else 0.0,
        },
        "generalized_rack_progress_m": progress,
        "max_observed_support_drift_m": (
            max(item["max_support_drift_m"] for item in metrics) if metrics else None
        ),
        "max_observed_coupling_error_m": (
            max(item["max_coupling_error_m"] for item in metrics) if metrics else None
        ),
        "max_observed_off_axis_rotation_rad": (
            max(item["max_off_axis_rotation_rad"] for item in metrics)
            if metrics
            else None
        ),
    }


def rack_pinion_certificate_model(adapter: AdapterSpec) -> dict[str, Any]:
    spec = rack_pinion_from_dict(adapter.family_spec)
    return {
        "adapter_id": adapter.adapter_id,
        "mechanism_kind": "rack_pinion",
        "mechanism_id": spec.mechanism_id,
        "topology_sha256": canonical_sha256(spec.to_dict()),
        "body_count": len(spec.bodies),
        "joint_count": 3,
        "loop_count": 1,
        "fixed_body_id": spec.fixed_body_id,
        "pinion_body_id": spec.pinion_body_id,
        "rack_body_id": spec.rack_body_id,
        "pinion_pitch_radius_m": spec.pinion_pitch_radius_m,
        "rack_axis_abs": list(spec.rack_axis_abs),
        "pinion_axis_abs": list(spec.pinion_axis_abs),
        "physical_support": "finite_bushings_and_finite_penalty_gear_mesh",
        "reduced_support": "ChLinkLockRevolute_ChLinkLockPrismatic_ChLinkMateRackPinion",
        "exact_tooth_or_guide_surface_contact_certified": False,
    }


def rack_pinion_replacement_topology(adapter: AdapterSpec) -> dict[str, Any]:
    spec = rack_pinion_from_dict(adapter.family_spec)
    return {
        "mechanism_kind": "rack_pinion",
        "mechanism_id": spec.mechanism_id,
        "family_spec_sha256": canonical_sha256(spec.to_dict()),
        "body_ids": [item.body_id for item in spec.bodies],
        "joint_ids": ["pinion_revolute", "rack_prismatic", "rack_pinion_coupling"],
        "loop_count": 1,
        "pinion_body_id": spec.pinion_body_id,
        "rack_body_id": spec.rack_body_id,
        "output_progress_range_m": [spec.home_progress_m, spec.outbound_progress_m],
        "pinion_pitch_radius_m": spec.pinion_pitch_radius_m,
        "ideal_links": [
            "ChLinkLockRevolute",
            "ChLinkLockPrismatic",
            "ChLinkMateRackPinion",
        ],
    }
