from __future__ import annotations

from typing import Any

from .adapters import AdapterSpec
from .detent_slider_spec import prismatic_detent_latch_from_dict
from .io import canonical_sha256
from .models import LoadCase


def detent_slider_load_envelope(
    load_cases: tuple[LoadCase, ...],
    comparisons: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    force_magnitudes = [
        phase.magnitude_si
        for load_case in load_cases
        for phase in load_case.phases
        if phase.wrench_kind == "force"
    ]
    masses = [load_case.moving_mass_kg for load_case in load_cases]
    metrics = [
        item["physical_metrics"]
        for item in comparisons
        if isinstance(item.get("physical_metrics"), dict)
    ]
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
        "applied_force_n": {
            "min_nonzero": (
                min(force_magnitudes) if force_magnitudes else 0.0
            ),
            "max": max(force_magnitudes) if force_magnitudes else 0.0,
        },
        "blade_position_range_m": (
            {
                "min": min(item["min_position_m"] for item in metrics),
                "max": max(item["max_position_m"] for item in metrics),
                "source": "observed_physical_detent_slider_position",
            }
            if metrics
            else None
        ),
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
    }


def detent_slider_certificate_model(
    adapter: AdapterSpec,
) -> dict[str, Any]:
    spec = prismatic_detent_latch_from_dict(adapter.family_spec)
    return {
        "adapter_id": adapter.adapter_id,
        "mechanism_id": spec.mechanism_id,
        "mechanism_kind": spec.mechanism_kind,
        "topology_sha256": canonical_sha256(spec.to_dict()),
        "body_count": len(spec.bodies),
        "fixed_body_id": spec.fixed_body_id,
        "input_body_id": spec.input_body_id,
        "output_body_id": spec.output_body_id,
        "detent_positions_m": list(spec.detent_interface.positions_m),
        "physical_guides": (
            "finite_conditional_ChLoadBodyBodyBushingGeneric"
        ),
        "physical_detent": "bound_STEP_surface_contact_only",
        "direct_output_actuation": False,
        "reduced_guides": "declared_ChLinkLockPrismatic_topology",
        "reduced_detent": "deterministic_ideal_reference_state_machine",
        "wear_life_certified": False,
    }


def detent_slider_replacement_topology(
    adapter: AdapterSpec,
) -> dict[str, Any]:
    spec = prismatic_detent_latch_from_dict(adapter.family_spec)
    return {
        "mechanism_id": spec.mechanism_id,
        "family_spec_sha256": canonical_sha256(spec.to_dict()),
        "body_ids": [body.body_id for body in spec.bodies],
        "joint_ids": [
            spec.blade_prismatic_joint.joint_id,
            spec.latch_press_joint.joint_id,
        ],
        "input_body_id": spec.input_body_id,
        "output_body_id": spec.output_body_id,
        "detent_positions_m": list(spec.detent_interface.positions_m),
        "detent_release_press_m": spec.detent_interface.release_press_m,
    }
