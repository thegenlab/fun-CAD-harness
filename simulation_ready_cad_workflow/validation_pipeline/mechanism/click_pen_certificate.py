from __future__ import annotations

from typing import Any

from .adapters import AdapterSpec
from .click_pen_spec import click_pen_cam_latch_from_dict
from .io import canonical_sha256
from .models import LoadCase


def click_pen_load_envelope(
    load_cases: tuple[LoadCase, ...],
    comparisons: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    forces = [
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
        "moving_mass_kg": {"min": min(masses), "max": max(masses)},
        "load_case_ids": [case.load_case_id for case in load_cases],
        "purposes": sorted({case.purpose for case in load_cases}),
        "gravity_abs_m_s2": None,
        "applied_button_force_n": {
            "min_nonzero": min(forces) if forces else 0.0,
            "max": max(forces) if forces else 0.0,
        },
        "refill_extension_range_m": (
            {
                "min": min(item["min_refill_extension_m"] for item in metrics),
                "max": max(item["max_refill_extension_m"] for item in metrics),
                "source": "observed_physical_click_pen_extension",
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


def click_pen_certificate_model(adapter: AdapterSpec) -> dict[str, Any]:
    spec = click_pen_cam_latch_from_dict(adapter.family_spec)
    return {
        "adapter_id": adapter.adapter_id,
        "mechanism_id": spec.mechanism_id,
        "mechanism_kind": spec.mechanism_kind,
        "topology_sha256": canonical_sha256(spec.to_dict()),
        "body_count": len(spec.bodies),
        "fixed_body_id": spec.fixed_body_id,
        "input_body_id": spec.input_body_id,
        "output_body_id": spec.output_body_id,
        "cam_body_id": spec.cam_body_id,
        "first_click_target_m": (
            spec.click_state_machine.first_click_target_m
        ),
        "index_angle_rad": spec.cam_cylindrical_joint.index_angle_rad,
        "physical_guides": (
            "finite_conditional_ChLoadBodyBodyBushingGeneric"
        ),
        "physical_cam_and_latch": "bound_STEP_surface_contact_only",
        "direct_output_actuation": False,
        "reduced_guides": (
            "declared_prismatic_and_cylindrical_ideal_topology"
        ),
        "reduced_click_state": "deterministic_ideal_reference_state_machine",
        "wear_life_certified": False,
    }


def click_pen_replacement_topology(adapter: AdapterSpec) -> dict[str, Any]:
    spec = click_pen_cam_latch_from_dict(adapter.family_spec)
    return {
        "mechanism_id": spec.mechanism_id,
        "family_spec_sha256": canonical_sha256(spec.to_dict()),
        "body_ids": [body.body_id for body in spec.bodies],
        "joint_ids": [
            spec.button_prismatic_joint.joint_id,
            spec.refill_prismatic_joint.joint_id,
            spec.cam_cylindrical_joint.joint_id,
        ],
        "input_body_id": spec.input_body_id,
        "output_body_id": spec.output_body_id,
        "cam_body_id": spec.cam_body_id,
        "first_click_target_m": (
            spec.click_state_machine.first_click_target_m
        ),
        "index_angle_rad": spec.cam_cylindrical_joint.index_angle_rad,
    }
