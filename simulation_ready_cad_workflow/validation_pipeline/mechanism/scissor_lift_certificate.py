"""Certificate model, load envelope, and replacement topology for the
``hydraulic_scissor_lift`` family.

Mirrors ``rack_pinion_certificate.py``: the load envelope summarizes the
certified physical-branch evidence (moving mass, applied cylinder forces,
observed platform progress, and worst-case joint drift / off-plane rotation /
platform level error), the certificate model describes the validated
mechanism, and the replacement topology is the hash-bound ideal-link model the
whole-machine replay executes.  The certificate JSON embeds
``family_spec_sha256`` so any spec edit invalidates the certificate.
"""

from __future__ import annotations

from typing import Any

from .adapters import AdapterSpec
from .io import canonical_sha256
from .models import LoadCase
from .scissor_lift_spec import expected_ideal_links, scissor_lift_from_dict


def scissor_lift_load_envelope(
    load_cases: tuple[LoadCase, ...], comparisons: tuple[dict[str, Any], ...]
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
    progress = (
        {
            "min": min(item["min_progress_m"] for item in metrics),
            "max": max(item["max_progress_m"] for item in metrics),
            "source": "observed_physical_branch_platform_progress",
        }
        if metrics
        else None
    )
    return {
        "moving_mass_kg": {"min": min(masses), "max": max(masses)},
        "load_case_ids": [item.load_case_id for item in load_cases],
        "purposes": sorted({item.purpose for item in load_cases}),
        "gravity_abs_m_s2": None,
        "applied_force_n": {
            "min_nonzero": min(force_magnitudes) if force_magnitudes else 0.0,
            "max": max(force_magnitudes) if force_magnitudes else 0.0,
        },
        "generalized_platform_progress_m": progress,
        "max_observed_joint_drift_m": (
            max(item["max_joint_drift_m"] for item in metrics) if metrics else None
        ),
        "max_observed_off_plane_rotation_rad": (
            max(item["max_off_plane_rotation_rad"] for item in metrics)
            if metrics
            else None
        ),
        "max_observed_platform_level_error_rad": (
            max(item["max_platform_level_error_rad"] for item in metrics)
            if metrics
            else None
        ),
        "max_observed_interpenetration_m": max(
            (
                float(item.get("max_illegal_interpenetration_m") or 0.0)
                for item in metrics
            ),
            default=None,
        ),
    }


def scissor_lift_certificate_model(adapter: AdapterSpec) -> dict[str, Any]:
    spec = scissor_lift_from_dict(adapter.family_spec)
    return {
        "adapter_id": adapter.adapter_id,
        "mechanism_kind": "hydraulic_scissor_lift",
        "mechanism_id": spec.mechanism_id,
        "topology_sha256": canonical_sha256(spec.to_dict()),
        "family_spec_sha256": canonical_sha256(spec.to_dict()),
        "body_count": len(spec.bodies),
        "joint_count": spec.joint_count,
        "loop_count": spec.loop_count,
        "fixed_body_id": spec.fixed_body_id,
        "output_body_id": spec.output_body_id,
        "output_axis_abs": list(spec.output_axis_abs),
        "actuator": {
            "kind": spec.actuator.kind,
            "barrel_body_id": spec.actuator.barrel_body_id,
            "rod_body_id": spec.actuator.rod_body_id,
            "free_axial_rotation": spec.actuator.free_axial_rotation,
            "length_limits_m": [
                spec.actuator.min_length_m,
                spec.actuator.max_length_m,
            ],
        },
        "physical_support": "finite_bushings_with_force_limited_stops",
        "reduced_support": "_".join(
            (
                f"{len(spec.revolute_joints)}xChLinkLockRevolute",
                f"{len(spec.prismatic_joints)}xChLinkLockRevolutePrismatic",
                "1xChLinkLockCylindrical",
            )
        ),
        "exact_joint_surface_contact_certified": False,
    }


def scissor_lift_replacement_topology(adapter: AdapterSpec) -> dict[str, Any]:
    spec = scissor_lift_from_dict(adapter.family_spec)
    return {
        "mechanism_kind": "hydraulic_scissor_lift",
        "mechanism_id": spec.mechanism_id,
        "family_spec_sha256": canonical_sha256(spec.to_dict()),
        "body_ids": [item.body_id for item in spec.bodies],
        "joint_ids": list(spec.joint_ids),
        "revolute_joint_ids": [item.joint_id for item in spec.revolute_joints],
        "prismatic_joint_ids": [item.joint_id for item in spec.prismatic_joints],
        "cylindrical_joint_id": spec.cylindrical_joint.joint_id,
        "loop_count": spec.loop_count,
        "output_body_id": spec.output_body_id,
        "output_progress_range_m": [spec.home_progress_m, spec.outbound_progress_m],
        "cylinder_length_limits_m": [
            spec.cylindrical_joint.min_length_m,
            spec.cylindrical_joint.max_length_m,
        ],
        "ideal_links": list(expected_ideal_links(spec)),
    }
