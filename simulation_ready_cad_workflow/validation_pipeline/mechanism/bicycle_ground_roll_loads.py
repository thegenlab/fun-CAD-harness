from __future__ import annotations

"""Validator-owned load program for straight-line bicycle rolling."""

from .adapters import adapter_from_dict, get_adapter
from .bicycle_ground_roll_spec import (
    BICYCLE_GROUND_ROLL_PROGRAM,
    BicycleGroundRollSpec,
    bicycle_ground_roll_from_dict,
)
from .models import LoadCase, LoadPhase, MechanismPlan, MechanismValidationError


_STEP_S = 0.001
_SAMPLE_INTERVAL_S = 0.02


def _spec_for_plan(plan: MechanismPlan) -> BicycleGroundRollSpec:
    if plan.family != "compound" or plan.adapter_id is None:
        raise MechanismValidationError(
            "LOAD_FAMILY_MISMATCH",
            "bicycle ground-roll loads require a compound mechanism plan",
        )
    adapter = (
        adapter_from_dict(plan.adapter_spec)
        if plan.adapter_spec is not None
        else get_adapter(plan.adapter_id)
    )
    family_spec = adapter.family_spec
    if isinstance(family_spec, dict):
        family_spec = {
            key: value
            for key, value in family_spec.items()
            if key != "collision_geometry"
        }
    return bicycle_ground_roll_from_dict(family_spec)


def default_bicycle_ground_roll_load_cases(
    plan: MechanismPlan,
) -> tuple[LoadCase, ...]:
    spec = _spec_for_plan(plan)
    moving_mass = sum(body.mass_kg for body in spec.bodies if not body.fixed)
    phases = (
        LoadPhase(
            phase_id="ground_contact_settle",
            wrench_kind="none",
            direction_abs=None,
            magnitude_si=0.0,
            duration_s=0.50,
        ),
        LoadPhase(
            phase_id="rear_wheel_powered_drive",
            wrench_kind="torque",
            direction_abs=spec.wheel_axis_abs,
            magnitude_si=spec.nominal_rear_drive_torque_nm,
            duration_s=0.70,
        ),
        LoadPhase(
            phase_id="passive_roll_observe",
            wrench_kind="none",
            direction_abs=None,
            magnitude_si=0.0,
            duration_s=0.10,
        ),
    )
    return (
        LoadCase(
            load_case_id=BICYCLE_GROUND_ROLL_PROGRAM,
            family="compound",
            program=BICYCLE_GROUND_ROLL_PROGRAM,
            purpose="nominal",
            phases=phases,
            moving_mass_kg=moving_mass,
            step_s=_STEP_S,
            sample_interval_s=_SAMPLE_INTERVAL_S,
        ),
    )
