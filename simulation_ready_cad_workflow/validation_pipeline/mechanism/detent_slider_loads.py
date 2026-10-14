from __future__ import annotations

import math

from .adapters import adapter_from_dict, get_adapter
from .detent_slider_spec import prismatic_detent_latch_from_dict
from .models import LoadCase, LoadPhase, MechanismPlan, MechanismValidationError


GUIDE_FREE_AXIS_DAMPING_N_S_M = 5.0
CONTROLLED_TRANSLATION_DAMPING_N_S_M = (
    2.0 * GUIDE_FREE_AXIS_DAMPING_N_S_M
)


def _weighted_direction(
    first: tuple[float, float, float],
    first_magnitude: float,
    second: tuple[float, float, float],
    second_magnitude: float,
) -> tuple[float, float, float]:
    value = tuple(
        first_magnitude * a + second_magnitude * b
        for a, b in zip(first, second)
    )
    length = math.sqrt(sum(item * item for item in value))
    return tuple(item / length for item in value)


def default_detent_slider_load_cases(
    plan: MechanismPlan,
) -> tuple[LoadCase, ...]:
    if plan.family != "compound" or plan.adapter_id is None:
        raise MechanismValidationError(
            "LOAD_FAMILY_MISMATCH",
            "detent slider loads require a compound mechanism plan",
        )
    adapter = (
        adapter_from_dict(plan.adapter_spec)
        if plan.adapter_spec is not None
        else get_adapter(plan.adapter_id)
    )
    spec = prismatic_detent_latch_from_dict(adapter.family_spec)
    moving_mass = sum(body.mass_kg for body in spec.bodies if not body.fixed)
    slide_axis = spec.blade_prismatic_joint.axis_abs
    press_axis = spec.latch_press_joint.axis_abs
    controlled_slide_force_n = min(
        spec.nominal_slide_force_n,
        (
            spec.detent_interface.capture_tolerance_m
            * CONTROLLED_TRANSLATION_DAMPING_N_S_M**2
            / (4.0 * moving_mass)
        ),
    )
    forward = _weighted_direction(
        slide_axis,
        controlled_slide_force_n,
        press_axis,
        spec.nominal_press_force_n,
    )
    backward = _weighted_direction(
        tuple(-item for item in slide_axis),
        controlled_slide_force_n,
        press_axis,
        spec.nominal_press_force_n,
    )
    drive_force = math.sqrt(
        controlled_slide_force_n**2 + spec.nominal_press_force_n**2
    )
    controlled_slide_speed_m_s = (
        controlled_slide_force_n
        / CONTROLLED_TRANSLATION_DAMPING_N_S_M
    )

    def prepress_phases(
        phase_id: str = "prepress_latch",
    ) -> tuple[LoadPhase, ...]:
        phases: list[LoadPhase] = []
        for percent, fraction in (
            (25, 0.25),
            (50, 0.50),
            (75, 0.75),
            (100, 1.0),
        ):
            press_force_n = spec.nominal_press_force_n * fraction
            phases.append(
                LoadPhase(
                    f"{phase_id}_ramp_{percent}",
                    "force",
                    press_axis,
                    press_force_n,
                    0.04,
                )
            )
        return tuple(phases)

    def move_phase(
        phase_id: str,
        *,
        direction: tuple[float, float, float],
        target_m: float,
        travel_m: float,
    ) -> LoadPhase:
        return LoadPhase(
            phase_id,
            "force",
            direction,
            drive_force,
            max(
                0.80,
                travel_m / controlled_slide_speed_m_s + 0.25,
            ),
            stop_condition=(
                "position_at_least"
                if direction == forward
                else "position_at_most"
            ),
            target_position_si=target_m,
            branch_termination_policy="each_branch_target",
        )

    def select(
        load_case_id: str,
        *,
        target_m: float,
        purpose: str,
        program: str,
    ) -> LoadCase:
        return LoadCase(
            load_case_id=load_case_id,
            family=plan.family,
            program=program,
            purpose=purpose,
            moving_mass_kg=moving_mass,
            step_s=0.0001,
            sample_interval_s=0.01,
            phases=(
                *prepress_phases(),
                move_phase(
                    "press_and_slide_to_detent",
                    direction=forward,
                    target_m=target_m,
                    travel_m=target_m,
                ),
                LoadPhase(
                    "release_into_detent",
                    "none",
                    None,
                    0.0,
                    0.18,
                ),
            ),
        )

    return (
        select(
            "nominal_working_detent",
            target_m=0.045,
            purpose="nominal",
            program="select_detent_position",
        ),
        LoadCase(
            load_case_id="nominal_working_detent_hold",
            family=plan.family,
            program="hold_detent_under_axial_load",
            purpose="nominal",
            moving_mass_kg=moving_mass,
            step_s=0.0001,
            sample_interval_s=0.01,
            phases=(
                *prepress_phases(),
                move_phase(
                    "press_and_slide_to_detent",
                    direction=forward,
                    target_m=0.045,
                    travel_m=0.045,
                ),
                LoadPhase(
                    "release_into_detent",
                    "none",
                    None,
                    0.0,
                    0.18,
                ),
                LoadPhase(
                    "axial_load_hold",
                    "force",
                    tuple(-item for item in slide_axis),
                    spec.adversarial_axial_force_n,
                    0.30,
                ),
            ),
        ),
        LoadCase(
            load_case_id="zero_input_detent",
            family=plan.family,
            program="zero_input_detent_slider",
            purpose="adversarial",
            moving_mass_kg=moving_mass,
            step_s=0.0001,
            sample_interval_s=0.01,
            phases=(
                LoadPhase("zero_input", "none", None, 0.0, 0.25),
            ),
        ),
        select(
            "held_out_second_detent",
            target_m=0.025,
            purpose="held_out",
            program="select_detent_position",
        ),
        select(
            "task_detent_extend",
            target_m=0.045,
            purpose="nominal",
            program="task_detent_extend",
        ),
        LoadCase(
            load_case_id="task_detent_hold_under_load",
            family=plan.family,
            program="task_detent_hold_under_load",
            purpose="nominal",
            moving_mass_kg=moving_mass,
            step_s=0.0001,
            sample_interval_s=0.01,
            phases=(
                *prepress_phases(),
                move_phase(
                    "press_and_slide_to_detent",
                    direction=forward,
                    target_m=0.045,
                    travel_m=0.045,
                ),
                LoadPhase(
                    "release_into_detent",
                    "none",
                    None,
                    0.0,
                    0.18,
                ),
                LoadPhase(
                    "axial_load_hold",
                    "force",
                    tuple(-item for item in slide_axis),
                    spec.adversarial_axial_force_n,
                    0.30,
                ),
            ),
        ),
        select(
            "task_detent_second_position",
            target_m=0.025,
            purpose="nominal",
            program="task_detent_second_position",
        ),
        LoadCase(
            load_case_id="task_detent_cycle",
            family=plan.family,
            program="task_detent_cycle",
            purpose="nominal",
            moving_mass_kg=moving_mass,
            step_s=0.0001,
            sample_interval_s=0.01,
            phases=(
                *prepress_phases("prepress_latch_for_extension"),
                move_phase(
                    "press_and_slide_to_outer_detent",
                    direction=forward,
                    target_m=0.065,
                    travel_m=0.065,
                ),
                LoadPhase(
                    "release_outer_detent",
                    "none",
                    None,
                    0.0,
                    0.16,
                ),
                LoadPhase(
                    "align_outer_detent_for_unlock",
                    "force",
                    slide_axis,
                    spec.nominal_slide_force_n,
                    0.12,
                    stop_condition="position_at_least",
                    target_position_si=0.065,
                    branch_termination_policy="each_branch_target",
                ),
                *prepress_phases("prepress_latch_for_retraction"),
                move_phase(
                    "press_and_retract_home",
                    direction=backward,
                    target_m=0.0,
                    travel_m=0.065,
                ),
                LoadPhase(
                    "release_home_detent",
                    "none",
                    None,
                    0.0,
                    0.18,
                ),
            ),
        ),
    )
