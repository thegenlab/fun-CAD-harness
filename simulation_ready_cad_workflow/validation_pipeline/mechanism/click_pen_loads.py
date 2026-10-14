from __future__ import annotations

from .adapters import adapter_from_dict, get_adapter
from .click_pen_spec import click_pen_cam_latch_from_dict
from .models import LoadCase, LoadPhase, MechanismPlan, MechanismValidationError


def default_click_pen_load_cases(
    plan: MechanismPlan,
) -> tuple[LoadCase, ...]:
    if plan.family != "compound" or plan.adapter_id is None:
        raise MechanismValidationError(
            "LOAD_FAMILY_MISMATCH",
            "click pen loads require a compound mechanism plan",
        )
    adapter = (
        adapter_from_dict(plan.adapter_spec)
        if plan.adapter_spec is not None
        else get_adapter(plan.adapter_id)
    )
    spec = click_pen_cam_latch_from_dict(adapter.family_spec)
    moving_mass = sum(body.mass_kg for body in spec.bodies if not body.fixed)
    press_axis = spec.button_prismatic_joint.axis_abs

    def click_phases(
        prefix: str,
        *,
        extended_start: bool,
    ) -> tuple[LoadPhase, ...]:
        if extended_start:
            # The returned button has about 7 mm of free take-up above a cam
            # held in the extended state.  Applying the ordinary short force
            # ramp makes the light button hit that cam at high speed.  A real
            # finger closes this lost motion before delivering the full click
            # force, so use a longer force-only approach for even clicks.
            press_phases = (
                LoadPhase(
                    f"{prefix}_press_takeup_06",
                    "force",
                    press_axis,
                    0.0625 * spec.nominal_press_force_n,
                    0.04,
                ),
                LoadPhase(
                    f"{prefix}_press_takeup_09",
                    "force",
                    press_axis,
                    0.09375 * spec.nominal_press_force_n,
                    0.06,
                ),
                LoadPhase(
                    f"{prefix}_press_ramp_12",
                    "force",
                    press_axis,
                    0.125 * spec.nominal_press_force_n,
                    0.04,
                ),
                LoadPhase(
                    f"{prefix}_press_ramp_25",
                    "force",
                    press_axis,
                    0.25 * spec.nominal_press_force_n,
                    0.04,
                ),
                LoadPhase(
                    f"{prefix}_press_ramp_50",
                    "force",
                    press_axis,
                    0.5 * spec.nominal_press_force_n,
                    0.03,
                ),
                LoadPhase(
                    f"{prefix}_press_ramp_75",
                    "force",
                    press_axis,
                    0.75 * spec.nominal_press_force_n,
                    0.02,
                ),
                LoadPhase(
                    f"{prefix}_press_full",
                    "force",
                    press_axis,
                    spec.nominal_press_force_n,
                    0.03,
                ),
            )
        else:
            press_phases = (
                LoadPhase(
                    f"{prefix}_press_ramp_12",
                    "force",
                    press_axis,
                    0.125 * spec.nominal_press_force_n,
                    0.02,
                ),
                LoadPhase(
                    f"{prefix}_press_ramp_25",
                    "force",
                    press_axis,
                    0.25 * spec.nominal_press_force_n,
                    0.02,
                ),
                LoadPhase(
                    f"{prefix}_press_ramp_50",
                    "force",
                    press_axis,
                    0.5 * spec.nominal_press_force_n,
                    0.02,
                ),
                LoadPhase(
                    f"{prefix}_press_ramp_75",
                    "force",
                    press_axis,
                    0.75 * spec.nominal_press_force_n,
                    0.02,
                ),
                LoadPhase(
                    f"{prefix}_press_full",
                    "force",
                    press_axis,
                    spec.nominal_press_force_n,
                    0.04,
                ),
            )
        return press_phases + (
            LoadPhase(
                f"{prefix}_release_ramp_50",
                "force",
                press_axis,
                0.50 * spec.nominal_press_force_n,
                0.01,
            ),
            LoadPhase(
                f"{prefix}_release_ramp_25",
                "force",
                press_axis,
                0.25 * spec.nominal_press_force_n,
                0.01,
            ),
            LoadPhase(
                f"{prefix}_release_ramp_12",
                "force",
                press_axis,
                0.125 * spec.nominal_press_force_n,
                0.01,
            ),
            LoadPhase(
                f"{prefix}_release_ramp_06",
                "force",
                press_axis,
                0.0625 * spec.nominal_press_force_n,
                0.01,
            ),
            LoadPhase(
                f"{prefix}_release_ramp_03",
                "force",
                press_axis,
                0.03125 * spec.nominal_press_force_n,
                0.01,
            ),
            LoadPhase(
                f"{prefix}_release_ramp_01",
                "force",
                press_axis,
                0.0125 * spec.nominal_press_force_n,
                0.01,
            ),
            LoadPhase(
                f"{prefix}_release",
                "none",
                None,
                0.0,
                # The button/cam stack still has downward momentum when the
                # finger force first reaches zero.  Retain a real unforced
                # settling window long enough for both return springs and the
                # barrel ramp to reverse that motion and seat the next index.
                0.64,
            ),
        )

    def case(
        load_case_id: str,
        *,
        click_count: int,
        program: str,
        purpose: str,
    ) -> LoadCase:
        phases = tuple(
            phase
            for click_index in range(click_count)
            for phase in click_phases(
                f"click_{click_index + 1}",
                extended_start=bool(click_index % 2),
            )
        )
        return LoadCase(
            load_case_id=load_case_id,
            family=plan.family,
            program=program,
            purpose=purpose,
            moving_mass_kg=moving_mass,
            step_s=0.0001,
            sample_interval_s=0.01,
            phases=phases,
        )

    return (
        case(
            "nominal_first_click",
            click_count=1,
            program="pen_first_click",
            purpose="nominal",
        ),
        case(
            "nominal_two_click_cycle",
            click_count=2,
            program="pen_two_click_cycle",
            purpose="nominal",
        ),
        LoadCase(
            load_case_id="zero_input_click_pen",
            family=plan.family,
            program="zero_input_click_pen",
            purpose="adversarial",
            moving_mass_kg=moving_mass,
            step_s=0.0002,
            sample_interval_s=0.01,
            phases=(
                LoadPhase("zero_input", "none", None, 0.0, 0.24),
            ),
        ),
        case(
            "task_pen_first_click",
            click_count=1,
            program="task_pen_first_click",
            purpose="nominal",
        ),
        case(
            "task_pen_second_click",
            click_count=2,
            program="task_pen_second_click",
            purpose="nominal",
        ),
        case(
            "task_pen_repeat_cycles",
            click_count=4,
            program="task_pen_repeat_cycles",
            purpose="held_out",
        ),
    )
