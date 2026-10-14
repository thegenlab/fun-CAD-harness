from __future__ import annotations

from .adapters import adapter_from_dict, get_adapter
from .clock_gear_train_spec import clock_gear_train_from_dict
from .models import LoadCase, LoadPhase, MechanismPlan, MechanismValidationError


def default_clock_gear_load_cases(plan: MechanismPlan) -> tuple[LoadCase, ...]:
    """Validator-owned programs for legacy and input-isolated clock trains."""

    if plan.family != "compound" or plan.adapter_id is None:
        raise MechanismValidationError(
            "LOAD_FAMILY_MISMATCH",
            "clock load generation requires a compound mechanism plan",
        )
    adapter = (
        adapter_from_dict(plan.adapter_spec)
        if plan.adapter_spec is not None
        else get_adapter(plan.adapter_id)
    )
    spec = clock_gear_train_from_dict(adapter.family_spec)
    moving_mass = sum(body.mass_kg for body in spec.bodies if not body.fixed)
    def cycle(
        load_case_id: str,
        *,
        torque_scale: float,
        purpose: str,
        program: str = "bidirectional_clock_train",
    ) -> LoadCase:
        alarm_program = program == "alarm_clock_cycle" and spec.alarm_isolated
        axis = (
            tuple(spec.effective_alarm_input_axis_abs)
            if alarm_program
            else tuple(spec.input_axis_abs)
        )
        input_torque = (
            spec.effective_nominal_alarm_input_torque_nm
            if alarm_program
            else spec.nominal_input_torque_nm
        )
        return_torque = (
            spec.effective_nominal_alarm_input_torque_nm
            if alarm_program
            else spec.return_input_torque_nm
        )
        return LoadCase(
            load_case_id=load_case_id,
            family=plan.family,
            program=program,
            purpose=purpose,
            moving_mass_kg=moving_mass,
            step_s=0.00005,
            sample_interval_s=0.05,
            phases=(
                LoadPhase(
                    "wind_and_advance",
                    "torque",
                    axis,
                    input_torque * torque_scale,
                    spec.max_torque_phase_s,
                ),
                LoadPhase("running_settle", "none", None, 0.0, spec.settle_phase_s),
                LoadPhase(
                    "controlled_return",
                    "torque",
                    tuple(-item for item in axis),
                    return_torque,
                    spec.max_return_phase_s,
                ),
                LoadPhase("home_settle", "none", None, 0.0, spec.settle_phase_s),
            ),
        )

    if spec.alarm_setting_clutch is not None:
        hour_axis = next(
            shaft.axis_abs
            for shaft in spec.shafts
            if shaft.body_id == spec.hour_body_id
        )
        manual_time_setting = LoadCase(
            load_case_id="task_manual_time_setting",
            family=plan.family,
            program="manual_time_setting_clock",
            purpose="nominal",
            moving_mass_kg=moving_mass,
            step_s=0.00005,
            sample_interval_s=0.05,
            phases=(
                LoadPhase(
                    "engage_and_set_forward",
                    "torque",
                    tuple(spec.setting_input_axis_abs),
                    0.05,
                    0.70,
                ),
                LoadPhase("setting_settle", "none", None, 0.0, 0.08),
                LoadPhase(
                    "set_reverse",
                    "torque",
                    tuple(-item for item in spec.setting_input_axis_abs),
                    0.05,
                    0.55,
                ),
                LoadPhase(
                    "setting_home_settle",
                    "none",
                    None,
                    0.0,
                    0.08,
                ),
            ),
        )
        alarm_setting_phases = (
            LoadPhase(
                "turn_hour_hand_to_alarm_target",
                "torque",
                tuple(hour_axis),
                0.60,
                1.20,
            ),
            LoadPhase(
                "alarm_setting_latched",
                "none",
                None,
                0.0,
                0.12,
            ),
        )
        scheduled_alarm_phases = (
            *alarm_setting_phases,
            LoadPhase(
                "restore_hour_hand_to_current_time",
                "torque",
                tuple(-item for item in hour_axis),
                0.60,
                1.20,
            ),
            LoadPhase(
                "alarm_target_ready",
                "none",
                None,
                0.0,
                0.12,
            ),
            LoadPhase(
                "advance_to_alarm_time",
                "torque",
                tuple(spec.input_axis_abs),
                spec.nominal_input_torque_nm,
                spec.max_torque_phase_s,
            ),
            LoadPhase(
                "ring_with_timekeeping",
                "torque",
                tuple(spec.input_axis_abs),
                spec.nominal_input_torque_nm,
                0.65,
            ),
            LoadPhase(
                "scheduled_alarm_settle",
                "none",
                None,
                0.0,
                0.08,
            ),
        )
        return (
            cycle(
                "task_three_hand_timekeeping",
                torque_scale=1.0,
                purpose="nominal",
                program="task_clock_cycle",
            ),
            manual_time_setting,
            cycle(
                "task_alarm_cycle",
                torque_scale=1.10,
                purpose="nominal",
                program="alarm_clock_cycle",
            ),
            LoadCase(
                load_case_id="task_alarm_time_setting",
                family=plan.family,
                program="hour_hand_alarm_setting_clock",
                purpose="nominal",
                moving_mass_kg=moving_mass,
                step_s=0.00005,
                sample_interval_s=0.05,
                phases=alarm_setting_phases,
            ),
            LoadCase(
                load_case_id="task_scheduled_alarm_trigger",
                family=plan.family,
                program="scheduled_alarm_clock",
                purpose="nominal",
                moving_mass_kg=moving_mass,
                step_s=0.00005,
                sample_interval_s=0.05,
                phases=scheduled_alarm_phases,
            ),
        )

    cases = (
        cycle("nominal_light", torque_scale=0.82, purpose="nominal"),
        cycle("nominal_loaded", torque_scale=1.15, purpose="nominal"),
        LoadCase(
            load_case_id="zero_input_gravity",
            family=plan.family,
            program="zero_input_clock_train",
            purpose="adversarial",
            moving_mass_kg=moving_mass,
            step_s=0.00005,
            sample_interval_s=0.05,
            phases=(LoadPhase("zero_input", "none", None, 0.0, 0.30),),
        ),
        LoadCase(
            load_case_id="shaft_transverse_probe",
            family=plan.family,
            program="transverse_clock_probe",
            purpose="adversarial",
            moving_mass_kg=moving_mass,
            step_s=0.00005,
            sample_interval_s=0.05,
            phases=(
                LoadPhase(
                    "off_axis_torque",
                    "torque",
                    (1.0, 0.0, 0.0),
                    spec.nominal_input_torque_nm * 0.18,
                    0.18,
                ),
                LoadPhase("probe_settle", "none", None, 0.0, 0.10),
            ),
        ),
        cycle("held_out_mid", torque_scale=0.94, purpose="held_out"),
        cycle(
            "task_clock_cycle",
            torque_scale=1.0,
            purpose="nominal",
            program="task_clock_cycle",
        ),
        LoadCase(
            load_case_id="task_manual_time_setting",
            family=plan.family,
            program="manual_time_setting_clock",
            purpose="nominal",
            moving_mass_kg=moving_mass,
            step_s=0.00005,
            sample_interval_s=0.05,
            phases=(
                LoadPhase(
                    "engage_and_set_forward",
                    "torque",
                    tuple(spec.setting_input_axis_abs),
                    spec.nominal_input_torque_nm,
                    0.70,
                ),
                LoadPhase("setting_settle", "none", None, 0.0, 0.08),
                LoadPhase(
                    "set_reverse",
                    "torque",
                    tuple(-item for item in spec.setting_input_axis_abs),
                    spec.return_input_torque_nm,
                    0.55,
                ),
                LoadPhase("setting_home_settle", "none", None, 0.0, 0.08),
            ),
        ),
        cycle(
            "task_alarm_cycle",
            torque_scale=1.10,
            purpose="nominal",
            program="alarm_clock_cycle",
        ),
    )
    if spec.scheduled_alarm is None:
        return cases
    advanced = (
        LoadCase(
            load_case_id="task_zero_time_and_resume",
            family=plan.family,
            program="manual_zero_and_resume_clock",
            purpose="nominal",
            moving_mass_kg=moving_mass,
            step_s=0.00005,
            sample_interval_s=0.05,
            phases=(
                LoadPhase(
                    "calibration_offset",
                    "torque",
                    tuple(spec.setting_input_axis_abs),
                    spec.nominal_input_torque_nm,
                    0.45,
                ),
                LoadPhase(
                    "calibration_return",
                    "torque",
                    tuple(-item for item in spec.setting_input_axis_abs),
                    max(
                        0.16 if spec.alarm_isolated else 0.12,
                        (2.00 if spec.alarm_isolated else 1.50)
                        * spec.return_input_torque_nm,
                    ),
                    2.50 if spec.alarm_isolated else 2.00,
                    stop_condition="position_at_most",
                    target_position_si=(
                        spec.home_output_angle_rad + 0.02
                    ),
                    branch_termination_policy="each_branch_target",
                ),
                LoadPhase("release_setting", "none", None, 0.0, 0.08),
                LoadPhase(
                    "post_zero_run",
                    "torque",
                    tuple(spec.input_axis_abs),
                    spec.nominal_input_torque_nm,
                    0.65,
                ),
                LoadPhase("post_zero_settle", "none", None, 0.0, 0.08),
            ),
        ),
        LoadCase(
            load_case_id="task_alarm_after_one_minute",
            family=plan.family,
            program="scheduled_alarm_clock",
            purpose="nominal",
            moving_mass_kg=moving_mass,
            step_s=0.00005,
            sample_interval_s=0.05,
            phases=(
                LoadPhase(
                    "set_alarm_for_one_minute",
                    "torque",
                    tuple(spec.setting_input_axis_abs),
                    spec.nominal_input_torque_nm,
                    0.16,
                ),
                LoadPhase("alarm_setting_latched", "none", None, 0.0, 0.08),
                LoadPhase(
                    "advance_to_alarm_time",
                    "torque",
                    tuple(spec.input_axis_abs),
                    spec.nominal_input_torque_nm,
                    spec.max_torque_phase_s,
                    stop_condition="progress_at_least",
                    target_progress_m=(
                        spec.scheduled_alarm.target_second_hand_travel_rad
                        - spec.scheduled_alarm.target_second_hand_tolerance_rad
                    ),
                    branch_termination_policy="each_branch_target",
                ),
                LoadPhase(
                    "ring_with_timekeeping",
                    "torque",
                    tuple(spec.input_axis_abs),
                    spec.nominal_input_torque_nm,
                    0.65,
                ),
                LoadPhase("scheduled_alarm_settle", "none", None, 0.0, 0.08),
            ),
        ),
    )
    return (*cases, *advanced)
