from __future__ import annotations

from .adapters import adapter_from_dict, get_adapter
from .models import LoadCase, LoadPhase, MechanismPlan, MechanismValidationError
from .rack_pinion_spec import rack_pinion_from_dict


RACK_PINION_NOMINAL_LIGHT_SCALE = 0.82
RACK_PINION_STATIC_BRAKE_FRACTION = 1.25
RACK_PINION_KINETIC_BRAKE_FRACTION = 0.05
RACK_PINION_BRAKE_DAMPING_NM_S_RAD = 1.5
RACK_PINION_OPERATOR_DAMPING_NM_S_RAD = 1.0
RACK_PINION_ACTUATOR_LIMIT_SCALE = 11.0 / 9.0
RACK_PINION_MIN_SETTLE_PHASE_S = 0.25
# User-approved anti-drop check: after the crank is released at the top,
# the gate must hold for at least this long with zero commanded input.
RACK_PINION_ANTI_DROP_HOLD_S = 0.5


def _cross(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def default_rack_pinion_load_cases(plan: MechanismPlan) -> tuple[LoadCase, ...]:
    if plan.family != "compound" or plan.adapter_id is None:
        raise MechanismValidationError(
            "LOAD_FAMILY_MISMATCH",
            "rack-pinion load generation requires a compound plan with family data",
        )
    adapter = (
        adapter_from_dict(plan.adapter_spec)
        if plan.adapter_spec is not None
        else get_adapter(plan.adapter_id)
    )
    spec = rack_pinion_from_dict(adapter.family_spec)
    moving_mass = sum(body.mass_kg for body in spec.bodies if not body.fixed)
    rack_mass = next(body.mass_kg for body in spec.bodies if body.body_id == spec.rack_body_id)
    gravity_force_axis_n = rack_mass * sum(
        gravity * axis
        for gravity, axis in zip(plan.gravity_abs_m_s2, spec.rack_axis_abs)
    )
    resisting_gravity_torque_nm = abs(
        gravity_force_axis_n * spec.pinion_pitch_radius_m
    )
    static_holding_torque_nm = (
        spec.nominal_input_torque_nm * RACK_PINION_STATIC_BRAKE_FRACTION
    )
    if static_holding_torque_nm + 1.0e-9 < resisting_gravity_torque_nm:
        raise MechanismValidationError(
            "RACK_PINION_HOLDING_TORQUE_INSUFFICIENT",
            "declared holding brake cannot support the rack under gravity: "
            f"{static_holding_torque_nm:.6g} N.m available, "
            f"{resisting_gravity_torque_nm:.6g} N.m required",
        )
    nominal_light_torque_nm = (
        spec.nominal_input_torque_nm * RACK_PINION_NOMINAL_LIGHT_SCALE
    )
    # The gate uses a normally-engaged holding brake whose release cam is
    # operated by the crank.  Intentional cranking sees only running drag;
    # releasing the crank makes the full holding capacity available.
    released_brake_drag_nm = (
        spec.nominal_input_torque_nm * RACK_PINION_KINETIC_BRAKE_FRACTION
    )
    light_breakaway_torque_nm = resisting_gravity_torque_nm + released_brake_drag_nm
    if nominal_light_torque_nm <= light_breakaway_torque_nm + 1.0e-9:
        raise MechanismValidationError(
            "RACK_PINION_LIFT_TORQUE_INSUFFICIENT",
            "light-load input cannot overcome gravity and the declared holding brake: "
            f"{nominal_light_torque_nm:.6g} N.m available, "
            f"more than {light_breakaway_torque_nm:.6g} N.m required",
        )
    settle_phase_s = max(spec.settle_phase_s, RACK_PINION_MIN_SETTLE_PHASE_S)
    approach_margin_m = 0.75 * spec.output_progress_tolerance_m
    lift_control_target_m = spec.outbound_progress_m - approach_margin_m
    lower_control_target_m = spec.home_progress_m + approach_margin_m
    # With the rack on the declared side of the pinion, x = -r*theta.
    outbound_axis = tuple(-item for item in spec.pinion_axis_abs)
    return_axis = spec.pinion_axis_abs
    basis = min(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        key=lambda item: abs(sum(a * b for a, b in zip(outbound_axis, item))),
    )
    transverse_axis = _cross(outbound_axis, basis)

    def cycle(
        load_case_id: str,
        *,
        torque_scale: float,
        purpose: str,
        task: bool = False,
    ) -> LoadCase:
        phases: list[LoadPhase] = [
            LoadPhase(
                "crank_lift",
                "torque",
                outbound_axis,
                spec.nominal_input_torque_nm * torque_scale,
                spec.max_torque_phase_s,
                stop_condition="position_at_least",
                target_position_si=lift_control_target_m,
            ),
            LoadPhase("upper_settle", "none", None, 0.0, settle_phase_s),
        ]
        if task:
            # prevent_gate_drop_after_release: hold the raised gate with the
            # crank released (zero commanded input) before lowering it.
            phases.append(
                LoadPhase(
                    "zero_input_hold", "none", None, 0.0, RACK_PINION_ANTI_DROP_HOLD_S
                )
            )
        phases.extend(
            (
                LoadPhase(
                    "crank_lower",
                    "torque",
                    return_axis,
                    spec.return_input_torque_nm,
                    spec.max_return_phase_s,
                    stop_condition="position_at_most",
                    target_position_si=lower_control_target_m,
                ),
                LoadPhase("home_settle", "none", None, 0.0, settle_phase_s),
            )
        )
        return LoadCase(
            load_case_id=load_case_id,
            family=plan.family,
            program=("task_rack_roundtrip" if task else "bidirectional_rack_pinion"),
            purpose=purpose,
            moving_mass_kg=moving_mass,
            step_s=0.0001,
            sample_interval_s=0.05,
            phases=tuple(phases),
        )

    return (
        cycle(
            "nominal_light",
            torque_scale=RACK_PINION_NOMINAL_LIGHT_SCALE,
            purpose="nominal",
        ),
        cycle("nominal_loaded", torque_scale=1.18, purpose="nominal"),
        LoadCase(
            load_case_id="zero_input_gravity",
            family=plan.family,
            program="zero_input_gravity",
            purpose="adversarial",
            moving_mass_kg=moving_mass,
            step_s=0.0001,
            sample_interval_s=0.05,
            phases=(LoadPhase("zero_input", "none", None, 0.0, 0.35),),
        ),
        LoadCase(
            load_case_id="transverse_rack_probe",
            family=plan.family,
            program="transverse_rack_probe",
            purpose="adversarial",
            moving_mass_kg=moving_mass,
            step_s=0.00002,
            sample_interval_s=0.05,
            phases=(
                LoadPhase(
                    "transverse_torque",
                    "torque",
                    transverse_axis,
                    spec.nominal_input_torque_nm * 0.20,
                    0.20,
                ),
                LoadPhase("transverse_settle", "none", None, 0.0, 0.10),
            ),
        ),
        cycle("held_out_mid", torque_scale=0.96, purpose="held_out"),
        cycle(
            "task_rack_roundtrip",
            torque_scale=1.0,
            purpose="nominal",
            task=True,
        ),
    )
