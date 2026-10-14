from __future__ import annotations

from .adapters import adapter_from_dict, get_adapter
from .models import LoadCase, LoadPhase, MechanismPlan, MechanismValidationError
from .multi_output_shaft_spec import multi_output_shaft_from_dict


def default_multi_output_shaft_load_cases(
    plan: MechanismPlan,
) -> tuple[LoadCase, ...]:
    """Reusable validator-owned programs for one input and multiple shafts."""

    if plan.family != "compound" or plan.adapter_id is None:
        raise MechanismValidationError(
            "LOAD_FAMILY_MISMATCH",
            "multi-output shaft loads require a compound mechanism plan",
        )
    adapter = (
        adapter_from_dict(plan.adapter_spec)
        if plan.adapter_spec is not None
        else get_adapter(plan.adapter_id)
    )
    spec = multi_output_shaft_from_dict(adapter.family_spec)
    moving_mass = sum(
        body.mass_kg for body in spec.bodies if not body.fixed
    )

    def cycle(
        load_case_id: str,
        *,
        torque_scale: float,
        purpose: str,
        program: str = "bidirectional_multi_output_shaft",
    ) -> LoadCase:
        return LoadCase(
            load_case_id=load_case_id,
            family=plan.family,
            program=program,
            purpose=purpose,
            moving_mass_kg=moving_mass,
            step_s=0.00005,
            sample_interval_s=0.01,
            phases=(
                LoadPhase(
                    "crank_forward",
                    "torque",
                    tuple(spec.input_axis_abs),
                    spec.nominal_input_torque_nm * torque_scale,
                    spec.max_torque_phase_s,
                ),
                LoadPhase(
                    "coast_after_forward",
                    "none",
                    None,
                    0.0,
                    spec.settle_phase_s,
                ),
                LoadPhase(
                    "crank_reverse",
                    "torque",
                    tuple(-item for item in spec.input_axis_abs),
                    spec.return_input_torque_nm,
                    spec.max_return_phase_s,
                ),
                LoadPhase(
                    "coast_after_reverse",
                    "none",
                    None,
                    0.0,
                    spec.settle_phase_s,
                ),
            ),
        )

    return (
        cycle(
            "nominal_light",
            torque_scale=0.82,
            purpose="nominal",
        ),
        cycle(
            "nominal_loaded",
            torque_scale=1.15,
            purpose="nominal",
        ),
        LoadCase(
            load_case_id="zero_input_gravity",
            family=plan.family,
            program="zero_input_multi_output_shaft",
            purpose="adversarial",
            moving_mass_kg=moving_mass,
            step_s=0.00005,
            sample_interval_s=0.01,
            phases=(
                LoadPhase(
                    "zero_input",
                    "none",
                    None,
                    0.0,
                    0.30,
                ),
            ),
        ),
        LoadCase(
            load_case_id="shaft_transverse_probe",
            family=plan.family,
            program="transverse_multi_output_shaft",
            purpose="adversarial",
            moving_mass_kg=moving_mass,
            step_s=0.00005,
            sample_interval_s=0.01,
            phases=(
                LoadPhase(
                    "off_axis_torque",
                    "torque",
                    (1.0, 0.0, 0.0),
                    spec.nominal_input_torque_nm * 0.18,
                    0.18,
                ),
                LoadPhase(
                    "probe_settle",
                    "none",
                    None,
                    0.0,
                    0.10,
                ),
            ),
        ),
        cycle(
            "held_out_mid",
            torque_scale=0.94,
            purpose="held_out",
        ),
        cycle(
            "task_egg_beater_cycle",
            torque_scale=1.0,
            purpose="nominal",
            program="task_egg_beater_cycle",
        ),
    )
