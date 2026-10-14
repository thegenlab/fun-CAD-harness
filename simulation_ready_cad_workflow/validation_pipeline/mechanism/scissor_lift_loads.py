"""Load-case generation for the ``hydraulic_scissor_lift`` family.

The six standard compound-family load cases parameterized by the scissor-lift
spec, mirroring ``loads.py`` (``default_compound_load_cases``) and
``rack_pinion_loads_v3.py``.  The generalized coordinate is the platform
reference-point height travel from home; the input wrench is the cylinder
axial force pair (``force`` phases whose ``direction_abs`` selects extension
versus retraction against the home cylinder axis).
"""

from __future__ import annotations

from .adapters import adapter_from_dict, get_adapter
from .models import LoadCase, LoadPhase, MechanismPlan, MechanismValidationError
from .scissor_lift_spec import ScissorLiftSpec, scissor_lift_from_dict


SCISSOR_LIFT_NOMINAL_LIGHT_SCALE = 0.82
SCISSOR_LIFT_NOMINAL_LOADED_SCALE = 1.18
SCISSOR_LIFT_HELD_OUT_SCALE = 0.96
SCISSOR_LIFT_TRANSVERSE_FRACTION = 0.20
# Lift-side margin sits closer to the outbound stop so the raised dwell
# latches higher above the tolerance band after the throttled cut; the
# home side keeps the larger margin so the retract cut leaves room for the
# gravity-assisted closing dip.
SCISSOR_LIFT_LIFT_APPROACH_MARGIN_FRACTION = 0.4
SCISSOR_LIFT_LOWER_APPROACH_MARGIN_FRACTION = 0.75


def _cross(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _unit(value: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = sum(item * item for item in value) ** 0.5
    if norm <= 1e-12:
        raise MechanismValidationError(
            "SCISSOR_LIFT_AXIS_INVALID", "scissor-lift axis cannot be zero"
        )
    return tuple(item / norm for item in value)


def cylinder_home_axis(spec: ScissorLiftSpec) -> tuple[float, float, float]:
    """Unit vector from the barrel anchor to the rod anchor at the home pose."""
    return _unit(
        tuple(
            rod - barrel
            for rod, barrel in zip(
                spec.actuator.rod_anchor_abs_m, spec.actuator.barrel_anchor_abs_m
            )
        )
    )


def default_scissor_lift_load_cases(plan: MechanismPlan) -> tuple[LoadCase, ...]:
    if plan.family != "compound" or plan.adapter_id is None:
        raise MechanismValidationError(
            "LOAD_FAMILY_MISMATCH",
            "scissor-lift load generation requires a compound plan with family data",
        )
    adapter = (
        adapter_from_dict(plan.adapter_spec)
        if plan.adapter_spec is not None
        else get_adapter(plan.adapter_id)
    )
    spec = scissor_lift_from_dict(adapter.family_spec)
    moving_mass = sum(body.mass_kg for body in spec.bodies if not body.fixed)
    extend_axis = cylinder_home_axis(spec)
    retract_axis = tuple(-item for item in extend_axis)
    # The transverse probe pushes along the scissor-plane normal so it excites
    # the stiff out-of-plane bushing directions instead of the free hinge.
    transverse_raw = _cross(extend_axis, spec.output_axis_abs)
    if sum(item * item for item in transverse_raw) <= 1e-12:
        basis = min(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            key=lambda item: abs(sum(a * b for a, b in zip(extend_axis, item))),
        )
        transverse_raw = _cross(extend_axis, basis)
    transverse_axis = _unit(transverse_raw)
    lift_control_target_m = (
        spec.outbound_progress_m
        - SCISSOR_LIFT_LIFT_APPROACH_MARGIN_FRACTION
        * spec.output_progress_tolerance_m
    )
    lower_control_target_m = (
        spec.home_progress_m
        + SCISSOR_LIFT_LOWER_APPROACH_MARGIN_FRACTION
        * spec.output_progress_tolerance_m
    )

    def cycle(
        load_case_id: str,
        *,
        force_scale: float,
        purpose: str,
        task: bool = False,
    ) -> LoadCase:
        return LoadCase(
            load_case_id=load_case_id,
            family=plan.family,
            program=("task_platform_roundtrip" if task else "bidirectional_scissor_lift"),
            purpose=purpose,
            moving_mass_kg=moving_mass,
            step_s=0.0001,
            sample_interval_s=0.005,
            phases=(
                LoadPhase(
                    "cylinder_extend",
                    "force",
                    extend_axis,
                    spec.nominal_input_force_n * force_scale,
                    spec.max_force_phase_s,
                    stop_condition="position_at_least",
                    target_position_si=lift_control_target_m,
                    branch_termination_policy="each_branch_target",
                ),
                LoadPhase("raised_settle", "none", None, 0.0, spec.settle_phase_s),
                LoadPhase(
                    "cylinder_retract",
                    "force",
                    retract_axis,
                    spec.return_input_force_n,
                    spec.max_return_phase_s,
                    stop_condition="position_at_most",
                    target_position_si=lower_control_target_m,
                    branch_termination_policy="each_branch_target",
                ),
                LoadPhase("home_settle", "none", None, 0.0, spec.settle_phase_s),
            ),
        )

    return (
        cycle(
            "nominal_light",
            force_scale=SCISSOR_LIFT_NOMINAL_LIGHT_SCALE,
            purpose="nominal",
        ),
        cycle(
            "nominal_loaded",
            force_scale=SCISSOR_LIFT_NOMINAL_LOADED_SCALE,
            purpose="nominal",
        ),
        LoadCase(
            load_case_id="zero_input_gravity",
            family=plan.family,
            program="zero_input_gravity",
            purpose="adversarial",
            moving_mass_kg=moving_mass,
            step_s=0.0001,
            sample_interval_s=0.005,
            phases=(LoadPhase("zero_input", "none", None, 0.0, 0.35),),
        ),
        LoadCase(
            load_case_id="transverse_scissor_probe",
            family=plan.family,
            program="transverse_scissor_probe",
            purpose="adversarial",
            moving_mass_kg=moving_mass,
            # The lateral probe excites the stiff bearing-seat directions, so it
            # deliberately uses a finer explicit integration step than free motion.
            step_s=0.00002,
            sample_interval_s=0.005,
            phases=(
                LoadPhase(
                    "transverse_force",
                    "force",
                    transverse_axis,
                    spec.nominal_input_force_n * SCISSOR_LIFT_TRANSVERSE_FRACTION,
                    0.20,
                ),
                LoadPhase("transverse_settle", "none", None, 0.0, 0.10),
            ),
        ),
        cycle("held_out_mid", force_scale=SCISSOR_LIFT_HELD_OUT_SCALE, purpose="held_out"),
        cycle(
            "task_platform_roundtrip",
            force_scale=1.0,
            purpose="nominal",
            task=True,
        ),
    )
