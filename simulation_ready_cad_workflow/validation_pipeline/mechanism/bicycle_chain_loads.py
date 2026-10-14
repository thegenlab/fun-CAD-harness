from __future__ import annotations

from .adapters import adapter_from_dict, get_adapter
from .bicycle_chain_spec import (
    BICYCLE_CHAIN_CRANK_PERTURBATION_PROGRAM,
    BICYCLE_CHAIN_CRANK_PERTURBATION_TORQUE_NM,
    BicycleChainSpec,
    bicycle_chain_from_dict,
)
from .models import LoadCase, LoadPhase, MechanismPlan, MechanismValidationError


_STEP_S = 0.00025
_SAMPLE_INTERVAL_S = 0.00025
_NOMINAL_TORQUE_NM = 3.20
_HELD_OUT_TORQUE_NM = 3.22
_EVIDENCE_TORQUES_NM = (_HELD_OUT_TORQUE_NM,)
_TASK_PHASES = (
    ("finite_engagement_settle", "torque", 0.60),
    ("measured_powered_drive", "torque", 1.60),
    ("passive_return", "none", 8.00),
    ("return_settle", "none", 2.00),
)


def _spec_for_plan(plan: MechanismPlan) -> BicycleChainSpec:
    if plan.family != "compound" or plan.adapter_id is None:
        raise MechanismValidationError(
            "LOAD_FAMILY_MISMATCH",
            "bicycle chain loads require a compound mechanism plan",
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
    return bicycle_chain_from_dict(family_spec)


def _moving_mass(spec: BicycleChainSpec) -> float:
    return sum(body.mass_kg for body in spec.bodies if not body.fixed)


def _task_program_values(spec: BicycleChainSpec) -> tuple[tuple[str, str, float], ...]:
    """Read the approved fixed program without introducing runtime defaults."""

    task_program = spec.task_program
    if task_program is None:
        raise MechanismValidationError(
            "BICYCLE_CHAIN_TASK_PROGRAM_MISSING",
            "the closed r04 family spec has no typed task program",
        )
    values = tuple(
        (
            str(phase.phase_id),
            str(phase.wrench_kind),
            float(phase.duration_s),
        )
        for phase in task_program.phases
    )
    if values != _TASK_PHASES:
        raise MechanismValidationError(
            "BICYCLE_CHAIN_TASK_PROGRAM_CHANGED",
            "the bicycle-chain task phases differ from the approved fixed "
            "0.6/1.6/8/2 second program",
        )
    return values


def _approved_evidence_torques(spec: BicycleChainSpec) -> tuple[float, ...]:
    domain = spec.held_out_validation_domain
    if domain is None:
        raise MechanismValidationError(
            "BICYCLE_CHAIN_HELD_OUT_DOMAIN_MISSING",
            "the closed r04 family spec has no typed held-out domain",
        )
    if domain.selection_authority != "validator_release_bound":
        raise MechanismValidationError(
            "BICYCLE_CHAIN_HELD_OUT_SELECTION_AUTHORITY_CHANGED",
            "the held-out torque must remain validator-owned and release-bound",
        )
    result = tuple(
        float(value)
        for value in domain.required_evidence_torque_nm
    )
    if result != _EVIDENCE_TORQUES_NM:
        raise MechanismValidationError(
            "BICYCLE_CHAIN_HELD_OUT_DOMAIN_CHANGED",
            "the release-bound held-out torque evidence point changed",
        )
    return result


def _roundtrip_phases(
    spec: BicycleChainSpec,
    *,
    torque_nm: float,
    direction_sign: float = 1.0,
) -> tuple[LoadPhase, ...]:
    if direction_sign not in {-1.0, 1.0}:
        raise MechanismValidationError(
            "BICYCLE_CHAIN_TASK_INPUT_DIRECTION_INVALID",
            "bicycle crank load direction must be exactly forward or reverse",
        )
    axis = tuple(
        direction_sign * float(value) for value in spec.axis_abs
    )
    phases: list[LoadPhase] = []
    for phase_id, wrench_kind, duration_s in _task_program_values(spec):
        phases.append(
            LoadPhase(
                phase_id=phase_id,
                wrench_kind=wrench_kind,
                direction_abs=axis if wrench_kind == "torque" else None,
                magnitude_si=torque_nm if wrench_kind == "torque" else 0.0,
                duration_s=duration_s,
            )
        )
    return tuple(phases)


def _case(
    spec: BicycleChainSpec,
    *,
    load_case_id: str,
    program: str,
    purpose: str,
    phases: tuple[LoadPhase, ...],
) -> LoadCase:
    return LoadCase(
        load_case_id=load_case_id,
        family="compound",
        program=program,
        purpose=purpose,
        moving_mass_kg=_moving_mass(spec),
        step_s=_STEP_S,
        sample_interval_s=_SAMPLE_INTERVAL_S,
        phases=phases,
    )


def default_bicycle_chain_load_cases(
    plan: MechanismPlan,
) -> tuple[LoadCase, ...]:
    """Return the exact r04 task program and its approved activation cases."""

    spec = _spec_for_plan(plan)
    (held_out_torque_nm,) = _approved_evidence_torques(spec)
    task_phases = _roundtrip_phases(spec, torque_nm=_NOMINAL_TORQUE_NM)

    return (
        _case(
            spec,
            load_case_id="task_bicycle_powered_roundtrip",
            program="task_bicycle_powered_roundtrip",
            purpose="nominal",
            phases=task_phases,
        ),
        _case(
            spec,
            load_case_id="held_out_positive_crank_load",
            program="task_bicycle_powered_roundtrip",
            purpose="held_out",
            phases=_roundtrip_phases(
                spec,
                torque_nm=held_out_torque_nm,
            ),
        ),
        _case(
            spec,
            load_case_id="zero_input_energy_audit",
            program="bicycle_zero_input_home_drift",
            purpose="adversarial",
            phases=(
                LoadPhase(
                    "zero_state_no_input_drift",
                    "none",
                    None,
                    0.0,
                    0.60,
                ),
            ),
        ),
        _case(
            spec,
            load_case_id="reverse_or_resisting_crank",
            program=BICYCLE_CHAIN_CRANK_PERTURBATION_PROGRAM,
            purpose="adversarial",
            phases=_roundtrip_phases(
                spec,
                torque_nm=BICYCLE_CHAIN_CRANK_PERTURBATION_TORQUE_NM,
                direction_sign=-1.0,
            ),
        ),
    )
